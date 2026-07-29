"""Trace schema: one flat record per request.

The brief requires four fields — end-to-end latency, retrieved chunk IDs, token
usage, and a relevance score for the retrieved context. The rest of this record
exists because those four alone cannot explain a bad answer: they say retrieval
returned nothing useful without saying which route was taken, how many times the
query was reformulated, or whether the grader agreed.

Three properties are deliberate:

- **Flat primitives, no internal imports.** This module is a leaf — it does not
  import `core`, `rag`, `kb` or `app`, so the token counts are ints rather than
  a `core.llm.Usage`. Reusing that type would pull the `openai` SDK into the
  logging path and make the trace schema move whenever a client does.
- **`from_state` reads final state, and the nodes know nothing about it.**
  `rag/state.py` records each provider call into `usage_log` as plain JSON
  precisely so this layer can total it afterwards, which is what keeps
  `rag/nodes/` free of a tracer import. State arrives as a `Mapping`, not as
  `rag.state.State` — a TypedDict is a `dict` at runtime, so the contract costs
  no import and no cycle. Every field defaults, so a request that failed half
  way still produces a record.
- **Nothing here can hold raw PII.** `redact` runs first and everything
  downstream reads the masked query, so `masked_query` is safe by construction —
  the field is named for the invariant so passing `raw_query` reads as the bug
  it is. `pii_kinds` records *what* was removed, never the values, matching
  `pii.PiiSpan`, which carries no matched text either.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any


def _now() -> str:
    """UTC, ISO-8601, seconds precision — sortable and unambiguous in a log."""
    return datetime.now(UTC).replace(microsecond=0).isoformat()


class Stopwatch:
    """Monotonic timer for end-to-end latency.

    `perf_counter` rather than `time()`: latency must not go backwards when the
    host adjusts its clock mid-request.
    """

    __slots__ = ("_start",)

    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def ms(self) -> float:
        return round((time.perf_counter() - self._start) * 1000, 2)


@dataclass(frozen=True, slots=True)
class Trace:
    """One request, start to finish.

    Every field defaults, so a request that dies half way still emits a usable
    record rather than raising inside the logger.
    """

    # --- identity ------------------------------------------------------
    session_id: str = ""
    timestamp: str = field(default_factory=_now)
    latency_ms: float = 0.0  # required by the brief

    # --- request -------------------------------------------------------
    # The PII-masked query. Never `raw_query`.
    masked_query: str = ""
    # Entity kinds only (PERSON, EMIRATES_ID, ...) — evidence that redaction
    # ran, with none of what it removed.
    pii_kinds: list[str] = field(default_factory=list)
    # The masked id from the resolved account record ("MAL-****-****-4417"),
    # empty when the turn carried no account context. Read from `account`,
    # never from `account_id` — the record carries no full number to leak.
    account_id_masked: str = ""

    # --- routing -------------------------------------------------------
    route: str = ""  # retrieve | refuse | answer
    route_reason: str = ""
    attempts: int = 0  # reformulate loops, 0 on a first-try hit

    # --- retrieval -----------------------------------------------------
    chunk_ids: list[str] = field(default_factory=list)  # required by the brief
    # What the answer actually cited, which is a subset: `generate` drops [n]
    # markers that point past the context it was given.
    cited_chunk_ids: list[str] = field(default_factory=list)
    # Required by the brief. The top-ranked chunk's score — the cross-encoder's
    # when reranking ran, its cosine otherwise. Both underlying values are kept
    # because they measure different things, and neither gates control flow.
    relevance_score: float = 0.0
    cosine_score: float = 0.0  # top-ranked chunk's vector similarity
    # None means no cross-encoder score reached the chunk: reranking was off, or
    # it failed open and left the candidates in cosine order.
    rerank_score: float | None = None
    # The grader's verdict — the value that actually decided the branch.
    # None when grading never ran (refusal or no-retrieval path).
    graded_relevant: bool | None = None

    # --- cost ----------------------------------------------------------
    # Totalled across every billed call: router, grader, query embedding,
    # rerank, generate — and each retry's repeat of the middle three.
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0  # required by the brief
    # Reranking bills per search unit and reports zero tokens, so a trace that
    # logged only tokens would report it as free. See core/rerank.py.
    search_units: int = 0
    cost_usd: float = 0.0
    # The per-call breakdown behind those totals, in call order — `usage_log`
    # verbatim. A total alone cannot show that a retrieval loop ran twice.
    calls: list[dict[str, Any]] = field(default_factory=list)

    # --- outcome -------------------------------------------------------
    outcome: str = ""  # answered | refused | no_answer | error
    error: str = ""  # exception summary when outcome == "error"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_state(
        cls,
        state: Mapping[str, Any],
        *,
        latency_ms: float,
        error: str = "",
    ) -> Trace:
        """Build the record from the graph's final state.

        Reads with `.get` throughout: `State` is `total=False`, and a turn that
        refused or died never sets most of these. Missing is the normal case,
        not an error — a trace must come out either way.
        """
        chunks = list(state.get("chunks") or [])
        # Position 0 is the best chunk: cosine order out of `retrieve`, the
        # cross-encoder's order once `rerank` has reordered them.
        top: Mapping[str, Any] = chunks[0] if chunks else {}
        cosine = float(top.get("score", 0.0))
        reranked = top.get("rerank_score")
        rerank_score = None if reranked is None else float(reranked)

        calls = [dict(call) for call in state.get("usage_log") or []]
        outcome = state.get("outcome") or ("error" if error else "")

        return cls(
            session_id=state.get("session_id", ""),
            latency_ms=latency_ms,
            masked_query=state.get("query", ""),
            pii_kinds=sorted({span["kind"] for span in state.get("pii_spans") or []}),
            account_id_masked=str((state.get("account") or {}).get("masked_id", "")),
            route=state.get("route", ""),
            route_reason=state.get("route_reason", ""),
            attempts=state.get("attempts", 0),
            chunk_ids=[chunk["chunk_id"] for chunk in chunks],
            cited_chunk_ids=[ref["chunk_id"] for ref in state.get("references") or []],
            relevance_score=cosine if rerank_score is None else rerank_score,
            cosine_score=cosine,
            rerank_score=rerank_score,
            graded_relevant=state.get("relevant"),
            prompt_tokens=sum(call.get("prompt_tokens", 0) for call in calls),
            completion_tokens=sum(call.get("completion_tokens", 0) for call in calls),
            total_tokens=sum(call.get("total_tokens", 0) for call in calls),
            search_units=sum(call.get("search_units", 0) for call in calls),
            cost_usd=round(sum(call.get("cost", 0.0) for call in calls), 8),
            calls=calls,
            outcome=outcome,
            error=error,
        )
