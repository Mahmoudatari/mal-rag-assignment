"""Unit tests for the trace schema and emitter.

Tracing has the same failure shape as chunking: nothing raises when it goes
wrong. A dropped field, a trace that never reaches stdout, or a raw identifier
that rides along in a query string all leave a working API behind them. So these
tests pin the properties that would otherwise fail silently:

- **the four fields the brief requires are in the emitted JSON** — under those
  names, so a rename cannot quietly delete a deliverable;
- **`from_state` reads the keys `rag/state.py` actually defines** — the two
  files are coupled by key name rather than by import, so nothing but a test
  stands between a renamed state key and a trace field that silently reads zero;
- **one request emits exactly one parseable line**, and a partially-filled trace
  still emits rather than raising inside the logger;
- **no PII survives into the log**, tested through the real redactor rather than
  by trusting the field name;
- **the package stays a leaf**, checked against the imports rather than the
  docstring that claims it.
"""

from __future__ import annotations

import ast
import io
import json
import time
from datetime import datetime
from pathlib import Path

import pytest

from observability import Stopwatch, Trace, emit, render

# The four the brief names explicitly. Everything else in the record is ours.
REQUIRED_FIELDS = ("latency_ms", "chunk_ids", "total_tokens", "relevance_score")


def emitted(trace: Trace) -> dict:
    """Emit to a buffer and hand back the parsed object."""
    buffer = io.StringIO()
    emit(trace, stream=buffer)
    return json.loads(buffer.getvalue())


# --- the required record ------------------------------------------------


@pytest.mark.parametrize("name", REQUIRED_FIELDS)
def test_brief_required_field_survives_to_the_log(name: str) -> None:
    assert name in emitted(Trace())


def test_account_context_logs_the_masked_id_and_nothing_else() -> None:
    """The trace reads `account`, whose record carries no full number — so the
    masked id is what a turn with account context logs, and a turn without one
    logs an empty string rather than omitting the field."""
    with_account = Trace.from_state(
        {"account": {"masked_id": "MAL-****-****-4417", "holdings": [{"product": "x"}]}},
        latency_ms=1.0,
    )
    assert emitted(with_account)["account_id_masked"] == "MAL-****-****-4417"
    assert "holdings" not in emitted(with_account)

    without = Trace.from_state({"account": None}, latency_ms=1.0)
    assert emitted(without)["account_id_masked"] == ""


def test_a_full_trace_logs_every_field_it_was_given() -> None:
    trace = Trace(
        session_id="s-1",
        latency_ms=812.5,
        masked_query="is [PERSON] able to settle murabaha early?",
        pii_kinds=["PERSON"],
        route="retrieve",
        route_reason="asks about murabaha settlement",
        attempts=1,
        chunk_ids=["murabaha-everyday-finance#027", "murabaha-everyday-finance#031"],
        relevance_score=0.91,
        cosine_score=0.76,
        graded_relevant=True,
        prompt_tokens=1200,
        completion_tokens=180,
        total_tokens=1380,
        search_units=1,
        cost_usd=0.0021,
        outcome="answered",
    )
    logged = emitted(trace)

    assert logged["chunk_ids"] == list(trace.chunk_ids)
    assert logged["route"] == "retrieve"
    assert logged["attempts"] == 1
    assert logged["graded_relevant"] is True
    assert logged["outcome"] == "answered"
    assert logged["relevance_score"] == pytest.approx(0.91)


def test_to_dict_round_trips_through_the_constructor() -> None:
    """The emitted object is the record, not a lossy view of it."""
    trace = Trace(session_id="s-1", latency_ms=4.2, chunk_ids=["a#001"])
    assert Trace(**trace.to_dict()) == trace


# --- building the record from final state --------------------------------


def answered_state() -> dict:
    """A turn that retrieved, reranked, graded relevant and answered."""
    return {
        "session_id": "s-1",
        "raw_query": "can Fatima Al Mansouri settle early?",
        "query": "can [PERSON] settle early?",
        "pii_spans": [{"kind": "PERSON", "start": 4, "end": 22}],
        "route": "retrieve",
        "route_reason": "murabaha early settlement",
        "search_query": "murabaha early settlement rebate",
        "chunks": [
            {
                "chunk_id": "murabaha-everyday-finance#027",
                "doc": "murabaha-everyday-finance",
                "text": "...",
                "score": 0.76,
                "rerank_score": 0.91,
            },
            {
                "chunk_id": "murabaha-everyday-finance#031",
                "doc": "murabaha-everyday-finance",
                "text": "...",
                "score": 0.71,
                "rerank_score": 0.64,
            },
        ],
        "relevant": True,
        "attempts": 0,
        "answer": "Yes [1].",
        "references": [
            {"n": 1, "doc": "murabaha-everyday-finance", "chunk_id": "murabaha-everyday-finance#027"}
        ],
        "outcome": "answered",
        "usage_log": [
            {
                "node": "router",
                "model": "google/gemini-3.5-flash-lite",
                "prompt_tokens": 300,
                "completion_tokens": 40,
                "total_tokens": 340,
                "cost": 0.0001,
            },
            {
                "node": "rerank",
                "model": "cohere/rerank-v3.5",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.002,
                "search_units": 1,
            },
            {
                "node": "generate",
                "model": "google/gemini-3.6-flash",
                "prompt_tokens": 1200,
                "completion_tokens": 140,
                "total_tokens": 1340,
                "cost": 0.0009,
            },
        ],
    }


def test_from_state_maps_the_whole_turn() -> None:
    trace = Trace.from_state(answered_state(), latency_ms=812.5)

    assert trace.session_id == "s-1"
    assert trace.latency_ms == 812.5
    assert trace.masked_query == "can [PERSON] settle early?"
    assert trace.pii_kinds == ["PERSON"]
    assert trace.route == "retrieve"
    assert trace.chunk_ids == [
        "murabaha-everyday-finance#027",
        "murabaha-everyday-finance#031",
    ]
    assert trace.cited_chunk_ids == ["murabaha-everyday-finance#027"]
    assert trace.graded_relevant is True
    assert trace.outcome == "answered"


def test_from_state_totals_every_billed_call() -> None:
    trace = Trace.from_state(answered_state(), latency_ms=1.0)

    assert trace.prompt_tokens == 1500
    assert trace.completion_tokens == 180
    assert trace.total_tokens == 1680
    assert trace.search_units == 1, "rerank bills search units, not tokens"
    assert trace.cost_usd == pytest.approx(0.0030)
    assert [call["node"] for call in trace.calls] == ["router", "rerank", "generate"]


def test_from_state_shows_what_a_retry_cost() -> None:
    """`usage_log` appends across the reformulate loop, so the totals include
    the second retrieval — a trace that overwrote would under-report it."""
    state = answered_state()
    state["attempts"] = 1
    state["usage_log"] = [*state["usage_log"], dict(state["usage_log"][0], node="grade")]

    trace = Trace.from_state(state, latency_ms=1.0)

    assert trace.attempts == 1
    assert trace.total_tokens == 2020
    assert len(trace.calls) == 4


def test_relevance_score_prefers_the_cross_encoder() -> None:
    trace = Trace.from_state(answered_state(), latency_ms=1.0)

    assert trace.relevance_score == pytest.approx(0.91)
    assert trace.cosine_score == pytest.approx(0.76)
    assert trace.rerank_score == pytest.approx(0.91)


def test_relevance_score_falls_back_to_cosine_when_reranking_did_not_run() -> None:
    """`rerank` fails open and leaves candidates in cosine order; the missing
    score is the signal, so the trace must not invent one."""
    state = answered_state()
    for chunk in state["chunks"]:
        del chunk["rerank_score"]

    trace = Trace.from_state(state, latency_ms=1.0)

    assert trace.rerank_score is None
    assert trace.relevance_score == pytest.approx(0.76)


def test_from_state_handles_a_turn_that_never_retrieved() -> None:
    """The refusal path sets no chunks, no grade and no references."""
    state = {"session_id": "s-1", "query": "who won the cup?", "route": "refuse", "outcome": "refused"}
    trace = Trace.from_state(state, latency_ms=120.0)

    assert trace.chunk_ids == []
    assert trace.relevance_score == 0.0
    assert trace.graded_relevant is None
    assert trace.outcome == "refused"


def test_from_state_survives_an_empty_state() -> None:
    """A request that died before `redact` still has to produce a record."""
    trace = Trace.from_state({}, latency_ms=3.0, error="ConnectionError")

    assert trace.outcome == "error", "an error with no outcome is still an outcome"
    assert trace.error == "ConnectionError"
    assert trace.total_tokens == 0


def test_from_state_reads_only_keys_the_state_schema_defines() -> None:
    """The one coupling `from_state` cannot get from the type system.

    `observability/` must not import `rag/`, so the two agree by key name alone.
    A rename in `rag/state.py` would leave this module reading a key nobody
    writes — no error, just a trace field stuck at its default.
    """
    from rag.state import Chunk, PIISpan, Reference, State, UsageEntry

    source = Path(__file__).resolve().parent.parent / "observability" / "trace.py"
    read = {
        node.args[0].value
        for node in ast.walk(ast.parse(source.read_text()))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "state"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }

    assert read, "the ast scan found no state.get(...) calls — it has stopped testing anything"
    assert read <= set(State.__annotations__)
    assert {"chunk_id", "score", "rerank_score"} <= set(Chunk.__annotations__)
    assert {"kind"} <= set(PIISpan.__annotations__)
    assert {"chunk_id"} <= set(Reference.__annotations__)
    assert {"prompt_tokens", "completion_tokens", "total_tokens", "cost", "search_units"} <= set(
        UsageEntry.__annotations__
    )


def test_from_state_leaves_raw_query_behind() -> None:
    """`raw_query` is the one key in state that still holds the customer's PII."""
    state = answered_state()
    line = render(Trace.from_state(state, latency_ms=1.0))

    assert "Fatima Al Mansouri" not in line
    assert "[PERSON]" in json.loads(line)["masked_query"]


# --- emission -----------------------------------------------------------


def test_emit_writes_exactly_one_newline_terminated_line() -> None:
    buffer = io.StringIO()
    emit(Trace(masked_query="a\nb"), stream=buffer)
    written = buffer.getvalue()

    assert written.endswith("\n")
    assert written.count("\n") == 1, "a multi-line query must not split the record"
    json.loads(written)


def test_emit_returns_the_line_it_wrote() -> None:
    buffer = io.StringIO()
    returned = emit(Trace(session_id="s-1"), stream=buffer)
    assert buffer.getvalue() == returned + "\n"


def test_emit_defaults_to_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """The deployment reads traces off stdout; nothing configures a handler."""
    emit(Trace(session_id="s-1"))
    assert json.loads(capsys.readouterr().out)["session_id"] == "s-1"


def test_a_half_finished_request_still_emits() -> None:
    """Every field defaults, so the error path logs rather than raising here."""
    logged = emitted(Trace(latency_ms=31.0, outcome="error", error="ConnectionError"))

    assert logged["outcome"] == "error"
    assert logged["chunk_ids"] == []
    assert logged["graded_relevant"] is None, "not graded is not the same as not relevant"


def test_render_is_compact() -> None:
    assert ", " not in render(Trace(chunk_ids=["a#001", "a#002"]))


# --- timing -------------------------------------------------------------


def test_stopwatch_measures_elapsed_milliseconds() -> None:
    watch = Stopwatch()
    time.sleep(0.02)
    assert watch.ms >= 20.0


def test_stopwatch_never_goes_backwards() -> None:
    watch = Stopwatch()
    first = watch.ms
    assert 0.0 <= first <= watch.ms


def test_timestamp_is_utc_iso8601() -> None:
    stamped = datetime.fromisoformat(Trace().timestamp)
    assert stamped.tzinfo is not None
    assert stamped.utcoffset().total_seconds() == 0


# --- cost ---------------------------------------------------------------


def test_reranking_is_not_reported_as_free() -> None:
    """`cohere/rerank-v3.5` bills search units and reports zero tokens. A record
    that carried only token counts would price the call at nothing."""
    logged = emitted(Trace(total_tokens=0, search_units=1, cost_usd=0.002))

    assert logged["search_units"] == 1
    assert logged["cost_usd"] == pytest.approx(0.002)


# --- the PII boundary ---------------------------------------------------


def test_no_pii_reaches_the_log() -> None:
    """Tested through the real redactor: the trace carries what the graph would
    actually put in it, not a hand-written placeholder string."""
    from pii import redact

    secret = "784-1990-1234567-6"
    result = redact(f"My Emirates ID is {secret} — can I settle early?")
    line = render(
        Trace(masked_query=result.text, pii_kinds=sorted(result.kinds), outcome="answered")
    )

    assert secret not in line
    assert secret.replace("-", "") not in line
    assert "EMIRATES_ID" in json.loads(line)["pii_kinds"]


def test_the_trace_records_that_redaction_happened() -> None:
    logged = emitted(Trace(pii_kinds=["EMIRATES_ID", "PERSON"]))
    assert logged["pii_kinds"] == ["EMIRATES_ID", "PERSON"]


# --- the boundary the docstring claims ----------------------------------


def test_observability_imports_nothing_internal() -> None:
    """A leaf by inspection of the imports, not by assertion in a docstring.

    `app/` and `rag/` both import this package; an import back the other way is
    a cycle, and importing `core` would pull the provider SDKs into the logging
    path.
    """
    forbidden = {"app", "core", "kb", "pii", "rag"}
    package = Path(__file__).resolve().parent.parent / "observability"

    for module in sorted(package.glob("*.py")):
        for node in ast.walk(ast.parse(module.read_text(), filename=str(module))):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                roots = [(node.module or "").split(".")[0]]
            else:
                continue
            assert not forbidden & set(roots), f"{module.name} imports {roots}"
