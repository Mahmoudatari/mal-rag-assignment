"""Graph state.

Deliberately free of LangGraph imports — this is a plain TypedDict so nodes and
evals can construct state without compiling a graph.
"""

from typing import Literal, NotRequired, TypedDict

Route = Literal["retrieve", "refuse", "answer"]
Outcome = Literal["answered", "refused", "no_answer"]


class Chunk(TypedDict):
    chunk_id: str
    doc: str
    text: str
    score: float  # cosine similarity, i.e. 1 - (pgvector <=> distance)
    # Cross-encoder score, present only once `rerank` has run — its absence is
    # how a trace shows reranking was off or failed. Trace data like `score`:
    # never gates control flow, and never serialized into the grader's prompt.
    rerank_score: NotRequired[float]


class ChatMessage(TypedDict):
    """One prior turn. Shape-compatible with `core.llm.Message`, redeclared
    rather than imported so `state.py` stays free of the OpenAI SDK.

    Content is always the PII-masked text — `raw_query` never enters history,
    which is checkpointed to Postgres for the life of the session.
    """

    role: str  # "user" | "assistant"
    content: str


class UsageEntry(TypedDict):
    """One provider call, appended by the node that made it.

    Plain JSON so the checkpointer can serialize it. The observability layer
    totals these from final state rather than nodes importing a tracer, which
    keeps `rag/nodes/` free of a dependency on `observability/`.
    """

    node: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost: float
    # Rerank only: cohere/rerank-v3.5 bills on search units and reports zero
    # tokens, so a trace reading tokens alone would report reranking as free.
    search_units: NotRequired[int]


class PIISpan(TypedDict):
    """Type and offsets only. Raw matched values are never stored or logged."""

    # Presidio entity names, matching the [KIND] placeholders left in `query`:
    # EMIRATES_ID | ACCOUNT_NUMBER | IBAN_CODE | PERSON | EMAIL_ADDRESS | ...
    kind: str
    start: int  # offsets index raw_query, which is discarded after redaction
    end: int


class Reference(TypedDict):
    n: int  # the inline [n] marker used in answer text
    doc: str
    chunk_id: str


class State(TypedDict, total=False):
    # --- turn input ---
    session_id: str
    raw_query: str

    # --- across turns ---
    # The only key `redact` does not reset. Everything else here is turn-scoped,
    # and the checkpointer persists this whole dict per thread_id, so without
    # that reset turn N+1 would inherit turn N's chunks, attempts and citations.
    history: list[ChatMessage]

    # --- redact ---
    query: str  # PII-masked; every downstream node reads this, never raw_query
    pii_spans: list[PIISpan]

    # --- router ---
    route: Route
    route_reason: str
    search_query: str  # history-resolved; empty unless route == "retrieve"

    # --- retrieve / grade ---
    chunks: list[Chunk]  # empty on the no-retrieval path; generate handles that
    relevant: bool
    grader_note: str  # why retrieval failed; feeds reformulate
    attempts: int  # incremented ONLY in reformulate
    tried_queries: list[str]  # seeded by router, appended by reformulate

    # --- generate ---
    answer: str
    references: list[Reference]
    outcome: Outcome

    # --- observability ---
    # One entry per provider call, in call order. Retries append rather than
    # overwrite, so a trace shows the retrieval loop's real cost.
    usage_log: list[UsageEntry]
