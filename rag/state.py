"""Graph state.

Deliberately free of LangGraph imports — this is a plain TypedDict so nodes and
evals can construct state without compiling a graph.
"""

from typing import Any, Literal, NotRequired, TypedDict

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
    # The full account number from the request. Like raw_query it is a lookup
    # input, not content: nothing renders it into a prompt, a trace or a
    # response — those read the masked fields inside `account` instead.
    account_id: str

    # --- across turns ---
    # The one key `redact` deliberately leaves alone. Everything else it touches
    # is turn-scoped, and the checkpointer persists this whole dict per
    # thread_id, so without that reset turn N+1 would inherit turn N's chunks,
    # attempts and citations.
    history: list[ChatMessage]
    # `masked_id` of the account context `history` was built under, "" for none.
    # History is the one place account-derived *text* outlives the turn that
    # produced it: `generate` appends its own answer, and that answer restates
    # whatever the rendered record said — contract reference, balance, arrears.
    # Clearing `account` does not unsay them, so those figures keep replaying
    # into every later router and generate prompt under an account that no
    # longer describes them. The account node compares this against the context
    # it just resolved and drops `history` when they disagree. Written on every
    # path like `account`, so it needs no place in redact's reset list.
    history_account: str

    # --- redact ---
    query: str  # PII-masked; every downstream node reads this, never raw_query
    pii_spans: list[PIISpan]

    # --- account ---
    # `accounts.lookup` result: the customer's synthetic record, or None when
    # the request carried no id or an unknown one. Not in redact's reset list
    # because the account node runs on every path and always writes it, so a
    # turn without an id cannot inherit the previous turn's account from the
    # checkpointer. Carries `masked_id`, never the full account number.
    account: dict[str, Any] | None

    # --- router ---
    route: Route
    route_reason: str
    search_query: str  # history-resolved; empty unless route == "retrieve"
    # The router's history-resolved standalone question ("can I use it for a
    # home?" → "can Mal Everyday Murabaha be used for home financing"), written
    # once per turn by the router and never overwritten by reformulate. This is
    # the stable question `grade` and `reformulate` work against, while
    # `search_query` mutates on every retry: grading against the rewrite would
    # let the retry loop approve its own drift, and rewriting from the raw turn
    # loses the subject the pronoun stood for.
    resolved_query: str

    # --- retrieve / grade ---
    chunks: list[Chunk]  # empty on the no-retrieval path; generate handles that
    # Chunk ids per retrieval run, pre-rerank — one inner list per search, so a
    # reformulate retry appends rather than overwrites (unlike `chunks`, which a
    # retry must replace). Ids only: the checkpointer serializes this dict per
    # thread, and 20 full texts per run is ~30KB of bloat with no reader. Like
    # `score` and `rerank_score`: trace/eval data, never gates control flow.
    candidate_log: list[list[str]]
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
