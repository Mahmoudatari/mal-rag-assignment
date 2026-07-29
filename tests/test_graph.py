"""The compiled graph, end to end, with every provider faked.

The node unit tests cover each `run()` in isolation. What they cannot see is
what only exists once the graph is compiled and checkpointed:

- **The per-turn reset.** The checkpointer persists one `State` per `thread_id`
  and nodes return only the keys they changed, so turn 2 starts from turn 1's
  dict. Without `redact` clearing the turn-scoped keys, a conversational second
  turn reaches `generate` still carrying turn 1's `chunks` and is answered as if
  it were grounded, and a turn after an exhausted retrieval starts at
  `attempts == 2` with no retries left. That is a two-turn property; a single
  `run()` call cannot show it.
- **The retry cycle.** `grade → reformulate → retrieve → rerank → grade` is a
  real loop, and `attempts < max_retrieval_attempts` is the only thing that ends
  it. The count that matters is how many times `retrieve` actually ran.
- **The edge predicates.** `after_router` maps route "answer" onto the
  `generate` node, and `after_grade` picks between three successors — wiring
  that lives in `rag/graph.py`, not in any node.

Every provider is faked per node module, so the queues cannot interleave: each
node's `FakeAsyncOpenAI` indexes its own call count, and `FakeAsyncOpenAI`
replays its last queued response once drained, which is what makes "the grader
always says no" a single queued reply.

The nodes are `async def`, so the graph is driven with `ainvoke` — a compiled
graph holding async nodes has no working sync `invoke`. `asyncio.run` per turn
is safe here because nothing in the harness outlives a loop: the fakes are
in-memory, and `InMemorySaver` and the compiled graph are both loop-agnostic,
which is what lets a two-turn test reuse one compiled graph across two runs.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from core.config import get_settings
from core.embeddings import EmbeddingClient
from core.llm import LLMClient
from core.rerank import RerankClient
from kb.store import Match
from rag import graph as graph_module
from rag.nodes import generate, grade, reformulate, rerank, retrieve, router
from tests.fakes import FakeAsyncOpenAI, embedding_response, rerank_response, response

MATCHES = [
    Match(
        chunk_id=f"murabaha-everyday-finance#{n:03d}",
        doc="murabaha-everyday-finance",
        section="Frequently Asked Questions",
        text=f"passage {n} about cost-plus financing",
        score=0.9 - n / 100,
    )
    for n in range(1, 6)
]


def _route(route: str, *, search_query: str = "", reason: str = "test") -> str:
    return json.dumps({"route": route, "reason": reason, "search_query": search_query})


def _verdict(relevant: bool, note: str = "") -> str:
    return json.dumps({"relevant": relevant, "note": note})


class Harness:
    """Every faked provider, plus the counters the assertions are actually about."""

    def __init__(self) -> None:
        self.searches: list[int] = []  # one entry per retrieve, holding its limit

    async def search(self, vector, limit):
        self.searches.append(limit)
        return MATCHES[:limit]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch):
    """Wire fakes into each node module and return the knobs a test needs.

    Patching the accessor *as imported into the node module* is what keeps the
    nodes themselves free of test seams — they still call `fast_llm()` and
    friends exactly as they do in production.
    """
    state = Harness()

    def build(*, routes, verdicts, rewrites=(), answers=("An answer [1].",)):
        router_fake = FakeAsyncOpenAI(responses=[response(r) for r in routes])
        grade_fake = FakeAsyncOpenAI(responses=[response(v) for v in verdicts])
        reformulate_fake = FakeAsyncOpenAI(
            responses=[response(json.dumps({"search_query": q})) for q in rewrites]
            or [response(json.dumps({"search_query": "a rewritten query"}))]
        )
        generate_fake = FakeAsyncOpenAI(responses=[response(a) for a in answers])
        embed_fake = FakeAsyncOpenAI(responses=[embedding_response([[0.1, 0.2, 0.3]])])
        rerank_fake = FakeAsyncOpenAI(
            responses=[rerank_response({1: 0.9, 0: 0.5, 2: 0.4, 3: 0.2}, search_units=1, cost=0.002)]
        )

        monkeypatch.setattr(
            router, "fast_llm", lambda: LLMClient("g", "fast", api_key="t", async_client=router_fake)
        )
        monkeypatch.setattr(
            grade, "fast_llm", lambda: LLMClient("g", "fast", api_key="t", async_client=grade_fake)
        )
        monkeypatch.setattr(
            reformulate, "fast_llm", lambda: LLMClient("g", "fast", api_key="t", async_client=reformulate_fake)
        )
        monkeypatch.setattr(
            generate, "answer_llm", lambda: LLMClient("g", "answer", api_key="t", async_client=generate_fake)
        )
        monkeypatch.setattr(
            retrieve,
            "embedding_client",
            lambda: EmbeddingClient("g", "emb", api_key="t", dimensions=3, async_client=embed_fake),
        )
        monkeypatch.setattr(retrieve, "asearch", state.search)
        monkeypatch.setattr(
            rerank,
            "rerank_client",
            lambda: RerankClient("cohere", "rr", api_key="t", async_client=rerank_fake),
        )

        state.router_fake = router_fake
        state.embed_fake = embed_fake
        state.rerank_fake = rerank_fake
        state.generate_fake = generate_fake
        return state

    return build


def _run(compiled, message: str, thread: str = "t1", account_id: str = "") -> dict:
    """One turn. `ainvoke` because six of the ten nodes are coroutines.

    Each call gets its own loop, which the checkpointer and the fakes are both
    indifferent to — the state that crosses turns is a plain dict in
    `InMemorySaver`, not a connection bound to a loop. `account_id` is always
    in the payload ("" when absent), mirroring what `app/` sends.
    """
    return asyncio.run(
        compiled.ainvoke(
            {"raw_query": message, "account_id": account_id},
            config={"configurable": {"thread_id": thread}},
        )
    )


def _nodes(result: dict) -> list[str]:
    return [entry["node"] for entry in result["usage_log"]]


# --- the two-turn property the checkpointer creates -------------------------


def test_second_turn_starts_clean_but_keeps_history(harness) -> None:
    """The bug this guards: turn 2 inheriting turn 1's retrieval.

    Turn 1 answers from context; turn 2 is conversational and must reach
    `generate` with no chunks, or an ungrounded reply is composed against the
    previous question's passages and cited as if it were grounded.
    """
    harness(
        routes=[_route("retrieve", search_query="murabaha profit rate"), _route("answer")],
        verdicts=[_verdict(True)],
        answers=["Murabaha is a cost-plus sale [1].", "You're welcome!"],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    first = _run(compiled, "what is the murabaha profit rate?")
    assert first["outcome"] == "answered"
    assert _nodes(first) == ["router", "retrieve", "rerank", "grade", "generate"]
    assert first["references"][0]["chunk_id"].startswith("murabaha-everyday-finance#")
    assert len(first["history"]) == 2

    second = _run(compiled, "thanks!")

    assert second["chunks"] == [], "turn 1's passages must not survive into turn 2"
    assert second["candidate_log"] == [], "turn 1's candidate ids must not survive either"
    assert second["references"] == []
    # Inherited, this would have grade and reformulate judging turn 2's
    # retrieval against turn 1's question.
    assert second["resolved_query"] == ""
    assert second["attempts"] == 0
    assert _nodes(second) == ["router", "generate"], "usage_log is per turn, not per session"
    assert len(second["history"]) == 4, "history is the one key that carries across turns"
    assert second["history"][0]["content"] == "what is the murabaha profit rate?"


def test_exhausted_retries_do_not_leak_into_the_next_turn(harness) -> None:
    """Turn 1 burns every retry; turn 2 must still get its own full budget."""
    h = harness(
        routes=[_route("retrieve", search_query="q1"), _route("retrieve", search_query="q2")],
        verdicts=[_verdict(False, "no rate given"), _verdict(False, "still nothing"),
                  _verdict(False, "nothing"), _verdict(True)],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    first = _run(compiled, "what is the fee?")
    assert first["outcome"] == "no_answer"
    assert first["attempts"] == get_settings().max_retrieval_attempts
    # One id-list per search: retries append to candidate_log while replacing
    # `chunks`, so the trace shows every candidate set the loop burned through.
    assert len(first["candidate_log"]) == len(h.searches)

    second = _run(compiled, "and the tenor?")

    # Turn 2's grader says yes first time, so `attempts` staying 0 is only
    # meaningful alongside the reset: inherited, it would have started at 2.
    assert second["attempts"] == 0
    assert second["outcome"] == "answered"
    assert len(second["candidate_log"]) == 1, "turn 2 logs its own single search only"


# --- the retry cycle --------------------------------------------------------


def test_failed_grade_reformulates_and_retries(harness) -> None:
    state = harness(
        routes=[_route("retrieve", search_query="murabaha fee")],
        verdicts=[_verdict(False, "the passages give terms, not the fee"), _verdict(True)],
        rewrites=["murabaha cost-plus profit margin disclosure"],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    result = _run(compiled, "what fee does murabaha carry?")

    assert result["outcome"] == "answered"
    assert result["attempts"] == 1, "reformulate is the only node that increments this"
    assert len(state.searches) == 2, "the retry must re-run retrieval, not reuse the graded set"
    assert result["tried_queries"] == ["murabaha fee", "murabaha cost-plus profit margin disclosure"]
    # `search_query` moves with the loop, `resolved_query` does not — the second
    # grade judged the retry against the customer's question, not the rewrite.
    assert result["search_query"] == "murabaha cost-plus profit margin disclosure"
    assert result["resolved_query"] == "murabaha fee"
    assert _nodes(result) == [
        "router", "retrieve", "rerank", "grade", "reformulate", "retrieve", "rerank", "grade", "generate"
    ]


def test_retry_loop_terminates_at_max_attempts(harness) -> None:
    """The grader never relents — `after_grade` must end the cycle, not spin."""
    state = harness(
        routes=[_route("retrieve", search_query="q")],
        verdicts=[_verdict(False, "not in these passages")],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    result = _run(compiled, "something the KB does not cover")

    attempts = get_settings().max_retrieval_attempts
    assert result["attempts"] == attempts
    assert len(state.searches) == attempts + 1, "one initial retrieval plus one per retry"
    assert result["outcome"] == "no_answer"
    assert "support@mal.example" in result["answer"]
    assert result["references"] == []


# --- the router's three-way branch ------------------------------------------


def test_refusal_never_touches_retrieval(harness) -> None:
    state = harness(routes=[_route("refuse", reason="not about Islamic finance")], verdicts=[])
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    result = _run(compiled, "what's the weather in Dubai?")

    assert result["outcome"] == "refused"
    assert state.searches == []
    assert state.embed_fake.calls == []
    assert state.rerank_fake.calls == []
    assert state.generate_fake.calls == [], "the refusal is a template, not a paid call"
    assert _nodes(result) == ["router"]


def test_answer_route_generates_without_context(harness) -> None:
    """`after_router` maps route "answer" onto `generate` — there is no separate node."""
    state = harness(routes=[_route("answer")], verdicts=[], answers=["Hello! I can help with Mal's products."])
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    result = _run(compiled, "hello")

    assert result["outcome"] == "answered"
    assert result["chunks"] == []
    assert state.searches == []
    system_prompt = state.generate_fake.last_call["messages"][0]["content"]
    assert system_prompt == generate.UNGROUNDED_SYSTEM, "no chunks means the ungrounded prompt"


# --- account context ---------------------------------------------------------


def test_account_context_reaches_the_prompts_and_does_not_linger(harness) -> None:
    """Turn 1 sends an account id: the router sees the product names, generate
    sees the full masked record, and no provider ever sees the raw number.
    Turn 2 on the same thread sends none: the account node's unconditional
    write must clear it — the account-shaped version of the chunks reset.

    Turn 1's answer restates the record on purpose. Clearing `account` alone
    leaves that sentence in `history`, which both later prompts replay, so the
    contract id survives the reset that was supposed to remove it — the whole
    reason the account node scopes history to `history_account`."""
    state = harness(
        routes=[
            _route("retrieve", search_query="murabaha remaining balance"),
            _route("retrieve", search_query="murabaha early settlement"),
        ],
        verdicts=[_verdict(True), _verdict(True)],
        answers=[
            "Seven instalments remain on contract MUR-2026-0417, AED 24,500 outstanding [1].",
            "Early settlement is allowed [1].",
        ],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    raw_id = "MAL-1001-2200-4417"
    first = _run(compiled, "how much is left on my contract?", account_id=raw_id)

    assert first["outcome"] == "answered"
    assert first["account"]["masked_id"] == "MAL-****-****-4417"
    router_prompt = state.router_fake.last_call["messages"][-1]["content"]
    assert "Murabaha everyday finance, Wakala savings" in router_prompt
    generate_prompt = state.generate_fake.last_call["messages"][-1]["content"]
    assert "Customer account context" in generate_prompt
    assert "MAL-****-****-4417" in generate_prompt
    assert "MUR-2026-0417" in generate_prompt, "the record's fields, not just its id"
    # The full number is a lookup key, not content — same containment the PII
    # test asserts for raw_query.
    for fake in (state.router_fake, state.embed_fake, state.rerank_fake, state.generate_fake):
        assert raw_id not in repr(fake.calls)

    second = _run(compiled, "can I settle early?")

    assert second["account"] is None, "turn 2 sent no id — turn 1's account must not survive"
    assert second["history_account"] == ""
    assert "account holds" not in state.router_fake.last_call["messages"][-1]["content"]
    assert "Customer account context" not in state.generate_fake.last_call["messages"][-1]["content"]
    # The rendered block being gone is not the property — the figures are. Turn
    # 1's answer restated them and went into history, which the router and
    # generate are both handed in full, so this is what the account node's
    # history drop is for. The whole message list, not just the last prompt.
    for fake in (state.router_fake, state.generate_fake):
        messages = json.dumps(fake.last_call["messages"])
        assert "MUR-2026-0417" not in messages, "turn 1's contract id must not replay via history"
        assert "24,500" not in messages
    assert len(second["history"]) == 2, "the conversation restarts, it does not accumulate"


# --- PII, across the whole graph --------------------------------------------


def test_raw_pii_reaches_no_provider_and_no_final_state(harness) -> None:
    """The redaction eval proves the node masks; this proves nothing re-introduces it."""
    state = harness(
        routes=[_route("retrieve", search_query="murabaha eligibility")],
        verdicts=[_verdict(True)],
        answers=["Eligibility depends on income [1]."],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    secret = "784-1990-1234567-6"
    result = _run(compiled, f"my Emirates ID is {secret}, am I eligible for murabaha?")

    for fake in (state.embed_fake, state.rerank_fake, state.generate_fake):
        assert secret not in repr(fake.calls)
    # The whole final state — history, usage_log, spans (offsets, never values)
    # and raw_query itself, which redact overwrites in its own superstep. This
    # dict is exactly what the checkpointer persists to Postgres per thread.
    assert result["raw_query"] == "", "redact must discard the raw text it masked"
    assert secret not in repr(result), "checkpointed state must not retain raw PII"
