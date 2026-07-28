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

        state.embed_fake = embed_fake
        state.rerank_fake = rerank_fake
        state.generate_fake = generate_fake
        return state

    return build


def _run(compiled, message: str, thread: str = "t1") -> dict:
    """One turn. `ainvoke` because six of the nine nodes are coroutines.

    Each call gets its own loop, which the checkpointer and the fakes are both
    indifferent to — the state that crosses turns is a plain dict in
    `InMemorySaver`, not a connection bound to a loop.
    """
    return asyncio.run(
        compiled.ainvoke({"raw_query": message}, config={"configurable": {"thread_id": thread}})
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
    assert second["references"] == []
    assert second["attempts"] == 0
    assert _nodes(second) == ["router", "generate"], "usage_log is per turn, not per session"
    assert len(second["history"]) == 4, "history is the one key that carries across turns"
    assert second["history"][0]["content"] == "what is the murabaha profit rate?"


def test_exhausted_retries_do_not_leak_into_the_next_turn(harness) -> None:
    """Turn 1 burns every retry; turn 2 must still get its own full budget."""
    harness(
        routes=[_route("retrieve", search_query="q1"), _route("retrieve", search_query="q2")],
        verdicts=[_verdict(False, "no rate given"), _verdict(False, "still nothing"),
                  _verdict(False, "nothing"), _verdict(True)],
    )
    compiled = graph_module.build_graph(checkpointer=InMemorySaver())

    first = _run(compiled, "what is the fee?")
    assert first["outcome"] == "no_answer"
    assert first["attempts"] == get_settings().max_retrieval_attempts

    second = _run(compiled, "and the tenor?")

    # Turn 2's grader says yes first time, so `attempts` staying 0 is only
    # meaningful alongside the reset: inherited, it would have started at 2.
    assert second["attempts"] == 0
    assert second["outcome"] == "answered"


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
    assert secret not in repr(result["history"])
    assert secret not in repr(result["usage_log"])
    assert secret not in repr(result["pii_spans"]), "spans carry offsets, never the value"
