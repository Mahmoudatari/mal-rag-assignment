"""Live smoke tests for the graph: real OpenRouter, real pgvector index.

Deselected by default. Run explicitly: `uv run pytest -m live`.

The faked tests prove the wiring; these prove the *prompts*, which fakes cannot
touch. Three things only a real model can answer:

- does `google/gemini-3.5-flash-lite` honour the strict schema and the
  retrieve-over-refuse bias, or does the cheap tier route badly under pressure;
- does the answering model actually emit `[n]` markers, given that citations are
  parsed out of free text rather than forced by a schema;
- does the ungrounded path hold the line when nothing grounds it — the main
  hallucination surface, and the one place a prompt is all that stands between a
  customer and an invented rate.

Assertions stay behavioural (a route, a marker, a chunk id) rather than exact
strings: a model is allowed to phrase things differently between runs, and a
test that pins wording would fail on prose rather than on a regression.

The graph tests read the deployed index but write nothing to it, so they carry
`live` alone — the `db` marker is for tests that call `prune()` or
`replace_document()`.

The nodes and the graph are async, so every call here is driven through
`asyncio.run`. That makes a fresh event loop per test, which is the one thing
this file has to manage deliberately: both the `AsyncOpenAI` transports and the
async connection pool hold resources bound to the loop that created them. See
`_fresh_async_transports` and `on_index` below.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Coroutine, Iterator
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from core import embeddings, llm, rerank
from core.config import get_settings
from kb import store
from rag.graph import build_graph
from rag.nodes import generate, router

live = pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
needs_db = pytest.mark.skipif(
    not get_settings().database_url, reason="needs DATABASE_URL pointing at the built index"
)

MARKER = re.compile(r"\[(\d+)\]")


@pytest.fixture(autouse=True)
def _fresh_async_transports() -> Iterator[None]:
    """Drop every cached client after each test, because the loop is gone.

    `asyncio.run` closes its loop on the way out, but an `AsyncOpenAI` outlives
    it: `_async_transport` is `lru_cache`d, and the httpx pool underneath it
    keeps connections alive that are bound to the loop that opened them. The
    next test's loop borrows one and raises on a future attached to a closed
    loop — intermittently, since it only bites when a connection is still warm.

    Both layers have to go. The accessors cache `LLMClient`/`EmbeddingClient`/
    `RerankClient` *instances*, each holding a transport reference of its own,
    so clearing `_async_transport` alone leaves live clients pointing at the
    dead one; clearing only the accessors rebuilds clients that fetch the same
    stale transport straight back out of its cache.
    """
    yield
    llm._async_transport.cache_clear()
    llm.fast_llm.cache_clear()
    llm.answer_llm.cache_clear()
    llm.llm_for.cache_clear()
    embeddings.embedding_client.cache_clear()
    rerank.rerank_client.cache_clear()


def on_index(coro: Coroutine[Any, Any, Any]) -> Any:
    """Open the async pool, await `coro`, close — all inside one event loop.

    `retrieve` reads the index through `kb.store.asearch`, which borrows from
    `async_pool()`. That pool is returned unopened by design — opening is a
    coroutine, so nothing can do it at import — and in production the app's
    lifespan owns both ends. There is no lifespan here, so a test that reaches
    retrieval has to own them itself, and within one loop: the pool binds to
    whichever loop opens it, which is the failure `close_async_pool` (it clears
    the cache as well as closing) exists to prevent.

    A whole test's worth of turns goes inside one call, so a multi-turn session
    shares the one pool rather than querying a closed loop's.
    """

    async def go() -> Any:
        await store.async_pool().open(wait=True, timeout=get_settings().db_pool_timeout)
        try:
            return await coro
        finally:
            await store.close_async_pool()

    return asyncio.run(go())


# --- router: the cheap tier under real conditions ---------------------------


@pytest.mark.live
@live
@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the profit rate on Murabaha everyday finance?", "retrieve"),
        ("Is Ijara auto financing halal?", "retrieve"),
        ("Who won the World Cup in 2022?", "refuse"),
        ("Hello!", "answer"),
    ],
)
def test_live_router_routes_the_obvious_cases(question: str, expected: str) -> None:
    result = asyncio.run(router.run({"query": question}))

    assert result["route"] == expected, result["route_reason"]
    assert result["route_reason"]
    if expected == "retrieve":
        # The rewrite is what `retrieve` embeds — blank would silently fall back
        # to the raw turn and lose the whole point of routing through a rewriter.
        assert result["search_query"]
        assert result["tried_queries"] == [result["search_query"]]
    else:
        assert result["search_query"] == ""


@pytest.mark.live
@live
def test_live_router_resolves_a_pronoun_against_history() -> None:
    """The reason the router reads history at all: "is it halal?" is not searchable."""
    history = [
        {"role": "user", "content": "Tell me about Murabaha everyday finance."},
        {"role": "assistant", "content": "Murabaha is a cost-plus sale where Mal buys the asset."},
    ]
    result = asyncio.run(router.run({"query": "is it halal?", "history": history}))

    assert result["route"] == "retrieve", result["route_reason"]
    assert "murabaha" in result["search_query"].lower(), result["search_query"]


@pytest.mark.live
@live
@pytest.mark.parametrize(
    "question",
    [
        "what happens if I pay late?",
        # Mal banking, but not obviously one of the six KB topics. The first
        # live run refused all three of these; a customer with a real problem
        # was told it was not this assistant's business. They belong in
        # retrieve, so an uncovered one ends at no_answer with a support
        # handover rather than a flat refusal.
        "How do I dispute a fraudulent card transaction on my Mal account?",
        "Can I get a home mortgage from Mal to buy a villa in Dubai?",
        "What is Mal's foreign exchange rate for converting USD to AED?",
    ],
)
def test_live_router_prefers_retrieve_when_the_question_is_marginal(question: str) -> None:
    """CLAUDE.md's bias, checked against the model rather than the prompt text.

    A false refusal stonewalls a real customer; a false retrieve is caught by
    the grader.
    """
    result = asyncio.run(router.run({"query": question}))

    assert result["route"] == "retrieve", result["route_reason"]


# --- generate: citations and the ungrounded guard ---------------------------


@pytest.mark.live
@live
def test_live_generate_cites_the_passages_it_used() -> None:
    chunks = [
        {
            "chunk_id": "murabaha-everyday-finance#012",
            "doc": "murabaha-everyday-finance",
            "text": (
                "Murabaha is a cost-plus sale. Mal buys the asset and resells it to the "
                "customer at a disclosed markup, payable in fixed instalments. The markup "
                "is fixed at contract signing and never changes."
            ),
            "score": 0.81,
        },
        {
            "chunk_id": "takaful-family-cover#003",
            "doc": "takaful-family-cover",
            "text": "Takaful is a cooperative arrangement funded by participant contributions.",
            "score": 0.44,
        },
    ]
    result = asyncio.run(
        generate.run({"query": "Can the Murabaha markup change later?", "chunks": chunks})
    )

    cited = {int(n) for n in MARKER.findall(result["answer"])}
    assert cited, f"no citation markers in: {result['answer']}"
    assert cited <= {1, 2}, "a marker past the passage count is a hallucinated citation"
    assert result["references"], "markers in the text must produce structured references"
    assert {ref["n"] for ref in result["references"]} == cited
    assert all(ref["chunk_id"] in {c["chunk_id"] for c in chunks} for ref in result["references"])
    assert result["outcome"] == "answered"
    # The answer is in the passage; the point is that it arrived with a citation.
    assert "1" in {str(n) for n in cited}


@pytest.mark.live
@live
def test_live_generate_declines_to_invent_a_fact_absent_from_the_passages() -> None:
    """The grounded prompt's other half: say what could not be confirmed."""
    chunks = [
        {
            "chunk_id": "murabaha-everyday-finance#004",
            "doc": "murabaha-everyday-finance",
            "text": "Murabaha finance is available to salaried and self-employed customers.",
            "score": 0.6,
        }
    ]
    result = asyncio.run(
        generate.run({"query": "What is the exact profit rate on Murabaha?", "chunks": chunks})
    )

    # No rate exists in that passage, so any digit-and-percent pair would be invented.
    assert not re.search(r"\d+(\.\d+)?\s*%", result["answer"]), result["answer"]


@pytest.mark.live
@live
def test_live_generate_makes_no_finance_claim_without_context() -> None:
    """The main hallucination surface: nothing grounds this path but the prompt."""
    result = asyncio.run(
        generate.run({"query": "What is the profit rate on your Murabaha product?", "chunks": []})
    )

    assert result["references"] == []
    assert not re.search(r"\d+(\.\d+)?\s*%", result["answer"]), result["answer"]
    assert not MARKER.search(result["answer"]), "nothing to cite on the no-retrieval path"


# --- the whole graph, against the deployed index ----------------------------


@pytest.mark.live
@needs_db
@live
def test_live_graph_answers_a_real_question_from_the_real_index() -> None:
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {"raw_query": "What is Murabaha and how does the markup work?"},
            config={"configurable": {"thread_id": "live-smoke-1"}},
        )
    )

    assert result["outcome"] == "answered", result.get("answer")
    assert result["chunks"], "retrieval returned nothing from the built index"
    assert len(result["chunks"]) <= get_settings().top_k
    assert result["references"], "a grounded answer must carry citations"
    retrieved = {chunk["chunk_id"] for chunk in result["chunks"]}
    assert all(ref["chunk_id"] in retrieved for ref in result["references"])

    # Every provider on the path reported its cost into the one trace field.
    nodes = [entry["node"] for entry in result["usage_log"]]
    assert nodes[0] == "router" and nodes[-1] == "generate"
    assert "retrieve" in nodes and "grade" in nodes
    if get_settings().rerank_enabled:
        assert "rerank" in nodes
        assert all("rerank_score" in chunk for chunk in result["chunks"])
    assert sum(entry["total_tokens"] for entry in result["usage_log"]) > 0


@pytest.mark.live
@needs_db
@live
@pytest.mark.parametrize(
    "question",
    [
        # Both halves are covered by the corpus, but the second half has no
        # stated formula. The first live run graded this not relevant three
        # times and dead-ended a plainly answerable question, so it is pinned:
        # partial-but-substantive coverage is relevant, and `generate` is the
        # node that discloses the gap.
        "What is Murabaha and how is the markup decided?",
        "What happens if I miss an Ijara payment?",
        "How does Takaful differ from conventional insurance?",
    ],
)
def test_live_graph_answers_rather_than_dead_ending(question: str) -> None:
    """The grader must not reject passages an answer can honestly be built from."""
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {"raw_query": question},
            config={"configurable": {"thread_id": f"live-grade-{hash(question)}"}},
        )
    )

    assert result["outcome"] == "answered", (
        f"dead-ended after {result['attempts']} retries: {result.get('grader_note')}"
    )
    assert result["references"]


@pytest.mark.live
@needs_db
@live
def test_live_graph_answers_an_account_figure_question_from_the_record() -> None:
    """The swagger sample request, pinned after it dead-ended live.

    "How much is left to pay" asks for an account fact no KB passage ever
    holds, so a passages-only grader is *correct* to fail it — and `reformulate`
    cannot rewrite its way to a chunk containing this customer's balance, so
    the turn burned both retries and landed at `no_answer` while `generate`,
    which renders the record and could answer, sat unreached one node away.
    The account-aware grader passes product-covering passages and leaves the
    figures to the record."""
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {
                "raw_query": "How much is left to pay on my Murabaha contract?",
                "account_id": "MAL-1001-2200-4417",
            },
            config={"configurable": {"thread_id": "live-account-1"}},
        )
    )

    assert result["outcome"] == "answered", (
        f"dead-ended after {result['attempts']} retries: {result.get('grader_note')}"
    )
    # The record's contract: 12 instalments of 1,090.00 against a 13,080.00
    # total, 5 paid. An answer naming none of these figures was written
    # without reading the record.
    figures = ("1,090", "13,080", "7,630", "5,450")
    assert any(fig in result["answer"] for fig in figures), result["answer"]


@pytest.mark.live
@needs_db
@live
def test_live_graph_answers_an_account_question_without_an_account() -> None:
    """The same question with no record attached must still not dead-end.

    `generate`'s grounded prompt owns this case — "say you cannot see their
    account details in this conversation" — but that instruction was
    unreachable while the grader failed every account-fact question before
    `generate` saw it. The honest outcome is an answer that says so, not the
    support handover for a question Mal's own app answers."""
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {"raw_query": "How much is left to pay on my Murabaha contract?"},
            config={"configurable": {"thread_id": "live-account-2"}},
        )
    )

    assert result["outcome"] == "answered", (
        f"dead-ended after {result['attempts']} retries: {result.get('grader_note')}"
    )


@pytest.mark.live
@needs_db
@live
def test_live_graph_hands_over_rather_than_inventing_an_uncovered_answer() -> None:
    """The other direction: a loosened grader must still stop a made-up answer.

    A home mortgage is a Mal-shaped question the corpus has no product for, so
    the honest outcome is the support handover — not a confident answer built
    out of the Murabaha goods-and-travel passages retrieval will return. The
    record is attached deliberately: an account outline in the grade prompt
    must not rubber-stamp a question about a product Mal does not offer, and
    no field in the record covers a mortgage either.
    """
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {
                "raw_query": "Can I get a home mortgage from Mal to buy a villa in Dubai?",
                "account_id": "MAL-1001-2200-4417",
            },
            config={"configurable": {"thread_id": "live-smoke-5"}},
        )
    )

    assert result["route"] == "retrieve", "a Mal banking question is not a refusal"
    assert result["outcome"] == "no_answer", result.get("answer")
    assert "support@mal.example" in result["answer"]


@pytest.mark.live
@needs_db
@live
def test_live_graph_refuses_an_out_of_scope_question() -> None:
    """The refusal eval's subject, end to end: no retrieval, no answer model."""
    compiled = build_graph(checkpointer=InMemorySaver())
    result = on_index(
        compiled.ainvoke(
            {"raw_query": "Write me a Python script to scrape a website."},
            config={"configurable": {"thread_id": "live-smoke-2"}},
        )
    )

    assert result["outcome"] == "refused", result.get("answer")
    assert result["chunks"] == []
    assert [entry["node"] for entry in result["usage_log"]] == ["router"]


@pytest.mark.live
@needs_db
@live
def test_live_graph_carries_a_session_across_two_turns() -> None:
    """Both halves of the session contract in one run: history carries, state resets."""
    compiled = build_graph(checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "live-smoke-3"}}

    async def session() -> dict:
        # Both turns in one loop: `on_index` opens the pool for the loop it runs
        # in and closes it on the way out, so a turn in a second `asyncio.run`
        # would be querying a pool bound to a loop that no longer exists.
        first = await compiled.ainvoke({"raw_query": "Tell me about Ijara auto financing."}, config)
        assert first["outcome"] == "answered", first.get("answer")
        return await compiled.ainvoke({"raw_query": "Who is responsible for insuring it?"}, config)

    second = on_index(session())

    assert second["outcome"] in {"answered", "no_answer"}, second.get("answer")
    assert len(second["history"]) == 4, "the first exchange must survive into turn 2"
    assert second["attempts"] <= get_settings().max_retrieval_attempts
    # The pronoun is only resolvable from turn 1 — an unresolved "it" would
    # embed as a query about nothing.
    assert "ijara" in second["search_query"].lower() or "vehicle" in second["search_query"].lower(), (
        second["search_query"]
    )
    assert [entry["node"] for entry in second["usage_log"]][0] == "router"


@pytest.mark.live
@needs_db
@live
def test_live_graph_masks_pii_before_any_provider_sees_it() -> None:
    """The PII requirement, asserted on the far side of a real request."""
    compiled = build_graph(checkpointer=InMemorySaver())
    secret = "784-1990-1234567-6"
    result = on_index(
        compiled.ainvoke(
            {"raw_query": f"My Emirates ID is {secret}. Am I eligible for Murabaha finance?"},
            config={"configurable": {"thread_id": "live-smoke-4"}},
        )
    )

    assert "[EMIRATES_ID]" in result["query"]
    assert secret not in result["query"]
    assert secret not in result["answer"]
    assert secret not in repr(result["history"])
    assert secret not in repr(result["usage_log"])
    assert secret not in repr(result["pii_spans"])
