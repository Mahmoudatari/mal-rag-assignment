"""Unit tests for the OpenRouter rerank client.

The thing worth defending here is the **index mapping**. Results come back
sorted by score carrying the input position, and the rerank node uses that
position to reorder whole `Chunk` dicts. Get it wrong and every citation points
at a different chunk than the one the answer was written from — which raises
nothing, reads as plausible, and defeats the grounding eval.

So: order, index bounds, and the fact that the echoed `document` text is never
read. No network; the transport is injected.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import get_settings
from core.llm import Model, Usage
from core.rerank import Ranking, RerankClient, RerankError, RerankResult, RerankUsage
from tests.fakes import FakeAsyncOpenAI, FakeOpenAI, rerank_response

DOCS = ["murabaha text", "ijara text", "sukuk text", "takaful text"]


def client(*responses, **kwargs) -> tuple[RerankClient, FakeOpenAI]:
    fake = FakeOpenAI(responses=list(responses))
    return RerankClient("cohere", "rerank-v3.5", client=fake, **kwargs), fake


def async_client(*responses, **kwargs) -> tuple[RerankClient, FakeAsyncOpenAI]:
    """`client()`'s twin — the same fake with a coroutine `post`, injected async-side."""
    fake = FakeAsyncOpenAI(responses=list(responses))
    return RerankClient("cohere", "rerank-v3.5", async_client=fake, **kwargs), fake


# --- any OpenRouter model, addressed as (provider, name) -----------------


@pytest.mark.parametrize(
    ("provider", "name"),
    [("cohere", "rerank-v3.5"), ("cohere", "rerank-english-v3.0"), ("jina", "jina-reranker-v2")],
)
def test_any_openrouter_model_is_addressable(provider: str, name: str) -> None:
    fake = FakeOpenAI(responses=[rerank_response({0: 0.9})])
    llm = RerankClient(provider, name, client=fake)

    result = llm.rerank("q", ["a"])

    assert llm.model == Model(provider, name)
    assert fake.last_call["model"] == f"{provider}/{name}"
    assert result.model == f"{provider}/{name}"


def test_from_slug_matches_the_two_argument_form() -> None:
    built = RerankClient.from_slug("cohere/rerank-v3.5", client=FakeOpenAI())
    assert built.model == Model("cohere", "rerank-v3.5")


@pytest.mark.parametrize("slug", ["rerank-v3.5", ""])
def test_unqualified_model_id_is_rejected(slug: str) -> None:
    with pytest.raises(ValueError, match="provider/name"):
        RerankClient.from_slug(slug, client=FakeOpenAI())


def test_missing_api_key_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        RerankClient("cohere", "rerank-v3.5", api_key="")


# --- the request ---------------------------------------------------------
# `/rerank` is not an OpenAI endpoint; it is reached through the SDK's generic
# `post()` so it shares the pool, the auth and the retry policy.


def test_it_posts_to_the_rerank_endpoint() -> None:
    llm, fake = client(rerank_response({0: 0.9}))
    llm.rerank("what is Ijara?", DOCS)
    assert fake.last_call["path"] == "/rerank"


def test_the_request_carries_query_and_documents() -> None:
    llm, fake = client(rerank_response({0: 0.9}))
    llm.rerank("what is Ijara?", DOCS)

    assert fake.last_call["query"] == "what is Ijara?"
    assert fake.last_call["documents"] == DOCS


def test_top_n_is_sent_when_given() -> None:
    """This is the cut from `retrieve_candidates` to `top_k`."""
    llm, fake = client(rerank_response({1: 0.9, 0: 0.5}))
    llm.rerank("q", DOCS, top_n=2)
    assert fake.last_call["top_n"] == 2


def test_top_n_is_omitted_when_not_given() -> None:
    llm, fake = client(rerank_response({0: 0.9}))
    llm.rerank("q", DOCS)
    assert "top_n" not in fake.last_call


@pytest.mark.parametrize("top_n", [0, -1])
def test_a_meaningless_top_n_is_rejected_before_the_call(top_n: int) -> None:
    llm, fake = client()
    with pytest.raises(ValueError, match="top_n"):
        llm.rerank("q", DOCS, top_n=top_n)
    assert fake.calls == []


@pytest.mark.parametrize("query", ["", "   \n "])
def test_a_blank_query_is_rejected_before_the_call(query: str) -> None:
    llm, fake = client()
    with pytest.raises(ValueError, match="empty query"):
        llm.rerank(query, DOCS)
    assert fake.calls == []


def test_no_documents_makes_no_request() -> None:
    """The API requires at least one document, and there is nothing to order."""
    llm, fake = client()
    result = llm.rerank("q", [])

    assert fake.calls == []
    assert result == RerankResult(rankings=[], usage=RerankUsage(), model="cohere/rerank-v3.5")


# --- the index mapping, which citations depend on ------------------------


def test_results_are_returned_best_first() -> None:
    llm, _ = client(rerank_response({0: 0.10, 1: 0.95, 2: 0.40}))
    result = llm.rerank("q", DOCS)

    assert result.order == [1, 2, 0]
    assert result.rankings[0] == Ranking(index=1, relevance_score=0.95)


def test_a_response_listed_out_of_order_is_still_sorted() -> None:
    """Ordering is the entire product of the call — it is not taken on trust."""
    llm, _ = client(rerank_response({2: 0.4, 0: 0.1, 1: 0.95}))
    assert llm.rerank("q", DOCS).order == [1, 2, 0]


def test_apply_reorders_the_callers_own_objects() -> None:
    """The rerank node reorders whole chunks, keeping chunk_id and doc intact."""
    chunks = [
        {"chunk_id": "murabaha-1", "text": "a"},
        {"chunk_id": "ijara-2", "text": "b"},
        {"chunk_id": "sukuk-3", "text": "c"},
    ]
    llm, _ = client(rerank_response({0: 0.1, 1: 0.9, 2: 0.5}))

    reordered = llm.rerank("q", [chunk["text"] for chunk in chunks]).apply(chunks)

    assert [chunk["chunk_id"] for chunk in reordered] == ["ijara-2", "sukuk-3", "murabaha-1"]


def test_the_echoed_document_text_is_never_read() -> None:
    """The echo drops chunk_id and doc, so reading it would break every citation.

    Here the API echoes text that matches nothing the caller sent; the mapping
    must come from `index` alone and be unaffected.
    """
    llm, _ = client(
        rerank_response({0: 0.2, 1: 0.9}, documents={0: "WRONG TEXT", 1: "ALSO WRONG"})
    )
    result = llm.rerank("q", ["first", "second"])

    assert result.apply(["first", "second"]) == ["second", "first"]


def test_apply_truncates_to_what_the_api_returned() -> None:
    """With top_n set, only top_n results come back — `apply` yields that many."""
    llm, _ = client(rerank_response({3: 0.9, 1: 0.7}))
    assert llm.rerank("q", DOCS, top_n=2).apply(DOCS) == ["takaful text", "ijara text"]


def test_scored_pairs_each_item_with_its_score_for_the_trace() -> None:
    llm, _ = client(rerank_response({1: 0.9, 0: 0.3}))
    assert llm.rerank("q", ["a", "b"]).scored(["a", "b"]) == [("b", 0.9), ("a", 0.3)]


@pytest.mark.parametrize("index", [4, 99, -1])
def test_an_out_of_range_index_is_fatal(index: int) -> None:
    """Unchecked, this is an IndexError at best and a wrong citation at worst."""
    llm, _ = client(rerank_response({index: 0.9}))
    with pytest.raises(RerankError, match="index"):
        llm.rerank("q", DOCS)


def test_a_duplicate_index_is_fatal() -> None:
    """One chunk returned twice would push a real one out of the top_k."""
    llm, _ = client(
        rerank_response({}) | {"results": [
            {"index": 1, "relevance_score": 0.9, "document": {"text": "x"}},
            {"index": 1, "relevance_score": 0.5, "document": {"text": "x"}},
        ]}
    )
    with pytest.raises(RerankError, match="twice"):
        llm.rerank("q", DOCS)


# --- malformed responses -------------------------------------------------


def test_missing_results_is_an_error() -> None:
    llm, _ = client({"id": "x", "model": "m"})
    with pytest.raises(RerankError, match="no results"):
        llm.rerank("q", DOCS)


def test_a_provider_error_on_a_200_is_surfaced() -> None:
    """Same OpenRouter behaviour the embedding client hit: 200 + an `error` object."""
    llm, _ = client({"error": {"message": "no endpoints found for cohere/rerank-v3.5", "code": 404}})
    with pytest.raises(RerankError, match="no endpoints found"):
        llm.rerank("q", DOCS)


@pytest.mark.parametrize(
    "results",
    [
        [{"relevance_score": 0.9}],  # no index
        [{"index": 0}],  # no score
        [{"index": "one", "relevance_score": 0.9}],  # index not a number
        ["not a dict"],
    ],
)
def test_a_malformed_result_is_an_error(results: list) -> None:
    llm, _ = client({"results": results})
    with pytest.raises(RerankError):
        llm.rerank("q", DOCS)


def test_a_non_object_response_is_an_error() -> None:
    llm, _ = client(["unexpected"])
    with pytest.raises(RerankError, match="expected an object"):
        llm.rerank("q", DOCS)


# --- usage, which the trace requires -------------------------------------


def test_usage_is_read_off_the_response() -> None:
    llm, _ = client(rerank_response({0: 0.9}, search_units=1, total_tokens=120, cost=0.002))
    assert llm.rerank("q", DOCS).usage == RerankUsage(
        search_units=1, total_tokens=120, cost=0.002
    )


def test_usage_survives_a_provider_that_omits_it() -> None:
    llm, _ = client(rerank_response({0: 0.9}, usage=False))
    assert llm.rerank("q", DOCS).usage == RerankUsage()


def test_search_units_are_kept_because_that_is_how_reranking_bills() -> None:
    llm, _ = client(rerank_response({0: 0.9}, search_units=3))
    assert llm.rerank("q", DOCS).usage.search_units == 3


def test_rerank_usage_folds_into_the_shared_type_for_one_trace_total() -> None:
    """`search_units` has no chat equivalent, so it is dropped by the fold, not lost —
    it stays on `RerankUsage` for the trace to log separately."""
    llm, _ = client(rerank_response({0: 0.9}, search_units=1, total_tokens=120, cost=0.002))
    usage = llm.rerank("q", DOCS).usage

    total = usage.as_usage() + Usage(100, 40, 140, 0.0004)
    assert total == Usage(100, 40, 260, pytest.approx(0.0024))


def test_top_score_is_the_best_relevance_score() -> None:
    """The trace's relevance score — a cross-encoder read, better than raw cosine."""
    llm, _ = client(rerank_response({0: 0.1, 2: 0.87}))
    assert llm.rerank("q", DOCS).top_score == 0.87


def test_top_score_of_nothing_is_zero_so_a_trace_always_has_a_number() -> None:
    llm, _ = client()
    assert llm.rerank("q", []).top_score == 0.0


# --- the async twin ------------------------------------------------------
# The graph node awaits `arerank`, so it gets the same coverage as its sync
# twin: guards, request shape, index mapping, usage and the 200-with-`error`
# reply. pytest-asyncio is not a dependency, so the coroutine is driven with
# `asyncio.run` from an ordinary sync test.


def test_arerank_returns_results_best_first() -> None:
    llm, _ = async_client(rerank_response({0: 0.10, 1: 0.95, 2: 0.40}))
    result = asyncio.run(llm.arerank("q", DOCS))

    assert result.order == [1, 2, 0]
    assert result.rankings[0] == Ranking(index=1, relevance_score=0.95)


def test_arerank_reorders_the_callers_own_objects() -> None:
    """Same contract as the sync path: `chunk_id` and `doc` survive the rerank."""
    chunks = [
        {"chunk_id": "murabaha-1", "text": "a"},
        {"chunk_id": "ijara-2", "text": "b"},
        {"chunk_id": "sukuk-3", "text": "c"},
    ]
    llm, _ = async_client(rerank_response({0: 0.1, 1: 0.9, 2: 0.5}))

    result = asyncio.run(llm.arerank("q", [chunk["text"] for chunk in chunks]))

    assert [chunk["chunk_id"] for chunk in result.apply(chunks)] == [
        "ijara-2",
        "sukuk-3",
        "murabaha-1",
    ]
    assert result.scored(chunks)[0] == (chunks[1], 0.9)


def test_arerank_posts_the_same_request_as_the_sync_call() -> None:
    llm, fake = async_client(rerank_response({1: 0.9, 0: 0.5}))
    asyncio.run(llm.arerank("what is Ijara?", DOCS, top_n=2))

    assert fake.last_call["path"] == "/rerank"
    assert fake.last_call["model"] == "cohere/rerank-v3.5"
    assert fake.last_call["query"] == "what is Ijara?"
    assert fake.last_call["documents"] == DOCS
    assert fake.last_call["top_n"] == 2


def test_arerank_omits_top_n_when_not_given() -> None:
    llm, fake = async_client(rerank_response({0: 0.9}))
    asyncio.run(llm.arerank("q", DOCS))
    assert "top_n" not in fake.last_call


def test_arerank_with_no_documents_makes_no_request() -> None:
    llm, fake = async_client()
    result = asyncio.run(llm.arerank("q", []))

    assert fake.calls == []
    assert result == RerankResult(rankings=[], usage=RerankUsage(), model="cohere/rerank-v3.5")


@pytest.mark.parametrize("query", ["", "   \n "])
def test_arerank_rejects_a_blank_query_before_the_call(query: str) -> None:
    """Validation is shared, not forked — a guard that held on one side only
    would be a rejected request on the path that skipped it."""
    llm, fake = async_client()
    with pytest.raises(ValueError, match="empty query"):
        asyncio.run(llm.arerank(query, DOCS))
    assert fake.calls == []


@pytest.mark.parametrize("top_n", [0, -1])
def test_arerank_rejects_a_meaningless_top_n_before_the_call(top_n: int) -> None:
    llm, fake = async_client()
    with pytest.raises(ValueError, match="top_n"):
        asyncio.run(llm.arerank("q", DOCS, top_n=top_n))
    assert fake.calls == []


def test_arerank_reads_usage_off_the_response() -> None:
    llm, _ = async_client(
        rerank_response({0: 0.9}, search_units=1, total_tokens=120, cost=0.002)
    )
    assert asyncio.run(llm.arerank("q", DOCS)).usage == RerankUsage(
        search_units=1, total_tokens=120, cost=0.002
    )


def test_arerank_surfaces_a_provider_error_on_a_200() -> None:
    llm, _ = async_client(
        {"error": {"message": "no endpoints found for cohere/rerank-v3.5", "code": 404}}
    )
    with pytest.raises(RerankError, match="no endpoints found"):
        asyncio.run(llm.arerank("q", DOCS))


@pytest.mark.parametrize("index", [4, 99, -1])
def test_arerank_rejects_an_out_of_range_index(index: int) -> None:
    llm, _ = async_client(rerank_response({index: 0.9}))
    with pytest.raises(RerankError, match="index"):
        asyncio.run(llm.arerank("q", DOCS))


def test_arerank_rejects_a_duplicate_index() -> None:
    llm, _ = async_client(
        rerank_response({}) | {"results": [
            {"index": 1, "relevance_score": 0.9, "document": {"text": "x"}},
            {"index": 1, "relevance_score": 0.5, "document": {"text": "x"}},
        ]}
    )
    with pytest.raises(RerankError, match="twice"):
        asyncio.run(llm.arerank("q", DOCS))


# --- one injected transport builds only that side ------------------------
# Same rule as `LLMClient`: the missing side raises rather than quietly opening
# a real connection behind a test's back.


def test_a_sync_only_client_has_no_async_transport() -> None:
    llm, fake = client(rerank_response({0: 0.9}))
    with pytest.raises(RerankError, match="async transport"):
        asyncio.run(llm.arerank("q", DOCS))
    assert fake.calls == []


def test_an_async_only_client_has_no_sync_transport() -> None:
    llm, fake = async_client(rerank_response({0: 0.9}))
    with pytest.raises(RerankError, match="sync transport"):
        llm.rerank("q", DOCS)
    assert fake.calls == []


# --- shape ---------------------------------------------------------------


def test_len_counts_the_rankings() -> None:
    llm, _ = client(rerank_response({0: 0.9, 1: 0.5}))
    assert len(llm.rerank("q", DOCS)) == 2


def test_repr_names_the_model() -> None:
    llm, _ = client()
    assert repr(llm) == "RerankClient('cohere/rerank-v3.5')"


def test_rerank_client_uses_the_configured_model() -> None:
    from core.rerank import rerank_client

    settings = get_settings()
    if not settings.openrouter_api_key:
        pytest.skip("needs a key to construct the real transport")

    built = rerank_client()
    assert str(built.model) == settings.rerank_model
    assert built is rerank_client()  # cached — one pool, not one per call


# --- live smoke test -----------------------------------------------------
# Deselected by default. Run explicitly: uv run pytest -m live


@pytest.mark.live
@pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
def test_live_rerank_puts_the_relevant_document_first() -> None:
    from core.rerank import rerank_client

    documents = [
        "Takaful is a cooperative insurance arrangement based on mutual guarantee.",
        "Murabaha is a cost-plus-profit sale in which the bank discloses its markup.",
        "The capital of France is Paris.",
    ]
    result = rerank_client().rerank("How does Murabaha markup work?", documents, top_n=2)

    assert len(result) == 2
    assert result.order[0] == 1
    assert result.apply(documents)[0].startswith("Murabaha")
    assert result.top_score > 0


@pytest.mark.live
@pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
def test_live_rerank_reports_what_it_cost() -> None:
    """`search_units` is the rerank billing unit and belongs in the trace."""
    from core.rerank import rerank_client

    usage = rerank_client().rerank("What is Sukuk?", ["Sukuk are asset-backed certificates."]).usage
    assert usage.search_units > 0 or usage.total_tokens > 0 or usage.cost > 0
