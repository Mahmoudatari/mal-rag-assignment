"""Unit tests for the `retrieve` and `rerank` nodes.

Both nodes are plain `async run(state) -> dict` functions with no LangGraph
imports, so they are exercised directly against `State` dicts, driven with
`asyncio.run`. No network and no database: `embedding_client`, `asearch`,
`rerank_client` and `get_settings` are all monkeypatched as imported into the
node modules.

What is worth defending here, per node:

- **retrieve** — the `Match` → `Chunk` projection must drop `section`, the
  candidate `limit` must track `rerank_enabled`, and `search_query` must win
  over `query` (the reformulate-retry seam). A blank query must short-circuit
  before any client is built, since that is the terminating condition for the
  reformulate/no_answer loop.
- **rerank** — the reorder must carry real `chunk_id`s, never the API's echoed
  `document` text (see `tests/test_rerank.py`, same failure mode: a wrong
  citation that raises nothing). A `RerankError` must fail open to cosine
  order rather than fail the turn, since reranking is an ordering optimisation
  and relevance stays the grader's call.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import Settings
from core.embeddings import EmbeddingClient
from core.rerank import RerankClient
from kb.store import Match
from rag.nodes import rerank, retrieve
from rag.state import Chunk, State
from tests.fakes import FakeAsyncOpenAI, embedding_response, rerank_response

CHUNKS: list[Chunk] = [
    Chunk(chunk_id="murabaha-everyday-finance#001", doc="murabaha-everyday-finance", text="murabaha text", score=0.5),
    Chunk(chunk_id="ijara-vehicle-finance#002", doc="ijara-vehicle-finance", text="ijara text", score=0.4),
    Chunk(chunk_id="sukuk-investment-certificates#003", doc="sukuk-investment-certificates", text="sukuk text", score=0.3),
]


# --- retrieve --------------------------------------------------------------


def _embedding_client(vector=(0.1, 0.2, 0.3), *, prompt_tokens=0, cost=None):
    fake = FakeAsyncOpenAI(responses=[embedding_response([list(vector)], prompt_tokens=prompt_tokens, cost=cost)])
    client = EmbeddingClient(
        "google", "emb-test", api_key="test", dimensions=len(vector), async_client=fake
    )
    return client, fake


async def _no_matches(vector: object, limit: int) -> list:
    """`asearch` stand-in for the tests that only assert on the request."""
    return []


def test_match_to_chunk_projection_drops_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chunk has no `section` field — it is the useless "Frequently Asked
    Questions" string for 73 of 202 chunks and citations never use it."""
    client, _ = _embedding_client()
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())
    match = Match(
        chunk_id="murabaha-everyday-finance#027",
        doc="murabaha-everyday-finance",
        section="Frequently Asked Questions",
        text="Q&A body",
        score=0.87,
    )
    async def fake_asearch(vector: object, limit: int) -> list:
        return [match]

    monkeypatch.setattr(retrieve, "asearch", fake_asearch)

    result = asyncio.run(retrieve.run(State(query="what is murabaha?")))

    [chunk] = result["chunks"]
    assert set(chunk.keys()) == {"chunk_id", "doc", "text", "score"}
    assert chunk == {
        "chunk_id": "murabaha-everyday-finance#027",
        "doc": "murabaha-everyday-finance",
        "text": "Q&A body",
        "score": 0.87,
    }


@pytest.mark.parametrize(
    ("rerank_enabled", "expected_limit"),
    [(True, 20), (False, 4)],
)
def test_limit_tracks_the_rerank_toggle(
    monkeypatch: pytest.MonkeyPatch, rerank_enabled: bool, expected_limit: int
) -> None:
    """With reranking off there is no second stage to cut candidates down, so
    `retrieve` must hand `generate` `top_k` directly rather than 20 passages."""
    client, _ = _embedding_client()
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(
        retrieve,
        "get_settings",
        lambda: Settings(rerank_enabled=rerank_enabled, retrieve_candidates=20, top_k=4),
    )
    seen: dict[str, int] = {}

    async def fake_asearch(vector: object, limit: int) -> list:
        seen["limit"] = limit
        return []

    monkeypatch.setattr(retrieve, "asearch", fake_asearch)

    asyncio.run(retrieve.run(State(query="q")))

    assert seen["limit"] == expected_limit


def test_search_query_is_preferred_over_query(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = _embedding_client()
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())
    monkeypatch.setattr(retrieve, "asearch", _no_matches)

    asyncio.run(retrieve.run(State(search_query="resolved query", query="raw query")))

    assert fake.last_call["input"] == ["resolved query"]


@pytest.mark.parametrize("state", [State(search_query="", query="fallback"), State(query="fallback")])
def test_query_is_the_fallback_when_search_query_is_absent_or_blank(
    monkeypatch: pytest.MonkeyPatch, state: State
) -> None:
    client, fake = _embedding_client()
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())
    monkeypatch.setattr(retrieve, "asearch", _no_matches)

    asyncio.run(retrieve.run(state))

    assert fake.last_call["input"] == ["fallback"]


@pytest.mark.parametrize("state", [State(search_query="   ", query=""), State()])
def test_a_blank_query_returns_empty_chunks_with_no_calls(
    monkeypatch: pytest.MonkeyPatch, state: State
) -> None:
    """Terminates the reformulate loop via grade's empty-chunks path rather
    than spending an embedding call and a search on nothing."""

    def explode_client() -> None:
        raise AssertionError("embedding client built for a blank query")

    async def explode_search(*args: object, **kwargs: object) -> None:
        raise AssertionError("asearch() called for a blank query")

    monkeypatch.setattr(retrieve, "embedding_client", explode_client)
    monkeypatch.setattr(retrieve, "asearch", explode_search)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())

    assert asyncio.run(retrieve.run(state)) == {"chunks": []}


def test_returned_chunks_replace_rather_than_accumulate(monkeypatch: pytest.MonkeyPatch) -> None:
    """On a reformulate retry this node re-runs with a new search_query — the
    stale candidate set from the failed attempt must not linger alongside it.

    `candidate_log` is the deliberate exception: one id-list appended per
    search, so the trace and the retrieval eval keep the first run."""
    client, _ = _embedding_client()
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())
    fresh = Match(
        chunk_id="ijara-vehicle-finance#002",
        doc="ijara-vehicle-finance",
        section="Overview",
        text="fresh",
        score=0.5,
    )
    async def fake_asearch(vector: object, limit: int) -> list:
        return [fresh]

    monkeypatch.setattr(retrieve, "asearch", fake_asearch)

    stale = Chunk(chunk_id="murabaha-everyday-finance#001", doc="murabaha-everyday-finance", text="stale", score=0.99)
    result = asyncio.run(
        retrieve.run(
            State(query="q", chunks=[stale], candidate_log=[["murabaha-everyday-finance#001"]])
        )
    )

    assert [c["chunk_id"] for c in result["chunks"]] == ["ijara-vehicle-finance#002"]
    assert result["candidate_log"] == [
        ["murabaha-everyday-finance#001"],
        ["ijara-vehicle-finance#002"],
    ]


def test_retrieve_usage_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _embedding_client(prompt_tokens=12, cost=0.0003)
    monkeypatch.setattr(retrieve, "embedding_client", lambda: client)
    monkeypatch.setattr(retrieve, "get_settings", lambda: Settings())
    monkeypatch.setattr(retrieve, "asearch", _no_matches)

    result = asyncio.run(retrieve.run(State(query="q", usage_log=[])))

    [entry] = result["usage_log"]
    assert entry["node"] == "retrieve"
    assert entry["model"] == "google/emb-test"
    assert entry["total_tokens"] == 12
    assert entry["cost"] == pytest.approx(0.0003)
    assert "search_units" not in entry


# --- rerank ------------------------------------------------------------


def _rerank_client(scores: dict[int, float], *, search_units: int = 0, cost: float | None = None):
    fake = FakeAsyncOpenAI(responses=[rerank_response(scores, search_units=search_units, cost=cost)])
    client = RerankClient("cohere", "rerank-v3.5", api_key="test", async_client=fake)
    return client, fake


def test_reorder_and_truncate_to_top_k_keeps_real_chunk_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    """The load-bearing property: citations are built from chunk_id/doc, which
    only survive if the node reorders its own objects rather than the echo."""
    # Simulates the server having already applied top_n=2 — only two results
    # come back, best first.
    client, _ = _rerank_client({2: 0.9, 0: 0.5})
    monkeypatch.setattr(rerank, "rerank_client", lambda: client)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True, top_k=2))

    result = asyncio.run(rerank.run(State(query="q", chunks=list(CHUNKS))))

    assert [c["chunk_id"] for c in result["chunks"]] == [
        "sukuk-investment-certificates#003",
        "murabaha-everyday-finance#001",
    ]


def test_rerank_score_is_added_and_cosine_score_is_preserved(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _rerank_client({1: 0.77, 0: 0.33})
    monkeypatch.setattr(rerank, "rerank_client", lambda: client)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True, top_k=2))

    result = asyncio.run(rerank.run(State(query="q", chunks=list(CHUNKS[:2]))))
    by_id = {c["chunk_id"]: c for c in result["chunks"]}

    assert by_id["ijara-vehicle-finance#002"]["rerank_score"] == pytest.approx(0.77)
    assert by_id["ijara-vehicle-finance#002"]["score"] == pytest.approx(0.4)  # cosine, unchanged
    assert by_id["murabaha-everyday-finance#001"]["rerank_score"] == pytest.approx(0.33)


def test_disabled_returns_empty_dict_with_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> None:
        raise AssertionError("rerank client constructed while rerank_enabled is False")

    monkeypatch.setattr(rerank, "rerank_client", explode)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=False))

    assert asyncio.run(rerank.run(State(query="q", chunks=list(CHUNKS)))) == {}


def test_no_chunks_returns_empty_dict_with_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode() -> None:
        raise AssertionError("rerank client constructed with no candidates to reorder")

    monkeypatch.setattr(rerank, "rerank_client", explode)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True))

    assert asyncio.run(rerank.run(State(query="q", chunks=[]))) == {}


def test_request_carries_query_documents_and_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    client, fake = _rerank_client({0: 0.9})
    monkeypatch.setattr(rerank, "rerank_client", lambda: client)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True, top_k=1))

    asyncio.run(rerank.run(State(query="what is Ijara?", chunks=list(CHUNKS))))

    assert fake.last_call["path"] == "/rerank"
    assert fake.last_call["query"] == "what is Ijara?"
    assert fake.last_call["documents"] == [c["text"] for c in CHUNKS]
    assert fake.last_call["top_n"] == 1


def test_usage_entry_records_search_units_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """cohere/rerank-v3.5 bills on search units and reports zero tokens — a
    trace reading only tokens would show reranking as free."""
    client, _ = _rerank_client({0: 0.9}, search_units=1, cost=0.002)
    monkeypatch.setattr(rerank, "rerank_client", lambda: client)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True, top_k=1))

    result = asyncio.run(rerank.run(State(query="q", chunks=CHUNKS[:1], usage_log=[])))

    [entry] = result["usage_log"]
    assert entry["node"] == "rerank"
    assert entry["search_units"] == 1
    assert entry["cost"] == pytest.approx(0.002)


def test_a_rerank_error_falls_back_to_cosine_order_top_k(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reranking is an ordering optimisation, not a correctness gate: a 200
    carrying an `error` object (OpenRouter's shape for some upstream
    rejections) must not fail the turn. The candidates are already in cosine
    order, so falling back is just truncating them — with no `rerank_score`,
    which is the trace's signal that this happened."""
    fake = FakeAsyncOpenAI(
        responses=[{"error": {"message": "no endpoints found for cohere/rerank-v3.5", "code": 404}}]
    )
    client = RerankClient("cohere", "rerank-v3.5", api_key="test", async_client=fake)
    monkeypatch.setattr(rerank, "rerank_client", lambda: client)
    monkeypatch.setattr(rerank, "get_settings", lambda: Settings(rerank_enabled=True, top_k=2))

    result = asyncio.run(rerank.run(State(query="q", chunks=list(CHUNKS), usage_log=[])))

    assert [c["chunk_id"] for c in result["chunks"]] == [
        "murabaha-everyday-finance#001",
        "ijara-vehicle-finance#002",
    ]
    assert all("rerank_score" not in c for c in result["chunks"])
    assert "usage_log" not in result
