"""Unit tests for the OpenRouter embedding client.

Two things here are worth more than the rest, because they are the failures
that raise nothing on their own:

- **width** — a vector that comes back at the wrong dimension corrupts the
  index at ingest and mismatches the column at query time;
- **order** — a batch zipped back against the wrong chunks attaches every
  embedding to the wrong text, and looks exactly like mediocre retrieval.

Both are asserted directly. No network: the transport is injected. The live
test is marked `live` and deselected unless asked for.

The async twin is driven through `asyncio.run` inside ordinary sync tests —
pytest-asyncio is not a dependency, and one loop per case costs nothing against
a fake transport that never opens a connection.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import get_settings
from core.embeddings import (
    Embedding,
    EmbeddingClient,
    EmbeddingDimensionError,
    EmbeddingError,
    Embeddings,
)
from core.llm import Model, Usage
from tests.fakes import FakeAsyncOpenAI, FakeOpenAI, embedding_response

DIMS = 4


def vec(seed: float, width: int = DIMS) -> list[float]:
    return [seed + i for i in range(width)]


def client(*responses, **kwargs) -> tuple[EmbeddingClient, FakeOpenAI]:
    kwargs.setdefault("dimensions", DIMS)
    fake = FakeOpenAI(responses=list(responses))
    return EmbeddingClient("google", "gemini-embedding-001", client=fake, **kwargs), fake


def async_client(*responses, **kwargs) -> tuple[EmbeddingClient, FakeAsyncOpenAI]:
    """`client()`'s twin — injected as `async_client=`, so only that side exists."""
    kwargs.setdefault("dimensions", DIMS)
    fake = FakeAsyncOpenAI(responses=list(responses))
    return (
        EmbeddingClient("google", "gemini-embedding-001", async_client=fake, **kwargs),
        fake,
    )


# --- any OpenRouter model, addressed as (provider, name) -----------------
# Same contract as the chat client: swapping embedding providers is two strings.


@pytest.mark.parametrize(
    ("provider", "name"),
    [
        ("google", "gemini-embedding-001"),
        ("openai", "text-embedding-3-small"),
        ("mistralai", "mistral-embed"),
        ("qwen", "qwen3-embedding-8b:free"),
    ],
)
def test_any_openrouter_model_is_addressable(provider: str, name: str) -> None:
    fake = FakeOpenAI(responses=[embedding_response([vec(0.0)])])
    llm = EmbeddingClient(provider, name, dimensions=DIMS, client=fake)

    result = llm.embed_query("Murabaha")

    assert llm.model == Model(provider, name)
    assert fake.last_call["model"] == f"{provider}/{name}"
    assert result.model == f"{provider}/{name}"


def test_from_slug_matches_the_two_argument_form() -> None:
    fake = FakeOpenAI()
    built = EmbeddingClient.from_slug("openai/text-embedding-3-small", client=fake)
    assert built.model == Model("openai", "text-embedding-3-small")


@pytest.mark.parametrize("slug", ["text-embedding-3-small", "", "gemini-embedding-001"])
def test_unqualified_model_id_is_rejected(slug: str) -> None:
    with pytest.raises(ValueError, match="provider/name"):
        EmbeddingClient.from_slug(slug, client=FakeOpenAI())


def test_missing_api_key_fails_at_construction() -> None:
    """Ingest should die on the first line, not part-way through the corpus."""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        EmbeddingClient("google", "gemini-embedding-001", api_key="")


# --- the request ---------------------------------------------------------


def test_dimensions_are_requested_not_assumed() -> None:
    """The model defaults to 3072; the pgvector column is 1536."""
    llm, fake = client(embedding_response([vec(0.0)]))
    llm.embed_query("Ijara")
    assert fake.last_call["dimensions"] == DIMS


def test_dimensions_are_omitted_when_unset() -> None:
    """No `dimensions` means take the model's default — don't send a null."""
    llm, fake = client(embedding_response([vec(0.0, 7)]), dimensions=None)
    llm.embed_query("Ijara")
    assert "dimensions" not in fake.last_call


def test_float_encoding_is_pinned_on_every_call() -> None:
    """Regression guard, and the only fake-transport test with a live cause.

    The SDK sends base64 when this is omitted, and OpenRouter's upstreams for
    this model do not all accept it. The failure is intermittent, not a clean
    per-provider split: 12 identical live calls gave 5 successes and 7 replies
    carrying `error: "... do not support base64 encoding_format"`. `float`
    measured 12/12. Nothing downstream would survive that being reintroduced,
    and a green test suite would not notice.
    """
    llm, fake = client(embedding_response([vec(0.0)]))
    llm.embed_query("Sukuk")
    assert fake.last_call["encoding_format"] == "float"

    llm.embed_documents(["a"])
    assert fake.last_call["encoding_format"] == "float"


def test_cost_is_requested_on_every_call() -> None:
    """Cost is a trace field; OpenRouter only returns it if asked."""
    llm, fake = client(embedding_response([vec(0.0)]))
    llm.embed_query("Sukuk")
    assert fake.last_call["extra_body"] == {"usage": {"include": True}}


def test_input_is_sent_as_a_list_even_for_one_query() -> None:
    llm, fake = client(embedding_response([vec(0.0)]))
    llm.embed_query("Takaful")
    assert fake.last_call["input"] == ["Takaful"]


def test_queries_and_documents_make_the_same_request() -> None:
    """Any asymmetry between the two sides is how the vector spaces drift apart."""
    llm, fake = client(embedding_response([vec(0.0)]))

    llm.embed_query("Wakala")
    query_call = dict(fake.last_call)
    llm.embed_documents(["Wakala"])

    assert fake.last_call == query_call


# --- width, asserted on every response -----------------------------------


def test_wrong_width_is_fatal() -> None:
    """A silent provider-side change must not reach the index."""
    llm, _ = client(embedding_response([vec(0.0, DIMS + 1)]))
    with pytest.raises(EmbeddingDimensionError, match="expected 4"):
        llm.embed_query("Murabaha")


def test_wrong_width_anywhere_in_a_batch_is_fatal() -> None:
    llm, _ = client(embedding_response([vec(0.0), vec(1.0), vec(2.0, DIMS - 1)]))
    with pytest.raises(EmbeddingDimensionError):
        llm.embed_documents(["a", "b", "c"])


def test_unconstrained_client_accepts_whatever_width_comes_back() -> None:
    """With no stated contract there is nothing to violate."""
    llm, _ = client(embedding_response([vec(0.0, 3072)]), dimensions=None)
    assert len(llm.embed_query("Sukuk")) == 3072


def test_empty_vector_is_an_error() -> None:
    llm, _ = client(embedding_response([[]]), dimensions=None)
    with pytest.raises(EmbeddingError):
        llm.embed_query("Sukuk")


def test_no_data_is_an_error() -> None:
    llm, _ = client(embedding_response([]))
    with pytest.raises(EmbeddingError, match="no embeddings"):
        llm.embed_query("Sukuk")


def test_a_provider_error_on_a_200_is_surfaced() -> None:
    """OpenRouter answers some upstream rejections with 200 + an `error` object.

    The SDK does not raise on those, so this message is the only account of what
    happened — it is what identifies e.g. an encoding_format the upstream refuses.
    """
    llm, _ = client(
        embedding_response([], error={"message": "does not support base64", "code": 400})
    )
    with pytest.raises(EmbeddingError, match="does not support base64"):
        llm.embed_query("Sukuk")


def test_a_short_response_is_an_error_not_a_silent_truncation() -> None:
    """Fewer vectors than inputs would misalign every chunk after the gap."""
    llm, _ = client(embedding_response([vec(0.0), vec(1.0)]))
    with pytest.raises(EmbeddingError, match="2 embeddings for 3 inputs"):
        llm.embed_documents(["a", "b", "c"])


# --- order, which ingest zips against its chunks -------------------------


def test_batch_preserves_input_order() -> None:
    llm, _ = client(embedding_response([vec(0.0), vec(10.0), vec(20.0)]))
    result = llm.embed_documents(["a", "b", "c"])
    assert result.vectors == [vec(0.0), vec(10.0), vec(20.0)]


def test_out_of_order_response_is_reordered_by_index() -> None:
    """The OpenAI schema carries `index` because order is not promised."""
    llm, _ = client(
        embedding_response([vec(20.0), vec(0.0), vec(10.0)], indices=[2, 0, 1])
    )
    result = llm.embed_documents(["a", "b", "c"])
    assert result.vectors == [vec(0.0), vec(10.0), vec(20.0)]


def test_order_is_preserved_across_batch_boundaries() -> None:
    llm, fake = client(
        embedding_response([vec(0.0), vec(10.0)]),
        embedding_response([vec(20.0), vec(30.0)]),
        embedding_response([vec(40.0)]),
        batch_size=2,
    )
    result = llm.embed_documents(["a", "b", "c", "d", "e"])

    assert len(fake.calls) == 3
    assert [call["input"] for call in fake.calls] == [["a", "b"], ["c", "d"], ["e"]]
    assert result.vectors == [vec(0.0), vec(10.0), vec(20.0), vec(30.0), vec(40.0)]


# --- batching ------------------------------------------------------------


def test_a_small_corpus_is_one_request() -> None:
    llm, fake = client(embedding_response([vec(0.0), vec(1.0)]), batch_size=64)
    llm.embed_documents(["a", "b"])
    assert len(fake.calls) == 1


def test_empty_document_list_makes_no_request() -> None:
    llm, fake = client()
    result = llm.embed_documents([])

    assert fake.calls == []
    assert result == Embeddings(vectors=[], usage=Usage(), model="google/gemini-embedding-001")


@pytest.mark.parametrize("text", ["", "   \n "])
def test_blank_text_is_rejected_before_it_costs_anything(text: str) -> None:
    """A blank input returns a plausible vector that means nothing."""
    llm, fake = client()

    with pytest.raises(ValueError):
        llm.embed_query(text)
    with pytest.raises(ValueError):
        llm.embed_documents(["fine", text])

    assert fake.calls == []


def test_batch_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        EmbeddingClient("google", "gemini-embedding-001", client=FakeOpenAI(), batch_size=0)


# --- usage, which the trace requires -------------------------------------


def test_usage_is_read_off_the_response() -> None:
    llm, _ = client(embedding_response([vec(0.0)], prompt_tokens=12, cost=0.00001))
    assert llm.embed_query("Ijara").usage == Usage(12, 0, 12, 0.00001)


def test_usage_survives_a_provider_that_omits_it() -> None:
    llm, _ = client(embedding_response([vec(0.0)], usage=False))
    assert llm.embed_query("Ijara").usage == Usage()


def test_usage_sums_across_batches_so_ingest_can_report_the_build_cost() -> None:
    llm, _ = client(
        embedding_response([vec(0.0), vec(1.0)], prompt_tokens=10, cost=0.001),
        embedding_response([vec(2.0)], prompt_tokens=5, cost=0.002),
        batch_size=2,
    )
    usage = llm.embed_documents(["a", "b", "c"]).usage

    assert usage.prompt_tokens == 15
    assert usage.total_tokens == 15
    assert usage.cost == pytest.approx(0.003)


def test_embedding_usage_adds_to_llm_usage_for_one_trace_total() -> None:
    """Same `Usage` type on both clients is the reason this works."""
    llm, _ = client(embedding_response([vec(0.0)], prompt_tokens=12, cost=0.00001))
    total = llm.embed_query("Ijara").usage + Usage(100, 40, 140, 0.0004)
    assert total == Usage(112, 40, 152, pytest.approx(0.00041))


# --- shape ---------------------------------------------------------------


def test_embed_query_returns_one_vector() -> None:
    llm, _ = client(embedding_response([vec(0.0)]))
    result = llm.embed_query("Murabaha")

    assert isinstance(result, Embedding)
    assert result.vector == vec(0.0)
    assert len(result) == DIMS


def test_embeddings_iterate_and_index_like_a_sequence() -> None:
    """`ingest` zips over this directly."""
    llm, _ = client(embedding_response([vec(0.0), vec(10.0)]))
    result = llm.embed_documents(["a", "b"])

    assert len(result) == 2
    assert result[1] == vec(10.0)
    assert list(result) == [vec(0.0), vec(10.0)]


def test_repr_names_the_model_and_the_width() -> None:
    llm, _ = client()
    assert repr(llm) == "EmbeddingClient('google/gemini-embedding-001', dimensions=4)"


# --- the async twin, which is the request-time path ----------------------
# `rag/nodes/retrieve.py` runs inside the request's event loop, so the query is
# embedded with `aembed_query`. Everything except the transport call itself is
# shared with the sync side, and these tests are what say so.


def test_aembed_query_returns_one_vector() -> None:
    llm, _ = async_client(embedding_response([vec(0.0)], prompt_tokens=12, cost=0.00001))
    result = asyncio.run(llm.aembed_query("Murabaha"))

    assert isinstance(result, Embedding)
    assert result.vector == vec(0.0)
    assert result.usage == Usage(12, 0, 12, 0.00001)
    assert result.model == "google/gemini-embedding-001"


def test_the_async_request_carries_the_same_pins_as_the_sync_one() -> None:
    """`_embed_kwargs` is shared; this is the check that it stayed that way.

    The `encoding_format` pin is the one that matters — a forked async copy
    without it would fail intermittently against OpenRouter's upstreams and
    pass every test that only exercised the sync side.
    """
    llm, fake = async_client(embedding_response([vec(0.0)]))
    asyncio.run(llm.aembed_query("Sukuk"))

    assert fake.last_call["encoding_format"] == "float"
    assert fake.last_call["dimensions"] == DIMS
    assert fake.last_call["extra_body"] == {"usage": {"include": True}}
    assert fake.last_call["input"] == ["Sukuk"]


def test_the_async_request_is_identical_to_the_sync_one() -> None:
    """Same vector space on both transports, asserted rather than assumed."""
    sync_llm, sync_fake = client(embedding_response([vec(0.0)]))
    async_llm, async_fake = async_client(embedding_response([vec(0.0)]))

    sync_llm.embed_query("Wakala")
    asyncio.run(async_llm.aembed_query("Wakala"))

    assert async_fake.last_call == sync_fake.last_call


def test_async_dimensions_are_omitted_when_unset() -> None:
    llm, fake = async_client(embedding_response([vec(0.0, 7)]), dimensions=None)
    asyncio.run(llm.aembed_query("Ijara"))
    assert "dimensions" not in fake.last_call


@pytest.mark.parametrize("text", ["", "   \n "])
def test_async_blank_text_is_rejected_before_it_costs_anything(text: str) -> None:
    llm, fake = async_client()

    with pytest.raises(ValueError):
        asyncio.run(llm.aembed_query(text))

    assert fake.calls == []


def test_async_wrong_width_is_fatal() -> None:
    """The width check is in shared plumbing — the async path must not skip it."""
    llm, _ = async_client(embedding_response([vec(0.0, DIMS + 1)]))
    with pytest.raises(EmbeddingDimensionError, match="expected 4"):
        asyncio.run(llm.aembed_query("Murabaha"))


def test_an_async_provider_error_on_a_200_is_surfaced() -> None:
    llm, _ = async_client(
        embedding_response([], error={"message": "does not support base64", "code": 400})
    )
    with pytest.raises(EmbeddingError, match="does not support base64"):
        asyncio.run(llm.aembed_query("Sukuk"))


# --- injecting one transport does not conjure the other ------------------
# A test that injects `client=` and then reaches an async method should get an
# exception naming the mistake, never a real `AsyncOpenAI` built from settings
# and a live key.


def test_a_sync_only_client_has_no_async_transport() -> None:
    llm, _ = client(embedding_response([vec(0.0)]))
    with pytest.raises(EmbeddingError, match="no async transport"):
        asyncio.run(llm.aembed_query("Sukuk"))


def test_an_async_only_client_has_no_sync_transport() -> None:
    llm, _ = async_client(embedding_response([vec(0.0)]))

    with pytest.raises(EmbeddingError, match="no sync transport"):
        llm.embed_query("Sukuk")
    with pytest.raises(EmbeddingError, match="no sync transport"):
        llm.embed_documents(["Sukuk"])


# --- the single configured accessor --------------------------------------


def test_embedding_client_uses_the_configured_model_and_width() -> None:
    """Ingest and retrieval both call this, which is what keeps them in one space."""
    from core.embeddings import embedding_client

    settings = get_settings()
    if not settings.openrouter_api_key:
        pytest.skip("needs a key to construct the real transport")

    built = embedding_client()
    assert str(built.model) == settings.embedding_model
    assert built.dimensions == settings.embedding_dimensions
    assert built is embedding_client()  # cached — one pool, not one per call


# --- live smoke test -----------------------------------------------------
# Deselected by default (`-m "not live"`). Run explicitly:
#     uv run pytest -m live
# This is the only check that the configured embedding model exists on
# OpenRouter and really honours `dimensions=1536`. A fake transport returns
# whatever width it is told to.


@pytest.mark.live
@pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
def test_live_embeddings_come_back_at_the_configured_width() -> None:
    from core.embeddings import embedding_client

    llm = embedding_client()
    settings = get_settings()

    query = llm.embed_query("What is Murabaha?")
    assert len(query.vector) == settings.embedding_dimensions
    assert query.usage.total_tokens > 0

    batch = llm.embed_documents(
        ["Murabaha is a cost-plus-profit sale.", "Ijara is a lease-based financing."]
    )
    assert len(batch) == 2
    assert all(len(vector) == settings.embedding_dimensions for vector in batch)


@pytest.mark.live
@pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
def test_live_related_text_scores_higher_than_unrelated() -> None:
    """The vectors have to be meaningful, not merely the right width."""
    from core.embeddings import embedding_client

    vectors = embedding_client().embed_documents(
        [
            "What is Murabaha financing?",
            "Murabaha is a cost-plus-profit sale used in Islamic banking.",
            "The capital of France is Paris.",
        ]
    )
    query, related, unrelated = vectors

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm = (sum(x * x for x in a) ** 0.5) * (sum(y * y for y in b) ** 0.5)
        return dot / norm

    assert cosine(query, related) > cosine(query, unrelated)
