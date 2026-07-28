"""OpenRouter embedding client — one class, any model.

Sits next to `llm.py` for the same reason: `kb/ingest.py` embeds documents at
build time and `rag/nodes/retrieve.py` embeds the query at request time, so the
client cannot live inside either without one importing across the other's
boundary. Those two callers also differ in transport: `retrieve` runs inside
the request's event loop and calls `aembed_query`, while ingest has no loop and
stays on the sync methods — the same twinning as `llm.py`, which is why both
transports are built here.

It is a separate module from `llm.py` rather than another method on `LLMClient`
because the two have different shapes — embeddings have no messages, no system
prompt, no schema, and no per-call temperature, but they do have a dimension
contract and batching. What they share (`Model`, `Usage`, the pooled transport)
is imported rather than duplicated, so a trace can total tokens across chat and
embedding calls with one type.

The dimension contract is the point of this module:

- **`dimensions` is requested, not assumed.** `google/gemini-embedding-001`
  returns 3072 by default; the pgvector column is 1536, and pgvector's `vector`
  type cannot be indexed above 2000. The parameter is OpenAI-compatible and
  OpenRouter forwards it.
- **The returned length is asserted on every call.** A provider that quietly
  stops honouring `dimensions` would otherwise corrupt the index at ingest, or
  return a query vector Postgres rejects at retrieval. Both should be loud.
- **Ingest and retrieval must use the same model.** That mismatch is the one
  failure with no error at all — documents and queries land in different vector
  spaces and search silently degrades to noise. Both sides construct their
  client from `embedding_client()`, which reads the single configured model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from core.config import get_settings

# Shared with the chat client on purpose. `Model` keeps `provider/name` parsing
# in one place, `Usage` lets a trace add an embedding call to an LLM call, and
# `_transport`/`_async_transport` are what make both clients share a connection
# pool per endpoint — one pool on each side, since the two SDK clients hold
# their own.
from core.llm import (
    Model,
    Usage,
    _async_transport,
    _provider_error,
    _transport,
    _usage,
)

# --- errors -------------------------------------------------------------
# As in `llm.py`: transport failures stay as the SDK's own exceptions. What is
# raised here is only what this layer can detect — a response that arrived fine
# and is unusable anyway.


class EmbeddingError(RuntimeError):
    """A response arrived but could not be used."""


class EmbeddingDimensionError(EmbeddingError):
    """The provider returned vectors of the wrong width.

    Fatal by design. At ingest it means the index would be built in a shape the
    column cannot hold; at retrieval it means the query is not comparable to
    what was indexed. Neither is worth continuing past.
    """


# --- value objects ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Embedding:
    """One vector plus what it cost."""

    vector: list[float]
    usage: Usage
    model: str

    def __len__(self) -> int:
        return len(self.vector)


@dataclass(frozen=True, slots=True)
class Embeddings:
    """A batch of vectors, in the order the inputs were given.

    Order is the contract: `kb/ingest.py` zips these back against its chunks, so
    a reordered batch would attach every embedding to the wrong text — a
    corruption that raises nothing and only shows up as bad retrieval.
    """

    vectors: list[list[float]]
    usage: Usage
    model: str

    def __len__(self) -> int:
        return len(self.vectors)

    def __iter__(self) -> Iterable[list[float]]:
        return iter(self.vectors)

    def __getitem__(self, index: int) -> list[float]:
        return self.vectors[index]


# --- client -------------------------------------------------------------


class EmbeddingClient:
    """Embedding calls against one OpenRouter model.

    One instance per model, reused across requests, built from a `(provider,
    name)` pair exactly like `LLMClient` — swapping to `openai/text-embedding-3-small`
    is a config change, not a code change. The transports are shared with the
    chat clients pointed at the same endpoint, sync and async alike.
    """

    def __init__(
        self,
        provider: str,
        name: str,
        *,
        dimensions: int | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        batch_size: int | None = None,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        """`provider` and `name` are the two halves of the OpenRouter model id.

        `dimensions` is sent with every request and enforced on every response.
        `None` means "take the model's default" — then whatever comes back is
        accepted and reported, since there is no stated contract to check against.

        `client` and `async_client` inject pre-built transports — used by the
        tests, and by any caller that wants to share a specific `OpenAI` or
        `AsyncOpenAI` instance. Injecting either builds *only* what was passed:
        calling the other side then raises `EmbeddingError` instead of quietly
        constructing a real network client behind a test's back.
        """
        settings = get_settings()
        self.model = Model(provider, name)
        self.dimensions = dimensions
        self.batch_size = settings.embedding_batch_size if batch_size is None else batch_size
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

        if client is not None or async_client is not None:
            self._client = client
            self._async_client = async_client
            return

        key = api_key if api_key is not None else settings.openrouter_api_key
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set — copy .env.example to .env and fill it in"
            )
        endpoint = base_url or settings.openrouter_base_url
        self._client = _transport(
            api_key=key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )
        self._async_client = _async_transport(
            api_key=key,
            base_url=endpoint,
            timeout=timeout,
            max_retries=max_retries,
        )

    @classmethod
    def from_slug(cls, slug: str, **kwargs: Any) -> EmbeddingClient:
        """Build from a `provider/name` string, which is how models are configured."""
        model = Model.parse(slug)
        return cls(model.provider, model.name, **kwargs)

    # --- calls ----------------------------------------------------------

    def embed_query(self, text: str) -> Embedding:
        """Embed one search query.

        Deliberately the same request as `embed_documents` makes — same model,
        same dimensions, no task-type hint. The OpenAI-compatible surface has no
        portable way to say "this is a query", and inventing an asymmetry here
        is how the two sides drift apart.
        """
        if not text.strip():
            raise ValueError("cannot embed empty text — a blank query is a bug upstream")
        batch = self._embed([text])
        return Embedding(vector=batch.vectors[0], usage=batch.usage, model=batch.model)

    async def aembed_query(self, text: str) -> Embedding:
        """`embed_query()` on the async transport — the request-time path.

        Same request as ingest makes, which is the symmetry that keeps queries
        and documents in one vector space.
        """
        if not text.strip():
            raise ValueError("cannot embed empty text — a blank query is a bug upstream")
        batch = await self._aembed([text])
        return Embedding(vector=batch.vectors[0], usage=batch.usage, model=batch.model)

    def embed_documents(self, texts: Sequence[str]) -> Embeddings:
        """Embed many chunks, in batches, preserving input order.

        Splitting at `batch_size` keeps a whole-corpus ingest from arriving as
        one oversized request; usage is summed across the batches so ingest can
        report what the build cost.

        There is deliberately no `aembed_documents`: batch embedding is
        ingest-only, and ingest runs synchronously with no event loop to yield
        to, so an async twin here would have no caller.
        """
        if not texts:
            return Embeddings(vectors=[], usage=Usage(), model=str(self.model))
        if any(not text.strip() for text in texts):
            raise ValueError("cannot embed empty text — check the chunker's output")

        vectors: list[list[float]] = []
        spent = Usage()
        for start in range(0, len(texts), self.batch_size):
            batch = self._embed(list(texts[start : start + self.batch_size]))
            vectors.extend(batch.vectors)
            spent = spent + batch.usage
        return Embeddings(vectors=vectors, usage=spent, model=str(self.model))

    # --- plumbing -------------------------------------------------------

    def _embed_kwargs(self, texts: list[str]) -> dict[str, Any]:
        """One request shape for both transports, so the two cannot fork.

        The `encoding_format` pin below is the reason this is a method rather
        than two copies: it is load-bearing, and a copy that lost it would fail
        only intermittently.
        """
        kwargs: dict[str, Any] = {
            "model": str(self.model),
            "input": texts,
            # Not a default worth inheriting. The `openai` SDK silently sends
            # `encoding_format="base64"` when the caller omits it (an internal
            # transfer optimisation, applied in `resources/embeddings.py` only).
            # OpenRouter serves `google/gemini-embedding-001` upstreams that do
            # not all accept it, and the result is *intermittent*: measured over
            # 12 identical calls, base64 succeeded 5 times and 7 came back
            # `200` carrying `error: "Google AI Studio embeddings do not support
            # base64 encoding_format"`. Same request, same second, different
            # outcome — an ingest would die part-way through the corpus and
            # succeed on the retry, which is the worst way to find this.
            # `float` measured 12/12 across every upstream OpenRouter attributed
            # ("Google" and "Google AI Studio"), and skips the SDK's decode step.
            "encoding_format": "float",
            # OpenRouter-only, same as the chat client: asks for the USD cost of
            # the call on the usage object.
            "extra_body": {"usage": {"include": True}},
        }
        if self.dimensions is not None:
            kwargs["dimensions"] = self.dimensions
        return kwargs

    def _embed(self, texts: list[str]) -> Embeddings:
        if self._client is None:
            raise EmbeddingError(
                f"{self.model} has no sync transport — it was built with async_client= only"
            )
        response = self._client.embeddings.create(**self._embed_kwargs(texts))
        vectors = self._vectors(response, expected=len(texts))
        return Embeddings(vectors=vectors, usage=_usage(response), model=str(self.model))

    async def _aembed(self, texts: list[str]) -> Embeddings:
        if self._async_client is None:
            raise EmbeddingError(
                f"{self.model} has no async transport — pass async_client= too, or "
                "construct without client= to get both"
            )
        response = await self._async_client.embeddings.create(**self._embed_kwargs(texts))
        vectors = self._vectors(response, expected=len(texts))
        return Embeddings(vectors=vectors, usage=_usage(response), model=str(self.model))

    def _vectors(self, response: Any, *, expected: int) -> list[list[float]]:
        data = getattr(response, "data", None)
        if not data:
            raise EmbeddingError(f"{self.model} returned no embeddings{_provider_error(response)}")
        if len(data) != expected:
            raise EmbeddingError(
                f"{self.model} returned {len(data)} embeddings for {expected} inputs"
            )

        # The OpenAI schema carries an explicit `index` per item and does not
        # promise response order. Sorting by it is what makes the order contract
        # on `Embeddings` true rather than hopeful.
        ordered = sorted(data, key=lambda item: getattr(item, "index", 0))
        vectors = [list(item.embedding) for item in ordered]

        for vector in vectors:
            if not vector:
                raise EmbeddingError(f"{self.model} returned an empty vector")
            if self.dimensions is not None and len(vector) != self.dimensions:
                raise EmbeddingDimensionError(
                    f"{self.model} returned {len(vector)}-dim vectors, expected "
                    f"{self.dimensions} — the index and the query would not match. "
                    "Check EMBEDDING_DIMENSIONS and whether the provider still "
                    "honours the `dimensions` parameter."
                )
        return vectors

    def __repr__(self) -> str:
        return f"EmbeddingClient({str(self.model)!r}, dimensions={self.dimensions})"


# --- the one configured model -------------------------------------------
# There is exactly one accessor, and both ingest and retrieval use it. That is
# the enforcement mechanism for "documents and queries share a vector space":
# no caller picks an embedding model, so no caller can pick a different one.


@lru_cache(maxsize=1)
def embedding_client() -> EmbeddingClient:
    """The configured embedding model, at the configured width."""
    settings = get_settings()
    return EmbeddingClient.from_slug(
        settings.embedding_model, dimensions=settings.embedding_dimensions
    )


__all__ = [
    "Embedding",
    "EmbeddingClient",
    "EmbeddingDimensionError",
    "EmbeddingError",
    "Embeddings",
    "embedding_client",
]
