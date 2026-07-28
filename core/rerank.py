"""OpenRouter rerank client — one class, any model.

The third OpenRouter surface, after chat and embeddings, and the second stage of
retrieval: `retrieve` pulls `retrieve_candidates` (20) by vector similarity and
this cuts them to `top_k` (4) with a cross-encoder that actually reads the query
against each document.

`POST /api/v1/rerank` is not an OpenAI endpoint, so the `openai` SDK has no
method for it. It is still reached *through* the SDK — `client.post(...)` with
`cast_to=object` — rather than through a second HTTP stack, which buys the
pooled connections, the bearer auth, the timeout and the SDK's own 429/5xx
backoff for free, and keeps one place where the OpenRouter base URL is set.
`OpenAI` and `AsyncOpenAI` expose that same generic `post`, so `arerank()` reaches
the endpoint exactly the same way and the async path stays inside the SDK too.

Two things here are deliberate and load-bearing:

- **The echoed `document` is ignored.** The API returns results sorted by score,
  each carrying `index` (the position in the input) and an echo of the document
  text. Reranking a `Chunk` and reading the echo back would return *text* and
  drop `chunk_id` and `doc` — the two fields citations are built from. So the
  index is the only thing read out of a result, and `apply()` reorders the
  caller's own objects. Nothing downstream ever depends on the echo.
- **Reranking must not gate control flow.** Same rule as cosine similarity:
  `relevance_score` is a trace field, not a filter. Relevance stays the LLM
  grader's decision, so this module has no threshold and no notion of a
  document being "not relevant enough" — it orders and truncates, nothing more.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, TypeVar

from core.config import get_settings

# Shared with the other two clients: `Model` keeps `provider/name` parsing in one
# place, `Usage` lets a trace total a rerank alongside its LLM and embedding
# calls, `_transport` and `_async_transport` share the two connection pools,
# `_provider_error` surfaces the 200-with-`error` replies OpenRouter sends for
# some upstream rejections.
from core.llm import Model, Usage, _async_transport, _provider_error, _transport

T = TypeVar("T")


# --- errors -------------------------------------------------------------
# As in the sibling clients: transport failures stay as the SDK's own
# exceptions. What is raised here is only a response that arrived and is
# unusable.


class RerankError(RuntimeError):
    """A rerank response arrived but could not be used."""


# --- value objects ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RerankUsage:
    """What one rerank call cost.

    `search_units` is Cohere's billing unit and has no equivalent on the chat or
    embedding endpoints, which is why this is its own type rather than the
    shared `Usage`. `as_usage()` folds it into the shared type so a trace can
    still total one request across all three clients.
    """

    search_units: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def as_usage(self) -> Usage:
        """The shared `Usage` view, for adding to LLM and embedding usage."""
        return Usage(total_tokens=self.total_tokens, cost=self.cost)


@dataclass(frozen=True, slots=True)
class Ranking:
    """One result: where the document was, and how well it scored."""

    index: int  # position in the documents passed in, NOT the rank
    relevance_score: float


@dataclass(frozen=True, slots=True)
class RerankResult:
    """Rankings best-first, plus what the call cost."""

    rankings: list[Ranking]
    usage: RerankUsage
    model: str

    def __len__(self) -> int:
        return len(self.rankings)

    @property
    def order(self) -> list[int]:
        """Input positions, best-first."""
        return [ranking.index for ranking in self.rankings]

    @property
    def top_score(self) -> float:
        """The cross-encoder score of the best document — a trace field.

        0.0 when nothing came back, so a trace always has a number to log.
        """
        return self.rankings[0].relevance_score if self.rankings else 0.0

    def apply(self, items: Sequence[T]) -> list[T]:
        """Reorder the caller's own objects, best-first, truncated to `top_n`.

        This is how the rerank node keeps whole `Chunk` dicts — with `chunk_id`
        and `doc` intact — instead of the bare text the API echoes back.
        """
        return [items[ranking.index] for ranking in self.rankings]

    def scored(self, items: Sequence[T]) -> list[tuple[T, float]]:
        """`apply()`, but pairing each item with its score for the trace."""
        return [(items[r.index], r.relevance_score) for r in self.rankings]


# --- client -------------------------------------------------------------


class RerankClient:
    """Rerank calls against one OpenRouter model.

    Built from a `(provider, name)` pair exactly like the chat and embedding
    clients, so swapping to another cross-encoder is a config change. `rerank()`
    and `arerank()` are twins over one set of guards; the graph node uses the
    async one.
    """

    def __init__(
        self,
        provider: str,
        name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.model = Model(provider, name)

        # Same injection rule as `LLMClient`: passing either transport builds
        # *only* what was passed, so calling the other side raises instead of
        # quietly constructing a real network client behind a test's back.
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
    def from_slug(cls, slug: str, **kwargs: Any) -> RerankClient:
        """Build from a `provider/name` string, which is how models are configured."""
        model = Model.parse(slug)
        return cls(model.provider, model.name, **kwargs)

    # --- calls ----------------------------------------------------------

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """Order `documents` by relevance to `query`, best first.

        `top_n` caps how many come back; `None` returns all of them. Passing a
        `top_n` equal to `len(documents)` is legal but pointless — the call then
        only permutes the set it was given, which is the failure mode to watch
        for when `retrieve_candidates` and `top_k` drift together.
        """
        body = self._request_body(query, documents, top_n)
        if body is None:
            return RerankResult(rankings=[], usage=RerankUsage(), model=str(self.model))
        if self._client is None:
            raise RerankError(
                f"{self.model} has no sync transport — it was built with async_client= only"
            )

        # `cast_to=object` returns the parsed JSON as a plain dict. A Pydantic
        # cast would be tidier but would raise on the 200-with-`error` replies
        # OpenRouter sends, losing the upstream's message — the one thing that
        # explains the failure.
        response = self._client.post("/rerank", body=body, cast_to=object)
        return RerankResult(
            rankings=self._rankings(response, count=len(documents)),
            usage=_rerank_usage(response),
            model=str(self.model),
        )

    async def arerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_n: int | None = None,
    ) -> RerankResult:
        """`rerank()` on the async transport — same guards, same response handling."""
        body = self._request_body(query, documents, top_n)
        if body is None:
            return RerankResult(rankings=[], usage=RerankUsage(), model=str(self.model))
        if self._async_client is None:
            raise RerankError(
                f"{self.model} has no async transport — pass async_client= too, or "
                "construct without client= to get both"
            )

        response = await self._async_client.post("/rerank", body=body, cast_to=object)
        return RerankResult(
            rankings=self._rankings(response, count=len(documents)),
            usage=_rerank_usage(response),
            model=str(self.model),
        )

    # --- plumbing -------------------------------------------------------

    def _request_body(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None,
    ) -> dict[str, Any] | None:
        """The validated request body, or `None` when there is nothing to ask.

        One request shape and one set of guards for both transports, so the two
        cannot fork — a validation rule that held on only one side would be a
        rejected request on the path that skipped it.
        """
        if not query.strip():
            raise ValueError("cannot rerank against an empty query")
        if not documents:
            # The API requires at least one document. Nothing to order anyway.
            return None
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")

        body: dict[str, Any] = {
            "model": str(self.model),
            "query": query,
            "documents": list(documents),
        }
        if top_n is not None:
            body["top_n"] = top_n
        return body

    def _rankings(self, response: Any, *, count: int) -> list[Ranking]:
        if not isinstance(response, dict):
            raise RerankError(f"{self.model} returned {type(response).__name__}, expected an object")

        results = response.get("results")
        if results is None:
            raise RerankError(f"{self.model} returned no results{_provider_error(response)}")
        if not isinstance(results, list):
            raise RerankError(f"{self.model} returned a non-list `results`")

        rankings: list[Ranking] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise RerankError(f"{self.model} returned a malformed result: {item!r}")
            try:
                index = int(item["index"])
                score = float(item["relevance_score"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RerankError(
                    f"{self.model} returned a result missing index/relevance_score: {item!r}"
                ) from exc

            # `index` is used to look up the caller's own objects, so a bad one
            # is either an IndexError later or — worse — a citation pointing at
            # the wrong chunk. Both are caught here instead.
            if not 0 <= index < count:
                raise RerankError(
                    f"{self.model} returned index {index} for {count} documents"
                )
            if index in seen:
                raise RerankError(f"{self.model} returned index {index} twice")
            seen.add(index)
            rankings.append(Ranking(index=index, relevance_score=score))

        # The API documents results as sorted by relevance, but the ordering is
        # the entire product of this call — asserting it costs nothing and means
        # `apply()` cannot silently hand back a worse order than it was given.
        rankings.sort(key=lambda ranking: ranking.relevance_score, reverse=True)
        return rankings

    def __repr__(self) -> str:
        return f"RerankClient({str(self.model)!r})"


def _rerank_usage(response: Any) -> RerankUsage:
    """Read usage off a rerank response, tolerating a provider that omits it."""
    usage = response.get("usage") if isinstance(response, dict) else None
    if not isinstance(usage, dict):
        return RerankUsage()

    def field(name: str, default: float = 0) -> Any:
        value = usage.get(name)
        return default if value is None else value

    return RerankUsage(
        search_units=int(field("search_units")),
        total_tokens=int(field("total_tokens")),
        cost=float(field("cost", 0.0)),
    )


# --- the configured model -----------------------------------------------


@lru_cache(maxsize=1)
def rerank_client() -> RerankClient:
    """The configured cross-encoder.

    Whether reranking runs at all is `rerank_enabled`, checked by the node —
    building a client does not commit to calling it.
    """
    return RerankClient.from_slug(get_settings().rerank_model)


__all__ = [
    "Ranking",
    "RerankClient",
    "RerankError",
    "RerankResult",
    "RerankUsage",
    "rerank_client",
]
