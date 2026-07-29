"""Cross-encoder reranking via OpenRouter: POST /api/v1/rerank.

Second stage of retrieval. `retrieve` pulls `retrieve_candidates` by vector
similarity; this reorders them and keeps `top_k`. Pointless unless candidates
exceed top_k — with both equal it only permutes the same set.

Request:  { model, query, documents[], top_n }
Response: { results[{ index, relevance_score, document }], usage{ search_units,
            total_tokens, cost } }

`relevance_score` here is the better value for the trace's required relevance
score than raw cosine — log both, and record `usage.cost` alongside token usage.

Skipped entirely when `rerank_enabled` is false, in which case `retrieve` should
return top_k directly.
"""

from openai import APIError

from core.config import get_settings
from core.rerank import RerankError, rerank_client
from rag.nodes._common import logged, usage_entry
from rag.state import Chunk, State


async def run(state: State) -> dict:
    """chunks (candidates) → chunks (reordered, truncated to top_k)."""
    chunks = state.get("chunks", [])
    settings = get_settings()
    if not settings.rerank_enabled or not chunks:
        # `retrieve` already returned top_k directly when reranking is off,
        # and there is nothing to reorder when it returned no candidates
        # either way. No client construction, no call.
        return {}

    query = state.get("search_query") or state.get("query", "")

    try:
        result = await rerank_client().arerank(
            query, [chunk["text"] for chunk in chunks], top_n=settings.top_k
        )
    except (RerankError, APIError):
        # Reranking is an ordering optimisation over a candidate set `retrieve`
        # has already produced in cosine order, and relevance stays the grader's
        # call — so nothing downstream needs this stage to have run. That makes
        # every way it can fail equivalent: whatever went wrong, the correct
        # response is to keep the candidates and truncate them. Failing the turn
        # would be reranking gating control flow, which is the one thing it must
        # never do. The missing `rerank_score` on every chunk is the signal a
        # trace reads to show this happened.
        #
        # Two classes, because they arrive by different routes. `RerankError` is
        # a reply that arrived and was unusable — OpenRouter's 200-with-`error`
        # shape for some upstream rejections, or an index the response should
        # not have contained. `openai.APIError` is the SDK's base for the rest:
        # `APIStatusError` (429, 5xx, 401) and `APIConnectionError`, whose
        # subclass is `APITimeoutError`, so timeouts are covered by the base and
        # need no separate arm. The SDK has already spent its own retries by the
        # time one of these surfaces, but falling open is not a further attempt
        # — it is skipping a stage the answer does not depend on.
        #
        # Deliberately not `except Exception`: a KeyError or TypeError here is a
        # bug in this module's own projection, and quietly serving cosine order
        # for it would hide the bug behind slightly worse answers forever.
        return {"chunks": chunks[: settings.top_k]}

    # Rebuilt from the caller's own chunk dicts via `scored`, never from the
    # echoed `document` text — the echo is the string that was sent, and
    # reading it back would return bare text with `chunk_id` and `doc` gone,
    # the two fields citations are built from.
    reranked: list[Chunk] = [
        {**chunk, "rerank_score": score} for chunk, score in result.scored(chunks)
    ]

    # cohere/rerank-v3.5 bills on search units and reports zero tokens; a trace
    # that only logged `as_usage()`'s tokens would read reranking as free.
    entry = usage_entry(
        "rerank",
        result.model,
        result.usage.as_usage(),
        search_units=result.usage.search_units,
    )
    return {"chunks": reranked, "usage_log": logged(state, entry)}
