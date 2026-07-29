"""Shared fixtures for the live evals: one compiled graph, one turn-runner.

The turn-runners replicate the proven pattern from `tests/test_nodes_live.py`:
the graph is driven with `ainvoke` (six nodes are coroutines — a compiled graph
holding async nodes has no working sync `invoke`), a turn that reads the index
owns the async pool for exactly its own lifetime, and the cached transports are
cleared before every turn because clients bind to the loop of their first
request and each `asyncio.run` here closes its loop on the way out.

Do not run the evals under xdist (`-n`): `run_golden`'s cache is per process,
so parallel workers would re-run turns — duplicate spend, though no
correctness issue.
"""

import os

# Before any other import: deepeval reads it at import time, and the grounding
# eval imports deepeval. Deliberately no deepeval import anywhere in this file —
# selecting only the free tests must never drag it in.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import asyncio
import uuid
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from core import embeddings, llm, rerank
from core.config import get_settings
from kb import store
from rag.graph import build_graph

# Same skip guards as `tests/test_nodes_live.py`. These are skipifs, not
# registered markers — eval files stack them alongside `pytest.mark.live`, the
# registered marker that keeps paid tests out of a default run.
live = pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
needs_db = pytest.mark.skipif(
    not get_settings().database_url, reason="needs DATABASE_URL pointing at the built index"
)


def _fresh_transports() -> None:
    """Drop every cached client — the loop they were built on is gone.

    Same two-layer clear as `tests/test_nodes_live.py`: the accessors cache
    client *instances* and `_async_transport` caches the httpx pool they share,
    so clearing either alone leaves something pointing at a dead loop.
    """
    llm._async_transport.cache_clear()
    llm.fast_llm.cache_clear()
    llm.answer_llm.cache_clear()
    llm.llm_for.cache_clear()
    embeddings.embedding_client.cache_clear()
    rerank.rerank_client.cache_clear()


def _turn(compiled: Any, question: str, *, with_db: bool) -> dict[str, Any]:
    """One first-turn conversation: fresh thread, own loop, own pool if needed."""
    _fresh_transports()

    async def go() -> dict[str, Any]:
        if with_db:
            await store.async_pool().open(wait=True, timeout=get_settings().db_pool_timeout)
        try:
            return await compiled.ainvoke(
                {"raw_query": question},
                config={"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}},
            )
        finally:
            if with_db:
                await store.close_async_pool()

    return asyncio.run(go())


@pytest.fixture(scope="session")
def run_golden():
    """question → final graph state, cached per question across eval files.

    Lazy — one selected test runs one turn — and run-once: the retrieval and
    grounding files parametrize over the same GOLDENS, so a full `-m live` run
    pays each golden exactly one graph turn.
    """
    # The two-tier retrieval assertions presume the funnel: `retrieve` pulls
    # `retrieve_candidates` and `rerank` cuts to `top_k`. With reranking off,
    # `retrieve` returns `top_k` directly and tier 1 would measure nothing.
    assert get_settings().rerank_enabled, "the retrieval evals require rerank_enabled"

    compiled = build_graph(checkpointer=InMemorySaver())
    cache: dict[str, dict[str, Any]] = {}

    def run(question: str) -> dict[str, Any]:
        if question not in cache:
            cache[question] = _turn(compiled, question, with_db=True)
        return cache[question]

    return run


@pytest.fixture(scope="session")
def invoke_turn():
    """question → final graph state, uncached, fresh thread per call.

    The default `with_db=False` keeps the refusal cases runnable with no
    DATABASE_URL — they must terminate at the router. A mis-route to retrieve
    then raises on the unopened pool, which is itself the failure signal.
    """
    compiled = build_graph(checkpointer=InMemorySaver())

    def run(question: str, *, with_db: bool = False) -> dict[str, Any]:
        return _turn(compiled, question, with_db=with_db)

    return run
