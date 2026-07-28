"""FastAPI entrypoint: lifespan wiring, POST /chat, GET /health.

The app is the graph's only caller and `observability/`'s only importer. Three
jobs live here and nowhere else:

- **Lifespan owns every loop-bound resource.** The async pool, the checkpointer
  (whose constructor grabs the running event loop) and the compiled graph are
  built inside `lifespan` because none of them may exist before the loop does.
  spaCy is warmed there too — ~1s of model load that must not land inside the
  first request's `redact`.
- **`/chat` does bookkeeping, not thinking.** Session id in, graph invoked,
  exactly one trace line out — emitted in `finally`, so the turn that crashed
  is the turn that is guaranteed to be logged.
- **The trace is the diagnostic channel.** The HTTP 500 body is deliberately
  generic: an exception raised before `redact` can carry the raw, unmasked
  message, and error bodies are an output surface the PII layer never sees.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request, Response

from app.schemas import ChatRequest, ChatResponse, HealthResponse
from core.config import get_settings
from kb.store import astats, async_pool, close_async_pool
from observability import Stopwatch, Trace, emit
from pii import detect
from rag.graph import acheckpointer, build_graph


async def serving_graph() -> Any:
    """Open the pool, run the checkpointer's DDL, compile. Once, at boot.

    The lifespan's one seam: tests monkeypatch this to serve a stub graph with
    no database behind it, and everything else about the app stays real.
    """
    settings = get_settings()
    await async_pool().open(wait=True, timeout=settings.db_pool_timeout)
    return build_graph(checkpointer=await acheckpointer())


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    detect("warm up")
    app.state.graph = await serving_graph()
    try:
        yield
    finally:
        # No-op when serving_graph was stubbed and the pool never opened.
        await close_async_pool()


app = FastAPI(title="Mal Islamic Finance Assistant", lifespan=lifespan)


@app.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """One turn of a stateful conversation.

    `session_id` does double duty: it goes into the invoke payload because no
    node writes it and the trace reads it from final state, and it is the
    checkpointer's `thread_id`, which is what carries history across turns.
    """
    session_id = payload.session_id or uuid4().hex
    watch = Stopwatch()
    error = ""
    # The fallback for the trace: a request that dies before the graph returns
    # still logs its session and latency rather than destroying the evidence.
    state: Mapping[str, Any] = {"session_id": session_id}
    try:
        state = await request.app.state.graph.ainvoke(
            {"raw_query": payload.message, "session_id": session_id},
            config={"configurable": {"thread_id": session_id}},
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise HTTPException(status_code=500, detail="internal error") from exc
    finally:
        emit(Trace.from_state(state, latency_ms=watch.ms, error=error))

    return ChatResponse(
        answer=state.get("answer", ""),
        references=state.get("references", []),
        outcome=state.get("outcome", "no_answer"),
        session_id=session_id,
    )


@app.get("/health", response_model=HealthResponse)
async def health(response: Response) -> HealthResponse:
    """Readiness, not liveness: 200 means a question could be answered now.

    Three checks, each of which has actually broken a RAG deployment somewhere:
    the database answers, the index has chunks in it, and every stored chunk
    was embedded by the configured model — `foreign_models` is the one failure
    that raises nothing at query time, it just quietly retrieves noise.
    """
    try:
        stats = await astats()
    except Exception as exc:
        response.status_code = 503
        return HealthResponse(
            status="degraded",
            documents=0,
            chunks=0,
            foreign_models=[],
            detail=f"database unavailable: {type(exc).__name__}",
        )

    foreign = list(stats.foreign_models(get_settings().embedding_model))
    if stats.is_empty:
        detail = "index is empty — run `uv run python -m kb.ingest`"
    elif foreign:
        detail = "part of the index was embedded by a model that is not configured"
    else:
        detail = ""

    if detail:
        response.status_code = 503
    return HealthResponse(
        status="degraded" if detail else "ok",
        documents=stats.documents,
        chunks=stats.chunks,
        foreign_models=foreign,
        detail=detail,
    )
