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


# The OpenAPI document (served at /openapi.json, rendered at /docs) is
# generated from this metadata plus the schemas' field descriptions — there is
# no hand-written spec file to drift from the code.
app = FastAPI(
    title="Mal Islamic Finance Assistant",
    version="1.0.0",
    summary=(
        "RAG assistant answering Islamic finance questions grounded in Mal's "
        "Sharia finance knowledge base, plus the customer's own account context."
    ),
    description=(
        "Ask about Mal's five Sharia finance topics — Murabaha everyday finance, "
        "Ijara auto lease-to-own, fractional Sukuk investing, Wakala savings, and "
        "the late-payment/charity policy — or about your own holdings by passing "
        "an `account_id`.\n\n"
        "**Grounding** — answers cite the knowledge base chunks they draw on via "
        "inline `[n]` markers plus a structured `references` list. Out-of-scope "
        "questions are refused; in-scope questions the knowledge base cannot "
        "answer are handed to support.\n\n"
        "**Sessions** — omit `session_id` on the first turn and pass the "
        "returned id back to keep a conversation going; history lives "
        "server-side. Ids are server-minted; invented ones are rejected.\n\n"
        "**PII** — customer identifiers in `message` are masked before reaching "
        "any model, log or trace.\n\n"
        "**Synthetic data** — all accounts and documents are fictitious. Demo "
        "account ids: `MAL-1001-2200-4417` (active Murabaha + Wakala savings), "
        "`MAL-2002-3300-8802` (Ijara lease + Sukuk holdings), "
        "`MAL-3003-4400-1103` (Murabaha in arrears)."
    ),
    openapi_tags=[
        {"name": "chat", "description": "Stateful conversation with the assistant."},
        {"name": "health", "description": "Readiness of the service and its index."},
    ],
    lifespan=lifespan,
)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    summary="One turn of a stateful conversation",
    response_description=(
        "The assistant's reply with citations, the outcome, and the session id "
        "to continue with. Refusals and hand-offs are 200s with the outcome "
        "named, not errors."
    ),
    responses={
        422: {"description": "Validation failure — empty message, or a malformed session_id/account_id."},
        500: {
            "description": (
                "Unexpected failure. The body is deliberately generic; detail "
                "goes to the trace log, never the response."
            ),
            "content": {"application/json": {"example": {"detail": "internal error"}}},
        },
    },
)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """One turn of a stateful conversation.

    `session_id` does double duty: it goes into the invoke payload because no
    node writes it and the trace reads it from final state, and it is the
    checkpointer's `thread_id`, which is what carries history across turns.
    Minted here when absent; when present the schema has already enforced the
    minted format, so every thread_id in the checkpointer is 122 bits of
    server-chosen randomness — a client cannot park a conversation on a
    guessable id for someone else to walk into.
    `account_id` rides along the same way — the account node resolves it into
    the customer's holdings; absent is sent as "" so the node always has the
    key to read and a turn without an id cannot inherit the previous turn's.
    """
    session_id = payload.session_id or uuid4().hex
    watch = Stopwatch()
    error = ""
    # The fallback for the trace: a request that dies before the graph returns
    # still logs its session and latency rather than destroying the evidence.
    state: Mapping[str, Any] = {"session_id": session_id}
    try:
        state = await request.app.state.graph.ainvoke(
            {
                "raw_query": payload.message,
                "session_id": session_id,
                "account_id": payload.account_id or "",
            },
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


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Readiness of the service and its index",
    response_description="All checks passing; the index is populated and current.",
    responses={
        503: {
            "model": HealthResponse,
            "description": (
                "Degraded — the database is unreachable, the index is empty, or "
                "part of it was embedded by an unconfigured model. `detail` "
                "names the failing check."
            ),
        },
    },
)
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
