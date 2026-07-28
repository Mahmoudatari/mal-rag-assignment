"""LangGraph wiring — the only module in the repo that imports langgraph.

Nodes are plain functions with no framework imports, so the pipeline stays
testable without compiling a graph and the framework can be swapped by
rewriting this file alone.
"""

import psycopg
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row

from core.config import get_settings
from kb.store import async_pool, database_url
from rag.nodes import (
    generate,
    grade,
    no_answer,
    redact,
    rerank,
    reformulate,
    refuse,
    retrieve,
    router,
)
from rag.state import State

# --- conditional edges -------------------------------------------------
# Routing predicates live here, not in the nodes: they are wiring decisions
# that read state and return an edge name.


def after_router(state: State) -> str:
    """router → retrieve | refuse | generate."""
    route = state["route"]
    return "generate" if route == "answer" else route


def after_grade(state: State) -> str:
    """grade → generate | reformulate | no_answer."""
    if state["relevant"]:
        return "generate"
    if state.get("attempts", 0) < get_settings().max_retrieval_attempts:
        return "reformulate"
    return "no_answer"


# --- session persistence -----------------------------------------------


async def acheckpointer() -> AsyncPostgresSaver:
    """The session store: LangGraph's own tables, in the KB's database.

    Async because the graph is invoked with `ainvoke` and a sync saver has no
    async methods to answer it with — and `AsyncPostgresSaver.__init__` grabs
    the running event loop, so this must only ever be called from inside one:
    the app's lifespan, in practice.

    `setup()` is LangGraph's migration runner, not a one-off install — it reads
    `checkpoint_migrations` for the applied version and runs what is missing, so
    calling it on every boot is both idempotent and how a version bump lands.
    It creates four tables of its own; `kb/schema.sql` is untouched by it and
    knows nothing about them.

    The DDL takes its own one-shot connection rather than `async_pool()`, the
    same carve-out `kb.store.apply_schema()` makes: opening the pool runs the
    `register_vector_async` configure hook, which needs the `vector` extension
    to already exist, so setup through the pool would make this DDL — which has
    nothing to do with pgvector — fail on a virgin database. It also keeps
    setup independent of whether the lifespan has opened the pool yet. The
    connection kwargs are the saver's own contract (`from_conn_string` uses
    exactly these): three of the migrations are `CREATE INDEX CONCURRENTLY`,
    which Postgres refuses inside a transaction block, hence `autocommit`; the
    migration-version read is `row["v"]`, hence `dict_row`.

    Only the DDL is special. The returned saver reads and writes through the
    shared async pool, so sessions do not cost a second connection budget. That
    does mean the caller must apply the KB schema first, which `app/` has to do
    regardless: nothing can serve a request without it.
    """
    async with await psycopg.AsyncConnection.connect(
        database_url(), autocommit=True, prepare_threshold=0, row_factory=dict_row
    ) as conn:
        await AsyncPostgresSaver(conn).setup()
    return AsyncPostgresSaver(async_pool())


# --- assembly ----------------------------------------------------------


def build_graph(checkpointer=None):
    """Compile the graph. Pass a Postgres checkpointer to persist session state."""
    g = StateGraph(State)

    g.add_node("redact", redact.run)
    g.add_node("router", router.run)
    g.add_node("retrieve", retrieve.run)
    g.add_node("rerank", rerank.run)
    g.add_node("grade", grade.run)
    g.add_node("reformulate", reformulate.run)
    g.add_node("generate", generate.run)
    g.add_node("refuse", refuse.run)
    g.add_node("no_answer", no_answer.run)

    g.add_edge(START, "redact")
    g.add_edge("redact", "router")
    g.add_conditional_edges("router", after_router, ["retrieve", "refuse", "generate"])
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade")
    g.add_conditional_edges(
        "grade", after_grade, ["generate", "reformulate", "no_answer"]
    )
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", END)
    g.add_edge("refuse", END)
    g.add_edge("no_answer", END)

    return g.compile(checkpointer=checkpointer)
