"""pgvector store: pool, schema, document replacement, similarity search.

The only module that talks to the knowledge-base tables. `kb/ingest.py` writes
through it at build time and `rag/nodes/retrieve.py` reads through it at request
time, so it sits in `kb/` and neither of them holds SQL.

Three things here are load-bearing rather than plumbing:

- **Writes replace a whole document, never a row.** `chunk_id` is positional
  (`murabaha-everyday-finance#027`), so editing one paragraph shifts the id of
  every chunk after it. A per-row upsert would leave the tail of the previous
  version behind under ids the new version no longer uses — orphaned text that
  still retrieves and still cites. `replace_document` deletes the document's
  chunks and rewrites them in one transaction.
- **Freshness is a hash of the chunks, plus the model that embedded them.**
  Either one changing invalidates the stored vectors. Hashing the source file
  instead would miss a change to `kb/chunking.py`; ignoring the model would keep
  vectors from a different vector space, which is the one failure that raises
  nothing anywhere (see `core/embeddings.py`).
- **Schema application does not go through the pool.** `register_vector` reads
  the `vector` type's OID out of the database, so it cannot run before
  `CREATE EXTENSION`. `apply_schema` therefore opens its own one-shot connection
  and the pool is only ever used afterwards. Getting this backwards makes a
  first deploy fail on an empty database and succeed on the retry.

There are two pools in this module, one per kind of process rather than one per
caller. The server opens `async_pool()` alone and hands it to the LangGraph
checkpointer; ingest, the CLI and the `-m db` tests open the sync `pool()`. No
process opens both, so `db_pool_max_size` bounds whichever one that process
uses. The split is the request path going async — `search` blocks an event loop,
`asearch` does not — and not a second connection budget.
"""

from __future__ import annotations

import atexit
import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector, register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool, ConnectionPool

from core.config import get_settings
from kb.chunking import Chunk

SCHEMA_PATH = Path(__file__).parent / "schema.sql"

# Duplicated in schema.sql, which cannot take a parameter for an identifier.
# `tests/test_store.py` asserts the two agree. Interpolated into SQL below as
# module constants only — nothing here ever formats a value into a query.
DOCUMENTS_TABLE = "sharia_documents"
CHUNKS_TABLE = "sharia_chunks"


class StoreError(RuntimeError):
    """The store could not be used."""


class SchemaNotAppliedError(StoreError):
    """The pgvector extension or the tables are missing."""


# --- value objects ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Match:
    """One retrieved chunk. `score` is cosine similarity, not distance.

    Postgres returns `<=>` as a distance in [0, 2]; this is `1 - distance`, so
    higher is more similar and the number reads the way "relevance score" does
    in the trace. It is observability only — relevance is the LLM grader's call,
    never a threshold on this value.
    """

    chunk_id: str
    doc: str
    section: str
    text: str
    score: float


@dataclass(frozen=True, slots=True)
class StoredDocument:
    """What the index already holds for one document — the skip decision."""

    doc: str
    content_hash: str
    embedding_model: str

    def is_current(self, *, content_hash: str, embedding_model: str) -> bool:
        """Both must match. A hash match alone would keep vectors from a
        previous embedding model, which is exactly the silent failure."""
        return self.content_hash == content_hash and self.embedding_model == embedding_model


@dataclass(frozen=True, slots=True)
class IndexStats:
    """Enough for `/health` to say something truer than 200."""

    documents: int
    chunks: int
    embedding_models: tuple[str, ...]

    @property
    def is_empty(self) -> bool:
        return self.chunks == 0

    def foreign_models(self, embedding_model: str) -> tuple[str, ...]:
        """Models in the index that are not the configured one.

        Non-empty means part of the index is in a different vector space from
        the queries, and those chunks will never be retrieved correctly. Kept as
        a query rather than an assertion so the caller decides whether that is
        a warning or a failed readiness check.
        """
        return tuple(m for m in self.embedding_models if m != embedding_model)


# --- freshness ----------------------------------------------------------


def content_hash(chunks: Sequence[Chunk]) -> str:
    """Fingerprint the exact strings that will be embedded.

    Over the chunks rather than the source file on purpose: the file is only one
    of the two inputs to what gets stored. Change `kb/chunking.py` — a new FAQ
    rule, a different header rendering — and the file is byte-identical while
    every stored chunk is wrong. Ids are folded in as well, so a pure reordering
    counts as a change; they are the citation keys, so their meaning moved even
    if the text did not.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


# --- connection ---------------------------------------------------------


def database_url() -> str:
    url = get_settings().database_url
    if not url:
        raise StoreError(
            "DATABASE_URL is not set — copy .env.example to .env and fill it in"
        )
    return url


def _configure(conn: psycopg.Connection) -> None:
    """Teach one pooled connection the `vector` type.

    Without this psycopg adapts a Python list as a Postgres array, which does
    not cast to `vector`, and the query fails on something that looks like a
    type error rather than a missing extension.
    """
    try:
        register_vector(conn)
    except psycopg.ProgrammingError as exc:
        raise SchemaNotAppliedError(
            "the `vector` extension is not installed in this database — run "
            "`uv run python -m kb.ingest`, which applies kb/schema.sql first"
        ) from exc


async def _configure_async(conn: psycopg.AsyncConnection) -> None:
    """`_configure` for the async pool. Same failure, same message."""
    try:
        await register_vector_async(conn)
    except psycopg.ProgrammingError as exc:
        raise SchemaNotAppliedError(
            "the `vector` extension is not installed in this database — run "
            "`uv run python -m kb.ingest`, which applies kb/schema.sql first"
        ) from exc


@lru_cache(maxsize=1)
def pool() -> ConnectionPool:
    """The sync pool, for build-time and CLI callers. Opened on first use.

    Ingest, the maintenance commands and the `-m db` tests run here; the server
    process uses `async_pool()` instead and never touches this one. Kept small
    anyway — a burst of ingest batches would otherwise open a connection each,
    and `db_pool_max_size` is the budget for whichever pool the process opens.
    """
    settings = get_settings()
    connections = ConnectionPool(
        database_url(),
        min_size=1,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout,
        configure=_configure,
        # Opened explicitly rather than in the constructor, which psycopg_pool
        # deprecates — it would connect on a background thread at import time.
        open=False,
        name="mal-kb",
    )
    connections.open(wait=True, timeout=settings.db_pool_timeout)
    # psycopg_pool prints a "couldn't stop thread" warning per worker if the
    # pool is still open at interpreter exit, which every CLI entrypoint would
    # otherwise trail. Registered once, since this function is cached, and
    # harmless if the caller closes first — close_pool() is idempotent.
    atexit.register(close_pool)
    return connections


def close_pool() -> None:
    """Release the pool. For app shutdown and for tests between databases."""
    if pool.cache_info().currsize:
        pool().close()
        pool.cache_clear()


@lru_cache(maxsize=1)
def async_pool() -> AsyncConnectionPool:
    """The request-time pool. Constructed here, opened by the app's lifespan.

    Unopened on return, and deliberately: opening is a coroutine, so there is no
    first-use hook that could do it the way `pool()` does, and nothing may
    connect at import. The lifespan calls `await async_pool().open(wait=True,
    timeout=...)` and `await close_async_pool()` — it owns both ends. There is no
    `atexit` counterpart for the same reason: nothing can be awaited there.

    The connection `kwargs` are the LangGraph `AsyncPostgresSaver` contract,
    which is why they are set pool-wide rather than by the checkpointer — it
    borrows connections from here and requires every one of them to arrive with
    autocommit on, `dict_row`, and prepared statements disabled.

    Pool-level autocommit is safe because nothing here needs a multi-statement
    transaction: this pool serves reads only — `asearch` and `astats` — plus the
    checkpointer's own writes, which manage their transactions themselves.
    `replace_document` and `prune` do depend on the implicit transaction psycopg
    opens per `with` block, and must never be routed through this pool; they
    stay on the sync one, where a failed rewrite rolls back instead of leaving a
    document with its old chunks deleted and none of its new ones written.
    """
    settings = get_settings()
    return AsyncConnectionPool(
        database_url(),
        min_size=1,
        max_size=settings.db_pool_max_size,
        timeout=settings.db_pool_timeout,
        configure=_configure_async,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
        # As in `pool()`: psycopg_pool deprecates opening in the constructor.
        open=False,
        name="mal-kb-async",
    )


async def close_async_pool() -> None:
    """Release the async pool. For app shutdown and for tests between loops.

    Clears the cache as well as closing, because a pool is bound to the event
    loop that opened it. A cached-but-closed pool would be handed to the next
    loop — the next test, or the next `--reload` cycle — which fails on tasks
    belonging to a loop that is gone.
    """
    if async_pool.cache_info().currsize:
        await async_pool().close()
        async_pool.cache_clear()


# --- schema -------------------------------------------------------------


def apply_schema() -> None:
    """Create the extension and the tables if they are absent.

    Runs on a one-shot connection outside the pool, because the pool's
    `configure` hook needs the `vector` type to already exist. Idempotent: every
    statement in schema.sql is `IF NOT EXISTS`.
    """
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with psycopg.connect(database_url()) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)  # type: ignore[arg-type]
        conn.commit()
    _verify_dimensions()


def _verify_dimensions() -> None:
    """Fail at startup if the column width and the configured width disagree.

    They would otherwise disagree at the first INSERT of an ingest — after the
    embedding calls have already been paid for. `schema.sql` hardcodes 1536
    because a .sql file cannot read settings, so this is the seam where the two
    can drift.
    """
    settings = get_settings()
    with psycopg.connect(database_url()) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.atttypmod
            FROM pg_attribute a
            JOIN pg_class c ON c.oid = a.attrelid
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relname = %s AND a.attname = 'embedding'
              AND n.nspname = current_schema() AND NOT a.attisdropped
            """,
            (CHUNKS_TABLE,),
        )
        row = cur.fetchone()

    if row is None:
        raise SchemaNotAppliedError(f"{CHUNKS_TABLE}.embedding does not exist")
    # pgvector stores the declared dimension directly in atttypmod, unlike the
    # built-in varlena types which add a header offset.
    stored = int(row[0])
    if stored != settings.embedding_dimensions:
        raise StoreError(
            f"{CHUNKS_TABLE}.embedding is vector({stored}) but EMBEDDING_DIMENSIONS "
            f"is {settings.embedding_dimensions}. Change the setting back, or drop "
            f"the table and edit kb/schema.sql — the stored vectors are the wrong "
            f"width either way and would have to be rebuilt."
        )


# --- writes -------------------------------------------------------------


def stored_documents() -> dict[str, StoredDocument]:
    """What is already indexed, keyed by document. Empty on a fresh database."""
    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"SELECT doc, content_hash, embedding_model FROM {DOCUMENTS_TABLE}"  # noqa: S608
        )
        return {row["doc"]: StoredDocument(**row) for row in cur.fetchall()}


def replace_document(
    doc: str,
    *,
    title: str,
    chunks: Sequence[Chunk],
    vectors: Sequence[Sequence[float]],
    embedding_model: str,
) -> None:
    """Rewrite one document's chunks, in a single transaction.

    Wholesale rather than per-row because `chunk_id` is positional: a document
    that loses a paragraph also loses its highest ids, and an upsert would leave
    those rows behind still holding the previous version's text.

    `chunks` and `vectors` are zipped strictly. A length or order mismatch here
    attaches every embedding to the wrong text — a corruption that raises
    nothing, indexes cleanly, and only shows up as retrieval that is mildly
    disappointing. `Embeddings` guarantees the order; this refuses to proceed
    without the length.
    """
    if len(chunks) != len(vectors):
        raise StoreError(
            f"{doc}: {len(chunks)} chunks but {len(vectors)} vectors — refusing to "
            "write, every embedding after the mismatch would land on the wrong text"
        )
    if not chunks:
        raise StoreError(f"{doc}: no chunks — a document that chunks to nothing is a bug")
    stray = {chunk.doc for chunk in chunks} - {doc}
    if stray:
        raise StoreError(f"{doc}: chunks from another document in the batch: {sorted(stray)}")

    rows = [
        (chunk.chunk_id, chunk.doc, chunk.section, chunk.text, Vector(vector))
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    with pool().connection() as conn, conn.cursor() as cur:
        # Parent first: the chunks' foreign key needs it to exist. `ingested_at`
        # is set explicitly rather than left to the column default, which only
        # applies on INSERT and would go stale on the UPDATE branch.
        cur.execute(
            f"""
            INSERT INTO {DOCUMENTS_TABLE} (doc, title, content_hash, embedding_model, ingested_at)
            VALUES (%s, %s, %s, %s, now())
            ON CONFLICT (doc) DO UPDATE SET
                title           = EXCLUDED.title,
                content_hash    = EXCLUDED.content_hash,
                embedding_model = EXCLUDED.embedding_model,
                ingested_at     = now()
            """,  # noqa: S608
            (doc, title, content_hash(chunks), embedding_model),
        )
        cur.execute(f"DELETE FROM {CHUNKS_TABLE} WHERE doc = %s", (doc,))  # noqa: S608
        cur.executemany(
            f"""
            INSERT INTO {CHUNKS_TABLE} (chunk_id, doc, section, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,  # noqa: S608
            rows,
        )


def prune(keep: Collection[str]) -> list[str]:
    """Drop documents that are no longer in `kb/documents/`. Cascades to chunks.

    Refuses an empty `keep` rather than treating it as "delete everything": the
    realistic way to arrive here with nothing is a wrong path or a failed glob,
    and wiping the index on a typo is not a recovery anyone wants at deploy time.
    """
    if not keep:
        raise StoreError(
            "prune() called with no documents to keep — that would empty the index. "
            "If the corpus really is empty, delete the rows by hand."
        )
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(
            f"DELETE FROM {DOCUMENTS_TABLE} WHERE doc <> ALL(%s) RETURNING doc",  # noqa: S608
            (list(keep),),
        )
        return sorted(row[0] for row in cur.fetchall())


# --- reads --------------------------------------------------------------


def search(vector: Sequence[float], limit: int) -> list[Match]:
    """Nearest chunks by cosine distance, closest first.

    `limit` is required and has no default: the caller decides whether it wants
    `retrieve_candidates` to feed the reranker or `top_k` directly, and a
    default here would quietly make that choice for it.

    Ordered by the `<=>` operator rather than by the computed score, which is
    the same order but is the form an ANN index could serve if the corpus ever
    grows enough to want one.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    with pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT chunk_id, doc, section, text,
                   1 - (embedding <=> %(query)s) AS score
            FROM {CHUNKS_TABLE}
            ORDER BY embedding <=> %(query)s
            LIMIT %(limit)s
            """,  # noqa: S608
            {"query": Vector(vector), "limit": limit},
        )
        return [
            Match(
                chunk_id=row["chunk_id"],
                doc=row["doc"],
                section=row["section"],
                text=row["text"],
                score=float(row["score"]),
            )
            for row in cur.fetchall()
        ]


def stats() -> IndexStats:
    """Document count, chunk count, and every embedding model in the index."""
    with pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {DOCUMENTS_TABLE}")  # noqa: S608
        documents = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute(f"SELECT count(*) FROM {CHUNKS_TABLE}")  # noqa: S608
        chunks = int(cur.fetchone()[0])  # type: ignore[index]
        cur.execute(
            f"SELECT DISTINCT embedding_model FROM {DOCUMENTS_TABLE} ORDER BY 1"  # noqa: S608
        )
        models = tuple(row[0] for row in cur.fetchall())
    return IndexStats(documents=documents, chunks=chunks, embedding_models=models)


# --- reads, async -------------------------------------------------------
# `search` and `stats` on the async pool — same queries, same contracts, for the
# request path, which cannot block an event loop on a socket. Kept as explicit
# twins rather than one implementation behind a bridge: `asyncio.to_thread`
# would hold a sync connection for the duration and put the two pools in every
# server process, and psycopg's sync and async cursors have no shared protocol
# to write once against. The cost is that the pairs must be changed together,
# which `tests/test_store.py` pins by asserting they return the same rows.


async def asearch(vector: Sequence[float], limit: int) -> list[Match]:
    """`search()` on the async pool — same query, same ordering, same contract.

    `limit` is required here for the same reason it is there: the caller picks
    `retrieve_candidates` or `top_k`, and a default would choose for it.
    """
    if limit < 1:
        raise ValueError(f"limit must be at least 1, got {limit}")

    async with async_pool().connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                f"""
                SELECT chunk_id, doc, section, text,
                       1 - (embedding <=> %(query)s) AS score
                FROM {CHUNKS_TABLE}
                ORDER BY embedding <=> %(query)s
                LIMIT %(limit)s
                """,  # noqa: S608
                {"query": Vector(vector), "limit": limit},
            )
            return [
                Match(
                    chunk_id=row["chunk_id"],
                    doc=row["doc"],
                    section=row["section"],
                    text=row["text"],
                    score=float(row["score"]),
                )
                for row in await cur.fetchall()
            ]


async def astats() -> IndexStats:
    """`stats()` on the async pool, for `/health`.

    Every column is aliased and read by name, unlike the sync version: this
    pool sets `row_factory=dict_row` for the checkpointer, so a positional
    `row[0]` read would raise a KeyError.
    """
    async with async_pool().connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(f"SELECT count(*) AS n FROM {DOCUMENTS_TABLE}")  # noqa: S608
            documents = int((await cur.fetchone())["n"])  # type: ignore[index]
            await cur.execute(f"SELECT count(*) AS n FROM {CHUNKS_TABLE}")  # noqa: S608
            chunks = int((await cur.fetchone())["n"])  # type: ignore[index]
            await cur.execute(
                f"SELECT DISTINCT embedding_model AS model FROM {DOCUMENTS_TABLE} ORDER BY 1"  # noqa: S608
            )
            models = tuple(row["model"] for row in await cur.fetchall())
    return IndexStats(documents=documents, chunks=chunks, embedding_models=models)


__all__ = [
    "CHUNKS_TABLE",
    "DOCUMENTS_TABLE",
    "IndexStats",
    "Match",
    "SchemaNotAppliedError",
    "StoreError",
    "StoredDocument",
    "apply_schema",
    "asearch",
    "astats",
    "async_pool",
    "close_async_pool",
    "close_pool",
    "content_hash",
    "database_url",
    "pool",
    "prune",
    "replace_document",
    "search",
    "stats",
    "stored_documents",
]
