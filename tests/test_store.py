"""Unit tests for the pgvector store.

Split in two. Most of this file needs no database: the freshness fingerprint,
the skip decision and the schema/constant agreement are all pure, and they are
where the store's actual reasoning lives.

The rest is marked `db` and deselected by default. Those tests write and delete,
so they refuse to run against `DATABASE_URL` — point `TEST_DATABASE_URL` at a
throwaway Postgres with pgvector:

    docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=pg -e POSTGRES_DB=maltest \\
        pgvector/pgvector:pg17
    TEST_DATABASE_URL=postgresql://postgres:pg@localhost:55432/maltest uv run pytest -m db

The guard is not ceremony: `prune` and `replace_document` delete rows, and the
deployed index is one connection string away.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Coroutine, Iterator
from typing import Any

import pytest

from kb import store
from kb.chunking import Chunk, load_corpus


def chunk(chunk_id: str = "doc#000", *, doc: str = "doc", text: str = "body") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc=doc, title="Title", section="Section", text=text)


# --- the schema file and the constants that name it ----------------------


def test_table_constants_match_the_schema_file() -> None:
    """The one duplication the design accepts — a .sql file cannot take an
    identifier as a parameter. This is what keeps it from drifting."""
    sql = store.SCHEMA_PATH.read_text(encoding="utf-8")

    assert f"CREATE TABLE IF NOT EXISTS {store.DOCUMENTS_TABLE}" in sql
    assert f"CREATE TABLE IF NOT EXISTS {store.CHUNKS_TABLE}" in sql


def test_schema_declares_the_configured_embedding_width() -> None:
    """schema.sql hardcodes the width because it cannot read settings, so this
    is the seam where the file and core/config.py can disagree. In a live
    database `_verify_dimensions` catches it; this catches it in CI."""
    from core.config import get_settings

    sql = store.SCHEMA_PATH.read_text(encoding="utf-8")

    assert f"vector({get_settings().embedding_dimensions})" in sql


# --- the freshness fingerprint -------------------------------------------
# What decides whether ingest re-embeds a document. Too loose and a deploy
# ships stale vectors; too tight and every deploy pays for the whole corpus.


def test_content_hash_is_stable_for_the_same_chunks() -> None:
    chunks = [chunk("doc#000"), chunk("doc#001", text="second")]

    assert store.content_hash(chunks) == store.content_hash(list(chunks))


def test_content_hash_changes_when_the_text_changes() -> None:
    before = [chunk("doc#000", text="profit rate is 4.5%")]
    after = [chunk("doc#000", text="profit rate is 5.5%")]

    assert store.content_hash(before) != store.content_hash(after)


def test_content_hash_changes_when_only_the_ids_move() -> None:
    """Reordering leaves every string in the corpus intact but repoints the
    citation keys, so the stored rows are wrong even though the text is not."""
    chunks = [chunk("doc#000", text="a"), chunk("doc#001", text="b")]
    swapped = [chunk("doc#000", text="b"), chunk("doc#001", text="a")]

    assert store.content_hash(chunks) != store.content_hash(swapped)


def test_content_hash_is_not_confusable_by_concatenation() -> None:
    """Fields are NUL-delimited, so "ab" + "c" cannot collide with "a" + "bc"."""
    one = [chunk("doc#000", text="ab"), chunk("doc#001", text="c")]
    two = [chunk("doc#000", text="a"), chunk("doc#001", text="bc")]

    assert store.content_hash(one) != store.content_hash(two)


def test_content_hash_covers_the_real_corpus_deterministically() -> None:
    assert store.content_hash(load_corpus()) == store.content_hash(load_corpus())


# --- the skip decision ---------------------------------------------------


def test_document_is_current_only_when_hash_and_model_both_match() -> None:
    stored = store.StoredDocument("doc", content_hash="abc", embedding_model="google/e-001")

    assert stored.is_current(content_hash="abc", embedding_model="google/e-001")
    assert not stored.is_current(content_hash="xyz", embedding_model="google/e-001")


def test_a_changed_embedding_model_invalidates_an_unchanged_document() -> None:
    """The failure this whole column exists for. Same text, same hash, but the
    stored vectors are in a different space from the queries — and nothing
    downstream raises, retrieval just quietly returns the wrong chunks."""
    stored = store.StoredDocument("doc", content_hash="abc", embedding_model="google/e-001")

    assert not stored.is_current(content_hash="abc", embedding_model="openai/text-embedding-3-small")


# --- index stats ---------------------------------------------------------


def test_foreign_models_names_only_the_models_that_are_not_configured() -> None:
    """A non-empty result means part of the index is in a different vector space
    from the queries — those chunks are unreachable however good the query is."""
    stats = store.IndexStats(2, 40, ("google/e-001", "openai/legacy"))

    assert stats.foreign_models("google/e-001") == ("openai/legacy",)


def test_a_fully_configured_index_reports_no_foreign_models() -> None:
    stats = store.IndexStats(5, 202, ("google/e-001",))

    assert stats.foreign_models("google/e-001") == ()
    assert not stats.is_empty


def test_an_empty_index_is_empty() -> None:
    assert store.IndexStats(0, 0, ()).is_empty


# --- against a real database ---------------------------------------------

@pytest.fixture(scope="module")
def database() -> Iterator[str]:
    """Point the store at TEST_DATABASE_URL, and refuse to share with the app's."""
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set — see this module's docstring")

    from core.config import get_settings

    get_settings.cache_clear()
    store.close_pool()
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("DATABASE_URL", url)
        get_settings.cache_clear()
        if get_settings().database_url != url:
            pytest.fail("settings did not pick up TEST_DATABASE_URL")
        store.apply_schema()
        yield url
        store.close_pool()
    get_settings.cache_clear()


@pytest.fixture
def clean(database: str) -> None:
    """Empty both tables. Deleting the parent cascades to the chunks."""
    with store.pool().connection() as conn, conn.cursor() as cur:
        cur.execute(f"DELETE FROM {store.DOCUMENTS_TABLE}")  # noqa: S608


def vector(seed: int, *, dimensions: int = 1536) -> list[float]:
    """A unit vector along one axis. Orthogonal to every other seed, which makes
    cosine similarity exactly 1.0 against itself and 0.0 against the rest."""
    values = [0.0] * dimensions
    values[seed % dimensions] = 1.0
    return values


def write(doc: str, chunks: list[Chunk], *, model: str = "test/model") -> None:
    store.replace_document(
        doc,
        title="Title",
        chunks=chunks,
        vectors=[vector(i) for i in range(len(chunks))],
        embedding_model=model,
    )


@pytest.mark.db
def test_schema_is_idempotent(database: object) -> None:
    store.apply_schema()
    store.apply_schema()


@pytest.mark.db
def test_chunks_round_trip_and_search_ranks_by_cosine(clean: None) -> None:
    chunks = [chunk(f"doc#{i:03d}", text=f"body {i}") for i in range(3)]
    write("doc", chunks)

    hits = store.search(vector(1), limit=3)

    # Only the winner's position is asserted: the other two vectors are both
    # orthogonal to the query, so their order is a tie Postgres may break either way.
    assert hits[0].chunk_id == "doc#001"
    assert hits[0].text == "body 1"
    assert hits[0].section == "Section"
    assert hits[0].score == pytest.approx(1.0)
    assert [hit.score for hit in hits[1:]] == pytest.approx([0.0, 0.0])


@pytest.mark.db
def test_search_limit_is_respected(clean: None) -> None:
    write("doc", [chunk(f"doc#{i:03d}", text=f"body {i}") for i in range(5)])

    assert len(store.search(vector(0), limit=2)) == 2


@pytest.mark.db
def test_replacing_a_shorter_document_drops_the_orphaned_tail(clean: None) -> None:
    """The reason writes are wholesale. `chunk_id` is positional, so a document
    that loses a paragraph loses its highest ids — an upsert would leave those
    rows behind, still retrievable and still citable."""
    write("doc", [chunk(f"doc#{i:03d}", text=f"body {i}") for i in range(5)])

    write("doc", [chunk(f"doc#{i:03d}", text=f"body {i}") for i in range(2)])

    assert store.stats().chunks == 2
    assert {hit.chunk_id for hit in store.search(vector(4), limit=5)} == {"doc#000", "doc#001"}


@pytest.mark.db
def test_replacing_one_document_leaves_the_others_alone(clean: None) -> None:
    write("a", [chunk("a#000", doc="a")])
    write("b", [chunk("b#000", doc="b")])

    write("a", [chunk("a#000", doc="a", text="rewritten")])

    assert store.stats() == store.IndexStats(2, 2, ("test/model",))


@pytest.mark.db
def test_stored_documents_reports_the_hash_that_was_written(clean: None) -> None:
    chunks = [chunk("doc#000", text="body")]
    write("doc", chunks)

    stored = store.stored_documents()["doc"]

    assert stored.is_current(content_hash=store.content_hash(chunks), embedding_model="test/model")


@pytest.mark.db
def test_prune_removes_absent_documents_and_cascades_to_their_chunks(clean: None) -> None:
    write("keep", [chunk("keep#000", doc="keep")])
    write("gone", [chunk(f"gone#{i:03d}", doc="gone") for i in range(3)])

    assert store.prune(keep=["keep"]) == ["gone"]
    assert store.stats() == store.IndexStats(1, 1, ("test/model",))


@pytest.mark.db
def test_prune_refuses_to_empty_the_index(clean: None) -> None:
    write("doc", [chunk("doc#000")])

    with pytest.raises(store.StoreError, match="empty the index"):
        store.prune(keep=[])

    assert store.stats().chunks == 1


@pytest.mark.db
def test_stats_reports_every_embedding_model_present(clean: None) -> None:
    write("a", [chunk("a#000", doc="a")], model="google/e-001")
    write("b", [chunk("b#000", doc="b")], model="openai/legacy")

    assert store.stats().foreign_models("google/e-001") == ("openai/legacy",)


@pytest.mark.db
def test_a_length_mismatch_refuses_to_write(clean: None) -> None:
    """The corruption with no symptom: one missing vector shifts every
    embedding after it onto the wrong text, and the index still builds."""
    chunks = [chunk(f"doc#{i:03d}") for i in range(3)]

    with pytest.raises(store.StoreError, match="wrong text"):
        store.replace_document(
            "doc", title="T", chunks=chunks, vectors=[vector(0)], embedding_model="test/model"
        )

    assert store.stats().chunks == 0


@pytest.mark.db
def test_chunks_from_another_document_refuse_to_write(clean: None) -> None:
    with pytest.raises(store.StoreError, match="another document"):
        store.replace_document(
            "doc",
            title="T",
            chunks=[chunk("other#000", doc="other")],
            vectors=[vector(0)],
            embedding_model="test/model",
        )


@pytest.mark.db
def test_a_document_that_chunks_to_nothing_refuses_to_write(clean: None) -> None:
    with pytest.raises(store.StoreError, match="no chunks"):
        store.replace_document(
            "doc", title="T", chunks=[], vectors=[], embedding_model="test/model"
        )


# --- the async twins ------------------------------------------------------
# Same database, same rows. `asearch` and `astats` promise "same query, same
# contract" as their sync siblings; these are the parity tests the design
# comment in kb/store.py points at.


def on_async_pool(coro: Coroutine[Any, Any, Any]) -> Any:
    """Open the async pool, await `coro`, close — all inside one event loop.

    One `asyncio.run` per test on purpose: the pool binds to the loop that
    opens it, so opening in one loop and querying from another is exactly the
    failure `close_async_pool` exists to prevent.
    """

    async def go() -> Any:
        await store.async_pool().open(wait=True, timeout=10.0)
        try:
            return await coro
        finally:
            await store.close_async_pool()

    return asyncio.run(go())


def test_asearch_rejects_a_limit_below_one() -> None:
    """The guard runs before the pool is touched, so no database is needed."""
    with pytest.raises(ValueError, match="at least 1"):
        asyncio.run(store.asearch([0.0], limit=0))


@pytest.mark.db
def test_asearch_returns_what_search_returns(clean: None) -> None:
    write("doc", [chunk(f"doc#{i:03d}", text=f"body {i}") for i in range(3)])

    sync_hits = store.search(vector(1), limit=3)
    async_hits = on_async_pool(store.asearch(vector(1), limit=3))

    # The winner is compared exactly; the two orthogonal also-rans are a tie
    # Postgres may break differently per query, so they are compared as a set.
    assert async_hits[0] == sync_hits[0]
    assert async_hits[0].score == pytest.approx(1.0)
    assert {hit.chunk_id for hit in async_hits} == {hit.chunk_id for hit in sync_hits}


@pytest.mark.db
def test_astats_returns_what_stats_returns(clean: None) -> None:
    write("a", [chunk("a#000", doc="a")], model="google/e-001")
    write("b", [chunk("b#000", doc="b")], model="openai/legacy")

    assert on_async_pool(store.astats()) == store.stats()


@pytest.mark.db
def test_the_async_pool_survives_a_fresh_event_loop(database: str) -> None:
    """`close_async_pool` clears the lru_cache as well as closing: a cached
    pool belongs to the loop that opened it, and handing it to the next loop
    fails on tasks from a loop that is gone. Two `asyncio.run` calls are
    exactly that situation — this is the --reload / TestClient regression."""
    assert on_async_pool(store.astats()) == on_async_pool(store.astats())


@pytest.mark.db
def test_the_real_corpus_fits_the_schema(clean: None) -> None:
    """Every chunking rule the corpus exercises, written through the real
    columns — NOT NULL, the non-empty CHECK, and the foreign key."""
    corpus = load_corpus()
    by_doc: dict[str, list[Chunk]] = {}
    for item in corpus:
        by_doc.setdefault(item.doc, []).append(item)

    for doc, chunks in by_doc.items():
        store.replace_document(
            doc,
            title=chunks[0].title,
            chunks=chunks,
            vectors=[vector(i) for i in range(len(chunks))],
            embedding_model="test/model",
        )

    assert store.stats() == store.IndexStats(len(by_doc), len(corpus), ("test/model",))
