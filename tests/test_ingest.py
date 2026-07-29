"""Unit tests for the build-time ingest.

No network and no database: the store's five entry points are monkeypatched and
the embedder is a fake that derives each vector from the text it was given, so a
mis-zipped batch is visible rather than merely possible.

What is worth testing here is the arithmetic of *not* doing work. Ingest runs on
every deploy and embedding is the only expensive step, so the skip decision is
the module — and it fails in two opposite directions:

- too loose, and a deploy ships an index whose vectors no longer match the
  documents, or match them in a different vector space;
- too tight, and every deploy re-embeds the whole corpus.

The prune keep-set gets the same attention, from both ends. It has to include the
documents that were *skipped*, not just the ones written this run; the version of
that line which passes every other test in this file deletes four fifths of the
index on the second deploy. And it may only be applied at all when the run read
the shipped corpus — pruning against a `--documents` directory means deleting
every document that directory does not happen to contain.

That second rule is why the prune tests take the `shipped` fixture: it points
`ingest.DOCUMENTS_DIR` at the temp corpus, which is what makes that corpus the
one the index is supposed to mirror.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.llm import Usage
from kb import ingest, store
from kb.chunking import Chunk, load_corpus

DIMS = 4

DOCUMENT = """# {title}

## Overview

Body of {name}.

## Fees

Fees for {name}.
"""


# --- fakes ---------------------------------------------------------------


def vector_for(text: str) -> list[float]:
    """A vector that identifies its input, so a mis-zip is an assertion failure
    rather than a plausible-looking index."""
    return [float(len(text)), float(sum(map(ord, text[:8]))), 0.0, 1.0]


class FakeEmbedder:
    """Shaped like `EmbeddingClient`, minus the network."""

    def __init__(self, model: str = "google/gemini-embedding-001") -> None:
        self.model = model
        self.batches: list[list[str]] = []

    def embed_documents(self, texts: list[str]):
        self.batches.append(list(texts))
        return _Embeddings(
            vectors=[vector_for(text) for text in texts],
            usage=Usage(prompt_tokens=10 * len(texts), total_tokens=10 * len(texts), cost=0.0001),
        )

    @property
    def calls(self) -> int:
        return len(self.batches)


class _Embeddings:
    def __init__(self, vectors: list[list[float]], usage: Usage) -> None:
        self.vectors = vectors
        self.usage = usage


class FakeStore:
    """Records what ingest asked the store to do."""

    def __init__(self) -> None:
        self.indexed: dict[str, store.StoredDocument] = {}
        self.writes: list[dict] = []
        self.pruned_keep: list[str] | None = None
        self.schema_applied = 0

    # --- the five entry points ingest uses ---
    def apply_schema(self) -> None:
        self.schema_applied += 1

    def stored_documents(self) -> dict[str, store.StoredDocument]:
        return dict(self.indexed)

    def replace_document(self, doc, *, title, chunks, vectors, embedding_model) -> None:
        # The real one raises on a length mismatch; mirrored here so a test that
        # would have corrupted a real index fails in the same place.
        assert len(chunks) == len(vectors), f"{doc}: {len(chunks)} chunks, {len(vectors)} vectors"
        self.writes.append(
            {
                "doc": doc,
                "title": title,
                "chunks": list(chunks),
                "vectors": list(vectors),
                "embedding_model": embedding_model,
            }
        )
        self.indexed[doc] = store.StoredDocument(
            doc=doc,
            content_hash=store.content_hash(chunks),
            embedding_model=embedding_model,
        )

    def prune(self, keep) -> list[str]:
        self.pruned_keep = sorted(keep)
        gone = sorted(set(self.indexed) - set(keep))
        for doc in gone:
            del self.indexed[doc]
        return gone

    def stats(self) -> store.IndexStats:
        return store.IndexStats(
            documents=len(self.indexed),
            chunks=sum(len(write["chunks"]) for write in self.writes),
            embedding_models=tuple(sorted({d.embedding_model for d in self.indexed.values()})),
        )

    # --- helpers for the tests ---
    @property
    def written_docs(self) -> list[str]:
        return [write["doc"] for write in self.writes]

    def mark_indexed(self, doc: str, *, chunks, model: str = "google/gemini-embedding-001") -> None:
        self.indexed[doc] = store.StoredDocument(
            doc=doc, content_hash=store.content_hash(chunks), embedding_model=model
        )


# --- fixtures ------------------------------------------------------------


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two small documents, four chunks. Deliberately not the real corpus —
    these tests are about the skip arithmetic, not about the documents."""
    for name, title in (("alpha", "Mal Alpha Product"), ("beta", "Mal Beta Product")):
        (tmp_path / f"{name}.md").write_text(
            DOCUMENT.format(title=title, name=name), encoding="utf-8"
        )
    return tmp_path


@pytest.fixture
def shipped(corpus: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The temp corpus, promoted to *the* corpus.

    Pruning is gated on the run covering `kb/documents/`, so a test about prune
    behaviour has to say which directory that is — otherwise it would be testing
    the gate instead of the keep-set.
    """
    monkeypatch.setattr(ingest, "DOCUMENTS_DIR", corpus)
    return corpus


@pytest.fixture
def fake_store(monkeypatch: pytest.MonkeyPatch) -> FakeStore:
    fake = FakeStore()
    for name in ("apply_schema", "stored_documents", "replace_document", "prune", "stats"):
        monkeypatch.setattr(store, name, getattr(fake, name))
    return fake


@pytest.fixture
def embedder(monkeypatch: pytest.MonkeyPatch) -> FakeEmbedder:
    fake = FakeEmbedder()
    monkeypatch.setattr(ingest, "embedding_client", lambda: fake)
    return fake


def chunks_of(corpus: Path, doc: str) -> list[Chunk]:
    return [chunk for chunk in load_corpus(corpus) if chunk.doc == doc]


# --- the first run -------------------------------------------------------


def test_first_run_embeds_everything(corpus, fake_store, embedder) -> None:
    report = ingest.ingest(corpus)

    assert report.embedded == ["alpha", "beta"]
    assert report.skipped == []
    assert report.chunks == 4
    assert fake_store.written_docs == ["alpha", "beta"]


def test_schema_is_applied_before_anything_is_written(corpus, fake_store, embedder) -> None:
    """The pool's configure hook registers the `vector` type, which cannot exist
    before `CREATE EXTENSION`. Getting this backwards fails a first deploy and
    succeeds on the retry."""
    ingest.ingest(corpus)

    assert fake_store.schema_applied == 1


def test_each_vector_lands_on_the_text_it_was_made_from(corpus, fake_store, embedder) -> None:
    """The corruption that raises nothing: a batch zipped back one position out
    attaches every embedding to the wrong chunk and indexes perfectly cleanly."""
    ingest.ingest(corpus)

    for write in fake_store.writes:
        for chunk, vector in zip(write["chunks"], write["vectors"], strict=True):
            assert vector == vector_for(chunk.text)


def test_documents_are_embedded_one_batch_per_document(corpus, fake_store, embedder) -> None:
    """Per document rather than per corpus, because each write is its own
    transaction — a failure part way through leaves the finished ones correct."""
    assert ingest.ingest(corpus).embedded == ["alpha", "beta"]
    assert embedder.calls == 2
    assert [len(batch) for batch in embedder.batches] == [2, 2]


def test_usage_totals_across_documents(corpus, fake_store, embedder) -> None:
    report = ingest.ingest(corpus)

    assert report.usage.total_tokens == 40  # 4 chunks x 10
    assert report.usage.cost == pytest.approx(0.0002)


# --- the skip decision ---------------------------------------------------


def test_unchanged_corpus_embeds_nothing(corpus, fake_store, embedder) -> None:
    ingest.ingest(corpus)
    embedder.batches.clear()

    report = ingest.ingest(corpus)

    assert report.embedded == []
    assert report.skipped == ["alpha", "beta"]
    assert report.chunks == 0
    assert embedder.calls == 0


def test_an_unchanged_corpus_needs_no_embedding_client_at_all(
    corpus, fake_store, monkeypatch
) -> None:
    """Which is what makes a no-op deploy free of both an API key and a network
    call: the model id is resolved from settings, not from a constructed client."""
    fake = FakeEmbedder()
    monkeypatch.setattr(ingest, "embedding_client", lambda: fake)
    ingest.ingest(corpus)

    def explode():
        raise AssertionError("built an embedding client on a no-op ingest")

    monkeypatch.setattr(ingest, "embedding_client", explode)

    assert len(ingest.ingest(corpus).skipped) == 2


def test_edited_document_is_re_embedded_and_its_neighbour_is_not(
    corpus, fake_store, embedder
) -> None:
    ingest.ingest(corpus)
    embedder.batches.clear()
    fake_store.writes.clear()
    (corpus / "beta.md").write_text(
        DOCUMENT.format(title="Mal Beta Product", name="beta, revised"), encoding="utf-8"
    )

    report = ingest.ingest(corpus)

    assert report.embedded == ["beta"]
    assert report.skipped == ["alpha"]
    assert embedder.calls == 1


def test_same_text_embedded_by_another_model_is_re_embedded(corpus, fake_store, embedder) -> None:
    """The hash matches and the text is identical, so a hash-only skip would
    keep the old vectors — and they are in a different vector space, which is
    the one failure that raises nothing anywhere."""
    fake_store.mark_indexed("alpha", chunks=chunks_of(corpus, "alpha"), model="openai/text-embedding-3-small")
    fake_store.mark_indexed("beta", chunks=chunks_of(corpus, "beta"))

    report = ingest.ingest(corpus)

    assert report.embedded == ["alpha"]
    assert report.skipped == ["beta"]


def test_a_reordering_that_leaves_the_text_alone_still_counts(
    corpus, fake_store, embedder
) -> None:
    """`chunk_id` is positional and is the citation key, so swapping two
    sections moves what every id after them means even though no word changed."""
    ingest.ingest(corpus)
    embedder.batches.clear()
    (corpus / "alpha.md").write_text(
        "# Mal Alpha Product\n\n## Fees\n\nFees for alpha.\n\n## Overview\n\nBody of alpha.\n",
        encoding="utf-8",
    )

    assert ingest.ingest(corpus).embedded == ["alpha"]


def test_force_re_embeds_everything(corpus, fake_store, embedder) -> None:
    ingest.ingest(corpus)
    embedder.batches.clear()

    report = ingest.ingest(corpus, force=True)

    assert report.embedded == ["alpha", "beta"]
    assert report.skipped == []
    assert embedder.calls == 2


# --- pruning -------------------------------------------------------------


def test_prune_keeps_the_documents_that_were_skipped(shipped, fake_store, embedder) -> None:
    """The keep-set is every document in the corpus, not the ones written this
    run. Passing only `report.embedded` passes every other test in this file and
    empties the index on the second deploy."""
    ingest.ingest(shipped)

    fake_store.pruned_keep = None
    report = ingest.ingest(shipped)

    assert report.skipped == ["alpha", "beta"]
    assert fake_store.pruned_keep == ["alpha", "beta"]
    assert set(fake_store.indexed) == {"alpha", "beta"}


def test_a_deleted_document_is_pruned(shipped, fake_store, embedder) -> None:
    ingest.ingest(shipped)
    (shipped / "beta.md").unlink()

    report = ingest.ingest(shipped)

    assert report.pruned == ["beta"]
    assert set(fake_store.indexed) == {"alpha"}


def test_pruning_happens_after_the_writes(shipped, fake_store, embedder, monkeypatch) -> None:
    """So a run that dies mid-corpus removes nothing — a failed deploy must not
    be able to empty the index."""
    monkeypatch.setattr(
        ingest.store,
        "replace_document",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("provider is down")),
    )

    with pytest.raises(RuntimeError, match="provider is down"):
        ingest.ingest(shipped)

    assert fake_store.pruned_keep is None


def test_another_directory_embeds_but_never_prunes(corpus, fake_store, embedder, capsys) -> None:
    """The destructive typo: `--documents ~/scratch` with DATABASE_URL still on
    the deployed index. One `.md` file in there is a non-empty keep-set, so
    `store.prune`'s own refusal lets it through and the five real documents are
    cascade-deleted along with all 202 chunks. Absence from some other directory
    is not evidence a document left the corpus, so this run may only add."""
    fake_store.mark_indexed("murabaha-everyday-finance", chunks=chunks_of(corpus, "alpha"))

    report = ingest.ingest(corpus)

    assert report.embedded == ["alpha", "beta"]
    assert fake_store.pruned_keep is None
    assert report.pruned == []
    assert "murabaha-everyday-finance" in fake_store.indexed
    assert "pruning    skipped" in capsys.readouterr().out


def test_the_default_directory_still_prunes(fake_store, embedder) -> None:
    """The other half: the gate must not disable pruning outright, or a document
    deleted from `kb/documents/` would stay in the index forever."""
    fake_store.mark_indexed("retired-product", chunks=[])

    report = ingest.ingest()

    assert report.pruned == ["retired-product"]
    assert fake_store.pruned_keep is not None
    assert "retired-product" not in fake_store.indexed


# --- titles --------------------------------------------------------------


def test_title_comes_from_the_hash_header(corpus, fake_store, embedder) -> None:
    ingest.ingest(corpus)

    assert [write["title"] for write in fake_store.writes] == [
        "Mal Alpha Product",
        "Mal Beta Product",
    ]


def test_title_falls_back_to_the_filename_when_there_is_no_header(
    tmp_path, fake_store, embedder
) -> None:
    """`title` is the only column in `sharia_documents` not derivable from the
    chunks; storing an empty one makes the row pointless."""
    (tmp_path / "headerless.md").write_text("Just prose, no header at all.\n", encoding="utf-8")

    ingest.ingest(tmp_path)

    assert fake_store.writes[0]["title"] == "headerless"


def test_titles_from_the_real_corpus(fake_store, embedder) -> None:
    """One test against the shipped documents, so a corpus that loses its `#`
    headers is caught here and not at deploy time."""
    ingest.ingest()

    assert len(fake_store.writes) == 5
    for write in fake_store.writes:
        assert write["title"].startswith("Mal ")


# --- refusals ------------------------------------------------------------


def test_an_empty_corpus_raises_rather_than_pruning(tmp_path, fake_store, embedder) -> None:
    """A wrong `--documents` path must not reach `prune`, which would be asked
    to keep nothing."""
    with pytest.raises(ingest.IngestError, match="nothing to ingest"):
        ingest.ingest(tmp_path)

    assert fake_store.pruned_keep is None
    assert fake_store.writes == []


def test_a_client_on_a_different_model_than_the_label_refuses_to_write(
    corpus, fake_store, monkeypatch
) -> None:
    """Rows labelled with one model while holding another model's vectors are
    indistinguishable from correct ones."""
    monkeypatch.setattr(
        ingest, "embedding_client", lambda: FakeEmbedder("openai/text-embedding-3-small")
    )

    with pytest.raises(ingest.IngestError, match="refusing to write"):
        ingest.ingest(corpus)

    assert fake_store.writes == []


def test_the_recorded_model_is_the_configured_one(corpus, fake_store, embedder) -> None:
    from core.config import get_settings

    ingest.ingest(corpus)

    for write in fake_store.writes:
        assert write["embedding_model"] == get_settings().embedding_model


# --- the CLI -------------------------------------------------------------


def test_main_reports_store_failures_without_a_traceback(monkeypatch, capsys) -> None:
    """`StoreError` carries its own instructions — "DATABASE_URL is not set,
    copy .env.example". A traceback buries them."""

    def unconfigured() -> None:
        raise store.StoreError("DATABASE_URL is not set")

    monkeypatch.setattr(store, "apply_schema", unconfigured)

    assert ingest.main([]) == 2
    assert "DATABASE_URL is not set" in capsys.readouterr().err


def test_main_returns_zero_on_success(corpus, fake_store, embedder) -> None:
    assert ingest.main(["--documents", str(corpus)]) == 0
    assert fake_store.written_docs == ["alpha", "beta"]


def test_main_passes_force_through(corpus, fake_store, embedder) -> None:
    ingest.main(["--documents", str(corpus)])
    embedder.batches.clear()

    ingest.main(["--documents", str(corpus), "--force"])

    assert embedder.calls == 2
