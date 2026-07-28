"""Build-time entrypoint: chunk `kb/documents/`, embed, write to pgvector.

    uv run python -m kb.ingest

Run on every deploy, which is what shapes the whole module: embedding is the
only step here that costs money and takes time, so the job is to do as little of
it as possible while never leaving a stale vector in the index.

- **The skip decision is per document, and it is two-part.** A document is left
  alone only when the hash of its rendered chunks *and* the model that embedded
  them both match. Hash alone would keep vectors from a previous embedding
  model — the one failure that raises nothing anywhere, because the documents
  and the queries just end up in different vector spaces and retrieval quietly
  turns to noise. See `kb/store.py` → `StoredDocument.is_current`.
- **The model id is resolved from settings, not from the client**, so an ingest
  where nothing changed makes no network calls and needs no API key. The client
  is built on first use — i.e. on the first document that actually has to be
  re-embedded — and its model is asserted against the same string then, since
  writing one id while embedding with another is the same silent failure by
  another route.
- **Writes are per document and each is its own transaction.** A failure part
  way through the corpus therefore leaves the documents already written fresh
  and correct, and re-running skips them and picks up where it stopped. That is
  why exceptions from the embedding calls are allowed to propagate rather than
  being collected: there is nothing to clean up.
- **Pruning happens after the writes.** A run that dies mid-corpus removes
  nothing, so a failed deploy cannot empty the index. `store.prune` separately
  refuses an empty keep-set, which covers the wrong-directory typo.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from core.config import get_settings
from core.embeddings import EmbeddingClient, embedding_client
from core.llm import Model, Usage
from kb import store
from kb.chunking import Chunk, load_corpus


class IngestError(RuntimeError):
    """The corpus could not be ingested."""


# --- report -------------------------------------------------------------


@dataclass(slots=True)
class IngestReport:
    """What one run did. Returned rather than only printed, so tests can assert
    on the skip decision without parsing stdout."""

    embedded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    pruned: list[str] = field(default_factory=list)
    # Chunks written this run, not the size of the index — a no-op ingest
    # reports 0 here and a full index in `stats`.
    chunks: int = 0
    usage: Usage = field(default_factory=Usage)
    stats: store.IndexStats | None = None

    def summary(self) -> str:
        parts = [
            f"{len(self.embedded)} embedded",
            f"{len(self.skipped)} unchanged",
        ]
        if self.pruned:
            parts.append(f"{len(self.pruned)} pruned")
        line = f"{', '.join(parts)} · {self.chunks} chunks written"
        if self.usage.total_tokens:
            line += f" · {self.usage.total_tokens:,} tokens"
        # 0.0 is what a provider that omits OpenRouter's cost extension returns,
        # and printing "$0.0000" for it would read as "this was free".
        if self.usage.cost:
            line += f" · ${self.usage.cost:.4f}"
        return line


# --- helpers ------------------------------------------------------------


def _group(corpus: list[Chunk]) -> dict[str, list[Chunk]]:
    """Chunks by document, in corpus order (which is filename order)."""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in corpus:
        grouped.setdefault(chunk.doc, []).append(chunk)
    return grouped


def _title(doc: str, chunks: list[Chunk]) -> str:
    """The document's `#` header.

    Falls back to the filename stem: text before the first header carries no
    title metadata, so a document that opens with prose would otherwise store an
    empty title. `title` is the one column in `sharia_documents` that is not
    derivable from the chunks, and an empty one makes the row pointless.
    """
    return next((chunk.title for chunk in chunks if chunk.title), doc)


def _checked(client: EmbeddingClient, expected: str) -> EmbeddingClient:
    """Assert the client embeds with the id being recorded against its vectors.

    Both come from `settings.embedding_model`, so this only fires if the two
    paths ever stop agreeing. Cheap, and the failure it guards is invisible:
    rows labelled with one model holding another model's vector space.
    """
    if str(client.model) != expected:
        raise IngestError(
            f"embedding client is {client.model} but rows would be recorded as "
            f"{expected} — refusing to write a label the vectors do not match"
        )
    return client


# --- the run ------------------------------------------------------------


def ingest(directory: Path | None = None, *, force: bool = False) -> IngestReport:
    """Bring the index in line with `kb/documents/`.

    `force` re-embeds everything, ignoring the hash. It is the escape hatch for
    "the index looks wrong and I do not trust it", not part of the normal path —
    a deploy relies on the skip.
    """
    settings = get_settings()

    # Before anything else, and outside the pool: `register_vector` in the
    # pool's configure hook reads the `vector` type's OID, so it cannot run
    # against a database where the extension does not exist yet. This also
    # verifies the column width matches `embedding_dimensions`, which is worth
    # knowing before paying for a corpus of embeddings.
    store.apply_schema()

    corpus = load_corpus(directory)
    if not corpus:
        raise IngestError(
            f"no chunks from {directory or 'kb/documents/'} — nothing to ingest. "
            "Either the path is wrong or the documents are empty."
        )
    documents = _group(corpus)

    # Resolved without touching the network: `Model.parse` normalises the same
    # way `EmbeddingClient.from_slug` does, so this is exactly the string the
    # client would report, and an all-skip run never needs a key.
    model = str(Model.parse(settings.embedding_model))
    indexed = store.stored_documents()
    client: EmbeddingClient | None = None
    report = IngestReport()

    print(f"{len(documents)} documents · {len(corpus)} chunks · {model}")

    for doc, chunks in documents.items():
        stored = indexed.get(doc)
        current = stored is not None and stored.is_current(
            content_hash=store.content_hash(chunks), embedding_model=model
        )
        if current and not force:
            report.skipped.append(doc)
            print(f"  unchanged  {doc}  ({len(chunks)} chunks)")
            continue

        if client is None:
            client = _checked(embedding_client(), model)

        print(f"  embedding  {doc}  ({len(chunks)} chunks)", flush=True)
        embeddings = client.embed_documents([chunk.text for chunk in chunks])
        # replace_document re-checks the length before writing; a mismatch there
        # would mean every vector after it lands on the wrong text.
        store.replace_document(
            doc,
            title=_title(doc, chunks),
            chunks=chunks,
            vectors=embeddings.vectors,
            embedding_model=model,
        )
        report.embedded.append(doc)
        report.chunks += len(chunks)
        report.usage = report.usage + embeddings.usage

    report.pruned = store.prune(documents.keys())
    for doc in report.pruned:
        print(f"  pruned     {doc}  (no longer in the corpus)")

    report.stats = store.stats()
    print(report.summary())
    print(
        f"index: {report.stats.documents} documents, {report.stats.chunks} chunks, "
        f"{', '.join(report.stats.embedding_models) or 'none'}"
    )

    # Should be empty by construction — a foreign model fails `is_current` and
    # gets re-embedded above. Non-empty means a document was left behind by a
    # partial run, and those chunks are unreachable rather than merely stale.
    foreign = report.stats.foreign_models(model)
    if foreign:
        print(
            f"warning: {', '.join(foreign)} still in the index — those chunks are in a "
            "different vector space and will not retrieve. Re-run to rebuild them.",
            file=sys.stderr,
        )

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kb.ingest",
        description="Chunk kb/documents/, embed, and write to pgvector.",
    )
    parser.add_argument(
        "--documents",
        type=Path,
        default=None,
        metavar="DIR",
        help="corpus directory (default: kb/documents/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-embed every document, ignoring the content hash",
    )
    args = parser.parse_args(argv)

    try:
        ingest(args.documents, force=args.force)
    except (IngestError, store.StoreError) as exc:
        # These carry their own instructions — a traceback would bury them.
        # Embedding and transport failures keep theirs, which is where the
        # useful detail lives for those.
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["IngestError", "IngestReport", "ingest", "main"]
