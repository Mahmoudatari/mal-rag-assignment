"""Chunking strategy. Shared by ingest and retrieval — single source of truth.

The corpus is hand-written markdown whose `##` sections are already the right
size: 135 sections, median 298 tokens, p90 414. Splitting those on a character
window would cut through rent tables and worked examples to fix a problem the
corpus does not have — at `chunk_size=800` chars it shatters them into 359
fragments, 102 of which fall under 100 tokens.

So sections are the unit, with one exception. Every `## Frequently Asked
Questions` block runs 1091-1559 tokens, several times the median, and each Q&A
pair inside it is atomic — a self-contained answer that stands alone. Splitting
those one pair per chunk takes the corpus to 201 chunks with a 482-token
maximum, and every boundary is one the author wrote.

That leaves no character-window splitting anywhere, which is why there is no
`chunk_size` or `chunk_overlap` here. `MAX_CHUNK_CHARS` is the backstop for the
case this strategy cannot handle on its own: a future document with a huge
section and no question markers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_text_splitters import MarkdownHeaderTextSplitter

DOCUMENTS_DIR = Path(__file__).parent / "documents"

# `#` and `##` only. There is no `###` anywhere in the corpus, and FAQ questions
# are bold lines rather than headers — deepening this list buys nothing, and
# QUESTION below is what actually reaches inside an FAQ.
#
# `strip_headers=True` because both header levels are re-attached by hand in
# `_render`. Letting the splitter keep them would put the `##` line in the body
# for section chunks but not for Q&A chunks, so the two would carry their
# context differently.
HEADERS = [("#", "title"), ("##", "section")]

_SECTION_SPLITTER = MarkdownHeaderTextSplitter(HEADERS, strip_headers=True)

# A line that is entirely bold. In this corpus that construct is used only for
# FAQ questions. It is matched instead of the header text because the FAQ
# sections are not named consistently — murabaha alone has "Frequently Asked
# Questions", "... — Servicing, Problems and Escalation" and "... — Products,
# Agency and Life Events", so a title match would silently miss two of them.
QUESTION = re.compile(r"^\*\*(?P<q>.+?)\*\*[ \t]*$", re.MULTILINE)

# Matching the pattern is not enough to mean "this is an FAQ". One prose
# section (murabaha, "Contract Variations After Acceptance") uses a single bold
# line as emphasis; requiring two keeps it intact. Splitting a section at its
# only marker would just decapitate it.
MIN_QUESTIONS = 2

# Backstop, not a tuning knob — nothing in this module splits on length, so
# without it an oversized section would embed silently truncated.
#
# The real limit is `google/gemini-embedding-001`'s 2048-token input. Measured
# in characters rather than tokens on purpose: there is no local tokenizer for
# that model, so a token count here would be a `tiktoken` estimate of a
# different tokenizer's output. 3 chars/token is deliberately pessimistic for
# English prose (~4 is typical), so this trips before the provider does. The
# largest chunk in the corpus today is 2,725 characters.
MAX_CHUNK_CHARS = 6_000


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit. `text` is what gets embedded and what reaches the
    model as context — the two must be the same string, or retrieval scores a
    document the answer never sees."""

    chunk_id: str
    doc: str
    title: str
    section: str
    text: str


def _render(title: str, section: str, body: str) -> str:
    """Re-attach both headers to the body.

    This is the reason a 52-token Q&A pair survives on its own. All five
    documents contain a section titled "Frequently Asked Questions", so that
    header alone carries no signal — it is the `#` title that says which
    product a bare question like "Can the amount I owe go up?" is about, both
    for the embedding and for the reranker reading query against document.
    """
    prefix = f"# {title}" if title else ""
    if section:
        prefix = f"{prefix}\n## {section}" if prefix else f"## {section}"
    return f"{prefix}\n\n{body}" if prefix else body


def _bodies(content: str) -> list[str]:
    """Split one section into bodies: itself, or one per Q&A pair."""
    marks = [m.start() for m in QUESTION.finditer(content)]
    if len(marks) < MIN_QUESTIONS:
        return [content]

    bounds = marks + [len(content)]
    # Anything before the first question — a lead-in paragraph — becomes its own
    # body rather than being repeated onto every pair. No FAQ section in the
    # current corpus has one, so this is a guard for documents added later.
    preamble = content[: marks[0]].strip()
    pairs = [content[bounds[i] : bounds[i + 1]].strip() for i in range(len(marks))]
    return [preamble, *pairs] if preamble else pairs


def chunk_document(text: str, doc: str) -> list[Chunk]:
    """Chunk one markdown document.

    `chunk_id` is positional (`doc#003`), so editing a document shifts the ids
    of everything after the edit. Ingest therefore replaces a document's rows
    wholesale rather than upserting them one by one.
    """
    chunks: list[Chunk] = []
    for parsed in _SECTION_SPLITTER.split_text(text):
        title = parsed.metadata.get("title", "")
        section = parsed.metadata.get("section", "")
        for body in _bodies(parsed.page_content):
            body = body.strip()
            if not body:
                continue  # a header with nothing under it
            rendered = _render(title, section, body)
            if len(rendered) > MAX_CHUNK_CHARS:
                raise ValueError(
                    f"{doc}: section {section!r} produced a {len(rendered)}-char "
                    f"chunk, over the {MAX_CHUNK_CHARS} limit. It has no question "
                    f"markers to split on — give it `##` subsections."
                )
            chunks.append(
                Chunk(
                    chunk_id=f"{doc}#{len(chunks):03d}",
                    doc=doc,
                    title=title,
                    section=section,
                    text=rendered,
                )
            )
    return chunks


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    """Chunk every document in `kb/documents/`. Sorted so ids are reproducible."""
    directory = directory or DOCUMENTS_DIR
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path.read_text(encoding="utf-8"), path.stem))
    return chunks


if __name__ == "__main__":
    # Inspection only — `uv run python -m kb.chunking`. Deliberately prints
    # rather than writing a file: chunks are derived from the documents, and a
    # persisted copy would be a second source of truth that goes stale silently.
    # Worse than usual here, because `chunk_id` is positional — a stale copy
    # would map citation ids to text the retriever never returned.
    import sys

    corpus = load_corpus()
    for chunk in corpus:
        rule = "─" * 78
        print(f"{rule}\n{chunk.chunk_id}  ({len(chunk.text)} chars)\n{rule}")
        print(chunk.text, end="\n\n")

    # To stderr, so a redirect captures only the chunks.
    widths = [len(c.text) for c in corpus]
    print(
        f"{len(corpus)} chunks · {sum(widths):,} chars · "
        f"max {max(widths)} of {MAX_CHUNK_CHARS}",
        file=sys.stderr,
    )
