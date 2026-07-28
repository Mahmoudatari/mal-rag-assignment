"""Unit tests for the chunking strategy.

Chunking has no runtime failure mode — a bad split raises nothing, embeds
cleanly and shows up weeks later as retrieval that is merely disappointing. So
these tests pin the two decisions the strategy rests on:

- **the FAQ split fires on FAQs and nothing else** — it keys on bold-line
  question markers rather than the header text, and the guard that keeps it off
  ordinary prose is a count of those markers;
- **every chunk carries both headers** — the thing that makes a 52-token Q&A
  pair retrievable on its own, since all five documents have a section called
  "Frequently Asked Questions" and that header alone says nothing.

Most tests run on inline markdown so they state their own inputs. The few that
read `kb/documents/` are marked as such: they are asserting properties of the
real corpus, which is what actually gets indexed.
"""

from __future__ import annotations

import pytest

from kb.chunking import (
    MAX_CHUNK_CHARS,
    MIN_QUESTIONS,
    Chunk,
    chunk_document,
    load_corpus,
)

FAQ = """# Product Title

Lead paragraph.

## Frequently Asked Questions

**First question?**
First answer.

**Second question?**
Second answer.

**Third question?**
Third answer.
"""


# --- both headers on every chunk -----------------------------------------
# The `#` title is the only thing distinguishing five identically-titled FAQ
# sections, and the splitter never inlines it — it only ever keeps the header
# that opens a section. So it is re-attached by hand, to section chunks and Q&A
# chunks alike.


def test_section_chunk_carries_title_and_section() -> None:
    text = "# Product Title\n\n## Fixed Total Price\n\nThe price is fixed.\n"

    (chunk,) = [c for c in chunk_document(text, "doc") if c.section]

    assert chunk.text == "# Product Title\n## Fixed Total Price\n\nThe price is fixed."


def test_every_qa_chunk_carries_the_headers_not_just_the_first() -> None:
    """The naive split would leave the header on pair one and strip it from the
    rest, so pairs 2..n would embed without ever naming the product."""
    chunks = chunk_document(FAQ, "doc")
    pairs = [c for c in chunks if c.section == "Frequently Asked Questions"]

    assert len(pairs) == 3
    for chunk in pairs:
        assert chunk.text.startswith("# Product Title\n## Frequently Asked Questions\n\n")


def test_preamble_before_the_first_header_keeps_the_title_alone() -> None:
    text = "# Product Title\n\nLead paragraph.\n\n## A Section\n\nBody.\n"

    first = chunk_document(text, "doc")[0]

    assert first.section == ""
    assert first.text == "# Product Title\n\nLead paragraph."


# --- the FAQ split, and what keeps it off everything else -----------------


def test_faq_section_splits_one_chunk_per_question() -> None:
    pairs = [c for c in chunk_document(FAQ, "doc") if c.section]

    bodies = [c.text.split("\n\n", 1)[1] for c in pairs]
    assert bodies == [
        "**First question?**\nFirst answer.",
        "**Second question?**\nSecond answer.",
        "**Third question?**\nThird answer.",
    ]


def test_a_single_bold_line_is_emphasis_and_does_not_split() -> None:
    """Real case: murabaha's "Contract Variations After Acceptance" opens with
    one bold line. Splitting at a section's only marker just decapitates it."""
    text = (
        "# Product Title\n\n## Contract Variations\n\n"
        "**Variations are permitted.**\nOnly the tenor may change.\n"
    )

    chunks = chunk_document(text, "doc")

    assert len(chunks) == 1
    assert "Only the tenor may change." in chunks[0].text


@pytest.mark.parametrize(
    "line",
    [
        "This has **bold** in the middle.",
        "**Leading bold** then prose continues here.",
        "| Tier | **Max** | Tenor |",
    ],
)
def test_inline_bold_is_not_a_question_marker(line: str) -> None:
    """Only a line that is *entirely* bold counts. Emphasis inside prose and
    inside table rows is everywhere in the corpus."""
    text = f"# T\n\n## S\n\n{line}\n\n{line}\n\n{line}\n"

    assert len(chunk_document(text, "doc")) == 1


def test_split_needs_at_least_MIN_QUESTIONS_markers() -> None:
    marker = "**Q?**\nA.\n\n"
    below = f"# T\n\n## S\n\n{marker * (MIN_QUESTIONS - 1)}"
    at = f"# T\n\n## S\n\n{marker * MIN_QUESTIONS}"

    assert len(chunk_document(below, "doc")) == 1
    assert len(chunk_document(at, "doc")) == MIN_QUESTIONS


def test_lead_in_before_the_first_question_becomes_its_own_chunk() -> None:
    """Not repeated onto every pair, which would duplicate it n times through
    the index. No FAQ in the corpus has one today; a new document might."""
    text = "# T\n\n## FAQ\n\nRead these first.\n\n**Q1?**\nA1.\n\n**Q2?**\nA2.\n"

    chunks = chunk_document(text, "doc")

    assert len(chunks) == 3
    assert chunks[0].text == "# T\n## FAQ\n\nRead these first."
    assert chunks[1].text.endswith("**Q1?**\nA1.")


# --- ids and empties ------------------------------------------------------


def test_chunk_ids_are_unique_and_ordered_within_a_document() -> None:
    ids = [c.chunk_id for c in chunk_document(FAQ, "murabaha")]

    assert ids == ["murabaha#000", "murabaha#001", "murabaha#002", "murabaha#003"]


def test_empty_sections_are_dropped_without_gapping_the_ids() -> None:
    """A header with nothing under it would otherwise embed as a bare title."""
    text = "# T\n\n## Empty\n\n## Has Body\n\nBody.\n"

    chunks = chunk_document(text, "doc")

    assert [c.section for c in chunks] == ["Has Body"]
    assert [c.chunk_id for c in chunks] == ["doc#000"]


def test_oversized_section_raises_rather_than_embedding_truncated() -> None:
    """Nothing in this module splits on length, so this is the only bound. A
    section this big with no question markers is unreachable by the strategy —
    it has to fail at ingest rather than silently at the provider."""
    text = f"# T\n\n## Huge\n\n{'word ' * MAX_CHUNK_CHARS}\n"

    with pytest.raises(ValueError, match="Huge"):
        chunk_document(text, "doc")


# --- the real corpus ------------------------------------------------------


@pytest.fixture(scope="module")
def corpus() -> list[Chunk]:
    return load_corpus()


def test_corpus_chunk_ids_are_globally_unique(corpus: list[Chunk]) -> None:
    """They are the citation keys the API returns, so a collision across
    documents would point a customer at the wrong policy."""
    assert len({c.chunk_id for c in corpus}) == len(corpus)


def test_only_faq_sections_split_in_the_real_corpus(corpus: list[Chunk]) -> None:
    """Including the two murabaha FAQs that are not called "Frequently Asked
    Questions" — matching on header text would have missed them."""
    counts: dict[tuple[str, str], int] = {}
    for chunk in corpus:
        counts[(chunk.doc, chunk.section)] = counts.get((chunk.doc, chunk.section), 0) + 1

    split = {section for (_, section), n in counts.items() if n > 1}
    assert split and all(s.startswith("Frequently Asked Questions") for s in split)


def test_every_corpus_chunk_is_within_the_embedding_bound(corpus: list[Chunk]) -> None:
    assert max(len(c.text) for c in corpus) <= MAX_CHUNK_CHARS


def test_every_corpus_chunk_names_its_document(corpus: list[Chunk]) -> None:
    for chunk in corpus:
        assert chunk.text.startswith(f"# {chunk.title}")
        assert chunk.title
