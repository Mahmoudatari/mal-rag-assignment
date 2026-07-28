"""The eval set's own guard rail.

`test_retrieval.py` and `test_grounding.py` both need an OpenRouter key, a
populated index and a working graph, so they are marked `live` and skipped in
ordinary runs. That would leave the anchors in `goldens.py` unchecked between
live runs — and the failure they guard against is an editor changing a heading
in `kb/documents/`, which is exactly the kind of change that gets made without
running the paid suite.

These tests are pure: `kb.chunking` needs no key, no database and no network, so
they run on every `uv run pytest` and fail the moment an anchor stops pointing at
exactly one chunk.
"""

import pytest

from kb.chunking import load_corpus
from evals.goldens import GOLDENS, AnchorError, Golden, resolve


@pytest.fixture(scope="module")
def corpus():
    """Chunked once — `load_corpus()` re-reads and re-splits all five documents."""
    return load_corpus()


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.doc}-{g.kind}")
def test_anchors_resolve_to_exactly_one_chunk(golden: Golden, corpus):
    """Every anchor still points at exactly one chunk of its document.

    Failure here means a document was edited: a heading was reworded, an FAQ
    question changed, or a section was split. Fix the anchor in `goldens.py` —
    do not paste in a chunk id, which is positional and will drift again.
    """
    ids = resolve(golden, corpus)
    assert len(ids) == len(golden.must_retrieve)
    assert len(set(ids)) == len(ids), f"anchors collapsed onto one chunk: {ids}"
    assert all(cid.startswith(f"{golden.doc}#") for cid in ids)


def test_unknown_anchor_raises(corpus):
    """The resolver fails loudly. A silent miss would make retrieval asserts no-ops."""
    broken = Golden(
        question="",
        doc=GOLDENS[0].doc,
        kind="lookup",
        must_retrieve=("## No Such Heading Exists Here",),
        must_contain="",
    )
    with pytest.raises(AnchorError):
        resolve(broken, corpus)


def test_covers_every_document_and_kind():
    """Four questions per document, one of each kind.

    Asserted rather than assumed: the set is meant to exercise plain lookups,
    figures buried in tables, the FAQ pairs that chunk differently from every
    other section, and multi-section synthesis. Losing a whole category to an
    edit would quietly narrow what the evals measure.
    """
    docs = {c.doc for c in load_corpus()}
    assert {g.doc for g in GOLDENS} == docs, "a document has no questions"

    for doc in docs:
        kinds = [g.kind for g in GOLDENS if g.doc == doc]
        assert sorted(kinds) == ["faq", "figure", "lookup", "synthesis"], doc


def test_questions_are_unique_and_self_contained():
    questions = [g.question for g in GOLDENS]
    assert len(set(questions)) == len(questions)
    # Each is asked as the first turn of a fresh session, so there is no history
    # for the router to resolve a trailing reference against.
    assert all(q.endswith("?") for q in questions)
