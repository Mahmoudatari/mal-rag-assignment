"""Detector: locate PII spans in free text.

Presidio's `AnalyzerEngine` does the work, with two custom recognizers
registered for the UAE-specific formats it does not ship (see `patterns.py`).

Two deliberate constraints:

- **No network at request time.** A bare `AnalyzerEngine()` tries to pip-install
  its spaCy model on first construction. The NLP engine is therefore configured
  explicitly against a model pinned in `pyproject.toml`, so the download happens
  at `uv sync` and never on the request path.
- **Spans carry no values.** `PiiSpan` is kind + offsets + score, matching what
  `rag/state.py` is allowed to hold. The matched text never leaves this module,
  so it cannot reach a prompt, a log, or a trace.

Names are detected, never assumed: nothing here is told who the customer is, so
spaCy's NER is the only defence for PERSON. That is why the pinned model is
`en_core_web_lg` — with no known-name fallback, NER recall *is* the recall.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

from core.config import get_settings
from pii.patterns import AccountNumberRecognizer, EmiratesIdRecognizer

PERSON = "PERSON"


@dataclass(frozen=True, slots=True)
class PiiSpan:
    """A detected identifier. Offsets index the *original* text.

    Carries no matched value by design — see module docstring.
    """

    kind: str
    start: int
    end: int
    score: float


@lru_cache(maxsize=1)
def _analyzer() -> AnalyzerEngine:
    """Build the engine once. Loading spaCy costs ~1s, so never per request."""
    settings = get_settings()
    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": settings.pii_spacy_model}
            ],
        }
    ).create_engine()

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(languages=["en"], nlp_engine=nlp_engine)
    registry.add_recognizer(EmiratesIdRecognizer())
    registry.add_recognizer(AccountNumberRecognizer())

    return AnalyzerEngine(
        nlp_engine=nlp_engine, registry=registry, supported_languages=["en"]
    )


# Tie-break order for overlapping spans of equal score, most specific first.
# A bare 10-digit number matches both ACCOUNT_NUMBER and PHONE_NUMBER at the
# same confidence; without an explicit order the winner is decided by sort
# stability, which is not a decision. Either way the span is masked, so this
# only settles which placeholder the answering model sees — and in a banking
# assistant an unprefixed digit run is far likelier an account than a phone.
_PRIORITY = (
    "EMIRATES_ID",
    "IBAN_CODE",
    "CREDIT_CARD",
    "ACCOUNT_NUMBER",
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    PERSON,
)


def _rank(kind: str) -> int:
    return _PRIORITY.index(kind) if kind in _PRIORITY else len(_PRIORITY)


# spaCy tags Mal's branded product names as PERSON — "Mal Digital Wakala",
# "Mal Everyday Murabaha", "Mal Ijara Muntahia Bittamleek" all come back at 0.85
# — and does it *inconsistently*: "What is Mal Everyday Murabaha?" survives
# while "Tell me about Mal Digital Wakala savings" becomes "[PERSON] savings".
# That is worse than a steady false positive, because it is flaky in production:
# one live turn masked the product, and the answering prompt's account-context
# rule then opened the reply with an irrelevant "I cannot see [PERSON]'s account
# details". A PERSON span made up entirely of brand vocabulary is dropped.
#
# **Every token must be brand vocabulary, not merely one of them.** A
# contains-"mal" test drops "Ahmed Mal Hassan" — a customer whose middle name
# happens to be the bank's token — and PERSON has no pattern recognizer behind
# it, so a dropped span is an unrecoverable leak, not a lower score. "Mal
# Digital Wakala" is all brand words and is dropped; any span carrying a word
# outside the vocabulary stays a PERSON and masks.
#
# **Scoped to PERSON on purpose.** Mal's account numbers are literally
# `MAL-nnnn-nnnn-nnnn`, so a blanket brand rule would unmask them. The
# pattern-based kinds match a *format*, and a format match is never brand
# vocabulary — they keep masking whatever their text contains.
#
# Whole tokens so real names are untouched: "Jamal", "Malik" and "Malak" never
# tokenize to a brand word. Lowercased because customers type lowercase.
#
# A module constant, not a setting: nothing outside this file has a reason to
# retune the bank's own vocabulary, and the repo only adds settings that earn it.
_BRAND_VOCAB = frozenset({
    "mal",
    # product-name words spaCy reads as part of a PERSON beside the bank's name
    "digital", "wakala", "everyday", "murabaha", "ijara", "muntahia",
    "bittamleek", "takaful", "family", "protection", "sukuk", "savings",
    # connector inside compound mentions ("Mal Digital Wakala and Mal Everyday
    # Murabaha" can come back as one span)
    "and",
})


def _is_brand_vocabulary(text: str) -> bool:
    # "'s" stripped first so "Mal's Wakala" reads as brand, not as a token "s".
    tokens = re.findall(r"[a-z]+", text.lower().replace("'s", " "))
    return bool(tokens) and all(t in _BRAND_VOCAB for t in tokens)


def _resolve_overlaps(spans: list[PiiSpan]) -> list[PiiSpan]:
    """Keep the strongest span where two overlap.

    Recognizers overlap by nature — an Emirates ID is also a long digit run, so
    ACCOUNT_NUMBER matches it too. Higher score wins, then the longer span so
    the mask covers the whole identifier rather than part of it, then entity
    specificity.
    """
    ordered = sorted(
        spans, key=lambda s: (-s.score, -(s.end - s.start), _rank(s.kind), s.start)
    )
    kept: list[PiiSpan] = []
    for span in ordered:
        if any(span.start < k.end and k.start < span.end for k in kept):
            continue
        kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def detect(
    text: str,
    *,
    entities: tuple[str, ...] | None = None,
    score_threshold: float | None = None,
) -> list[PiiSpan]:
    """Return non-overlapping PII spans, ordered by position.

    `entities` is an allowlist: Presidio's full recognizer set fires low-score
    guesses at irrelevant formats (US driver's licence, and so on) that are pure
    noise for a UAE bank.
    """
    if not text:
        return []

    settings = get_settings()
    allowed = tuple(entities if entities is not None else settings.pii_entities)
    threshold = (
        score_threshold
        if score_threshold is not None
        else settings.pii_score_threshold
    )

    results = _analyzer().analyze(
        text=text,
        language="en",
        entities=list(allowed),
        score_threshold=threshold,
    )
    spans = [PiiSpan(r.entity_type, r.start, r.end, r.score) for r in results]

    # Drop branded product names (see `_BRAND_VOCAB`) before overlap
    # resolution, not after: a discarded PERSON span must not go on suppressing
    # a real identifier that overlaps it. Reading the matched text here keeps
    # the module's rule intact — it is already in scope, and it still never
    # leaves.
    spans = [
        s
        for s in spans
        if not (s.kind == PERSON and _is_brand_vocabulary(text[s.start : s.end]))
    ]
    return _resolve_overlaps(spans)
