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
    return _resolve_overlaps(spans)
