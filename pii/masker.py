"""Masking/redaction of detected spans.

`presidio-anonymizer` is a separate package and is deliberately not used: the
substitution is a few lines, and owning it keeps the placeholder format ours.
Placeholders are `[ENTITY_TYPE]` because the downstream model needs to know
*what* was removed — "transfer from [ACCOUNT_NUMBER] to [ACCOUNT_NUMBER]" stays
answerable in a way that a uniform `[REDACTED]` does not.

Masking is one-way. No mapping back to the original values is stored anywhere,
which is what makes it safe to hand the result to a prompt, a log, or a trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pii.detector import PiiSpan, detect


@dataclass(frozen=True, slots=True)
class Redaction:
    """Masked text plus the spans that were removed (kind and offsets only).

    `spans` index the *original* text, not `text`. They exist for tracing —
    which is why they never carry the matched values.
    """

    text: str
    spans: list[PiiSpan] = field(default_factory=list)

    @property
    def found_pii(self) -> bool:
        return bool(self.spans)

    @property
    def kinds(self) -> set[str]:
        return {s.kind for s in self.spans}


def placeholder(kind: str) -> str:
    return f"[{kind}]"


def mask(text: str, spans: list[PiiSpan]) -> str:
    """Replace each span with its placeholder.

    Applied right-to-left so that each replacement leaves the offsets of the
    spans still to be processed valid.
    """
    masked = text
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        masked = masked[: span.start] + placeholder(span.kind) + masked[span.end :]
    return masked


def redact(text: str) -> Redaction:
    """Detect and mask in one call — the entry point the graph's `redact` node uses."""
    spans = detect(text)
    return Redaction(text=mask(text, spans), spans=spans)
