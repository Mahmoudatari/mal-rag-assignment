"""PII detection and masking. Pure — no network, no LLM, no secrets."""

from pii.detector import PiiSpan, detect
from pii.masker import Redaction, mask, redact
from pii.patterns import ACCOUNT_NUMBER, EMIRATES_ID

__all__ = [
    "ACCOUNT_NUMBER",
    "EMIRATES_ID",
    "PiiSpan",
    "Redaction",
    "detect",
    "mask",
    "redact",
]
