"""Regex patterns: Emirates ID, account numbers.

Presidio ships recognizers for the international identifiers we care about —
`IBAN_CODE` already validates UAE IBANs (AE + 21 chars, mod-97) at full score,
and PERSON / EMAIL_ADDRESS / PHONE_NUMBER are built in. Only the two UAE- and
Mal-specific formats are defined here.

Checksums *raise* confidence, they never reject. A failed checksum still yields
a match at the base score: over-redacting a number that merely looks like an
Emirates ID is harmless, while a missed one is a data leak. It also means a
wrong assumption about the check-digit algorithm cannot cause a false negative.
"""

from presidio_analyzer import Pattern, PatternRecognizer

EMIRATES_ID = "EMIRATES_ID"
ACCOUNT_NUMBER = "ACCOUNT_NUMBER"

# 784-YYYY-NNNNNNN-C — 15 digits: 784 (UAE ISO country code), birth year, 7
# random digits, check digit. Separators are conventional — customers type
# dashes, spaces, dots, slashes and underscores — so accept any grouping.
_EMIRATES_ID_PATTERN = r"\b784[-\s./_]?\d{4}[-\s./_]?\d{7}[-\s./_]?\d\b"

# Bank-internal account numbers: 9+ digits, optionally grouped by any common
# separator. Deliberately broad — the score starts low and Presidio's context
# enhancer lifts it when account-ish words appear nearby, which is what
# separates "my account 1234567890" from a long number that happens to be
# something else.
#
# The lookbehind keeps this off international phone numbers: a leading `+` is a
# strong phone signal, and without it "+971501234567" matches both recognizers
# at the same score and the winner comes down to sort order. No `\b` anchors:
# they made glued shorthands ("acct1234567890") unmatchable and let a stray
# trailing letter truncate the match so only part of an identifier was masked —
# a partial mask leaks the remainder. The lookarounds exclude exactly what the
# guard is about (digits and `+`) and nothing else. The upper bound is wide so
# an over-long run masks whole rather than to a 24-character prefix.
_ACCOUNT_PATTERN = r"(?<![+\d])\d[\d\s./_-]{7,40}\d(?!\d)"

_ACCOUNT_CONTEXT = [
    "account",
    "acct",
    "iban",
    "balance",
    "statement",
    "transfer",
    "deposit",
    "card",
]


def luhn_valid(digits: str) -> bool:
    """Standard Luhn mod-10 check over a digit string."""
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def digits_only(text: str) -> str:
    return "".join(c for c in text if c.isdigit())


class EmiratesIdRecognizer(PatternRecognizer):
    """Emirates ID. Shape match detects; a valid Luhn check promotes to certain.

    `validate_result` returns True (score -> 1.0) or None (keep the base score)
    and never False, so a checksum mismatch lowers confidence without dropping
    the span.
    """

    def __init__(self) -> None:
        super().__init__(
            supported_entity=EMIRATES_ID,
            patterns=[Pattern("emirates_id", _EMIRATES_ID_PATTERN, 0.6)],
            context=["emirates", "eid", "identity", "identification"],
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        digits = digits_only(pattern_text)
        if len(digits) != 15:
            return None
        return True if luhn_valid(digits) else None


class AccountNumberRecognizer(PatternRecognizer):
    """Bank account numbers, scored low and lifted by surrounding context."""

    def __init__(self) -> None:
        super().__init__(
            supported_entity=ACCOUNT_NUMBER,
            patterns=[Pattern("account_number", _ACCOUNT_PATTERN, 0.4)],
            context=_ACCOUNT_CONTEXT,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        # Runs with too few digits once separators are stripped are grouped
        # prose (dates, "1 000 000"), not accounts — rejecting them is safe
        # because nothing identifier-shaped is that short. Too-long runs are
        # NEVER rejected: Presidio drops a result its validator returns False
        # for, so rejecting a 20-digit paste would forward the likeliest real
        # identifier in the message to the LLM unmasked.
        return None if len(digits_only(pattern_text)) >= 9 else False
