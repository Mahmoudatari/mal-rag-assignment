"""Eval: PII is masked before reaching the LLM.

The assertion that matters is *absence*: the raw identifier must not survive
into the text handed downstream. Everything else here guards a specific way
that property has been seen to break — a checksum used as a gate, an overlap
resolved by masking only part of an identifier, a span object that quietly
retains the value it was supposed to remove.
"""

import subprocess
import sys

import pytest

from pii import EMIRATES_ID, PiiSpan, detect, mask, redact
from pii.patterns import ACCOUNT_NUMBER, luhn_valid
from rag.nodes import redact as redact_node

# 784-YYYY-NNNNNNN-C. The trailing 6 makes this Luhn-valid; any other final
# digit does not, which is what the "checksums never gate" test relies on.
VALID_EID = "784-1990-1234567-6"
INVALID_LUHN_EID = "784-1990-1234567-1"
UAE_IBAN = "AE070331234567890123456"
ACCOUNT = "1234567890"


# --- the core requirement: known PII inputs are masked -------------------


@pytest.mark.parametrize(
    ("text", "secret", "kind"),
    [
        (f"My Emirates ID is {VALID_EID}", VALID_EID, "EMIRATES_ID"),
        (f"Emirates ID {VALID_EID.replace('-', '')}", VALID_EID.replace("-", ""), "EMIRATES_ID"),
        (f"my account number is {ACCOUNT}", ACCOUNT, "ACCOUNT_NUMBER"),
        (f"transfer from {UAE_IBAN} today", UAE_IBAN, "IBAN_CODE"),
        ("reach me at fatima@example.ae", "fatima@example.ae", "EMAIL_ADDRESS"),
        ("call me on +971501234567", "+971501234567", "PHONE_NUMBER"),
        ("my card is 4111111111111111", "4111111111111111", "CREDIT_CARD"),
        ("I am Fatima Al Mansouri", "Fatima Al Mansouri", "PERSON"),
    ],
)
def test_known_pii_is_masked(text: str, secret: str, kind: str) -> None:
    result = redact(text)
    assert secret not in result.text, f"{kind} survived redaction"
    assert f"[{kind}]" in result.text
    assert kind in result.kinds


def test_multiple_identifiers_in_one_message() -> None:
    text = (
        f"Hi, I'm Fatima Al Mansouri, Emirates ID {VALID_EID}, "
        f"account {ACCOUNT}, email fatima@example.ae"
    )
    result = redact(text)

    for secret in ("Fatima Al Mansouri", VALID_EID, ACCOUNT, "fatima@example.ae"):
        assert secret not in result.text
    assert {"PERSON", EMIRATES_ID, ACCOUNT_NUMBER, "EMAIL_ADDRESS"} <= result.kinds


def test_no_digit_run_from_an_identifier_survives() -> None:
    """A partially-masked identifier is still a leak.

    Guards the overlap resolution: EMIRATES_ID and ACCOUNT_NUMBER both match a
    15-digit run, and resolving that badly can mask one half and emit the other.
    """
    result = redact(f"Emirates ID {VALID_EID}, account {ACCOUNT}")
    digits_left = "".join(c for c in result.text if c.isdigit())
    assert digits_left == ""


# --- names, which rest entirely on NER -----------------------------------
# Nothing tells this module who the customer is, so spaCy is the only thing
# between a name and the prompt. These cases are therefore load-bearing rather
# than illustrative, and they are the reason the pinned model is `lg`.


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("I am Fatima Al Mansouri", "Fatima Al Mansouri"),
        ("this is Ahmed Khalid speaking", "Ahmed Khalid"),
        ("Hi, this is Ahmed. What is my balance?", "Ahmed"),
        ("Omar Bin Rashid here, account 1234567890", "Omar Bin Rashid"),
        ("Please transfer to Yusuf Abdullah", "Yusuf Abdullah"),
        # Lower-cased input is the classic NER weak spot — customers type this way.
        ("my name is mohammed al hashimi", "mohammed al hashimi"),
    ],
)
def test_names_are_detected_and_masked(text: str, name: str) -> None:
    result = redact(text)
    assert name not in result.text
    assert "[PERSON]" in result.text


def test_third_party_name_is_masked_too() -> None:
    """Not just the customer: anyone they mention is PII we must not forward."""
    assert "Khalid Al Nuaimi" not in redact("send 500 to Khalid Al Nuaimi").text


# --- checksums raise confidence, they never reject -----------------------


def test_luhn_invalid_emirates_id_is_still_masked() -> None:
    """A wrong checksum assumption must not become a false negative."""
    assert not luhn_valid(INVALID_LUHN_EID.replace("-", ""))
    result = redact(f"my id is {INVALID_LUHN_EID}")
    assert INVALID_LUHN_EID not in result.text


def test_luhn_valid_emirates_id_scores_higher_than_invalid() -> None:
    def eid_score(value: str) -> float:
        spans = detect(f"Emirates ID {value}", entities=(EMIRATES_ID,))
        return max(s.score for s in spans)

    assert eid_score(VALID_EID) > eid_score(INVALID_LUHN_EID)


def test_luhn_valid() -> None:
    assert luhn_valid("4111111111111111")
    assert not luhn_valid("4111111111111112")


# --- spans must not carry the values they removed ------------------------


def test_span_carries_offsets_only() -> None:
    assert set(PiiSpan.__dataclass_fields__) == {"kind", "start", "end", "score"}


def test_no_span_field_contains_the_secret() -> None:
    for span in detect(f"Emirates ID {VALID_EID}"):
        for name in PiiSpan.__dataclass_fields__:
            assert VALID_EID not in str(getattr(span, name))


def test_spans_index_the_original_text() -> None:
    text = f"my id {VALID_EID} ok"
    spans = detect(text)
    assert [text[s.start : s.end] for s in spans] == [VALID_EID]


# --- precision: ordinary questions must survive intact -------------------


@pytest.mark.parametrize(
    "text",
    [
        "Is Murabaha permissible for a car purchase?",
        "What is the difference between Sukuk and a conventional bond?",
        "Explain how Ijara works for home finance.",
        "Does Takaful cover medical expenses?",
    ],
)
def test_clean_finance_questions_are_untouched(text: str) -> None:
    result = redact(text)
    assert result.text == text
    assert not result.found_pii


def test_empty_input() -> None:
    result = redact("")
    assert result.text == ""
    assert not result.found_pii


# --- masking mechanics ---------------------------------------------------


def test_mask_applies_right_to_left_without_corrupting_offsets() -> None:
    text = "aaa bbb ccc"
    spans = [PiiSpan("X", 0, 3, 1.0), PiiSpan("Y", 8, 11, 1.0)]
    assert mask(text, spans) == "[X] bbb [Y]"


def test_mask_placeholder_names_the_entity() -> None:
    """Downstream needs to know *what* was removed to stay able to answer."""
    result = redact(f"send from {ACCOUNT} to 9876543210 account")
    assert result.text.count(f"[{ACCOUNT_NUMBER}]") == 2


def test_detected_spans_never_overlap() -> None:
    spans = detect(f"Emirates ID {VALID_EID} account {ACCOUNT} iban {UAE_IBAN}")
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end <= later.start


# --- the graph node ------------------------------------------------------
# Tested directly, without compiling a graph — nodes are plain functions.

# Being the first node, redact also starts the turn: it resets the turn-scoped
# keys so a checkpointed session cannot carry the previous turn's retrieval into
# this one. `history` is not here — it is the one key that survives a turn.
TURN_RESET_KEYS = {
    "search_query",
    "chunks",
    "relevant",
    "grader_note",
    "attempts",
    "tried_queries",
    "references",
    "usage_log",
}
NODE_KEYS = {"query", "pii_spans"} | TURN_RESET_KEYS


def test_node_masks_raw_query_into_query() -> None:
    out = redact_node.run({"raw_query": f"I'm Fatima, ID {VALID_EID}"})
    assert VALID_EID not in out["query"]
    assert "Fatima" not in out["query"]


def test_node_returns_only_the_keys_it_changed() -> None:
    """Nodes return partial state; anything extra here would be a leak surface."""
    out = redact_node.run({"raw_query": f"account {ACCOUNT}"})
    assert set(out) == NODE_KEYS


def test_node_clears_the_previous_turn_but_keeps_history() -> None:
    """State is checkpointed per session, so turn N+1 starts from turn N's dict."""
    stale = {
        "raw_query": "and what about Ijara?",
        "search_query": "murabaha profit rate",
        "chunks": [{"chunk_id": "murabaha#001", "doc": "murabaha", "text": "x", "score": 0.9}],
        "relevant": True,
        "grader_note": "nothing missing",
        "attempts": 2,
        "tried_queries": ["murabaha profit rate"],
        "references": [{"n": 1, "doc": "murabaha", "chunk_id": "murabaha#001"}],
        "usage_log": [{"node": "router", "model": "m", "prompt_tokens": 1,
                       "completion_tokens": 1, "total_tokens": 2, "cost": 0.0}],
        "history": [{"role": "user", "content": "is murabaha halal?"}],
    }
    out = redact_node.run(stale)

    assert out["attempts"] == 0
    assert out["chunks"] == []
    assert out["references"] == []
    assert out["usage_log"] == []
    assert out["tried_queries"] == []
    assert out["search_query"] == ""
    assert out["grader_note"] == ""
    assert out["relevant"] is False
    assert "history" not in out, "history is the one key that must survive the turn"


def test_reset_lists_are_not_shared_between_turns() -> None:
    """A module-level constant would hand every turn the same list objects."""
    first = redact_node.run({"raw_query": "one"})
    second = redact_node.run({"raw_query": "two"})
    assert first["chunks"] is not second["chunks"]
    assert first["usage_log"] is not second["usage_log"]


def test_node_never_writes_raw_query_anywhere() -> None:
    raw = f"my id is {VALID_EID}"
    out = redact_node.run({"raw_query": raw})
    assert raw not in repr(out)
    assert VALID_EID not in repr(out)


def test_node_spans_carry_kind_and_offsets_only() -> None:
    out = redact_node.run({"raw_query": f"Emirates ID {VALID_EID}"})
    assert out["pii_spans"]
    for span in out["pii_spans"]:
        assert set(span) == {"kind", "start", "end"}


def test_node_handles_a_turn_with_no_pii() -> None:
    text = "Is Murabaha permissible?"
    out = redact_node.run({"raw_query": text})
    assert out["query"] == text
    assert out["pii_spans"] == []


def test_node_handles_missing_raw_query() -> None:
    out = redact_node.run({})
    assert out["query"] == ""
    assert out["pii_spans"] == []
    assert set(out) == NODE_KEYS


# --- the module must run standalone in CI --------------------------------


def test_pii_imports_without_llm_or_database_dependencies() -> None:
    """`pii/` stays pure: no API key, no DB, so this eval runs anywhere.

    Asserted as a dependency boundary rather than a comment, because the way
    this regresses is an innocent-looking import added months later.
    """
    probe = (
        "import sys; import pii; "
        "leaked = {m for m in ('openai', 'psycopg', 'langgraph') if m in sys.modules}; "
        "assert not leaked, leaked; print('clean')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr
    assert "clean" in proc.stdout
