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
# Mal's own account format, and the reason the brand carve-out is scoped to
# PERSON rather than applied to every kind — see the branded-name test below.
MAL_ACCOUNT = "MAL-1001-2200-4417"


# --- the core requirement: known PII inputs are masked -------------------


@pytest.mark.parametrize(
    ("text", "secret", "kind"),
    [
        (f"My Emirates ID is {VALID_EID}", VALID_EID, "EMIRATES_ID"),
        (f"Emirates ID {VALID_EID.replace('-', '')}", VALID_EID.replace("-", ""), "EMIRATES_ID"),
        # Separators are whatever the customer typed, not just dashes — the
        # dot/slash/underscore forms all bypassed detection entirely once.
        ("My Emirates ID is 784.1990.1234567.6", "784.1990.1234567.6", "EMIRATES_ID"),
        ("my id is 784/1990/1234567/6", "784/1990/1234567/6", "EMIRATES_ID"),
        ("my id is 784_1990_1234567_6", "784_1990_1234567_6", "EMIRATES_ID"),
        (f"my account number is {ACCOUNT}", ACCOUNT, "ACCOUNT_NUMBER"),
        # Longer than 18 digits must mask, not drop: the length validator once
        # *rejected* out-of-range runs, and a rejected span is a total leak.
        ("my account is 1234 5678 9012 3456 7890", "1234 5678 9012 3456 7890", "ACCOUNT_NUMBER"),
        ("my account is 9876543210987654321", "9876543210987654321", "ACCOUNT_NUMBER"),
        # Glued to a shorthand prefix — `\b` anchors once made this unmatchable.
        ("acct1234567890 please", "1234567890", "ACCOUNT_NUMBER"),
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


@pytest.mark.parametrize(
    "text",
    [
        f"Emirates ID {VALID_EID}, account {ACCOUNT}",
        # A trailing letter once truncated the match: the Emirates recognizer's
        # final \b failed on the "x", the account recognizer took the first 14
        # digits, and the identifier's check digit went to the model in clear.
        f"my emirates id is {VALID_EID}x",
        "acct1234567890",
        # 25 digits: longer than any single pattern's old upper bound — must
        # mask whole, not to a prefix.
        "ref 12345 67890 12345 67890 12345",
    ],
)
def test_no_digit_run_from_an_identifier_survives(text: str) -> None:
    """A partially-masked identifier is still a leak.

    Guards the overlap resolution and the match boundaries: EMIRATES_ID and
    ACCOUNT_NUMBER both match a 15-digit run, and resolving or anchoring that
    badly can mask one part and emit the other.
    """
    result = redact(text)
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


# Branded product names are the expensive false positive: spaCy reads
# "Mal Digital Wakala" as a PERSON at 0.85, and did it *intermittently* — bare
# and in most sentences, but not in "What is Mal Everyday Murabaha?". A live
# turn came through as "[PERSON] savings" and the answer opened by apologising
# for not being able to see "[PERSON]'s account details". Both the bare and
# in-sentence forms are pinned, because the bug was that the two disagreed.


@pytest.mark.parametrize(
    "text",
    [
        "Mal Digital Wakala",
        "Mal Everyday Murabaha",
        "Mal Ijara Muntahia Bittamleek",
        "Mal Takaful Family Protection",
        "Tell me about Mal Digital Wakala savings",
        "What is Mal Everyday Murabaha?",
        "How does Mal Ijara Muntahia Bittamleek work?",
        "Does Mal Digital Wakala pay profit?",
        "Compare Mal Digital Wakala and Mal Everyday Murabaha",
    ],
)
def test_branded_product_names_are_not_masked(text: str) -> None:
    result = redact(text)
    assert result.text == text, "the bank's own product name was read as a customer"
    assert not result.found_pii


@pytest.mark.parametrize(
    ("text", "name", "product"),
    [
        (
            "I am Ahmed Hassan and I want Mal Digital Wakala",
            "Ahmed Hassan",
            "Mal Digital Wakala",
        ),
        (
            "Ahmed Hassan asked about Mal Everyday Murabaha",
            "Ahmed Hassan",
            "Mal Everyday Murabaha",
        ),
    ],
)
def test_a_real_name_still_masks_beside_a_branded_product(
    text: str, name: str, product: str
) -> None:
    """The carve-out drops one PERSON span, not the entity."""
    result = redact(text)
    assert name not in result.text
    assert "[PERSON]" in result.text
    assert product in result.text
    assert result.text.count("[PERSON]") == 1


@pytest.mark.parametrize(
    "name",
    ["Jamal Al Farsi", "Malik Rahman", "Malak Ibrahim"],
)
def test_names_merely_containing_mal_are_unaffected(name: str) -> None:
    """The brand must stand alone as a token, not as a substring."""
    result = redact(f"my name is {name}")
    assert name not in result.text
    assert "[PERSON]" in result.text


@pytest.mark.parametrize(
    ("text", "name"),
    [
        ("my full name is Ahmed Mal Hassan", "Ahmed Mal Hassan"),
        ("I am Mal Ahmed", "Mal Ahmed"),
        ("Hala Mal Saeed here, what is my balance?", "Hala Mal Saeed"),
    ],
)
def test_a_name_containing_the_brand_token_still_masks(text: str, name: str) -> None:
    """The carve-out fires only when *every* token is brand vocabulary.

    A contains-"mal" test dropped these spans wholesale — and PERSON has no
    pattern recognizer behind it, so a dropped span is an unrecoverable leak.
    """
    result = redact(text)
    assert name not in result.text
    assert "[PERSON]" in result.text


@pytest.mark.parametrize(
    "text",
    [
        MAL_ACCOUNT,
        f"My account is {MAL_ACCOUNT}",
        f"account {MAL_ACCOUNT} balance please",
        f"Hi I am Sara Ahmed, my account {MAL_ACCOUNT} needs help",
    ],
)
def test_mal_prefixed_account_numbers_still_mask(text: str) -> None:
    """The carve-out is scoped to PERSON precisely so this keeps working.

    Mal's account ids carry the bank's name, so a blanket contains-"Mal" rule
    would exempt every one of them. Pattern-based kinds match a format, and a
    format match is never brand vocabulary.
    """
    result = redact(text)
    assert MAL_ACCOUNT not in result.text
    assert f"[{ACCOUNT_NUMBER}]" in result.text
    assert ACCOUNT_NUMBER in result.kinds
    assert not any(c.isdigit() for c in result.text)


def test_a_name_beside_a_mal_account_number_masks_too() -> None:
    result = redact(f"Hi I am Sara Ahmed, my account {MAL_ACCOUNT} needs help")
    assert "Sara Ahmed" not in result.text
    assert {"PERSON", ACCOUNT_NUMBER} <= result.kinds


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
    "resolved_query",
    "chunks",
    "candidate_log",
    "relevant",
    "grader_note",
    "attempts",
    "tried_queries",
    "references",
    "usage_log",
}
# `raw_query` is in the write set on purpose: the node overwrites it with ""
# so the checkpointed state never retains the unmasked text.
NODE_KEYS = {"query", "pii_spans", "raw_query"} | TURN_RESET_KEYS


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
        "candidate_log": [["murabaha#001", "murabaha#002"]],
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
    assert out["candidate_log"] == []
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
    assert first["candidate_log"] is not second["candidate_log"]
    assert first["usage_log"] is not second["usage_log"]


def test_node_discards_raw_query_from_the_merged_state() -> None:
    """The property that matters is on the *merged* dict, not the partial return.

    The checkpointer persists `{**previous_state, **node_writes}` per thread —
    asserting on the partial return alone would pass even if raw_query were
    silently retained, because a partial return never contains it.
    """
    raw = f"my id is {VALID_EID}"
    out = redact_node.run({"raw_query": raw})
    merged = {"raw_query": raw} | out
    assert merged["raw_query"] == ""
    assert raw not in repr(merged)
    assert VALID_EID not in repr(merged)


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
