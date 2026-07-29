"""Unit tests for the accounts package and the account node.

Everything here is pure — the store is dicts, the node is a lookup — so the
tests are about the contract, not behaviour under IO: the record never carries
the full account number (the leak-by-construction guarantee the prompts and
trace rely on), lookups hand out copies rather than the module's own dicts
(state is checkpointed, so a shared object would let one turn edit history),
and the node always writes `account` and `history_account`, which together stop
a turn without an id inheriting the previous turn's context — the record via the
unconditional write, the *answers derived from* the record via the history drop.
"""

from __future__ import annotations

import re

import pytest

from accounts import ACCOUNTS, field_summary, lookup, product_names, render
from rag.nodes import account as account_node

KNOWN_ID = "MAL-1001-2200-4417"
KNOWN_MASK = "MAL-****-****-4417"
OTHER_ID = "MAL-2002-3300-8802"

# What the leak looks like: `generate`'s own answer, restating the record.
HISTORY = [
    {"role": "user", "content": "how much is left on my contract?"},
    {"role": "assistant", "content": "Seven instalments remain on contract MUR-****-0417."},
]


# --- the store's invariants -------------------------------------------------


def test_every_record_masks_all_but_the_last_four_digits() -> None:
    for account_id, record in ACCOUNTS.items():
        assert record["masked_id"] == f"MAL-****-****-{account_id[-4:]}"


@pytest.mark.parametrize("account_id", sorted(ACCOUNTS))
def test_the_full_account_number_appears_nowhere_inside_the_record(account_id: str) -> None:
    """The guarantee prompts and traces lean on: they render the record, and
    the record has nothing to leak — only the key holds the full number."""
    assert account_id not in repr(ACCOUNTS[account_id])


def test_no_unmasked_reference_appears_in_any_record() -> None:
    """Contract and lease references are identifiers like the account number,
    and the record is rendered verbatim into prompts — so they exist only in
    display-masked form (`MUR-****-0417`), by construction."""
    assert not re.search(r"(?:MUR|IJR)-\d", repr(ACCOUNTS))


def test_every_holding_names_its_product() -> None:
    for record in ACCOUNTS.values():
        assert record["holdings"], "a customer with no holdings has no context to attach"
        for holding in record["holdings"]:
            assert holding["product"]


def test_the_demo_ids_documented_in_the_api_schema_exist() -> None:
    """The OpenAPI description promises three demo accounts; a renamed key here
    would leave the documentation pointing at ids that silently resolve to
    no-account-context."""
    from app.schemas import ChatRequest

    documented = re.findall(r"MAL-\d{4}-\d{4}-\d{4}", ChatRequest.model_fields["account_id"].description)
    assert set(documented) == set(ACCOUNTS)


# --- lookup -----------------------------------------------------------------


def test_lookup_returns_the_record_for_a_known_id() -> None:
    record = lookup(KNOWN_ID)
    assert record is not None
    assert record["masked_id"] == "MAL-****-****-4417"


@pytest.mark.parametrize("account_id", ["", "MAL-9999-9999-9999", "nonsense"])
def test_lookup_is_none_for_empty_or_unknown_ids(account_id: str) -> None:
    assert lookup(account_id) is None


def test_lookup_hands_out_a_copy_not_the_stores_own_dict() -> None:
    """The caller's return value is checkpointed state — mutating it must not
    edit what every later lookup in the process sees."""
    first = lookup(KNOWN_ID)
    first["holdings"].clear()
    first["masked_id"] = "tampered"

    second = lookup(KNOWN_ID)
    assert second["masked_id"] == "MAL-****-****-4417"
    assert second["holdings"]


# --- rendering --------------------------------------------------------------


def test_render_carries_the_masked_id_and_every_holding_field() -> None:
    record = lookup(KNOWN_ID)
    block = render(record)

    assert "MAL-****-****-4417" in block
    for holding in record["holdings"]:
        assert holding["product"] in block
        for key in holding:
            if key != "product":
                assert key in block


def test_product_names_lists_each_holding_once() -> None:
    assert product_names(lookup(KNOWN_ID)) == ["Murabaha everyday finance", "Wakala savings"]


def test_field_summary_carries_every_field_name_and_no_value() -> None:
    """The grader's view of the record: availability without figures."""
    record = lookup(KNOWN_ID)
    outline = field_summary(record)

    for holding in record["holdings"]:
        assert holding["product"] in outline
        for key, value in holding.items():
            if key != "product":
                assert key in outline
                assert str(value) not in outline
    assert "MAL-****-****-4417" not in outline


# --- the node ---------------------------------------------------------------


def test_node_resolves_a_known_id_and_writes_the_record_and_its_context() -> None:
    result = account_node.run({"account_id": KNOWN_ID})
    assert set(result) == {"account", "history_account"}
    assert result["account"]["masked_id"] == KNOWN_MASK
    assert result["history_account"] == KNOWN_MASK


@pytest.mark.parametrize("state", [{}, {"account_id": ""}, {"account_id": "MAL-9999-9999-9999"}])
def test_node_always_writes_account_even_when_there_is_nothing_to_attach(state: dict) -> None:
    """Always writing None is the anti-carryover mechanism: a node that skipped
    the key would leave the previous turn's account in checkpointed state."""
    assert account_node.run(state) == {"account": None, "history_account": ""}


# --- history scoped to the account that produced it -------------------------


def test_first_account_of_the_session_keeps_the_conversation() -> None:
    """The asymmetry, deliberately: nothing said before an account was attached
    can restate a record, so wiping here would cost context and protect nothing."""
    result = account_node.run({"account_id": KNOWN_ID, "history_account": "", "history": HISTORY})

    assert "history" not in result, "anonymous → account must not restart the conversation"
    assert result["history_account"] == KNOWN_MASK


def test_switching_account_drops_the_previous_customers_figures() -> None:
    """`account` alone is cleared by the unconditional write; the contract id in
    turn 1's answer is not, and history is what replays it into turn 2."""
    result = account_node.run(
        {"account_id": OTHER_ID, "history_account": KNOWN_MASK, "history": HISTORY}
    )

    assert result["history"] == []
    assert result["history_account"] == "MAL-****-****-8802"


def test_a_further_turn_on_the_same_account_keeps_its_history() -> None:
    """The ordinary multi-turn case — resetting here would break every follow-up
    question the account path exists to answer."""
    result = account_node.run(
        {"account_id": KNOWN_ID, "history_account": KNOWN_MASK, "history": HISTORY}
    )

    assert "history" not in result
    assert result["history_account"] == KNOWN_MASK


@pytest.mark.parametrize("account_id", ["", "MAL-9999-9999-9999"])
def test_dropping_the_account_also_drops_its_history(account_id: str) -> None:
    """A turn that omits the id is exactly the turn the figures would be replayed
    into: `account` is None, yet the prompt still carries "contract MUR-****-0417"
    from the assistant message. Unknown ids resolve to no context and count too."""
    result = account_node.run(
        {"account_id": account_id, "history_account": KNOWN_MASK, "history": HISTORY}
    )

    assert result["history"] == []
    assert result["history_account"] == ""
