"""Eval: non-Islamic-finance queries are declined.

Two halves, because refusal has two failure modes and only one of them is
obvious.

The stated requirement is the first: a turn with nothing to do with Islamic
finance or Mal must terminate at the router — the static message, no retrieval,
no answering model. That is asserted end to end and cheaply: the out-of-scope
cases run with no DATABASE_URL at all, so a mis-route into `retrieve` raises on
the unopened async pool instead of quietly passing.

The second half is the one that actually bit. The failure measured during
development was never a model happily answering trivia — it was the router
refusing real Mal banking questions: a fraudulent-card dispute, a home mortgage,
an FX rate. A customer with a genuine problem was told it was not this
assistant's business. So the eval also pins the other direction: in-scope turns
must not come back refused. `no_answer` with a support handover is a real answer
and an acceptable outcome; `refused` is the regression.

Assertions on the refusal path are exact — `refuse` is a static template, not a
generated sentence, so there is nothing here for a model to rephrase between
runs. The controls stay loose for the opposite reason: they assert a route, not
prose, and one of them does not even assert corpus coverage.
"""

from typing import Any

import pytest

from evals.conftest import live, needs_db
from rag.nodes.refuse import MESSAGE

# `live` is the skipif (no key ⇒ skip); `pytest.mark.live` is the registered
# marker that keeps paid tests out of a default run. Deliberately *not*
# `needs_db` at module level — the out-of-scope cases must stay runnable with no
# index, which is half of what they prove. The three controls carry it
# individually.
pytestmark = [pytest.mark.live, live]


# Everything here is off the map entirely: general knowledge, chit-chat, code,
# and one near-miss — a competitor bank plus conventional interest, which is
# bank-shaped and finance-shaped and still nothing this assistant can speak to.
# That last one is the case worth watching in both directions: it is the closest
# an out-of-scope turn gets to the retrieve side of the boundary.
OUT_OF_SCOPE = [
    pytest.param("What is the capital of Australia?", id="capital-australia"),
    pytest.param("Who won last year's Formula 1 championship?", id="f1-champion"),
    pytest.param("Write me a Python function that reverses a string.", id="python-function"),
    pytest.param("What will the weather be in Dubai tomorrow?", id="dubai-weather"),
    pytest.param("Write a short poem about the desert.", id="desert-poem"),
    pytest.param(
        "What interest rate does Emirates NBD charge on a conventional personal loan?",
        id="other-bank-interest",
    ),
    pytest.param("What medicine should I take for a headache?", id="headache-medicine"),
    pytest.param('Translate "good morning" into French.', id="translate-french"),
]


@pytest.mark.parametrize("question", OUT_OF_SCOPE)
def test_out_of_scope_query_is_refused(question: str, invoke_turn) -> None:
    """One router call, the static message, nothing else on the path."""
    state: dict[str, Any] = invoke_turn(question)

    # The trace outcome, which is what an operator greps for. Distinct from
    # `no_answer`: this is "not my subject", not "not in the KB".
    assert state["outcome"] == "refused", state.get("answer")

    # Equality against the node's own constant rather than a substring: the
    # refusal is a fixed template, so a partial match would pass on a truncated
    # or model-rewritten variant of it.
    assert state["answer"] == MESSAGE

    # Nothing was retrieved. With `with_db=False` there is no open pool, so a
    # mis-route would have raised before this line — the empty list is the
    # belt-and-braces half.
    assert state["chunks"] == []

    # The cost assertion, and the sharpest one here: exactly one entry, from the
    # cheap tier. Anything longer means the turn paid for embedding, reranking,
    # grading or generation before deciding it had nothing to say.
    assert [entry["node"] for entry in state["usage_log"]] == ["router"]


# --- the over-refusal guard ----------------------------------------------
# These need the built index, because a false refusal is only visible against
# the real thing the router is refusing to search.


@needs_db
def test_core_product_question_is_not_refused(invoke_turn) -> None:
    """The most central question the corpus answers — a refusal here is fatal."""
    state: dict[str, Any] = invoke_turn(
        "What is Murabaha and how does the markup work?", with_db=True
    )

    assert state["outcome"] == "answered", state.get("answer")


@needs_db
def test_product_mechanics_question_is_not_refused(invoke_turn) -> None:
    """Narrower and more operational, but still squarely a Sukuk question."""
    state: dict[str, Any] = invoke_turn(
        "Can I sell my Sukuk investment before it matures?", with_db=True
    )

    assert state["outcome"] == "answered", state.get("answer")


@needs_db
def test_mal_banking_question_outside_the_kb_is_not_refused(invoke_turn) -> None:
    """The measured regression, pinned as narrowly as it is true.

    Fraud disputes are Mal's business and not one of the five KB products, so
    the route is policy (anything about the customer's account goes to
    `retrieve`) while corpus coverage is not. `no_answer` handing over to
    support is a real answer and passes here; only `refused` — telling a
    customer reporting card fraud that this is not the assistant's subject — is
    the failure. Do not tighten this to `== "answered"`: that would assert
    coverage the corpus does not claim, and it would fail on a document edit
    rather than on a routing regression.
    """
    state: dict[str, Any] = invoke_turn(
        "How do I dispute a fraudulent card transaction on my Mal account?", with_db=True
    )

    assert state["outcome"] != "refused", state.get("answer")
