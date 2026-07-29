"""Unit tests for the three terminal answering nodes: generate, refuse, no_answer.

`generate` is the one node here that calls an LLM, so most of its tests are
about the request it builds (grounded vs. ungrounded system prompt, numbered
passages, history) and about the citation post-processing, which is pure string
handling and the main way this node could quietly hallucinate a reference.
`refuse` and `no_answer` are static templates — their tests are about the fixed
contract (exact key set, determinism, history) rather than prose.

No network: `answer_llm` is monkeypatched to a real `LLMClient` wired to
`FakeAsyncOpenAI`, exactly as `tests/test_llm.py` and `tests/test_nodes_router.py`
build one. `generate` is `async def`, so each call is driven with `asyncio.run`;
`refuse` and `no_answer` stay synchronous.
"""

from __future__ import annotations

import asyncio

from core.config import get_settings
from core.llm import LLMClient
from rag.nodes import generate, no_answer, refuse
from tests.fakes import FakeAsyncOpenAI, response


def fake_client(*responses, **kwargs) -> tuple[LLMClient, FakeAsyncOpenAI]:
    fake = FakeAsyncOpenAI(responses=list(responses))
    return LLMClient("google", "flash", async_client=fake, **kwargs), fake


def chunk(n: int, doc: str = "murabaha-everyday-finance") -> dict:
    return {"chunk_id": f"{doc}#{n:03d}", "doc": doc, "text": f"passage {n} text", "score": 0.5}


# --- citations ------------------------------------------------------------


def test_references_built_only_from_markers_actually_cited(monkeypatch) -> None:
    llm, fake = fake_client(response("A sale, not a loan [1][3]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    chunks = [chunk(1), chunk(2), chunk(3)]
    result = asyncio.run(generate.run({"query": "what is Murabaha?", "chunks": chunks, "history": []}))

    assert result["answer"] == "A sale, not a loan [1][3]."
    assert result["references"] == [
        {"n": 1, "doc": chunks[0]["doc"], "chunk_id": chunks[0]["chunk_id"]},
        {"n": 3, "doc": chunks[2]["doc"], "chunk_id": chunks[2]["chunk_id"]},
    ]


def test_out_of_range_marker_is_stripped_from_text_and_absent_from_references(monkeypatch) -> None:
    llm, fake = fake_client(response("Something not in the passages [7]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    chunks = [chunk(1), chunk(2), chunk(3), chunk(4)]
    result = asyncio.run(generate.run({"query": "q", "chunks": chunks, "history": []}))

    assert "[7]" not in result["answer"]
    # The space that preceded the marker goes with it — a stripped citation
    # should leave prose, not "the passages ." with a gap before the full stop.
    assert result["answer"] == "Something not in the passages."
    assert result["references"] == []


def test_repeated_marker_is_referenced_once(monkeypatch) -> None:
    llm, fake = fake_client(response("It is a sale [1], not a loan, definitely a sale [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    chunks = [chunk(1), chunk(2)]
    result = asyncio.run(generate.run({"query": "q", "chunks": chunks, "history": []}))

    assert result["references"] == [{"n": 1, "doc": chunks[0]["doc"], "chunk_id": chunks[0]["chunk_id"]}]


def test_empty_chunks_produce_no_references_even_with_a_stray_marker(monkeypatch) -> None:
    """Unconditional: the no-retrieval path never carries citations, no matter what the model writes."""
    llm, fake = fake_client(response("Hi! I can help with [1] Islamic finance questions."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    result = asyncio.run(generate.run({"query": "hi", "chunks": [], "history": []}))

    assert result["references"] == []
    # With no chunks every marker is out of range, so the same strip that
    # catches a hallucinated citation catches this one — a [1] left in an
    # ungrounded reply points at nothing and reads as if it were grounded.
    assert "[1]" not in result["answer"]
    assert result["answer"] == "Hi! I can help with Islamic finance questions."


def test_blank_query_never_posts_an_empty_user_message(monkeypatch) -> None:
    """A turn masked down to nothing routes to "answer" and lands here blank.

    `complete` guards its own blank *reply*, not a blank prompt, so an empty
    user message would go out to the provider as-is.
    """
    llm, fake = fake_client(response("Hello! How can I help?"))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    asyncio.run(generate.run({"query": "   ", "chunks": [], "history": []}))

    assert fake.last_call["messages"][-1]["content"].strip()


# --- prompt construction ----------------------------------------------------


def test_grounded_prompt_numbers_the_passages_and_carries_the_question(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    chunks = [chunk(1, doc="ijara-auto-lease"), chunk(2, doc="sukuk-fractional")]
    asyncio.run(generate.run({"query": "is Ijara halal?", "chunks": chunks, "history": []}))

    user_message = fake.last_call["messages"][-1]["content"]
    assert "[1] (ijara-auto-lease)" in user_message
    assert "[2] (sukuk-fractional)" in user_message
    assert "passage 1 text" in user_message
    assert "is Ijara halal?" in user_message


def test_empty_chunks_selects_the_ungrounded_system_prompt(monkeypatch) -> None:
    llm, fake = fake_client(response("Hi, how can I help?"))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    asyncio.run(generate.run({"query": "hello", "chunks": [], "history": []}))

    system_message = fake.last_call["messages"][0]["content"]
    assert system_message == generate.UNGROUNDED_SYSTEM
    # Distinguishing phrase: forbids the exact thing the grounded prompt requires.
    assert "Do NOT state any substantive finance claim" in system_message


def test_chunks_present_selects_the_grounded_system_prompt(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    asyncio.run(generate.run({"query": "what is Murabaha?", "chunks": [chunk(1)], "history": []}))

    assert fake.last_call["messages"][0]["content"] == generate.GROUNDED_SYSTEM


def test_grounded_prompt_places_the_account_block_between_passages_and_question(monkeypatch) -> None:
    llm, fake = fake_client(response("Seven instalments remain."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    account = {
        "masked_id": "MAL-****-****-4417",
        "holdings": [{"product": "Murabaha everyday finance", "instalments_paid": 5}],
    }
    asyncio.run(
        generate.run(
            {"query": "how much is left?", "chunks": [chunk(1)], "history": [], "account": account}
        )
    )

    user_message = fake.last_call["messages"][-1]["content"]
    assert "Customer account context" in user_message
    assert "MAL-****-****-4417" in user_message
    assert "instalments_paid: 5" in user_message
    # Order: passages, account, question — the question stays last.
    assert (
        user_message.index("passage 1 text")
        < user_message.index("MAL-****-****-4417")
        < user_message.index("how much is left?")
    )


def test_ungrounded_prompt_carries_the_account_block_too(monkeypatch) -> None:
    llm, fake = fake_client(response("Hello! You hold a Murabaha contract with us."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    account = {
        "masked_id": "MAL-****-****-4417",
        "holdings": [{"product": "Murabaha everyday finance"}],
    }
    asyncio.run(generate.run({"query": "hi", "chunks": [], "history": [], "account": account}))

    assert fake.last_call["messages"][0]["content"] == generate.UNGROUNDED_SYSTEM
    user_message = fake.last_call["messages"][-1]["content"]
    assert "Customer account context" in user_message
    assert user_message.rstrip().endswith("hi")


def test_no_account_means_no_account_block(monkeypatch) -> None:
    """None and absent alike — the block must not render an empty shell that
    invites the model to fill it in."""
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    asyncio.run(
        generate.run({"query": "q", "chunks": [chunk(1)], "history": [], "account": None})
    )

    assert "Customer account context" not in fake.last_call["messages"][-1]["content"]


def test_history_is_passed_to_the_call(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    history = [
        {"role": "user", "content": "what is Murabaha?"},
        {"role": "assistant", "content": "A cost-plus sale."},
    ]
    asyncio.run(generate.run({"query": "is it halal?", "chunks": [chunk(1)], "history": history}))

    roles = [m["role"] for m in fake.last_call["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert fake.last_call["messages"][1]["content"] == "what is Murabaha?"


# --- history bookkeeping -----------------------------------------------------


def test_history_is_appended_and_capped_at_history_max_messages(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    cap = get_settings().history_max_messages
    seeded = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"turn {i}"}
        for i in range(cap + 6)
    ]
    result = asyncio.run(
        generate.run({"query": "new question", "chunks": [chunk(1)], "history": seeded})
    )

    assert len(result["history"]) == cap
    assert result["history"][-2] == {"role": "user", "content": "new question"}
    assert result["history"][-1]["role"] == "assistant"
    # Same text as the reply minus its citation markers — see the test below.
    assert result["history"][-1]["content"] == "An answer."


def test_history_stores_the_answer_without_its_citation_markers(monkeypatch) -> None:
    """The reply keeps its markers, the history copy does not.

    A marker numbers *this* turn's passages. Replayed into the next turn, whose
    passages are renumbered with different meanings, a restated fact carries its
    stale number along and `_citations` resolves it against the new chunk list —
    a reference to an unrelated chunk that raises nothing and traces as genuine.
    """
    llm, fake = fake_client(response("Murabaha is a sale [1]. The fee is fixed [2]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    chunks = [chunk(1), chunk(2)]
    result = asyncio.run(generate.run({"query": "what is Murabaha?", "chunks": chunks, "history": []}))

    assert result["answer"] == "Murabaha is a sale [1]. The fee is fixed [2]."
    stored = result["history"][-1]["content"]
    assert "[1]" not in stored and "[2]" not in stored
    # The strip takes each marker's preceding space with it, so what is replayed
    # is prose — not "a sale ." with a gap, and no doubled spaces mid-sentence.
    assert stored == "Murabaha is a sale. The fee is fixed."


def test_history_carries_the_masked_query_not_a_stray_raw_value(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    result = asyncio.run(
        generate.run(
            {
                "raw_query": "my account is 12345",
                "query": "my account is [ACCOUNT_NUMBER]",
                "chunks": [chunk(1)],
                "history": [],
            }
        )
    )

    assert result["history"][-2] == {"role": "user", "content": "my account is [ACCOUNT_NUMBER]"}


# --- outcome and usage --------------------------------------------------


def test_outcome_is_answered(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    result = asyncio.run(generate.run({"query": "q", "chunks": [chunk(1)], "history": []}))

    assert result["outcome"] == "answered"


def test_usage_entry_uses_the_answer_model(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1].", prompt_tokens=50, completion_tokens=20))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    result = asyncio.run(
        generate.run({"query": "q", "chunks": [chunk(1)], "history": [], "usage_log": []})
    )

    assert len(result["usage_log"]) == 1
    entry = result["usage_log"][0]
    assert entry["node"] == "generate"
    assert entry["model"] == "google/flash"
    assert entry["prompt_tokens"] == 50
    assert entry["completion_tokens"] == 20


def test_usage_log_is_appended_not_replaced(monkeypatch) -> None:
    llm, fake = fake_client(response("An answer [1]."))
    monkeypatch.setattr(generate, "answer_llm", lambda: llm)

    prior = [{"node": "grade", "model": "n/a", "prompt_tokens": 0, "completion_tokens": 0,
              "total_tokens": 0, "cost": 0.0}]
    result = asyncio.run(
        generate.run({"query": "q", "chunks": [chunk(1)], "history": [], "usage_log": prior})
    )

    assert len(result["usage_log"]) == 2
    assert result["usage_log"][0] is prior[0]


# --- refuse -------------------------------------------------------------


def test_refuse_returns_exactly_the_contracted_keys() -> None:
    result = refuse.run({"query": "what's the weather?", "history": []})
    assert set(result) == {"answer", "outcome", "history"}
    assert result["outcome"] == "refused"


def test_refuse_names_products_the_customer_can_ask_about() -> None:
    result = refuse.run({"query": "tell me a joke", "history": []})
    assert "Murabaha" in result["answer"]
    assert "?" not in result["answer"]  # statement only, no trailing question


def test_refuse_is_deterministic_across_calls() -> None:
    first = refuse.run({"query": "what's the weather?", "history": []})
    second = refuse.run({"query": "what's the weather?", "history": []})
    assert first["answer"] == second["answer"]


def test_refuse_appends_history() -> None:
    result = refuse.run({"query": "off topic", "history": [{"role": "user", "content": "hi"}]})
    assert result["history"][-2] == {"role": "user", "content": "off topic"}
    assert result["history"][-1] == {"role": "assistant", "content": result["answer"]}


def test_refuse_works_on_a_near_empty_state() -> None:
    result = refuse.run({})
    assert result["outcome"] == "refused"
    assert result["history"] == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": result["answer"]},
    ]


# --- no_answer ------------------------------------------------------------


def test_no_answer_returns_exactly_the_contracted_keys() -> None:
    result = no_answer.run({"query": "what's the fee for X?", "history": []})
    assert set(result) == {"answer", "outcome", "history"}
    assert result["outcome"] == "no_answer"


def test_no_answer_points_at_obviously_fake_support_contact() -> None:
    result = no_answer.run({"query": "what's the fee for X?", "history": []})
    assert "support@mal.example" in result["answer"]


def test_no_answer_is_deterministic_across_calls() -> None:
    first = no_answer.run({"query": "what's the fee for X?", "history": []})
    second = no_answer.run({"query": "what's the fee for X?", "history": []})
    assert first["answer"] == second["answer"]


def test_no_answer_appends_history() -> None:
    result = no_answer.run(
        {"query": "obscure question", "history": [{"role": "user", "content": "hi"}]}
    )
    assert result["history"][-2] == {"role": "user", "content": "obscure question"}
    assert result["history"][-1] == {"role": "assistant", "content": result["answer"]}


def test_no_answer_works_on_a_near_empty_state() -> None:
    result = no_answer.run({})
    assert result["outcome"] == "no_answer"
    assert result["history"] == [
        {"role": "user", "content": ""},
        {"role": "assistant", "content": result["answer"]},
    ]
