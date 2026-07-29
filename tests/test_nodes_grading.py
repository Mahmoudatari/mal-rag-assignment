"""Unit tests for the grade and reformulate nodes.

`grade` makes one structured call over the retrieved chunks; `reformulate`
makes one structured call to rewrite the search query. Both run on the cheap
model, so these tests wire `fast_llm` to a real `LLMClient` backed by
`FakeAsyncOpenAI`, exactly as `tests/test_llm.py` and `tests/test_nodes_router.py`
build one — no network. Both nodes are `async def`, so each call is driven with
`asyncio.run`.
"""

from __future__ import annotations

import asyncio
import json

from core.llm import LLMClient
from rag.nodes import grade, reformulate
from rag.state import Chunk
from tests.fakes import FakeAsyncOpenAI, response


def fake_client(*responses, **kwargs) -> tuple[LLMClient, FakeAsyncOpenAI]:
    fake = FakeAsyncOpenAI(responses=list(responses))
    return LLMClient("google", "gemini-3.5-flash-lite", async_client=fake, **kwargs), fake


def chunk(
    chunk_id: str,
    doc: str,
    text: str,
    *,
    score: float = 0.87654,
    rerank_score: float | None = 0.99321,
) -> Chunk:
    c: Chunk = {"chunk_id": chunk_id, "doc": doc, "text": text, "score": score}
    if rerank_score is not None:
        c["rerank_score"] = rerank_score
    return c


def grade_reply(relevant: bool, note: str = "") -> str:
    return json.dumps({"relevant": relevant, "note": note})


def rewrite_reply(search_query: str) -> str:
    return json.dumps({"search_query": search_query})


# --- grade: verdicts -----------------------------------------------------


def test_relevant_verdict_returns_exact_key_set(monkeypatch) -> None:
    llm, fake = fake_client(response(grade_reply(True)))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    result = asyncio.run(
        grade.run(
            {
                "query": "what is the profit rate on Murabaha financing?",
                "chunks": [
                    chunk(
                        "murabaha#003",
                        "murabaha-everyday-finance",
                        "The profit rate is fixed at origination.",
                    )
                ],
            }
        )
    )

    assert result["relevant"] is True
    assert result["grader_note"] == ""
    assert set(result) == {"relevant", "grader_note", "usage_log"}


def test_not_relevant_verdict_carries_a_note(monkeypatch) -> None:
    llm, fake = fake_client(response(grade_reply(False, "missing the early-settlement fee")))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    result = asyncio.run(
        grade.run(
            {
                "query": "what is the early settlement fee for Ijara?",
                "chunks": [chunk("ijara#010", "ijara-auto-lease", "Ijara payments are monthly.")],
            }
        )
    )

    assert result["relevant"] is False
    assert result["grader_note"] == "missing the early-settlement fee"
    assert set(result) == {"relevant", "grader_note", "usage_log"}


# --- grade: empty chunks short-circuit -----------------------------------


def test_empty_chunks_short_circuits_with_no_llm_call(monkeypatch) -> None:
    llm, fake = fake_client()
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    result = asyncio.run(grade.run({"query": "what is Sukuk?", "chunks": []}))

    assert fake.calls == []
    assert result["relevant"] is False
    assert result["grader_note"] != ""
    assert set(result) == {"relevant", "grader_note"}


# --- grade: score must never reach the prompt ----------------------------


def test_scores_never_appear_in_the_grader_prompt(monkeypatch) -> None:
    """Relevance is the grader's call alone — a score in the prompt would let
    it re-derive a threshold decision from a number instead of reading the text."""
    llm, fake = fake_client(response(grade_reply(True)))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    asyncio.run(
        grade.run(
            {
                "query": "what is the minimum Sukuk investment?",
                "chunks": [
                    chunk(
                        "sukuk#002",
                        "fractional-sukuk-investing",
                        "The minimum investment is AED 500.",
                        score=0.87654,
                        rerank_score=0.99321,
                    )
                ],
            }
        )
    )

    call_repr = repr(fake.last_call)
    assert "0.87654" not in call_repr
    assert "0.99321" not in call_repr
    # The passage text and the question must still be there — this is not a
    # blanket "nothing gets through" check, only scores are excluded.
    assert "AED 500" in call_repr
    assert "minimum Sukuk investment" in call_repr


# --- grade: which question is graded -------------------------------------


def test_grader_reads_the_resolved_query_not_the_elliptical_turn(monkeypatch) -> None:
    """The bug this guards: grading an elliptical turn against the passages
    retrieved for its resolved form. "can I use it for a home?" names no
    subject, so the grader sees Murabaha passages against a question that could
    be about anything and reads a topic mismatch — dead-ending an answerable
    question at `no_answer`."""
    llm, fake = fake_client(response(grade_reply(True)))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    asyncio.run(
        grade.run(
            {
                "query": "can I use it for a home?",
                "resolved_query": "can Mal Everyday Murabaha be used for home financing",
                # The retry's rewrite must not be what gets graded either — that
                # would let the loop approve its own drift.
                "search_query": "Mal Ijara home finance real estate mortgage policy",
                "chunks": [
                    chunk(
                        "murabaha#014",
                        "murabaha-everyday-finance",
                        "Murabaha covers goods and travel, not property.",
                    )
                ],
            }
        )
    )

    call_repr = repr(fake.last_call)
    assert "can Mal Everyday Murabaha be used for home financing" in call_repr
    assert "can I use it for a home?" not in call_repr
    assert "Mal Ijara home finance" not in call_repr


def test_grader_falls_back_to_the_masked_query_without_a_resolved_one(monkeypatch) -> None:
    """States built by hand in evals — and any turn predating the key — carry
    `query` alone, which must still be what is graded."""
    llm, fake = fake_client(response(grade_reply(True)))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    asyncio.run(
        grade.run(
            {
                "query": "what is the minimum Sukuk investment?",
                "chunks": [
                    chunk("sukuk#002", "fractional-sukuk-investing", "The minimum is AED 500.")
                ],
            }
        )
    )

    assert "what is the minimum Sukuk investment?" in repr(fake.last_call)


# --- grade: usage ----------------------------------------------------------


def test_grade_logs_a_usage_entry_for_the_grade_node(monkeypatch) -> None:
    llm, fake = fake_client(
        response(grade_reply(True), prompt_tokens=55, completion_tokens=8)
    )
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    result = asyncio.run(
        grade.run(
            {
                "query": "what is Wakala?",
                "chunks": [chunk("wakala#001", "wakala-savings", "Wakala is an agency contract.")],
                "usage_log": [],
            }
        )
    )

    assert len(result["usage_log"]) == 1
    entry = result["usage_log"][0]
    assert entry["node"] == "grade"
    assert entry["prompt_tokens"] == 55
    assert entry["completion_tokens"] == 8
    assert entry["total_tokens"] == 63


def test_grade_usage_log_is_appended_not_replaced(monkeypatch) -> None:
    llm, fake = fake_client(response(grade_reply(True)))
    monkeypatch.setattr(grade, "fast_llm", lambda: llm)

    prior = [
        {
            "node": "router",
            "model": "n/a",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }
    ]
    result = asyncio.run(
        grade.run(
            {
                "query": "what is Takaful?",
                "chunks": [
                    chunk("takaful#001", "takaful-cover", "Takaful is Sharia-compliant insurance.")
                ],
                "usage_log": prior,
            }
        )
    )

    assert len(result["usage_log"]) == 2
    assert result["usage_log"][0] is prior[0]


# --- reformulate: attempts -------------------------------------------------


def test_attempts_increments_from_absent(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Murabaha profit rate structure")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "what is the profit rate?",
                "search_query": "profit rate",
                "grader_note": "missing the specific rate",
                "tried_queries": ["profit rate"],
            }
        )
    )

    assert result["attempts"] == 1


def test_attempts_increments_from_an_existing_value(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Ijara early settlement charge")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "what is the early settlement fee?",
                "search_query": "settlement fee",
                "grader_note": "missing the fee amount",
                "attempts": 1,
                "tried_queries": ["settlement fee"],
            }
        )
    )

    assert result["attempts"] == 2


# --- reformulate: tried_queries ---------------------------------------------


def test_new_query_is_appended_to_tried_queries(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Wakala agency fee structure")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "how much does Wakala cost?",
                "search_query": "Wakala cost",
                "grader_note": "missing the fee",
                "tried_queries": ["Wakala cost"],
            }
        )
    )

    assert result["tried_queries"] == ["Wakala cost", "Wakala agency fee structure"]
    assert result["search_query"] == "Wakala agency fee structure"


def test_prompt_carries_the_grader_note_and_tried_queries(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Takaful contribution schedule")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    asyncio.run(
        reformulate.run(
            {
                "query": "how much are Takaful payments?",
                "search_query": "Takaful payments",
                "grader_note": "missing the contribution amount",
                "tried_queries": ["Takaful payments", "Takaful cost"],
            }
        )
    )

    call_repr = repr(fake.last_call)
    assert "missing the contribution amount" in call_repr
    assert "Takaful payments" in call_repr
    assert "Takaful cost" in call_repr


# --- reformulate: which question it rewrites from ---------------------------


def test_rewrite_is_steered_by_the_resolved_query_not_the_elliptical_turn(monkeypatch) -> None:
    """"Always keep the subject of the customer's question in the query" is only
    obeyable if the question shown has one. Fed "can I use it for a home?", the
    rewrite has no product to keep and drifts onto whichever one the words
    suggest — live, that turned a Murabaha turn into an Ijara mortgage search by
    the second retry."""
    llm, fake = fake_client(response(rewrite_reply("Murabaha property purchase eligibility")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    asyncio.run(
        reformulate.run(
            {
                "query": "can I use it for a home?",
                "resolved_query": "can Mal Everyday Murabaha be used for home financing",
                "search_query": "Murabaha home purchase",
                "grader_note": "the passages cover goods and travel, not property",
                "tried_queries": ["Murabaha home purchase"],
            }
        )
    )

    call_repr = repr(fake.last_call)
    assert "can Mal Everyday Murabaha be used for home financing" in call_repr
    assert "can I use it for a home?" not in call_repr
    # The tried query still reaches the prompt — it fills a different field.
    assert "Murabaha home purchase" in call_repr


def test_rewrite_falls_back_to_the_masked_query_without_a_resolved_one(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Wakala agency fee structure")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    asyncio.run(
        reformulate.run(
            {
                "query": "how much does Wakala cost?",
                "search_query": "Wakala cost",
                "grader_note": "missing the fee",
                "tried_queries": ["Wakala cost"],
            }
        )
    )

    assert "how much does Wakala cost?" in repr(fake.last_call)


# --- reformulate: blank rewrite fallback ------------------------------------


def test_blank_rewrite_falls_back_to_the_masked_query(monkeypatch) -> None:
    """A blank rewrite must not reach `retrieve`'s `embed_query`, which raises
    on empty text — the masked customer query is the floor."""
    llm, fake = fake_client(response(rewrite_reply("   ")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "what is Murabaha?",
                "search_query": "Murabaha",
                "grader_note": "too vague",
                "tried_queries": ["Murabaha"],
            }
        )
    )

    assert result["search_query"] == "what is Murabaha?"
    assert result["tried_queries"] == ["Murabaha", "what is Murabaha?"]


# --- reformulate: colliding rewrite is accepted, not looped -----------------


def test_colliding_rewrite_is_accepted_with_exactly_one_model_call(monkeypatch) -> None:
    """A rewrite identical to one already tried is appended as-is — `attempts`
    already bounds the retry cycle, so this must not re-call the model."""
    llm, fake = fake_client(response(rewrite_reply("Ijara lease terms")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "what are Ijara lease terms?",
                "search_query": "Ijara lease terms",
                "grader_note": "still missing specifics",
                "tried_queries": ["Ijara lease terms"],
            }
        )
    )

    assert len(fake.calls) == 1
    assert result["search_query"] == "Ijara lease terms"
    assert result["tried_queries"] == ["Ijara lease terms", "Ijara lease terms"]


# --- reformulate: usage -----------------------------------------------------


def test_reformulate_logs_a_usage_entry_for_the_reformulate_node(monkeypatch) -> None:
    llm, fake = fake_client(
        response(rewrite_reply("Sukuk minimum ticket size"), prompt_tokens=30, completion_tokens=6)
    )
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    result = asyncio.run(
        reformulate.run(
            {
                "query": "what is the minimum Sukuk investment?",
                "search_query": "Sukuk minimum investment",
                "grader_note": "missing the minimum amount",
                "tried_queries": ["Sukuk minimum investment"],
                "usage_log": [],
            }
        )
    )

    assert len(result["usage_log"]) == 1
    entry = result["usage_log"][0]
    assert entry["node"] == "reformulate"
    assert entry["prompt_tokens"] == 30
    assert entry["completion_tokens"] == 6
    assert entry["total_tokens"] == 36


def test_reformulate_usage_log_is_appended_not_replaced(monkeypatch) -> None:
    llm, fake = fake_client(response(rewrite_reply("Wakala savings profit share")))
    monkeypatch.setattr(reformulate, "fast_llm", lambda: llm)

    prior = [
        {
            "node": "grade",
            "model": "n/a",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
        }
    ]
    result = asyncio.run(
        reformulate.run(
            {
                "query": "how does Wakala profit sharing work?",
                "search_query": "Wakala profit sharing",
                "grader_note": "missing the share percentage",
                "tried_queries": ["Wakala profit sharing"],
                "usage_log": prior,
            }
        )
    )

    assert len(result["usage_log"]) == 2
    assert result["usage_log"][0] is prior[0]
