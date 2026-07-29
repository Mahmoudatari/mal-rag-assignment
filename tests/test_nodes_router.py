"""Unit tests for the router node.

The router makes one structured call and never answers, so these tests are
about the request it builds (history placement, the query it reads) and the
partial state it returns for each route — not about prose quality, which is
`generate`'s problem.

No network: `fast_llm` is monkeypatched to a real `LLMClient` wired to
`FakeAsyncOpenAI`, exactly as `tests/test_llm.py` builds one. The node is
`async def`, so each call is driven with `asyncio.run`.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from core.llm import EmptyCompletionError, LLMClient
from rag.nodes import router
from tests.fakes import FakeAsyncOpenAI, response


def fake_client(*responses, **kwargs) -> tuple[LLMClient, FakeAsyncOpenAI]:
    fake = FakeAsyncOpenAI(responses=list(responses))
    return LLMClient("google", "gemini-3.5-flash-lite", async_client=fake, **kwargs), fake


def decision(route: str, reason: str = "r", search_query: str = "") -> str:
    return json.dumps({"route": route, "reason": reason, "search_query": search_query})


# --- retrieve --------------------------------------------------------------


def test_retrieve_carries_the_resolved_query_and_seeds_tried_queries(monkeypatch) -> None:
    llm, fake = fake_client(
        response(decision("retrieve", "needs product facts", "is Murabaha financing permissible"))
    )
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "is it halal?", "history": []}))

    assert result["route"] == "retrieve"
    assert result["search_query"] == "is Murabaha financing permissible"
    assert result["tried_queries"] == ["is Murabaha financing permissible"]
    # Written once here and never by reformulate: `search_query` is what the next
    # retrieval embeds, `resolved_query` is the question grade and reformulate
    # keep judging against while the retry loop rewrites the search text.
    assert result["resolved_query"] == "is Murabaha financing permissible"
    assert set(result) == {
        "route", "route_reason", "search_query", "resolved_query", "tried_queries", "usage_log"
    }


def test_retrieve_falls_back_to_the_raw_query_when_the_rewrite_is_blank(monkeypatch) -> None:
    """A blank rewrite must not reach `retrieve`'s `embed_query`, which raises on empty text."""
    llm, fake = fake_client(response(decision("retrieve", "ok", "   ")))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "what about Ijara?", "history": []}))

    assert result["search_query"] == "what about Ijara?"
    assert result["tried_queries"] == ["what about Ijara?"]
    # The copy is taken after the fallback, so the two never disagree.
    assert result["resolved_query"] == "what about Ijara?"


# --- answer / refuse ---------------------------------------------------


@pytest.mark.parametrize("route", ["answer", "refuse"])
def test_non_retrieve_routes_blank_the_search_query(monkeypatch, route: str) -> None:
    llm, fake = fake_client(response(decision(route, "no product facts needed")))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "thanks!", "history": []}))

    assert result["route"] == route
    assert result["search_query"] == ""
    assert result["tried_queries"] == []
    # `grade` is unreachable on both routes, so there is no question to resolve —
    # blanked exactly like search_query and tried_queries.
    assert result["resolved_query"] == ""
    assert set(result) == {
        "route", "route_reason", "search_query", "resolved_query", "tried_queries", "usage_log"
    }


# --- history --------------------------------------------------------------


def test_history_is_passed_between_system_prompt_and_current_turn(monkeypatch) -> None:
    llm, fake = fake_client(response(decision("retrieve", "ok", "is Murabaha halal")))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    history = [
        {"role": "user", "content": "what is Murabaha?"},
        {"role": "assistant", "content": "A cost-plus sale."},
    ]
    asyncio.run(router.run({"query": "is it halal?", "history": history}))

    messages = fake.last_call["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "what is Murabaha?"
    assert messages[2]["content"] == "A cost-plus sale."
    assert messages[-1]["content"] == "is it halal?"


# --- blank query short-circuit ------------------------------------------


@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_short_circuits_with_no_llm_call(monkeypatch, query: str) -> None:
    """A turn that was entirely PII masks down to nothing — no call, no usage entry."""
    llm, fake = fake_client()
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": query, "history": []}))

    assert fake.calls == []
    assert result["route"] == "answer"
    assert result["search_query"] == ""
    assert result["tried_queries"] == []
    assert result["resolved_query"] == ""
    assert set(result) == {"route", "route_reason", "search_query", "resolved_query", "tried_queries"}


# --- fail-open --------------------------------------------------------------


def test_structured_output_error_fails_open_to_retrieve(monkeypatch) -> None:
    """The router is on every request; failing open beats killing the turn.

    `structured_retries` defaults to 1, i.e. two attempts total, so two
    non-JSON replies are queued to exhaust the client's internal retry before
    `StructuredOutputError` reaches the node.
    """
    llm, fake = fake_client(response("not json"), response("still not json"))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "is it halal?", "history": []}))

    assert len(fake.calls) == 2
    assert result["route"] == "retrieve"
    assert result["search_query"] == "is it halal?"
    assert result["tried_queries"] == ["is it halal?"]
    # No rewrite happened, so the masked turn is the only standalone question
    # available — unresolved, but better than leaving grade and reformulate blank.
    assert result["resolved_query"] == "is it halal?"
    assert set(result) == {"route", "route_reason", "search_query", "resolved_query", "tried_queries"}


def test_a_200_with_error_reply_fails_open_to_retrieve(monkeypatch) -> None:
    """OpenRouter's other failure shape: HTTP 200, an `error` object, no choices.

    The SDK does not raise on it and the reply parses with `choices` as `None`,
    which used to reach the node as `EmptyCompletionError` — an `LLMError` that
    was neither retried by the client nor caught here, so the documented
    fail-open was skipped and a valid customer question got a 500. Queued once
    and replayed for both attempts, since the fake serves its last response
    after the queue drains.
    """
    llm, fake = fake_client(
        response(None, choices=False, error={"message": "upstream rate-limited", "code": 429})
    )
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "is it halal?", "history": []}))

    assert len(fake.calls) == 2
    assert result["route"] == "retrieve"
    assert result["search_query"] == "is it halal?"
    assert result["tried_queries"] == ["is it halal?"]
    assert result["resolved_query"] == "is it halal?"
    assert set(result) == {"route", "route_reason", "search_query", "resolved_query", "tried_queries"}


def test_the_fail_open_covers_every_llm_layer_failure_not_only_the_schema_one(monkeypatch) -> None:
    """Pins the handler being `LLMError` rather than `StructuredOutputError`.

    The test above cannot do it: the client now converts an exhausted
    200-with-error into `StructuredOutputError`, so it passes under the narrow
    handler too. This one raises the flavour that used to escape both layers
    directly, which is the only way the widening is visible from here.
    """

    class Failing:
        async def astructured(self, *args, **kwargs):
            raise EmptyCompletionError("response contained no choices — provider said: down")

    monkeypatch.setattr(router, "fast_llm", Failing)

    result = asyncio.run(router.run({"query": "is it halal?", "history": []}))

    assert result["route"] == "retrieve"
    assert result["search_query"] == "is it halal?"


# --- usage -------------------------------------------------------------


def test_usage_entry_is_logged_for_the_router_node(monkeypatch) -> None:
    llm, fake = fake_client(
        response(
            decision("retrieve", "ok", "sukuk minimum investment"),
            prompt_tokens=40,
            completion_tokens=12,
        )
    )
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    result = asyncio.run(router.run({"query": "sukuk?", "history": [], "usage_log": []}))

    assert len(result["usage_log"]) == 1
    entry = result["usage_log"][0]
    assert entry["node"] == "router"
    assert entry["prompt_tokens"] == 40
    assert entry["completion_tokens"] == 12
    assert entry["total_tokens"] == 52


def test_usage_log_is_appended_not_replaced(monkeypatch) -> None:
    llm, fake = fake_client(response(decision("answer", "greeting")))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    prior = [{"node": "redact", "model": "n/a", "prompt_tokens": 0, "completion_tokens": 0,
              "total_tokens": 0, "cost": 0.0}]
    result = asyncio.run(router.run({"query": "hi", "history": [], "usage_log": prior}))

    assert len(result["usage_log"]) == 2
    assert result["usage_log"][0] is prior[0]


# --- PII boundary --------------------------------------------------------


def test_raw_query_never_reaches_the_llm_call(monkeypatch) -> None:
    """The node must read `query` (masked) only — `raw_query` must never leak into a call."""
    llm, fake = fake_client(response(decision("retrieve", "ok", "resolved query")))
    monkeypatch.setattr(router, "fast_llm", lambda: llm)

    asyncio.run(
        router.run(
            {
                "raw_query": "my Emirates ID is 784-1990-1234567-1, my name is Fatima Al Suwaidi",
                "query": "my Emirates ID is [EMIRATES_ID], my name is [PERSON]",
                "history": [],
            }
        )
    )

    assert "784-1990-1234567-1" not in repr(fake.calls)
    assert "Fatima Al Suwaidi" not in repr(fake.calls)
