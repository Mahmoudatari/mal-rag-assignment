"""Unit tests for the API layer.

The graph is stubbed at the `serving_graph` seam — `app/`'s job is everything
around the graph, and that is what is tested: session-id handling, the
thread_id config, the one-trace-line-per-request contract (including the
request that crashed), and `/health`'s three readiness checks. Graph behaviour
itself is tests/test_graph.py's problem; checkpointer persistence is exercised
there too.

`TestClient` runs the real lifespan, so every test here also proves the app
boots and shuts down with the pool seam stubbed out.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app import main
from core.config import get_settings
from kb.store import IndexStats

# A plausible final state for an answered turn. The trace assertions read the
# same keys `Trace.from_state` does, so this doubles as a fixture proving the
# app hands the *final* state to the tracer, not the input payload.
ANSWERED: dict[str, Any] = {
    "query": "what is murabaha financing?",
    "route": "retrieve",
    "route_reason": "product question",
    "chunks": [
        {
            "chunk_id": "murabaha-everyday-finance#001",
            "doc": "murabaha-everyday-finance",
            "text": "Murabaha is cost-plus financing.",
            "score": 0.91,
            "rerank_score": 0.74,
        }
    ],
    "relevant": True,
    "answer": "Murabaha is cost-plus financing [1].",
    "references": [
        {"n": 1, "doc": "murabaha-everyday-finance", "chunk_id": "murabaha-everyday-finance#001"}
    ],
    "outcome": "answered",
    "usage_log": [
        {
            "node": "router",
            "model": "google/gemini-3.5-flash-lite",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0001,
        }
    ],
}


class StubGraph:
    """Records every invocation; answers with a canned final state or raises."""

    def __init__(self) -> None:
        self.invocations: list[tuple[dict[str, Any], dict[str, Any]]] = []
        self.error: Exception | None = None

    async def ainvoke(self, payload: dict[str, Any], config: dict[str, Any] | None = None) -> dict[str, Any]:
        self.invocations.append((payload, config or {}))
        if self.error is not None:
            raise self.error
        return {**ANSWERED, "session_id": payload["session_id"]}


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch) -> StubGraph:
    stub = StubGraph()

    async def stubbed() -> StubGraph:
        return stub

    monkeypatch.setattr(main, "serving_graph", stubbed)
    return stub


@pytest.fixture
def client(graph: StubGraph) -> Iterator[TestClient]:
    with TestClient(main.app) as test_client:
        yield test_client


def trace_lines(capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


# Client-sent session ids must be in the server-minted format (uuid4().hex, 32
# lowercase hex chars) — the id is the only key to a conversation, so anything
# guessable is rejected at validation. Fixed literals, not uuid4() per test:
# deterministic ids keep assertion failures reproducible.
SID = "8f14e45fceea167a8f14e45fceea167a"
SID_TRACE = "aced0000000000000000000000000000"
SID_ERR = "beef0000000000000000000000000000"


# --- /chat ----------------------------------------------------------------


def test_chat_answers_and_echoes_the_session_id(client: TestClient, graph: StubGraph) -> None:
    reply = client.post("/chat", json={"message": "what is murabaha?", "session_id": SID})

    assert reply.status_code == 200
    body = reply.json()
    assert body["answer"] == ANSWERED["answer"]
    assert body["references"] == ANSWERED["references"]
    assert body["outcome"] == "answered"
    assert body["session_id"] == SID

    payload, config = graph.invocations[0]
    assert payload == {"raw_query": "what is murabaha?", "session_id": SID, "account_id": ""}
    assert config["configurable"]["thread_id"] == SID


def test_chat_mints_a_session_id_when_none_is_given(client: TestClient, graph: StubGraph) -> None:
    reply = client.post("/chat", json={"message": "hello"})

    minted = reply.json()["session_id"]
    assert len(minted) == 32  # uuid4().hex
    payload, config = graph.invocations[0]
    assert payload["session_id"] == minted
    assert config["configurable"]["thread_id"] == minted


def test_two_turns_with_one_session_share_one_thread(client: TestClient, graph: StubGraph) -> None:
    """The app's half of statefulness: same session id → same thread_id, so
    the checkpointer (tested in test_graph.py) sees one continuous thread."""
    client.post("/chat", json={"message": "what is murabaha?", "session_id": SID})
    client.post("/chat", json={"message": "is it halal?", "session_id": SID})

    configs = [config["configurable"]["thread_id"] for _, config in graph.invocations]
    assert configs == [SID, SID]


def test_chat_emits_one_trace_line_per_request(
    client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    client.post("/chat", json={"message": "what is murabaha?", "session_id": SID_TRACE})

    lines = trace_lines(capsys)
    assert len(lines) == 1
    trace = lines[0]
    assert trace["session_id"] == SID_TRACE
    assert trace["latency_ms"] >= 0
    # The four required fields, by name — and chunk_ids proves the app handed
    # the graph's *final* state to the tracer, not the request payload.
    assert trace["chunk_ids"] == ["murabaha-everyday-finance#001"]
    assert trace["total_tokens"] == 15
    assert trace["relevance_score"] == pytest.approx(0.74)


def test_a_graph_crash_returns_a_generic_500_and_still_emits_a_trace(
    client: TestClient, graph: StubGraph, capsys: pytest.CaptureFixture[str]
) -> None:
    """The crashed turn is the one that must be logged — and the HTTP body must
    stay generic, because a pre-redact exception can carry the raw message."""
    graph.error = RuntimeError("boom")

    reply = client.post("/chat", json={"message": "what is murabaha?", "session_id": SID_ERR})

    assert reply.status_code == 500
    assert reply.json() == {"detail": "internal error"}

    lines = trace_lines(capsys)
    assert len(lines) == 1
    assert lines[0]["session_id"] == SID_ERR
    assert "RuntimeError: boom" in lines[0]["error"]


def test_account_id_rides_the_invoke_payload(client: TestClient, graph: StubGraph) -> None:
    """`account_id` follows the `session_id` pattern: no node writes it, so the
    app puts it in the payload — and always sends the key ("" when absent), so
    the account node overwrites rather than inherits across turns."""
    client.post(
        "/chat",
        json={"message": "how much is left?", "account_id": "MAL-1001-2200-4417"},
    )

    payload, _ = graph.invocations[0]
    assert payload["account_id"] == "MAL-1001-2200-4417"


@pytest.mark.parametrize(
    "payload",
    [
        {"message": ""},
        {},
        {"message": "x", "session_id": ""},
        # Client-invented ids are the session-hijack vector: anyone posting the
        # same guessable string would share the thread. Only the minted format
        # passes.
        {"message": "x", "session_id": "abc123"},
        {"message": "x", "session_id": "s-1"},
        {"message": "x", "session_id": "8F14E45FCEEA167A8F14E45FCEEA167A"},
        {"message": "x", "account_id": "not-an-account"},
        {"message": "x", "account_id": "MAL-1-2-3"},
    ],
)
def test_invalid_requests_are_rejected_before_the_graph_runs(
    client: TestClient,
    graph: StubGraph,
    capsys: pytest.CaptureFixture[str],
    payload: dict[str, Any],
) -> None:
    reply = client.post("/chat", json=payload)

    assert reply.status_code == 422
    assert graph.invocations == []
    assert trace_lines(capsys) == []


# --- /health --------------------------------------------------------------


def set_stats(monkeypatch: pytest.MonkeyPatch, result: IndexStats | Exception) -> None:
    async def stubbed() -> IndexStats:
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(main, "astats", stubbed)


def test_health_is_ok_when_the_index_is_populated_and_current(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_stats(monkeypatch, IndexStats(5, 202, (get_settings().embedding_model,)))

    reply = client.get("/health")

    assert reply.status_code == 200
    assert reply.json() == {
        "status": "ok",
        "documents": 5,
        "chunks": 202,
        "foreign_models": [],
        "detail": "",
    }


def test_health_degrades_on_an_empty_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_stats(monkeypatch, IndexStats(0, 0, ()))

    reply = client.get("/health")

    assert reply.status_code == 503
    assert "ingest" in reply.json()["detail"]


def test_health_degrades_when_part_of_the_index_is_in_another_vector_space(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one failure that raises nothing at query time: chunks embedded by an
    unconfigured model retrieve as noise. /health is where it becomes visible."""
    set_stats(
        monkeypatch, IndexStats(5, 202, (get_settings().embedding_model, "openai/legacy"))
    )

    reply = client.get("/health")

    assert reply.status_code == 503
    assert reply.json()["foreign_models"] == ["openai/legacy"]


def test_health_degrades_when_the_database_is_unreachable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_stats(monkeypatch, RuntimeError("connection refused"))

    reply = client.get("/health")

    assert reply.status_code == 503
    assert reply.json()["detail"] == "database unavailable: RuntimeError"


# --- the OpenAPI document ---------------------------------------------------
# The spec at /openapi.json is the documented API contract (Swagger UI renders
# it at /docs). These assert the parts a consumer actually relies on, so a
# refactor that drops a description or an error response fails here rather
# than silently shipping an undocumented API.


def test_openapi_documents_both_endpoints_with_summaries(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert spec["info"]["title"] == "Mal Islamic Finance Assistant"
    assert spec["info"]["description"]
    chat = spec["paths"]["/chat"]["post"]
    health = spec["paths"]["/health"]["get"]
    assert chat["summary"] and chat["description"]
    assert health["summary"] and health["description"]


def test_openapi_documents_the_error_responses(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    assert set(spec["paths"]["/chat"]["post"]["responses"]) == {"200", "422", "500"}
    assert set(spec["paths"]["/health"]["get"]["responses"]) == {"200", "503"}


def test_openapi_documents_account_id_with_pattern_and_demo_ids(client: TestClient) -> None:
    spec = client.get("/openapi.json").json()

    request_schema = spec["components"]["schemas"]["ChatRequest"]
    account_id = request_schema["properties"]["account_id"]
    field = account_id.get("anyOf", [account_id])[0]  # Optional renders as anyOf
    assert field["pattern"] == r"^MAL-\d{4}-\d{4}-\d{4}$"
    assert "MAL-1001-2200-4417" in account_id["description"]
    assert request_schema["examples"]
