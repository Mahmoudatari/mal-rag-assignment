"""Unit tests for the OpenRouter client.

Grouped by what they defend. The client is one indirection over an HTTP call,
so most of these assert the *request* — a bad request here is a 400 on every
turn in production, and the response half is the SDK's problem, not ours.

No network: the transport is injected. The one test that does hit OpenRouter is
marked `live` and deselected unless asked for.
"""

from __future__ import annotations

import asyncio
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from core.config import get_settings
from core.llm import (
    Completion,
    EmptyCompletionError,
    LLMClient,
    LLMError,
    Model,
    Structured,
    StructuredOutputError,
    Usage,
    _strict_schema,
)
from tests.fakes import FakeAsyncOpenAI, FakeOpenAI, response


class Route(BaseModel):
    """Stands in for the router's real output schema."""

    route: Literal["retrieve", "refuse", "answer"]
    reason: str
    search_query: str = ""


class Nested(BaseModel):
    inner: Route
    tags: list[str] = Field(default_factory=list)


def client(*responses, **kwargs) -> tuple[LLMClient, FakeOpenAI]:
    fake = FakeOpenAI(responses=list(responses))
    return LLMClient("google", "gemini-3.5-flash-lite", client=fake, **kwargs), fake


def async_client(*responses, **kwargs) -> tuple[LLMClient, FakeAsyncOpenAI]:
    """`client()`'s twin: same model, same queue, the async transport instead.

    The `a*` tests are ordinary sync functions driving `asyncio.run(...)` — this
    repo has no `pytest-asyncio`, and adding a plugin to await four coroutines
    would buy nothing the one-liner does not.
    """
    fake = FakeAsyncOpenAI(responses=list(responses))
    return LLMClient("google", "gemini-3.5-flash-lite", async_client=fake, **kwargs), fake


# --- the model id is a (provider, name) pair -----------------------------
# The whole point of the split: swapping providers is two strings, not a code
# change. These cases are the ones a config typo would land on.


@pytest.mark.parametrize(
    ("provider", "name"),
    [
        ("google", "gemini-3.6-flash"),
        ("google", "gemini-3.5-flash-lite"),
        ("openai", "gpt-4o-mini"),
        ("anthropic", "claude-haiku-4.5"),
        ("mistralai", "mistral-small"),
        # Names may carry further slashes and suffixes; only the first slash splits.
        ("meta-llama", "llama-3.1-8b-instruct:free"),
    ],
)
def test_any_openrouter_model_round_trips(provider: str, name: str) -> None:
    slug = f"{provider}/{name}"
    assert Model.parse(slug) == Model(provider, name)
    assert str(Model.parse(slug)) == slug


def test_client_sends_the_model_it_was_built_with() -> None:
    llm, fake = client()
    llm.complete("hi")
    assert fake.last_call["model"] == "google/gemini-3.5-flash-lite"


def test_from_slug_matches_the_two_argument_form() -> None:
    fake = FakeOpenAI()
    assert LLMClient.from_slug("openai/gpt-4o-mini", client=fake).model == Model(
        "openai", "gpt-4o-mini"
    )


@pytest.mark.parametrize("slug", ["gpt-4o-mini", "", "  ", "gemini-3.6-flash"])
def test_unqualified_model_id_is_rejected(slug: str) -> None:
    """OpenRouter has no bare model ids — fail here, not with a 404 per request."""
    with pytest.raises(ValueError, match="provider/name"):
        Model.parse(slug)


@pytest.mark.parametrize(("provider", "name"), [("", "x"), ("x", ""), ("a/b", "c")])
def test_malformed_pairs_are_rejected(provider: str, name: str) -> None:
    with pytest.raises(ValueError):
        Model(provider, name)


def test_with_model_switches_model_and_keeps_the_transport() -> None:
    llm, fake = client()
    other = llm.with_model("google/gemini-3.6-flash")

    other.complete("hi")
    assert fake.last_call["model"] == "google/gemini-3.6-flash"
    assert llm.model.name == "gemini-3.5-flash-lite"


def test_with_model_keeps_the_async_transport_too() -> None:
    """Dropping the async half here would rebuild a real client on the next await."""
    sync_fake, async_fake = FakeOpenAI(), FakeAsyncOpenAI()
    llm = LLMClient(
        "google", "gemini-3.5-flash-lite", client=sync_fake, async_client=async_fake
    )
    other = llm.with_model("google/gemini-3.6-flash")

    asyncio.run(other.acomplete("hi"))
    assert async_fake.last_call["model"] == "google/gemini-3.6-flash"
    assert sync_fake.calls == []


def test_missing_api_key_fails_at_construction() -> None:
    """A missing key should surface at startup, not as a 401 mid-conversation."""
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        LLMClient("google", "gemini-3.6-flash", api_key="")


# --- message assembly ----------------------------------------------------


def test_system_prompt_leads_and_user_prompt_trails() -> None:
    llm, fake = client()
    llm.complete("what is Ijara?", system="You route questions.")
    assert fake.last_call["messages"] == [
        {"role": "system", "content": "You route questions."},
        {"role": "user", "content": "what is Ijara?"},
    ]


def test_history_sits_between_system_and_current_turn() -> None:
    """Order is load-bearing: the router resolves 'is it halal?' against history."""
    llm, fake = client()
    llm.complete(
        "is it halal?",
        system="S",
        history=[
            {"role": "user", "content": "what is Murabaha?"},
            {"role": "assistant", "content": "A cost-plus sale."},
        ],
    )
    assert [m["role"] for m in fake.last_call["messages"]] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert fake.last_call["messages"][-1]["content"] == "is it halal?"


def test_no_system_prompt_means_no_system_message() -> None:
    llm, fake = client()
    llm.complete("hi")
    assert [m["role"] for m in fake.last_call["messages"]] == ["user"]


def test_acomplete_assembles_the_same_request_as_complete() -> None:
    """The async path must build the request through the same `_create_kwargs`.

    Asserted off `last_call` in the same terms as the sync tests above, because
    a forked request shape is exactly the drift the shared helper exists to
    prevent — and it would only ever show up in production, where the request
    path is the async one.
    """
    llm, fake = async_client()
    asyncio.run(
        llm.acomplete(
            "is it halal?",
            system="S",
            history=[
                {"role": "user", "content": "what is Murabaha?"},
                {"role": "assistant", "content": "A cost-plus sale."},
            ],
        )
    )
    assert fake.last_call["messages"] == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "what is Murabaha?"},
        {"role": "assistant", "content": "A cost-plus sale."},
        {"role": "user", "content": "is it halal?"},
    ]
    assert fake.last_call["model"] == "google/gemini-3.5-flash-lite"
    assert fake.last_call["temperature"] == 0.0
    assert fake.last_call["extra_body"] == {"usage": {"include": True}}


# --- completions ---------------------------------------------------------


def test_complete_returns_the_text() -> None:
    llm, _ = client(response("Ijara is a lease."))
    assert llm.complete("q") == Completion(
        text="Ijara is a lease.", usage=Usage(), model="google/gemini-3.5-flash-lite"
    )


def test_acomplete_returns_the_text_usage_and_model() -> None:
    """The async twin returns the same `Completion`, cost extension included."""
    llm, _ = async_client(
        response("Ijara is a lease.", prompt_tokens=120, completion_tokens=30, cost=0.00042)
    )
    assert asyncio.run(llm.acomplete("q")) == Completion(
        text="Ijara is a lease.",
        usage=Usage(120, 30, 150, 0.00042),
        model="google/gemini-3.5-flash-lite",
    )


@pytest.mark.parametrize("content", [None, "", "   \n "])
def test_empty_completion_raises(content: str | None) -> None:
    """Silence is a failure, not an answer — a blank reply must not become one."""
    llm, _ = client(response(content))
    with pytest.raises(EmptyCompletionError):
        llm.complete("q")


@pytest.mark.parametrize("content", [None, "", "   \n "])
def test_acomplete_empty_completion_raises(content: str | None) -> None:
    llm, _ = async_client(response(content))
    with pytest.raises(EmptyCompletionError):
        asyncio.run(llm.acomplete("q"))


def test_temperature_defaults_to_zero_and_is_overridable_per_call() -> None:
    llm, fake = client()
    llm.complete("q")
    assert fake.last_call["temperature"] == 0.0

    llm.complete("q", temperature=0.7)
    assert fake.last_call["temperature"] == 0.7


def test_max_tokens_is_omitted_unless_set() -> None:
    llm, fake = client()
    llm.complete("q")
    assert "max_tokens" not in fake.last_call

    llm.complete("q", max_tokens=256)
    assert fake.last_call["max_tokens"] == 256


def test_cost_is_requested_on_every_call() -> None:
    """Cost is a trace field; OpenRouter only returns it if asked."""
    llm, fake = client()
    llm.complete("q")
    assert fake.last_call["extra_body"] == {"usage": {"include": True}}


# --- usage, which the trace requires -------------------------------------


def test_usage_is_read_off_the_response() -> None:
    llm, _ = client(response("ok", prompt_tokens=120, completion_tokens=30, cost=0.00042))
    assert llm.complete("q").usage == Usage(120, 30, 150, 0.00042)


def test_usage_survives_a_provider_that_omits_it() -> None:
    """Missing usage must not take the request down with it."""
    llm, _ = client(response("ok", usage=False))
    assert llm.complete("q").usage == Usage()


def test_usage_without_cost_reports_zero_cost() -> None:
    """`cost` is an OpenRouter extension, absent on a plain OpenAI-compatible host."""
    llm, _ = client(response("ok", prompt_tokens=10, completion_tokens=5))
    assert llm.complete("q").usage == Usage(10, 5, 15, 0.0)


def test_usage_adds_so_a_trace_can_total_a_request() -> None:
    total = Usage(10, 5, 15, 0.001) + Usage(20, 10, 30, 0.002)
    assert total == Usage(30, 15, 45, pytest.approx(0.003))


# --- structured output ---------------------------------------------------


def test_structured_returns_the_validated_model() -> None:
    llm, _ = client(
        response('{"route": "retrieve", "reason": "finance", "search_query": "ijara"}')
    )
    result = llm.structured("q", Route)

    assert isinstance(result, Structured)
    assert result.data == Route(route="retrieve", reason="finance", search_query="ijara")


def test_structured_asks_for_strict_json_schema() -> None:
    llm, fake = client(response('{"route": "refuse", "reason": "r", "search_query": ""}'))
    llm.structured("q", Route)

    fmt = fake.last_call["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Route"
    assert fmt["json_schema"]["strict"] is True


def test_astructured_returns_the_validated_model() -> None:
    llm, _ = async_client(
        response('{"route": "retrieve", "reason": "finance", "search_query": "ijara"}')
    )
    result = asyncio.run(llm.astructured("q", Route))

    assert isinstance(result, Structured)
    assert result.data == Route(route="retrieve", reason="finance", search_query="ijara")


def test_astructured_asks_for_strict_json_schema() -> None:
    """Strict mode is what the router depends on, and it is per-request — the
    async path has to send it too."""
    llm, fake = async_client(
        response('{"route": "refuse", "reason": "r", "search_query": ""}')
    )
    asyncio.run(llm.astructured("q", Route))

    fmt = fake.last_call["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["name"] == "Route"
    assert fmt["json_schema"]["strict"] is True


def test_completions_do_not_request_a_schema() -> None:
    llm, fake = client()
    llm.complete("q")
    assert "response_format" not in fake.last_call


def test_fenced_json_is_still_parsed() -> None:
    """Some models fence their JSON regardless of strict mode."""
    llm, _ = client(
        response('```json\n{"route": "answer", "reason": "greeting"}\n```')
    )
    assert llm.structured("q", Route).data.route == "answer"


def test_structured_retries_a_malformed_reply_then_succeeds() -> None:
    llm, fake = client(
        response("sorry, I cannot do that", prompt_tokens=10, completion_tokens=4),
        response('{"route": "retrieve", "reason": "ok"}', prompt_tokens=10, completion_tokens=6),
    )
    result = llm.structured("q", Route)

    assert result.data.route == "retrieve"
    assert len(fake.calls) == 2
    # Failed attempts still cost money, so the trace must see those tokens too.
    assert result.usage == Usage(20, 10, 30, 0.0)


def test_astructured_retries_a_malformed_reply_then_succeeds() -> None:
    """The retry loop is written out twice, so its usage accounting is pinned twice."""
    llm, fake = async_client(
        response("sorry, I cannot do that", prompt_tokens=10, completion_tokens=4),
        response('{"route": "retrieve", "reason": "ok"}', prompt_tokens=10, completion_tokens=6),
    )
    result = asyncio.run(llm.astructured("q", Route))

    assert result.data.route == "retrieve"
    assert len(fake.calls) == 2
    assert result.usage == Usage(20, 10, 30, 0.0)


def test_structured_gives_up_after_the_configured_retries() -> None:
    llm, fake = client(response("not json"), structured_retries=1)
    with pytest.raises(StructuredOutputError, match="Route"):
        llm.structured("q", Route)
    assert len(fake.calls) == 2


def test_astructured_gives_up_after_the_configured_retries() -> None:
    llm, fake = async_client(response("not json"), structured_retries=1)
    with pytest.raises(StructuredOutputError, match="Route"):
        asyncio.run(llm.astructured("q", Route))
    assert len(fake.calls) == 2


def test_structured_retries_are_configurable() -> None:
    llm, fake = client(response("not json"), structured_retries=0)
    with pytest.raises(StructuredOutputError):
        llm.structured("q", Route)
    assert len(fake.calls) == 1


def test_json_that_violates_the_schema_is_a_failure_not_a_default() -> None:
    """Valid JSON with a bogus route must not be silently coerced into one."""
    llm, _ = client(response('{"route": "banana", "reason": "?"}'), structured_retries=0)
    with pytest.raises(StructuredOutputError):
        llm.structured("q", Route)


def test_structured_carries_the_system_prompt_and_history() -> None:
    llm, fake = client(response('{"route": "answer", "reason": "hi"}'))
    llm.structured(
        "is it halal?",
        Route,
        system="You route.",
        history=[{"role": "user", "content": "what is Sukuk?"}],
    )
    assert [m["role"] for m in fake.last_call["messages"]] == ["system", "user", "user"]


# --- schema tightening ---------------------------------------------------
# Strict providers reject a schema that allows extra properties or omits any
# property from `required`. Pydantic emits both, so this rewrite is the
# difference between structured output working and a 400 on every request.


def test_strict_schema_closes_objects_and_requires_every_field() -> None:
    schema = _strict_schema(Route.model_json_schema())
    assert schema["additionalProperties"] is False
    # search_query has a default, so Pydantic leaves it out of `required`.
    assert set(schema["required"]) == {"route", "reason", "search_query"}


def test_strict_schema_reaches_nested_models() -> None:
    schema = _strict_schema(Nested.model_json_schema())
    nested = schema["$defs"]["Route"]
    assert nested["additionalProperties"] is False
    assert set(nested["required"]) == {"route", "reason", "search_query"}


def test_strict_schema_preserves_the_field_constraints() -> None:
    """Tightening must not flatten the enum — that is what pins the route values."""
    schema = _strict_schema(Route.model_json_schema())
    assert schema["properties"]["route"]["enum"] == ["retrieve", "refuse", "answer"]


def test_strict_schema_is_valid_json() -> None:
    """It is serialized into the request body, so it has to survive json.dumps."""
    import json

    json.dumps(_strict_schema(Nested.model_json_schema()))


# --- transport sharing ---------------------------------------------------


def test_transport_is_shared_between_clients_on_the_same_endpoint() -> None:
    """Connection pools are the reason to cache; a client per model would not."""
    from core.llm import _transport

    args = {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 60.0,
        "max_retries": 2,
    }
    assert _transport(**args) is _transport(**args)


def test_async_transport_is_cached_per_endpoint_and_keyed_on_its_arguments() -> None:
    """Its own cache, because the two SDK clients hold separate pools."""
    from core.llm import _async_transport

    args = {
        "api_key": "test-key",
        "base_url": "https://openrouter.ai/api/v1",
        "timeout": 60.0,
        "max_retries": 2,
    }
    assert _async_transport(**args) is _async_transport(**args)
    assert _async_transport(**{**args, "base_url": "https://example.test/v1"}) is not (
        _async_transport(**args)
    )


def test_injecting_only_a_sync_transport_makes_the_async_calls_fail_loudly() -> None:
    """Silently building a real `AsyncOpenAI` here would let a test hit the network."""
    llm, _ = client()
    with pytest.raises(LLMError, match="async transport"):
        asyncio.run(llm.acomplete("x"))


def test_injecting_only_an_async_transport_makes_the_sync_calls_fail_loudly() -> None:
    llm, _ = async_client()
    with pytest.raises(LLMError, match="sync transport"):
        llm.complete("x")


def test_repr_names_the_model_and_hides_nothing_secret() -> None:
    llm, _ = client()
    assert repr(llm) == "LLMClient('google/gemini-3.5-flash-lite')"


# --- live smoke test -----------------------------------------------------
# Deselected by default (`-m "not live"`). Run explicitly:
#     uv run pytest -m live
# This is the check that the configured model ids exist and the key works —
# the failure everything else in this file is blind to, since a fake transport
# will happily serve a model that OpenRouter has never heard of.


@pytest.mark.live
@pytest.mark.skipif(
    not get_settings().openrouter_api_key, reason="needs a real OPENROUTER_API_KEY"
)
def test_live_structured_call_against_the_configured_fast_model() -> None:
    from core.llm import fast_llm

    result = fast_llm().structured(
        "Customer asks: what is Murabaha? Route this.",
        Route,
        system=(
            "You route customer questions for an Islamic finance assistant. "
            "Reply with a route, a short reason, and a search query."
        ),
    )
    assert result.data.route in {"retrieve", "refuse", "answer"}
    assert result.usage.total_tokens > 0
