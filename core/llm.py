"""OpenRouter chat client — one class, any model.

Every LLM call in the graph goes through here. OpenRouter is OpenAI-compatible,
so the transport is the `openai` SDK pointed at `openrouter_base_url`; what this
module adds on top is the three things the nodes actually need and the SDK does
not give directly:

- **A model is (provider, name).** OpenRouter addresses every model as
  `provider/name`, so that pair *is* the client's identity — construct one per
  model rather than passing a model string into every call. `google/gemini-3.6-flash`
  and `openai/gpt-4o-mini` differ only in those two strings, which is what keeps
  the two-tier split in `config.py` a configuration change and not a code change.
- **Structured output that is actually enforced.** `structured()` sends a strict
  JSON schema derived from a Pydantic model and validates the reply back into
  it. The router and grader are useless if their output has to be parsed by
  hand — a router that returns prose has no route.
- **Usage on every reply.** Token counts and cost are required trace fields, so
  they come back attached to the result instead of being fished out of a raw
  response object at the call site.

Sync and async are twins here, not alternatives. The request path is async end
to end — async endpoint → `ainvoke` → `async def` nodes → `acomplete()` /
`astructured()` — so the hot calls never block the event loop. The sync methods
stay because build-time work (`kb/ingest`) runs without a loop and unit tests
drive collaborators directly; the two sides share every piece of plumbing
except the transport call itself, so behaviour cannot drift between them.

This module holds no prompts. Prompts belong to the nodes that own them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Literal, TypedDict, TypeVar

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel, ValidationError

from core.config import get_settings

T = TypeVar("T", bound=BaseModel)

Role = Literal["system", "user", "assistant"]


class Message(TypedDict):
    role: Role
    content: str


# --- errors -------------------------------------------------------------
# Transport failures (timeouts, 429s, 5xx) stay as the SDK's own exceptions —
# they are already well-typed and the SDK retries them. What is raised here is
# only what is specific to *this* layer: a reply that arrived fine and was
# unusable anyway.


class LLMError(RuntimeError):
    """A reply arrived but could not be used."""


class EmptyCompletionError(LLMError):
    """The model returned no content — usually a filter or a truncated reply."""


class StructuredOutputError(LLMError):
    """The model's reply did not validate against the requested schema."""


# --- value objects ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Model:
    """An OpenRouter model id, split at the slash.

    Kept as a pair rather than one string so a typo lands here, at construction,
    instead of as a 404 on the request path.
    """

    provider: str
    name: str

    def __post_init__(self) -> None:
        if not self.provider or not self.name:
            raise ValueError(
                f"model needs both a provider and a name, got {self.provider!r}/{self.name!r}"
            )
        if "/" in self.provider:
            raise ValueError(f"provider must not contain '/': {self.provider!r}")

    @classmethod
    def parse(cls, slug: str) -> Model:
        """`"google/gemini-3.6-flash"` → `Model("google", "gemini-3.6-flash")`.

        Splits on the first slash only: names may contain further slashes and
        suffixes (`meta-llama/llama-3.1-8b-instruct:free`).
        """
        provider, sep, name = slug.strip().partition("/")
        if not sep:
            raise ValueError(
                f"model id must be 'provider/name', got {slug!r} — "
                "OpenRouter has no unqualified model ids"
            )
        return cls(provider, name)

    def __str__(self) -> str:
        return f"{self.provider}/{self.name}"


@dataclass(frozen=True, slots=True)
class Usage:
    """Token counts and cost for one call.

    `cost` is an OpenRouter extension (USD, requested via `usage.include`), not
    part of the OpenAI schema — it is 0.0 on providers that omit it. Adds so a
    trace can total a whole request across the four or five calls it makes.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
            self.cost + other.cost,
        )


@dataclass(frozen=True, slots=True)
class Completion:
    """A text reply plus what it cost."""

    text: str
    usage: Usage
    model: str


@dataclass(frozen=True, slots=True)
class Structured(Generic[T]):
    """A validated reply plus what it cost. `data` is the Pydantic model."""

    data: T
    usage: Usage
    model: str


# --- strict JSON schema -------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)


def _strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Tighten a Pydantic JSON schema until a strict provider will accept it.

    Strict structured output requires every object to forbid extra properties
    and to list every property as required. Pydantic emits neither: optional
    fields are omitted from `required`, and `additionalProperties` is left
    unset. Providers reject the schema outright rather than relaxing it, so the
    rewrite happens here — recursively, since nested models arrive under `$defs`.
    """
    if schema.get("type") == "object":
        properties = schema.get("properties", {})
        schema["additionalProperties"] = False
        schema["required"] = list(properties)

    for key in ("properties", "$defs", "definitions"):
        for sub in schema.get(key, {}).values():
            _strict_schema(sub)

    for key in ("items", "additionalProperties"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            _strict_schema(sub)

    for key in ("anyOf", "oneOf", "allOf", "prefixItems"):
        for sub in schema.get(key, []):
            _strict_schema(sub)

    return schema


def _response_format(schema: type[BaseModel]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "strict": True,
            "schema": _strict_schema(schema.model_json_schema()),
        },
    }


def _strip_fence(text: str) -> str:
    """Unwrap ```json fences.

    Strict mode should make this dead code. It is not: models that were
    instruction-tuned to fence their JSON sometimes do it anyway, and losing a
    router decision to three backticks is not a trade worth making.
    """
    match = _FENCE.match(text)
    return match.group(1) if match else text.strip()


# --- client -------------------------------------------------------------


class LLMClient:
    """Chat and structured-output calls against one OpenRouter model.

    One instance per model, reused across requests. The underlying `OpenAI` and
    `AsyncOpenAI` objects each hold a connection pool and are shared between
    instances that talk to the same endpoint (see `_transport` and
    `_async_transport`), so building a client is cheap. Every call has a sync
    and an async form (`complete`/`acomplete`, `structured`/`astructured`);
    graph nodes use the async ones.
    """

    def __init__(
        self,
        provider: str,
        name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        structured_retries: int = 1,
        client: Any | None = None,
        async_client: Any | None = None,
    ) -> None:
        """`provider` and `name` are the two halves of the OpenRouter model id.

        `client` and `async_client` inject pre-built transports — used by the
        tests, and by any caller that wants to share a specific `OpenAI` or
        `AsyncOpenAI` instance. Injecting either builds *only* what was passed:
        calling the other side then raises `LLMError` instead of quietly
        constructing a real network client behind a test's back.

        `temperature` defaults to 0: three of the four callers (router, grader,
        reformulate) want the same answer for the same input, and the fourth
        (generate) is quoting retrieved text back at the customer.
        """
        settings = get_settings()
        self.model = Model(provider, name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.structured_retries = structured_retries

        if client is not None or async_client is not None:
            self._client = client
            self._async_client = async_client
            return

        key = api_key if api_key is not None else settings.openrouter_api_key
        if not key:
            raise ValueError(
                "OPENROUTER_API_KEY is not set — copy .env.example to .env and fill it in"
            )
        endpoint = base_url or settings.openrouter_base_url
        self._client = _transport(
            api_key=key, base_url=endpoint, timeout=timeout, max_retries=max_retries
        )
        self._async_client = _async_transport(
            api_key=key, base_url=endpoint, timeout=timeout, max_retries=max_retries
        )

    @classmethod
    def from_slug(cls, slug: str, **kwargs: Any) -> LLMClient:
        """Build from a `provider/name` string, which is how models are configured."""
        model = Model.parse(slug)
        return cls(model.provider, model.name, **kwargs)

    def with_model(self, slug: str) -> LLMClient:
        """Same transports and settings, different model. Shares the connection pools."""
        model = Model.parse(slug)
        return LLMClient(
            model.provider,
            model.name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            structured_retries=self.structured_retries,
            client=self._client,
            async_client=self._async_client,
        )

    # --- calls ----------------------------------------------------------

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: list[Message] | tuple[Message, ...] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Free-text reply.

        `history` is prior turns, inserted between the system prompt and the
        current one. Callers pass PII-masked text only — this layer does not
        redact, it is downstream of the node that does.
        """
        response = self._create(
            self._messages(prompt, system, history),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._completion(response)

    async def acomplete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: list[Message] | tuple[Message, ...] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """`complete()` on the async transport — same request, same checks."""
        response = await self._acreate(
            self._messages(prompt, system, history),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return self._completion(response)

    def structured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        history: list[Message] | tuple[Message, ...] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        """Reply validated into `schema`.

        Asks for strict JSON schema output and validates the result, so callers
        get a typed object or an exception — never a string to guess at.

        A reply that fails validation is retried `structured_retries` times.
        Retrying is worth it because the failure is usually a stray token rather
        than a misunderstanding, and because the alternative for the router is a
        request with no route at all. Usage from failed attempts is included in
        the returned total: the tokens were spent either way, and a trace that
        hides them under-reports cost.
        """
        messages = self._messages(prompt, system, history)
        response_format = _response_format(schema)
        spent = Usage()
        last_error: Exception | None = None

        for _ in range(self.structured_retries + 1):
            response = self._create(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            spent = spent + _usage(response)
            try:
                data = schema.model_validate_json(_strip_fence(self._content(response)))
            # `EmptyCompletionError` is listed explicitly because it is a
            # `RuntimeError`, so neither of the other two covers it — and a reply
            # with no choices at all (OpenRouter's 200-with-`error`) is the same
            # kind of transient failure as a malformed one: worth one more
            # attempt, and worth reporting as `StructuredOutputError`, which is
            # the type the router's fail-open is written against. `last_error`
            # carries the provider's own message into that final raise.
            except (ValidationError, ValueError, EmptyCompletionError) as exc:
                last_error = exc
                continue
            return Structured(data=data, usage=spent, model=str(self.model))

        raise StructuredOutputError(
            f"{self.model} did not return valid {schema.__name__} after "
            f"{self.structured_retries + 1} attempts: {last_error}"
        ) from last_error

    async def astructured(
        self,
        prompt: str,
        schema: type[T],
        *,
        system: str | None = None,
        history: list[Message] | tuple[Message, ...] = (),
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Structured[T]:
        """`structured()` on the async transport — same schema, same retry loop.

        The loop is duplicated rather than shared through a callback because
        the only difference is the `await`; everything that could drift — the
        request kwargs, the fence-stripping, the usage accounting — lives in
        helpers both versions call.
        """
        messages = self._messages(prompt, system, history)
        response_format = _response_format(schema)
        spent = Usage()
        last_error: Exception | None = None

        for _ in range(self.structured_retries + 1):
            response = await self._acreate(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
            spent = spent + _usage(response)
            try:
                data = schema.model_validate_json(_strip_fence(self._content(response)))
            # Same tuple as `structured`, for the same reasons — these two lines
            # are the pair most worth keeping identical, since the async side is
            # the one the request path actually runs.
            except (ValidationError, ValueError, EmptyCompletionError) as exc:
                last_error = exc
                continue
            return Structured(data=data, usage=spent, model=str(self.model))

        raise StructuredOutputError(
            f"{self.model} did not return valid {schema.__name__} after "
            f"{self.structured_retries + 1} attempts: {last_error}"
        ) from last_error

    # --- plumbing -------------------------------------------------------

    @staticmethod
    def _messages(
        prompt: str,
        system: str | None,
        history: list[Message] | tuple[Message, ...],
    ) -> list[Message]:
        messages: list[Message] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def _create_kwargs(
        self,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """One request shape for both transports, so the two cannot fork."""
        kwargs: dict[str, Any] = {
            "model": str(self.model),
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
            # OpenRouter-only: asks for the USD cost of the call on the usage
            # object. Cheaper than pricing tokens ourselves per model.
            "extra_body": {"usage": {"include": True}},
        }
        limit = self.max_tokens if max_tokens is None else max_tokens
        if limit is not None:
            kwargs["max_tokens"] = limit
        if response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    def _create(
        self,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        if self._client is None:
            raise LLMError(
                f"{self.model} has no sync transport — it was built with async_client= only"
            )
        return self._client.chat.completions.create(
            **self._create_kwargs(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        )

    async def _acreate(
        self,
        messages: list[Message],
        *,
        temperature: float | None,
        max_tokens: int | None,
        response_format: dict[str, Any] | None = None,
    ) -> Any:
        if self._async_client is None:
            raise LLMError(
                f"{self.model} has no async transport — pass async_client= too, or "
                "construct without client= to get both"
            )
        return await self._async_client.chat.completions.create(
            **self._create_kwargs(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
            )
        )

    def _completion(self, response: Any) -> Completion:
        text = self._content(response)
        if not text.strip():
            raise EmptyCompletionError(
                f"{self.model} returned an empty completion{_provider_error(response)}"
            )
        return Completion(text=text, usage=_usage(response), model=str(self.model))

    @staticmethod
    def _content(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            # A reply with no choices is usually not an empty answer at all: it is
            # OpenRouter passing an upstream rejection back as HTTP 200 with an
            # `error` object, which the SDK does not raise on. The suffix is the
            # only place that message survives — same call the embeddings and
            # rerank clients make on their own no-payload path.
            raise EmptyCompletionError(
                f"response contained no choices{_provider_error(response)}"
            )
        return getattr(choices[0].message, "content", None) or ""

    def __repr__(self) -> str:
        return f"LLMClient({str(self.model)!r})"


def _usage(response: Any) -> Usage:
    """Read usage off a response, tolerating providers that omit it."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return Usage()

    def field(name: str, default: float = 0) -> Any:
        value = getattr(usage, name, None)
        return default if value is None else value

    return Usage(
        prompt_tokens=int(field("prompt_tokens")),
        completion_tokens=int(field("completion_tokens")),
        total_tokens=int(field("total_tokens")),
        cost=float(field("cost", 0.0)),
    )


def _provider_error(response: Any) -> str:
    """Pull the upstream's own complaint out of a response that carries one.

    OpenRouter answers some upstream rejections with HTTP 200 and an `error`
    object rather than a status code, so the SDK does not raise and this message
    is the only thing that says what went wrong. Dropping it turns a one-line
    diagnosis ("this provider does not support base64 encoding_format") into a
    debugging session.

    Lives here with `Model`, `Usage` and `_transport` because it is shared by
    every client in `core/` — the behaviour is OpenRouter's, not any one
    endpoint's. Accepts both SDK response objects and the plain dicts that
    `client.post(..., cast_to=object)` returns for non-OpenAI endpoints.
    """
    error = response.get("error") if isinstance(response, dict) else getattr(response, "error", None)
    if not error:
        return ""
    message = error.get("message") if isinstance(error, dict) else getattr(error, "message", None)
    return f" — provider said: {message or error}"


@lru_cache(maxsize=8)
def _transport(*, api_key: str, base_url: str, timeout: float, max_retries: int) -> OpenAI:
    """One `OpenAI` object per endpoint, so its connection pool is shared.

    Retries here cover transport failures (429, 5xx, timeouts) with the SDK's
    own backoff. Schema-validation retries are a separate concern and live in
    `structured()`.
    """
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        default_headers={"X-Title": "Mal Islamic Finance Assistant"},
    )


@lru_cache(maxsize=8)
def _async_transport(
    *, api_key: str, base_url: str, timeout: float, max_retries: int
) -> AsyncOpenAI:
    """`_transport`'s async twin — one `AsyncOpenAI` per endpoint.

    A separate cache because the two SDK clients hold separate connection
    pools. The async pool binds to the event loop that first uses it, which is
    fine under uvicorn — one loop for the process lifetime — but not across
    repeated `asyncio.run()` calls: a keep-alive connection from a closed loop
    raises on reuse. Tests that create a loop per case must either inject fakes
    (unit tests do) or clear this cache between loops (the live suite does).
    """
    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=max_retries,
        default_headers={"X-Title": "Mal Islamic Finance Assistant"},
    )


# --- the two configured tiers -------------------------------------------
# Nodes call these rather than constructing clients: the split between the
# cheap model and the answering model is the main cost lever, and it stays a
# config change only as long as no node hardcodes a model id.


@lru_cache(maxsize=1)
def fast_llm() -> LLMClient:
    """Router, grader, reformulate — short structured calls on every request."""
    return LLMClient.from_slug(get_settings().fast_model)


@lru_cache(maxsize=1)
def answer_llm() -> LLMClient:
    """`generate` only — the one call whose prose the customer reads."""
    return LLMClient.from_slug(get_settings().answer_model)


@lru_cache(maxsize=8)
def llm_for(slug: str) -> LLMClient:
    """Escape hatch for a model outside the two tiers. Cached per slug."""
    return LLMClient.from_slug(slug)


__all__ = [
    "Completion",
    "EmptyCompletionError",
    "LLMClient",
    "LLMError",
    "Message",
    "Model",
    "Structured",
    "StructuredOutputError",
    "Usage",
    "answer_llm",
    "fast_llm",
    "llm_for",
]
