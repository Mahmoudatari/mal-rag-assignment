"""A stand-in for the `openai` client, shaped like the part we call.

`LLMClient` takes its transport by injection, so the tests drive it with this
instead of the network. It records every request, which is what most of the
assertions are actually about — the interesting failures in this module are
malformed *requests* (a schema a provider will reject, a message order that
loses the system prompt), not malformed replies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


def response(
    content: str | None,
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    total_tokens: int | None = None,
    cost: float | None = None,
    usage: bool = True,
) -> SimpleNamespace:
    """Build a minimal chat-completion response.

    `usage=False` models a provider that omits the block entirely, and `cost=None`
    one that returns usage without OpenRouter's cost extension.
    """
    usage_obj = None
    if usage:
        fields: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if total_tokens is None else total_tokens
            ),
        }
        if cost is not None:
            fields["cost"] = cost
        usage_obj = SimpleNamespace(**fields)

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=usage_obj,
    )


def embedding_response(
    vectors: list[list[float]],
    *,
    prompt_tokens: int = 0,
    total_tokens: int | None = None,
    cost: float | None = None,
    usage: bool = True,
    indices: list[int] | None = None,
    error: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Build a minimal embeddings response.

    `indices` overrides the per-item `index` field, which is how the
    out-of-order case is modelled: the OpenAI schema carries an index precisely
    because response order is not promised.
    """
    usage_obj = None
    if usage:
        fields: dict[str, Any] = {
            "prompt_tokens": prompt_tokens,
            "total_tokens": prompt_tokens if total_tokens is None else total_tokens,
        }
        if cost is not None:
            fields["cost"] = cost
        usage_obj = SimpleNamespace(**fields)

    positions = list(range(len(vectors))) if indices is None else indices
    return SimpleNamespace(
        data=[
            SimpleNamespace(index=position, embedding=vector)
            for position, vector in zip(positions, vectors, strict=True)
        ],
        usage=usage_obj,
        # OpenRouter attaches this to some upstream rejections *instead of* a
        # status code, so the response arrives as a 200 the SDK will not raise on.
        error=error,
    )


def rerank_response(
    scores: dict[int, float],
    *,
    search_units: int = 0,
    total_tokens: int = 0,
    cost: float | None = None,
    usage: bool = True,
    documents: dict[int, str] | None = None,
    provider: str = "Cohere",
) -> dict[str, Any]:
    """Build a rerank response: `{input index: relevance_score}`.

    A plain dict, not a `SimpleNamespace`, because the rerank endpoint is
    reached with `cast_to=object` and really does come back as parsed JSON.

    `documents` sets the echoed text. It defaults to a marker no test asserts
    on, since ignoring the echo — and reordering the caller's own objects by
    `index` instead — is the contract worth defending.
    """
    body: dict[str, Any] = {
        "id": "orid-test",
        "model": "cohere/rerank-v3.5",
        "provider": provider,
        "results": [
            {
                "index": index,
                "relevance_score": score,
                "document": {"text": (documents or {}).get(index, f"echo-{index}")},
            }
            for index, score in scores.items()
        ],
    }
    if usage:
        block: dict[str, Any] = {"search_units": search_units, "total_tokens": total_tokens}
        if cost is not None:
            block["cost"] = cost
        body["usage"] = block
    return body


@dataclass
class FakeOpenAI:
    """Returns queued responses in order, replaying the last one once drained.

    Serves `chat.completions.create`, `embeddings.create` and the generic
    `post()` used for `/rerank` from the same queue and the same call log — a
    given test only exercises one of them, and sharing the recorder keeps the
    assertions identical across all three files.
    """

    responses: list[Any] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._next()

    def post(self, path: str, *, body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        """The SDK's generic escape hatch, used for non-OpenAI endpoints.

        The body is flattened into the call record so assertions read the same
        as they do for chat and embeddings; `path` rides alongside it.
        """
        self.calls.append({"path": path, **(body or {})})
        return self._next()

    def _next(self) -> Any:
        if self.error is not None:
            raise self.error
        if not self.responses:
            return response("ok")
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    @property
    def last_call(self) -> dict[str, Any]:
        return self.calls[-1]


@dataclass
class FakeAsyncOpenAI(FakeOpenAI):
    """`FakeOpenAI` with coroutine surfaces, for driving the `a*` methods.

    Injected as `async_client=` where the parent goes to `client=`. Same queue,
    same call log, same response builders — the sync/async split lives entirely
    in the clients' transport call, so the fakes differ only there too.
    """

    def __post_init__(self) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._acreate))
        self.embeddings = SimpleNamespace(create=self._acreate)

    async def _acreate(self, **kwargs: Any) -> Any:
        return self._create(**kwargs)

    async def post(self, path: str, *, body: dict[str, Any] | None = None, **kwargs: Any) -> Any:
        return FakeOpenAI.post(self, path, body=body, **kwargs)
