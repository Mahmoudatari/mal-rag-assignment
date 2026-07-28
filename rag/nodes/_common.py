"""Bookkeeping shared by the nodes: usage entries and conversation history.

Pure functions over state — no clients, no prompts, no LangGraph. They exist
because the append-and-cap logic would otherwise be retyped in three terminal
nodes and the usage projection in five, and both have the same failure mode:
mutating the list already in state instead of returning a new one. Nodes return
partial state, so an in-place `.append()` would edit the checkpointed object
before the graph decided the node succeeded.
"""

from core.config import get_settings
from core.llm import Usage
from rag.state import ChatMessage, State, UsageEntry


def usage_entry(node: str, model: str, usage: Usage, *, search_units: int | None = None) -> UsageEntry:
    """Project a provider call's usage into the JSON shape state carries.

    `search_units` is rerank's billing unit and is omitted for every other
    caller, rather than written as 0 — absent means "not billed that way".
    """
    entry = UsageEntry(
        node=node,
        model=model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        cost=usage.cost,
    )
    if search_units is not None:
        entry["search_units"] = search_units
    return entry


def logged(state: State, entry: UsageEntry) -> list[UsageEntry]:
    """The turn's usage log with `entry` appended, as a new list."""
    return [*state.get("usage_log", []), entry]


def appended_history(state: State, query: str, answer: str) -> list[ChatMessage]:
    """History with this turn's exchange appended, oldest dropped past the cap.

    `query` must be the masked text — this list is checkpointed and replayed
    into later prompts, so raw PII here would outlive the turn that leaked it.
    """
    history = [
        *state.get("history", []),
        ChatMessage(role="user", content=query),
        ChatMessage(role="assistant", content=answer),
    ]
    cap = get_settings().history_max_messages
    return history[-cap:] if cap > 0 else []
