"""Compose the answer. Handles both the retrieval and no-retrieval paths.

`chunks` empty means the router chose "answer" — a conversational or meta turn.
That path is ungrounded by construction, so the prompt must forbid substantive
finance claims there.

Citations are inline [n] markers plus a structured references list. Keep the
display-index → chunk_id mapping so traces record real chunk IDs. No follow-up
questions.
"""

import re

from core.llm import answer_llm
from rag.nodes._common import appended_history, logged, usage_entry
from rag.state import Chunk, Reference, State

# Chunks present: answer strictly from the numbered passages. The [n] marker is
# the whole citation mechanism (no markdown anchors — there is no frontend), so
# it has to be attached to the fact it supports, not bolted on at the end.
GROUNDED_SYSTEM = """You are the answering assistant for Mal, an Islamic finance bank.

Answer the customer's question using ONLY the numbered context passages below.
Every factual claim must carry an inline citation marker, e.g. "Murabaha is a \
cost-plus sale [1]." Use the number of the passage the fact came from — if a \
sentence draws on more than one passage, cite all of them, e.g. "[1][3]".

If the passages do not contain a fact the question needs, say plainly that it \
could not be confirmed from Mal's guides rather than inventing a figure, rate, \
term or ruling. Never guess at a number or a Sharia judgement that is not in the \
passages.

The question may contain masked placeholders like [ACCOUNT_NUMBER] or [PERSON] \
— these stand in for the customer's own identifiers, redacted before reaching \
you. Leave any such placeholder exactly as written; never invent a value for it.

Do not ask the customer a follow-up question — end with the answer, not a prompt \
for more information."""

# No chunks: the router sent a conversational or meta turn straight to
# `generate`. Nothing grounds this reply, so the one job of this prompt is
# closing off the main hallucination surface CLAUDE.md calls out — a fabricated
# rate or ruling with no retrieved text behind it.
UNGROUNDED_SYSTEM = """You are the answering assistant for Mal, an Islamic finance bank.

No knowledge base passages were retrieved for this turn — it is a greeting, \
thanks, or a question about what you can help with, not a request for product \
facts. Reply naturally to that, and explain briefly that you can answer \
questions about Mal's Islamic finance products (Murabaha, Ijara, Sukuk, Wakala, \
Takaful) and the customer's own account.

Do NOT state any substantive finance claim: no figures, rates, fees, tenors, \
eligibility rules or Sharia rulings. Anything like that requires retrieved \
context you do not have here — if the customer is actually asking about product \
facts, say you'd need to look that up rather than answering from memory.

Do not cite anything — there is no context to cite. Do not ask the customer a \
follow-up question."""

_USER_TEMPLATE = """Context passages:
{passages}

Customer question:
{query}"""

# The router short-circuits a turn that masked down to nothing to route="answer",
# which lands here with a blank query. `complete` guards its own blank *reply*,
# not a blank prompt, so without this the graph would post an empty user message
# to the provider and take whatever it made of it.
_EMPTY_TURN = "The customer sent an empty message. Greet them and say what you can help with."

# The leading `\s*` is what makes stripping an invalid marker clean: dropping the
# match removes the space that preceded it too, so "a loan [7]." does not become
# "a loan .". Valid markers are re-emitted as matched, whitespace included.
_MARKER = re.compile(r"\s*\[(\d+)\]")


def _render_passages(chunks: list[Chunk]) -> str:
    blocks = [f"[{i}] ({chunk['doc']})\n{chunk['text']}" for i, chunk in enumerate(chunks, start=1)]
    return "\n\n".join(blocks)


def _citations(answer: str, chunks: list[Chunk]) -> tuple[str, list[Reference]]:
    """Strip out-of-range [n] markers and build references from the rest.

    A marker the model invented past `len(chunks)` is a hallucinated citation —
    left in place it reads as grounded when it points at nothing, so it is
    removed from the text as well as excluded from `references`. Built from the
    markers actually present, in ascending order, so a trace never claims a
    chunk was cited when the model never wrote its number.

    Runs on the no-retrieval path too, where `chunks` is empty and therefore
    *every* marker is out of range. That path is told not to cite at all, so a
    marker appearing there is precisely the hallucinated-citation case this
    strips — handling it in one path rather than two.
    """
    valid: list[int] = []
    seen: set[int] = set()

    def replace(match: re.Match[str]) -> str:
        n = int(match.group(1))
        if not 1 <= n <= len(chunks):
            return ""
        if n not in seen:
            seen.add(n)
            valid.append(n)
        return match.group(0)

    cleaned = _MARKER.sub(replace, answer).strip()
    references = [
        Reference(n=n, doc=chunks[n - 1]["doc"], chunk_id=chunks[n - 1]["chunk_id"])
        for n in sorted(valid)
    ]
    return cleaned, references


async def run(state: State) -> dict:
    """chunks (possibly empty) → answer, references, outcome="answered"."""
    query = state.get("query", "")
    chunks = state.get("chunks", [])
    history = state.get("history", [])

    if chunks:
        system = GROUNDED_SYSTEM
        prompt = _USER_TEMPLATE.format(passages=_render_passages(chunks), query=query)
    else:
        system = UNGROUNDED_SYSTEM
        prompt = query.strip() or _EMPTY_TURN

    # EmptyCompletionError propagates rather than being papered over: a
    # fabricated apology string here would itself enter history via
    # appended_history below and pollute every later turn's prompt.
    result = await answer_llm().acomplete(prompt, system=system, history=history)

    answer, references = _citations(result.text, chunks)
    entry = usage_entry("generate", result.model, result.usage)

    return {
        "answer": answer,
        "references": references,
        "outcome": "answered",
        "history": appended_history(state, query, answer),
        "usage_log": logged(state, entry),
    }
