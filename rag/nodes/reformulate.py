"""Retry-only query rewrite, steered by the grader's note.

Distinct from the router's contextualization: that resolves a query against
history, this repairs one that retrieved badly.

Owns the `attempts` counter — it is incremented here and nowhere else.
Guard against rewrites that converge on an already-tried query.
"""

from pydantic import BaseModel, Field

from core.llm import fast_llm
from rag.nodes._common import logged, usage_entry
from rag.state import State

# The grader's note names what was missing; this call's whole job is turning
# that into search text that will surface a different slice of the corpus. A
# rewrite that just rephrases the same words re-runs the same vector search and
# wastes the retry, so the prompt pushes explicitly for different vocabulary.
SYSTEM = """You rewrite a search query for Mal, an Islamic finance bank's \
knowledge base, after a first retrieval came back inadequate.

You are given the customer's original question, the search query that was just \
tried, a grader's note on what that retrieval was missing, and every search \
query already tried this turn. Write a new search query likely to surface \
different passages: use synonyms, the product's formal Arabic name where one \
exists (e.g. Murabaha, Ijara, Sukuk, Wakala, Takaful), or the underlying \
concept the grader's note points at, rather than repeating the same wording. \
Never repeat a query already tried — if every rewording is exhausted, phrase it \
from a different angle instead.

Always keep the subject of the customer's question in the query. If they asked \
about Murabaha, the rewrite still names Murabaha; a query of generic finance \
vocabulary with the product dropped searches the wrong documents entirely. \
Change how the question is asked, never what it is about.

The customer's underlying question never changes here — only the search \
wording does."""

_USER_TEMPLATE = """Customer's original question:
{query}

Search query just tried:
{search_query}

Grader's note on why that retrieval was inadequate:
{grader_note}

Search queries already tried this turn (do not repeat these):
{tried_queries}"""


class Rewrite(BaseModel):
    search_query: str = Field(
        description=(
            "A new search query for the knowledge base, worded differently from "
            "every query already tried — different vocabulary, the product's "
            "formal name, or the underlying concept named in the grader's note. "
            "Never identical to a previously tried query, and always still about "
            "the same product or subject the customer asked about."
        )
    )


async def run(state: State) -> dict:
    """search_query + grader_note → new search_query, attempts + 1."""
    tried_queries = state.get("tried_queries", [])
    # "Always keep the subject of the customer's question in the query" is only
    # obeyable if the question shown has a subject. The raw turn often does not
    # — "can I use it for a home?" — and a rewrite built from that drifts onto
    # whichever product the words suggest, silently switching topic mid-loop.
    # The router's resolved question carries the subject and, unlike
    # `search_query`, does not change under us on each retry. `query` remains
    # the fallback for hand-built states in tests and evals.
    prompt = _USER_TEMPLATE.format(
        query=state.get("resolved_query") or state.get("query", ""),
        search_query=state.get("search_query", ""),
        grader_note=state.get("grader_note", ""),
        tried_queries="\n".join(tried_queries) if tried_queries else "(none)",
    )
    # StructuredOutputError propagates: same reasoning as `grade` — a guessed
    # rewrite inside a retry loop is worse than failing the request outright.
    result = await fast_llm().astructured(prompt, Rewrite, system=SYSTEM)
    entry = usage_entry("reformulate", result.model, result.usage)

    # A blank rewrite must not reach `retrieve`'s `embed_query`, which raises on
    # empty text — the masked customer query is the floor, same fallback the
    # router uses for its own rewrite.
    search_query = result.data.search_query.strip() or state.get("query", "")

    # A rewrite that lands on a query already tried is accepted as-is rather
    # than looped on: `attempts` (incremented below, only here) already bounds
    # this cycle via `after_grade`, and a retry loop nested inside the retry
    # loop is the failure mode to avoid, not the collision itself. The
    # duplicate entry in `tried_queries` is honest trace data — it shows the
    # rewrite converged rather than hiding that fact.
    return {
        "search_query": search_query,
        "attempts": state.get("attempts", 0) + 1,
        "tried_queries": [*tried_queries, search_query],
        "usage_log": logged(state, entry),
    }
