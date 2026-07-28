"""Terminal: query is out of scope.

"I can only answer questions about Islamic finance and your Mal account."
Distinct from `no_answer` — different message, different trace outcome. This is
the node the refusal eval targets.
"""

from rag.nodes._common import appended_history
from rag.state import State

# Static template, no LLM call: the refusal eval wants a deterministic string
# for a fixed decision the router already made, and paying for a model call to
# phrase one fixed sentence contradicts the "keep LLM costs minimal" constraint.
# Names a couple of the five products so the customer knows what to ask
# instead, rather than leaving them at a dead end. Statement only — no
# trailing question, matching the "no follow-up questions" rule for every
# terminal node.
MESSAGE = (
    "I can only answer questions about Islamic finance and your Mal account — "
    "things like Murabaha everyday finance, Ijara auto lease-to-own, fractional "
    "Sukuk investing, Wakala savings or Takaful cover. Please ask about one of "
    "those and I'll be glad to help."
)


def run(state: State) -> dict:
    """→ answer, outcome="refused"."""
    query = state.get("query", "")
    return {
        "answer": MESSAGE,
        "outcome": "refused",
        "history": appended_history(state, query, MESSAGE),
    }
