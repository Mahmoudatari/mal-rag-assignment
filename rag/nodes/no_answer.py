"""Terminal: in scope, but retrieval was exhausted without relevant context.

Points the customer at support using obviously-fake placeholder contact details
(support@mal.example) so stub data is never mistaken for real.
"""

from rag.nodes._common import appended_history
from rag.state import State

# Static template, no LLM call: same reasoning as `refuse` — this is a fixed
# terminal message for a decision `grade` already made, and the retries were
# the place to spend LLM calls trying to answer, not this dead end.
# support@mal.example is an IANA reserved example domain, and the phone number
# is an obvious placeholder — neither could be mistaken for a real Mal contact.
MESSAGE = (
    "I couldn't find this in Mal's guides, and I don't want to guess at an "
    "answer that affects your finances. Please contact Mal support at "
    "support@mal.example or +971-800-000-0000 and they'll be able to help."
)


def run(state: State) -> dict:
    """→ answer, outcome="no_answer"."""
    query = state.get("query", "")
    return {
        "answer": MESSAGE,
        "outcome": "no_answer",
        "history": appended_history(state, query, MESSAGE),
    }
