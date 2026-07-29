"""Detect and mask PII. Always the first node — nothing downstream sees raw input.

Delegates to the pure `pii` package so the redaction eval can test the logic
without constructing a graph.

The node's only real job is the boundary: `raw_query` in, `query` out, and every
node after this one reads `query`. It writes no other key, so there is no path
by which the unmasked text reaches the router, a prompt, or a trace.

Confidence scores are dropped here on purpose. `pii.PiiSpan` carries one, but
`state.PIISpan` is kind and offsets only — state is checkpointed to Postgres and
persists for the life of the session, so it holds what tracing needs and no more.

Being first also makes this the node that starts the turn, so it resets the
turn-scoped keys. The checkpointer persists one State per `thread_id` and nodes
write only what they change, so without this a second turn inherits the first
one's: a conversational turn would reach `generate` carrying the previous turn's
`chunks` and be treated as grounded, and a turn following an exhausted retrieval
would start at `attempts == 2` and get no retries at all. `history` is the one
key deliberately left alone.
"""

from pii import redact as redact_text
from rag.state import PIISpan, State


def _fresh_turn() -> dict:
    """Zero values for every turn-scoped key.

    Built per call rather than held as a module constant: the lists would
    otherwise be shared by every turn in the process, and a node that appended
    to one in place would edit them all. `route`, `answer` and `outcome` are
    absent because every path through the graph writes them.
    """
    return {
        "search_query": "",
        # Cleared with the rest: an inherited `resolved_query` would make grade
        # and reformulate judge turn N+1's retrieval against turn N's question,
        # which is the exact failure this key exists to prevent.
        "resolved_query": "",
        "chunks": [],
        "candidate_log": [],
        "relevant": False,
        "grader_note": "",
        "attempts": 0,
        "tried_queries": [],
        "references": [],
        "usage_log": [],
    }


def run(state: State) -> dict:
    """raw_query → query (masked) + pii_spans (kind and offsets only), turn reset."""
    result = redact_text(state.get("raw_query", ""))
    return {
        **_fresh_turn(),
        "query": result.text,
        "pii_spans": [
            PIISpan(kind=span.kind, start=span.start, end=span.end)
            for span in result.spans
        ],
    }
