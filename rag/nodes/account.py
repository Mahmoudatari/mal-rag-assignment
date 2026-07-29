"""Attach the customer's account context. Second node, right after `redact`.

A separate node rather than app-side injection so the graph owns everything a
prompt is built from: the trace's `calls`/state view stays complete, and a
future real lookup (an accounts service instead of `accounts.ACCOUNTS`) changes
this node alone. Pure and synchronous like `redact` — a dict lookup with no IO
— so langgraph runs it in a worker thread under `ainvoke` and nothing blocks
the loop.

Always writes `account` and `history_account`, None and "" included. Nodes write
only what they change and the checkpointer persists the whole dict per thread, so
a node that skipped the write on "no id this turn" would leave the previous
turn's account attached — the same silent-carryover failure `redact`'s reset
exists for, handled here by unconditional writing rather than by joining the
reset list.

**Clearing `account` is not enough on its own**, which is why `history_account`
exists. `generate` appends its answer to `history`, the one key that survives a
turn, and that answer restates whatever the rendered record said — contract
reference, outstanding balance, arrears. Dropping the record leaves the sentences
derived from it in place, replayed into every later router and generate prompt.
So this node also drops `history` when the account context changes, and the
conversation restarts rather than carrying another context's figures forward.

The rule is deliberately asymmetric. Anonymous → account keeps history: no
account facts can be in it yet, only the customer's own earlier questions.
Account → a different account, or → no account, wipes it: the accumulated text
describes a customer whose record is no longer attached, and `account_id` is a
per-request field, so on the next turn that is somebody else's context.

An unknown id is not an error: `account` is None and `generate`'s prompts say
account details could not be found this session. The API does not 404 on it —
whether an account exists is not something an unauthenticated endpoint should
confirm or deny.
"""

from accounts import lookup
from rag.state import State


def run(state: State) -> dict:
    """account_id → account (the record, or None) + history_account, scoping history."""
    record = lookup(state.get("account_id", ""))
    context = record["masked_id"] if record else ""
    written = {"account": record, "history_account": context}

    previous = state.get("history_account", "")
    if previous and previous != context:
        # Only ever narrows: an empty `previous` means nothing in history was
        # written under an account, so there is nothing to leak forward.
        written["history"] = []
    return written
