"""Render an account record into the plain-text block prompts carry.

One renderer, used by `generate` (the full block) and — as `product_names` —
by the router (one line, on the hottest call). Kept here rather than in the
nodes so the block's shape is testable without a prompt around it, and so the
two prompts cannot drift into rendering the same record two different ways.

The output is deliberately `key: value` lines, not prose: the model treats the
block as data to quote from, and a field it cannot find reads as absent rather
than being paraphrased into existence.
"""

from collections.abc import Mapping
from typing import Any


def product_names(account: Mapping[str, Any]) -> list[str]:
    """The products this customer holds, for the router's one-line summary."""
    return [h["product"] for h in account.get("holdings", []) if h.get("product")]


def field_summary(account: Mapping[str, Any]) -> str:
    """Each holding's product and field names, one line per holding — no values.

    What the grader sees: its verdict needs "is the asked-about personal fact
    available downstream", which the field names answer. The figures themselves
    belong to `generate`, the only node that answers, and leaving them out
    keeps the every-retrieval grade call small."""
    lines = []
    for holding in account.get("holdings", []):
        fields = ", ".join(key for key in holding if key != "product")
        lines.append(f"{holding.get('product', 'holding')}: {fields}")
    return "\n".join(lines)


def render(account: Mapping[str, Any]) -> str:
    """The account as a prompt block: masked id, then each holding's fields."""
    lines = [f"Account {account.get('masked_id', '')}"]
    for holding in account.get("holdings", []):
        lines.append("")
        lines.append(f"{holding.get('product', 'holding')}:")
        lines.extend(
            f"  {key}: {value}" for key, value in holding.items() if key != "product"
        )
    return "\n".join(lines)
