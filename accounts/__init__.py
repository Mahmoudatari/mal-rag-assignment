"""Synthetic customer account context — pure, like `pii/`.

No network, no database, no secrets: the account node and its tests import this
with nothing configured. Re-exports the whole surface, which `core/__init__`
deliberately does not do — the reason not to (a convenience import dragging the
OpenAI SDK in) does not exist here, and `pii/` set the precedent for pure leaf
packages.
"""

from accounts.render import product_names, render
from accounts.store import ACCOUNTS, Account, lookup

__all__ = ["ACCOUNTS", "Account", "lookup", "product_names", "render"]
