"""Structured tracing. Leaf module — imports nothing from app/, kb/, rag/.

Re-exporting here is safe in a way it is not in `core/__init__.py`: nothing
under this package imports a provider SDK, a database driver or settings, so
`from observability import Trace` pulls in the standard library and nothing else.
"""

from observability.logger import emit, render
from observability.trace import Stopwatch, Trace

__all__ = [
    "Stopwatch",
    "Trace",
    "emit",
    "render",
]
