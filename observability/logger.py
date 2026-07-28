"""JSON trace emitter — one line per request, on stdout.

Deliberately not the `logging` module. A trace is a required deliverable rather
than diagnostics, and `logging` makes delivery conditional on configuration that
lives somewhere else: with no handler installed, `logger.info(...)` is silently
dropped, because `logging.lastResort` only handles WARNING and above. That
failure is invisible — the app serves fine and the traces simply never appear.
Writing to the stream directly has no such mode, needs no setup, and is what the
platform collects anyway (Railway captures stdout).

`settings.log_level` therefore governs the app's ordinary logging, not this.
"""

from __future__ import annotations

import json
import sys
from typing import TextIO

from observability.trace import Trace


def render(trace: Trace) -> str:
    """The trace as a single-line JSON string.

    `default=str` is a backstop, not a feature: every field of `Trace` is a
    primitive, so nothing should reach it. It is here so that an unserializable
    value degrades to its repr in the log instead of raising out of `emit` and
    taking a request that had already succeeded down with it.
    """
    return json.dumps(trace.to_dict(), separators=(",", ":"), default=str)


def emit(trace: Trace, stream: TextIO | None = None) -> str:
    """Write one JSON line and return it.

    Returned so callers and tests can assert on exactly what was written without
    capturing stdout.

    The line is built first and written in a single `write` call. On the async
    request path every `emit` runs on the event-loop thread, where two writes
    cannot interleave — but stdout is still shared with worker threads (the
    graph's sync nodes under `ainvoke`, uvicorn's own logging), and the
    single-write discipline costs nothing, so it stays. The write itself is a
    one-line buffered syscall: not worth a thread hop to keep off the loop.
    """
    line = render(trace)
    out = sys.stdout if stream is None else stream
    out.write(line + "\n")
    out.flush()
    return line
