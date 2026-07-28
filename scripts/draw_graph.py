"""Render the compiled graph as a Mermaid diagram and a PNG.

Dev tooling, not product: lives in scripts/ (no __init__.py) precisely so it can
never be imported and is never shipped in the wheel — the diagram comes from the
same `build_graph()` the app runs, so it can't drift from the real wiring the
way a hand-drawn one would.

    uv run python scripts/draw_graph.py

Writes docs/graph.mmd and docs/graph.png. The Mermaid source is written first
and entirely locally; the PNG step posts that source to mermaid.ink (the
library's default renderer — the payload is node names and edges, nothing
else), so offline the .mmd still lands and the PNG step reports its failure.
"""

from pathlib import Path

from rag.graph import build_graph

DOCS = Path(__file__).resolve().parent.parent / "docs"


def main() -> None:
    DOCS.mkdir(exist_ok=True)
    # No checkpointer: persistence changes nothing about the topology, and the
    # bare compile keeps this runnable without Postgres.
    graph = build_graph().get_graph()

    mmd = DOCS / "graph.mmd"
    mmd.write_text(graph.draw_mermaid())
    print(f"wrote {mmd}")

    png = DOCS / "graph.png"
    try:
        png.write_bytes(graph.draw_mermaid_png())
    except Exception as error:  # noqa: BLE001 — renderer is remote; .mmd already saved
        print(f"PNG render failed ({error}); the Mermaid source in {mmd} is complete")
        raise SystemExit(1)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
