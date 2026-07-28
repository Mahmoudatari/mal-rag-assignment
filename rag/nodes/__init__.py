"""Graph nodes.

Every node is a function `run(state: State) -> dict` returning only the keys
it changed. The six that do IO (router, retrieve, rerank, grade, reformulate,
generate) are `async def` and await their clients; the three pure ones
(redact, refuse, no_answer) stay plain `def` — langgraph runs those in a
worker thread under `ainvoke`, so neither kind blocks the event loop. No
langgraph imports — wiring lives in rag/graph.py.
"""
