"""pgvector top-k similarity search over the Sharia knowledge base.

Re-entered on retry from `reformulate`; returned chunks REPLACE any previous
set rather than accumulating.
"""

from core.config import get_settings
from core.embeddings import embedding_client
from kb.store import asearch
from rag.nodes._common import logged, usage_entry
from rag.state import Chunk, State


async def run(state: State) -> dict:
    """search_query → chunks (with cosine similarity scores)."""
    query = state.get("search_query") or state.get("query", "")
    if not query.strip():
        # The router only sets search_query on the retrieve route, so an empty
        # query here means there is nothing to search for. No embedding call,
        # no search — grade sees empty chunks and drives the
        # reformulate/no_answer loop, which is what terminates this on
        # `attempts` rather than looping forever. This is also why
        # `aembed_query`'s own blank-text guard is effectively unreachable here.
        return {"chunks": []}

    # With reranking off there is no second stage to cut the candidate set
    # down, so pulling `retrieve_candidates` (20) would hand `generate` 20
    # passages instead of the configured `top_k` (4).
    settings = get_settings()
    limit = settings.retrieve_candidates if settings.rerank_enabled else settings.top_k

    emb = await embedding_client().aembed_query(query)
    matches = await asearch(emb.vector, limit=limit)

    # Match carries `section`; Chunk deliberately does not. It is the useless
    # string "Frequently Asked Questions" for 73 of the 202 chunks and citations
    # are built from chunk_id/doc/text alone.
    chunks: list[Chunk] = [
        Chunk(chunk_id=match.chunk_id, doc=match.doc, text=match.text, score=match.score)
        for match in matches
    ]

    entry = usage_entry("retrieve", emb.model, emb.usage)
    # Replaces rather than appends to any chunks already in state: on a
    # reformulate retry this node runs again with a new search_query, and the
    # previous (failed) candidate set must not linger alongside the new one.
    return {"chunks": chunks, "usage_log": logged(state, entry)}
