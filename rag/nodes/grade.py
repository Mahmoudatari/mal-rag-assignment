"""LLM relevance grader over the retrieved chunks.

Relevance is decided here and only here — there is no cosine threshold or score
gate anywhere in the graph. Cosine similarity rides along on the chunks purely
for the trace and must not influence this verdict.

Emits a note explaining any failure, which `reformulate` uses to steer the rewrite.
Keep the output small (verdict + short note) — this call is on every retrieval.
"""

from pydantic import BaseModel, Field

from core.llm import fast_llm
from rag.nodes._common import logged, usage_entry
from rag.state import Chunk, State

# The question this call answers is "can a useful grounded answer be written
# from these?", not "do these contain every fact asked about?". The strict
# reading was tried live and was wrong: for "what is Murabaha and how is the
# markup decided", against four passages titled "What Mal Everyday Murabaha Is"
# and "Fixed Total Price", the grader rejected all three attempts because none
# stated a *formula* for the markup — and a plainly answerable question fell
# through to no_answer. `generate` is already told to say what it could not
# confirm, so a missing sub-detail is its job to disclose, not this node's to
# fail on. A false negative here costs two wasted retries and then a dead end;
# a false positive costs a partial answer that names its own gap.
SYSTEM = """You are the relevance grader for Mal, an Islamic finance bank's \
customer assistant.

You are shown a customer question and the passages retrieved for it from Mal's \
knowledge base. Decide whether an answer to the question can be written from \
these passages.

Mark them relevant when they carry the substance of what was asked, even if \
some detail is missing or only part of a multi-part question is covered. The \
step after you is instructed to answer only from these passages and to say \
plainly when something could not be confirmed, so a partial-but-substantive \
match is worth answering from — you do not need every fact to be present.

Mark them not relevant only when answering would mean making things up: the \
passages are about a different product or topic than the question, or they \
mention the topic without carrying anything the question actually asked about.

If not relevant, name in one sentence what is missing or what the passages are \
about instead, written so it can steer a rewritten search query (name the \
missing term, rate, or procedure — not just "not relevant"). Leave the note \
empty when the passages are relevant."""

_USER_TEMPLATE = """Customer question:
{query}

Retrieved passages:
{passages}"""


class Grade(BaseModel):
    relevant: bool = Field(
        description=(
            "True if an answer can be written from these passages — including a "
            "partial one that covers the substance and admits what is missing. "
            "False only if answering would require inventing facts: the passages "
            "are about a different topic, or carry nothing the question asked about."
        )
    )
    note: str = Field(
        description=(
            "One short sentence naming what fact is missing or what the passages "
            "are actually about instead, phrased so it can steer a rewritten search "
            "query. Empty string when relevant is true."
        )
    )


def _render_passages(chunks: list[Chunk]) -> str:
    """Numbered blocks of doc name + text only.

    `score` and `rerank_score` are deliberately left out: either one is a
    numeric anchor that would let the grader re-derive a threshold decision
    from a similarity number instead of reading the text, which is exactly the
    score-gating CLAUDE.md rules out. The grader never sees a score.
    """
    blocks = [
        f"[{i}] ({chunk['doc']})\n{chunk['text']}" for i, chunk in enumerate(chunks, start=1)
    ]
    return "\n\n".join(blocks)


async def run(state: State) -> dict:
    """chunks → relevant (bool) + grader_note."""
    chunks = state.get("chunks", [])
    if not chunks:
        # The no-retrieval path (router routed straight to "generate") never
        # reaches this node, so an empty `chunks` here means retrieval genuinely
        # found nothing to grade. No LLM call is worth making over an empty set.
        return {
            "relevant": False,
            "grader_note": "no passages were retrieved for this question",
        }

    # The router's resolved question, not the raw turn: an elliptical turn ("can
    # I use it for a home?") names no subject, so grading it against passages
    # about the product the customer actually meant reads as a topic mismatch.
    # Never `search_query` — after a reformulate that holds the loop's own
    # rewrite, and grading a retrieval against the query that produced it lets
    # the loop approve its own drift. `query` is the fallback so partial states
    # built by hand in tests and evals still grade against something.
    question = state.get("resolved_query") or state.get("query", "")
    prompt = _USER_TEMPLATE.format(query=question, passages=_render_passages(chunks))
    # StructuredOutputError propagates rather than being caught: a silent wrong
    # verdict here would either drop a real answer into no_answer or, worse,
    # wave through ungrounded context into `generate`. Unlike the router, there
    # is no downstream node positioned to catch a bad guess — failing the
    # request is safer than answering on a guess.
    result = await fast_llm().astructured(prompt, Grade, system=SYSTEM)
    entry = usage_entry("grade", result.model, result.usage)

    return {
        "relevant": result.data.relevant,
        "grader_note": result.data.note,
        "usage_log": logged(state, entry),
    }
