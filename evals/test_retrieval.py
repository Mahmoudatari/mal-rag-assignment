"""Eval: the right chunks come back — scored on ids, not on text.

The bonus eval `goldens.py` describes. It measures the half of the pipeline the
DeepEval metrics in `test_grounding.py` structurally cannot see: those are handed
the chunk *text* the pipeline retrieved and judge the answer against it, so a
turn that retrieved plausible-but-wrong passages and answered faithfully from
them scores well. Only chunk ids distinguish the correct chunk from a neighbour
that happens to support the same claim, and `resolve()` produces those ids from
the same `kb.chunking` ingest used — no hand-maintained id list to drift.

Scored in two tiers, because the pipeline is a funnel and the two stages fail
differently. Tier 1 asks whether the embedding search surfaced every anchor into
the candidate set (`retrieve_candidates`, 20) — a miss there is unrecoverable,
nothing downstream can rank a chunk it was never handed. Tier 2 asks whether the
cross-encoder kept the *primary* anchor inside `top_k` (4) — a miss there is
reranking mis-ordering a set that was correct on arrival. Requiring all four
anchors of a synthesis question inside a top-4 would assert that reranking is
perfect rather than that it works, so only the primary is held to it.

Deterministic and free once the turn has run: `run_golden` caches per question
across eval files, so a full `-m live` run pays one graph turn per golden no
matter how many files parametrize over GOLDENS. Every assertion here is set
membership over strings — no judge, no second opinion, no flake.
"""

import pytest

from core.config import get_settings
from evals.conftest import live, needs_db
from evals.goldens import GOLDENS, Golden, resolve
from kb.chunking import load_corpus

# Every test drives a real graph turn through `run_golden`: OpenRouter calls plus
# a populated index. `live` is the registered marker that keeps this out of a
# default run; the two skipifs make an opt-in run degrade to a skip rather than
# an error when the key or the database URL is missing.
pytestmark = [pytest.mark.live, live, needs_db]


@pytest.fixture(scope="module")
def corpus():
    """Chunked once — `load_corpus()` re-reads and re-splits all five documents."""
    return load_corpus()


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.doc}-{g.kind}")
def test_routes_to_retrieval(golden: Golden, run_golden):
    """Every golden is an in-scope question that must reach the index.

    Asserted on its own, ahead of the retrieval tiers, because a refused turn
    never writes `candidate_log` or `chunks` — without this the regression
    surfaces two tests later as an empty-list or KeyError failure that reads like
    a retrieval bug. A golden that gets refused is a router regression: these are
    all Mal product questions drawn from the corpus itself, so `refuse` is wrong
    by construction. See CLAUDE.md — `refuse` is for turns with nothing to do
    with Islamic finance or Mal at all.
    """
    final = run_golden(golden.question)
    assert final["route"] == "retrieve", (
        f"routed to {final['route']!r}: {final.get('route_reason', '')}"
    )


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.doc}-{g.kind}")
def test_all_anchors_in_candidate_set(golden: Golden, corpus, run_golden):
    """Tier 1: vector search surfaced every anchor into the first candidate set.

    Index `[0]` is the first retrieval pass, deliberately. A reformulate retry
    rewrites the query, so a later pass measures the grader's note and the
    rewrite rather than how well the index answers the router's rendering of the
    golden's actual question. There is intentionally no assertion that only one
    pass ran: whether the grader accepts the first set is a judgement call that
    varies run to run, and pinning the retry count would make this flaky over
    behaviour it is not measuring.
    """
    final = run_golden(golden.question)
    expected = set(resolve(golden, corpus))
    first_pass = final["candidate_log"][0]

    # Catches two failures at once: an index that silently lost rows (a partial
    # ingest returns fewer candidates and everything downstream still works), and
    # the degenerate configuration CLAUDE.md warns about — candidates == top_k,
    # where reranking permutes a set it was supposed to be selecting from and
    # tier 1 collapses into tier 2.
    assert len(first_pass) == get_settings().retrieve_candidates

    missing = expected - set(first_pass)
    assert not missing, f"anchors never reached rerank: {sorted(missing)}"


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.doc}-{g.kind}")
def test_primary_survives_rerank(golden: Golden, corpus, run_golden):
    """Tier 2: the primary anchor is in the top_k `generate` actually consumed.

    Read off final `chunks` rather than a rerank-specific log on purpose: this is
    the set the answer was written from and the citations point at, which is the
    outcome-relevant question even when a retry replaced an earlier set. Tier 1
    already pins where the *first* pass landed, so the two together distinguish
    "search missed it" from "rerank dropped it".
    """
    final = run_golden(golden.question)
    primary = resolve(golden, corpus)[0]
    chunks = final["chunks"]

    assert len(chunks) == get_settings().top_k

    # `rag/nodes/rerank.py` fails open on a RerankError, returning the candidates
    # in cosine order truncated to top_k. That path leaves `rerank_score` unset,
    # which is the only signal separating it from a successful rerank — without
    # this the tier-2 result is a cosine top-4 wearing a cross-encoder's name.
    unscored = [c["chunk_id"] for c in chunks if "rerank_score" not in c]
    assert not unscored, f"rerank failed open, cosine order only: {unscored}"

    assert primary in {c["chunk_id"] for c in chunks}, (
        f"primary anchor {primary} dropped by rerank; kept "
        f"{[(c['chunk_id'], round(c['rerank_score'], 3)) for c in chunks]}"
    )
