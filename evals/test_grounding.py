"""Eval: answers stay grounded in retrieved context.

DeepEval's `FaithfulnessMetric` is the assignment's "no hallucination beyond
retrieved context" written as a measurement: it decomposes the answer into
claims and checks that none of them contradicts or exceeds the passages the
pipeline retrieved. `AnswerRelevancyMetric` guards the other end — an answer
can be perfectly faithful by saying almost nothing, so relevancy scores how
much of it actually addresses the question asked.

Both are judged by a small LLM against the chunks *this turn* retrieved, which
is what keeps the eval set cheap: there are no hand-written gold answers to
re-review every time a document is edited. The price is that neither metric can
see chunk ids, so a turn that retrieved plausible-but-wrong passages and
answered faithfully from them scores well here — `test_retrieval.py` covers that
half, deterministically and for free.
"""

import os
from functools import lru_cache

import pytest

# Imported ahead of deepeval on purpose. `evals/conftest.py` sets
# DEEPEVAL_TELEMETRY_OPT_OUT, and deepeval acts on it at *import* time: its
# telemetry module creates a `.deepeval` directory and initialises Sentry as a
# module-level side effect. Under pytest the conftest is loaded before any test
# module, so this ordering only matters when the module is imported directly —
# which is exactly how the cheap import smoke test runs it.
from evals.conftest import live, needs_db

from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.models import OpenRouterModel
from deepeval.test_case import LLMTestCase

from core.config import get_settings
from evals.goldens import GOLDENS, Golden

# Two costs per test: the graph turn itself and the judge grading it. `live` is
# the registered marker that keeps both out of a default run; the two skipifs
# make an opt-in run degrade to a skip rather than an error when the key or the
# database URL is missing.
pytestmark = [pytest.mark.live, live, needs_db]


@lru_cache
def judge() -> OpenRouterModel:
    """The judge model, built once and shared by every metric instance.

    `OpenRouterModel`, never `GPTModel`: `GPTModel` matches the model name
    against a curated capability table, and a name it does not know falls back
    to a default entry with every capability flag `None` — the structured-output
    and json-mode gates then both fail and the judge silently degrades to
    un-schema'd calls with no cost tracking. `OpenRouterModel` has no such table.
    It takes the full OpenRouter slug, tries a strict JSON-schema
    `response_format` first with a parse fallback, and takes explicit per-token
    pricing, which is required here because OpenRouter reports `usage.cost` only
    when a request opts in and deepeval's gateway does not.

    The two prices are for the default model above them — override the env var
    and update both, or the reported spend is fiction. The knob is an
    environment variable rather than a `core/config.py` field because `core/`
    ships in the wheel and `evals/` does not: a shipped setting whose only
    consumer is unshipped code crosses the packaging boundary this repo polices.
    Key and base URL still come from `get_settings()`, so there is no second
    path to the secret.
    """
    s = get_settings()
    return OpenRouterModel(
        model=os.environ.get("EVAL_JUDGE_MODEL", "openai/gpt-5.6-luna"),
        api_key=s.openrouter_api_key,
        base_url=s.openrouter_base_url,
        cost_per_input_token=0.50e-6,
        cost_per_output_token=3.00e-6,
    )


@pytest.mark.parametrize("golden", GOLDENS, ids=lambda g: f"{g.doc}-{g.kind}")
def test_answer_is_grounded_in_retrieved_context(golden: Golden, run_golden) -> None:
    """Every claim in the answer is supported by the chunks it was written from."""
    final = run_golden(golden.question)

    # Deterministic assertions first, before a single judge token is spent: each
    # of these is a way the turn can have nothing gradeable to hand the metrics,
    # and finding that out from a free assert beats paying to be told.
    assert final["outcome"] == "answered", (
        f"outcome {final['outcome']!r} — these goldens are all drawn from the "
        f"corpus, so anything but an answer is a routing or grading regression"
    )
    assert final["answer"].strip(), "answered with empty text"
    # A real assertion, not a formality. `_citations` in `rag/nodes/generate.py`
    # builds this list from the [n] markers the model actually wrote, so an
    # answer that cites nothing produces `[]` while still counting as
    # "answered" — grounded-looking text with no stated provenance. (That the
    # cited ids are a subset of the retrieved ones is true by construction in
    # that same helper; asserting it would be a tautology.)
    assert final["references"], "answer carries no [n] citation markers"

    case = LLMTestCase(
        input=golden.question,
        actual_output=final["answer"],
        # Exactly what `generate` was given — the metrics judge the answer
        # against the pipeline's own retrieval, not against an ideal context.
        retrieval_context=[c["text"] for c in final["chunks"]],
    )

    # Fresh metric instances per case: metrics carry the score, reason and
    # extracted claims from their last `measure` call, so reusing one across
    # parametrised cases would grade a case against another's state. Only the
    # model is shared, and it is stateless.
    #
    # Sync throughout (`async_mode=False`, `run_async=False`) — this is plain
    # pytest with no anyio plugin, and deepeval's async path would want its own
    # event loop inside a test that is already running graph turns through
    # `asyncio.run`. Sequential judge calls cost seconds, not dollars.
    #
    # Thresholds: 0.8 faithfulness tolerates one flagged claim out of the
    # typical four to eight, which is roughly the judge's noise floor on the
    # "could not be confirmed from Mal's guides" disclaimers the answer prompt
    # mandates — they read like unsupported claims to a claim extractor. 0.7
    # relevancy leaves the same headroom, since those disclaimers are honest but
    # not directly responsive. Recalibrate only on evidence from the
    # `include_reason` output of a failing run, never to turn a red run green.
    assert_test(
        test_case=case,
        metrics=[
            FaithfulnessMetric(
                threshold=0.8, model=judge(), async_mode=False, include_reason=True
            ),
            AnswerRelevancyMetric(
                threshold=0.7, model=judge(), async_mode=False, include_reason=True
            ),
        ],
        run_async=False,
    )
