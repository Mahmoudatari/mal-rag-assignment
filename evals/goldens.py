"""The eval question set — 20 questions, four per knowledge base document.

Drives two different evals from one set of questions:

- **Generation** (`test_grounding.py`) — DeepEval's `FaithfulnessMetric` and
  `AnswerRelevancyMetric` judge the answer against the chunks the pipeline
  actually retrieved. Neither needs a gold answer, which is what keeps this set
  cheap to maintain and is why the three `Contextual*` metrics are out of scope:
  they need `expected_output`, a hand-written answer per question that has to be
  re-reviewed whenever the corpus changes.
- **Retrieval** (`test_retrieval.py`) — did the right chunks come back at all?
  That is measured on chunk *ids*, which the DeepEval metrics never see: they are
  handed chunk text and have no way to tell a correct chunk from a plausible one
  that happens to support the same claim. It is also deterministic and free.

**`must_retrieve` holds anchors, not chunk ids, and that is the load-bearing
decision here.** `chunk_id` is positional (`murabaha-everyday-finance#027`), so
inserting a paragraph into a document shifts every id after it. A hardcoded id
would then point at neighbouring text — and because the neighbour is usually
from the same section and still plausible, the eval would go on passing while
asserting something that is no longer true. Anchors are verbatim strings the
document's author wrote, so they move *with* the content, and `resolve()` maps
them to ids at test time through the same `kb.chunking` the ingest uses. An
anchor that stops resolving raises instead of silently mismeasuring.

Every question is answerable from **one** document. A question needing two
products would be graded against a retrieval set that can only be half right, so
a low score would indict the corpus rather than the pipeline.

Questions are in customer voice and self-contained — each is asked as the first
turn of a fresh session, so a pronoun with no antecedent would be resolved by the
router against empty history. None contains PII: `test_pii_redaction.py` covers
that, and a name here would put the redactor in the middle of a grounding
measurement. None is out of scope: `test_refusal.py` covers that. No yes/no
questions — answer relevancy scores the proportion of relevant statements in the
answer, which is degenerate for a one-word reply.
"""

from dataclasses import dataclass
from typing import Literal

from kb.chunking import Chunk, load_corpus

Kind = Literal["lookup", "figure", "faq", "synthesis"]


class AnchorError(LookupError):
    """An anchor matched no chunk, or more than one.

    Fatal rather than skipped, for the same reason a bad rerank index is fatal:
    an anchor that quietly resolves to nothing turns a retrieval assertion into
    a no-op that keeps reporting success.
    """


@dataclass(frozen=True, slots=True)
class Golden:
    """One eval question, plus enough provenance to debug a failure.

    Deliberately not `deepeval.dataset.Golden`: that type carries dataset,
    tracing and Confident AI plumbing this set has no use for, and the test
    builds its own `LLMTestCase` anyway.
    """

    question: str
    doc: str
    kind: Kind

    #: Verbatim substrings, each unique within `doc`, identifying the chunks that
    #: must come back. A `## Heading` line works because `kb.chunking._render`
    #: re-attaches both headers into every chunk's text; the FAQ pairs need their
    #: whole-line bold question instead, since all 73 of them share the section
    #: name "Frequently Asked Questions".
    #:
    #: **The first entry is the primary anchor** — the one chunk that alone
    #: carries most of the answer. Retrieval is scored in two tiers because the
    #: pipeline is a funnel: all anchors must survive into the candidate set
    #: (`retrieve_candidates`, 20), but only the primary is required to survive
    #: reranking into `top_k` (4). Demanding all four anchors of a synthesis
    #: question inside a top-4 would be asserting that reranking is perfect
    #: rather than that it is working.
    must_retrieve: tuple[str, ...]

    #: The facts a grounded answer has to state, taken from the source document
    #: and spot-checked against it. **Nothing asserts on this** — faithfulness is
    #: measured against retrieved chunks, not against this string. It is here for
    #: triaging a failed run by eye, and as the starting point for
    #: `expected_output` if the contextual metrics are ever added.
    must_contain: str


def resolve(golden: Golden, corpus: list[Chunk] | None = None) -> tuple[str, ...]:
    """Anchors → chunk ids, in `must_retrieve` order so index 0 stays primary.

    Takes the corpus from `kb.chunking`, which is pure: no database, no network,
    no API key. That matters twice over — it means the id an anchor resolves to
    is by construction the id ingest wrote, and it means a broken anchor is
    caught by a test that runs in CI with no credentials.
    """
    corpus = corpus if corpus is not None else load_corpus()
    in_doc = [c for c in corpus if c.doc == golden.doc]
    if not in_doc:
        raise AnchorError(f"no chunks for document {golden.doc!r}")

    ids: list[str] = []
    for anchor in golden.must_retrieve:
        hits = [c for c in in_doc if anchor in c.text]
        if len(hits) != 1:
            raise AnchorError(
                f"anchor {anchor!r} matched {len(hits)} chunks in {golden.doc!r} "
                f"(expected exactly 1) — the document was probably edited"
            )
        ids.append(hits[0].chunk_id)
    return tuple(ids)


GOLDENS: tuple[Golden, ...] = (
    # --- ijara-auto-lease-to-own -------------------------------------------
    Golden(
        question=(
            "I'm thinking about applying for a car lease-to-own plan with Mal — "
            "what do I actually need to qualify, and how long does approval usually take?"
        ),
        doc="ijara-auto-lease-to-own",
        kind="lookup",
        must_retrieve=("## Eligibility and Limits",),
        must_contain=(
            "Age 21-65 at maturity; min income AED 8,000 salaried / AED 15,000 self-employed; "
            "valid UAE licence and Mal account; 20%/25% down payment (new/used); max finance "
            "AED 600,000; tenor 12-60 months; debt burden ratio capped at 50%; decision "
            "typically within four working hours."
        ),
    ),
    Golden(
        question=(
            "I'm looking at a AED 120,000 car over 48 months with the minimum 20% down "
            "payment — can you break down how my monthly rent works before and after the "
            "first six-month rate reset?"
        ),
        doc="ijara-auto-lease-to-own",
        kind="figure",
        must_retrieve=("## Worked Example: A 48-Month Rent Schedule",),
        must_contain=(
            "Fixed rent AED 2,000/month; first period 6.50% (4.25% benchmark + 2.25% margin) "
            "on AED 96,000 = AED 520 variable, AED 2,520 total; after reset 7.25% but "
            "outstanding cost falls to AED 84,000 = AED 507.50 variable, AED 2,507.50 total, "
            "so total rent falls despite the higher rate."
        ),
    ),
    Golden(
        question=(
            "I might need to drive my financed car over to Saudi Arabia for a trip — "
            "what do I need to arrange beforehand?"
        ),
        doc="ijara-auto-lease-to-own",
        kind="faq",
        must_retrieve=("**Can I drive the car to Saudi Arabia or Oman?**",),
        must_contain=(
            "Owner's NOC from Mal since Mal owns the vehicle; request at least five working "
            "days ahead; bilingual NOC valid up to 30 days, GCC only; AED 100 fee with first "
            "per 12 months free; Takaful covers UAE and Oman as standard, other GCC states "
            "need a border orange-card extension reimbursed by Mal."
        ),
    ),
    Golden(
        question=(
            "Why does Mal cover things like engine or gearbox repairs on my leased car, "
            "and how exactly do I end up owning the vehicle once I've paid everything off?"
        ),
        doc="ijara-auto-lease-to-own",
        kind="synthesis",
        must_retrieve=(
            "## Maintenance Responsibility Split",
            "## End of Term: Executing the Transfer Undertaking",
            "## Structure and Ownership",
            "## Why the Transfer Promise Is a Separate Undertaking",
        ),
        must_contain=(
            "Mal owns the vehicle throughout and bears ownership risk, hence it pays for "
            "major/structural maintenance and Takaful while the customer covers routine wear; "
            "transfer happens via a separate wa'd (gift deed or AED 100 token sale) executed "
            "only once final rent clears, with registration reissued within about five working days."
        ),
    ),
    # --- late-payment-and-charity-policy -----------------------------------
    Golden(
        question=(
            "I'm going through a rough patch financially and I'm worried I'll miss a payment "
            "or two on my Mal financing soon — what kind of support can I actually ask Mal "
            "for before things get worse?"
        ),
        doc="late-payment-and-charity-policy",
        kind="lookup",
        must_retrieve=("## Genuine Hardship Support",),
        must_contain=(
            "Early disclosure treated as cooperation, not default; three routes (deferral of "
            "1-2 instalments, restructuring to a longer tenor, payment holiday up to 3 months); "
            "no late payment charge for months covered by approved hardship; light evidence "
            "required; decision targeted within two working days."
        ),
    ),
    Golden(
        question=(
            "I have a Mal Everyday Murabaha plan with a total price of AED 24,000 paid in "
            "AED 1,000 monthly instalments, and I ended up paying instalment 7 fifteen days "
            "after it was due — how much extra did that cost me and where does that money go?"
        ),
        doc="late-payment-and-charity-policy",
        kind="figure",
        must_retrieve=("## Worked Example: One Missed Instalment",),
        must_contain=(
            "Flat AED 100 charge, not scaled to the 15-day delay; the debt itself does not "
            "grow; AED 1,100 paid, allocated AED 1,000 to the instalment first then AED 100 "
            "to the charge; the AED 100 is booked same-day to the Wujuh al-Khair charity "
            "account; resulting balance AED 17,000 across 17 instalments."
        ),
    ),
    Golden(
        question=(
            "My direct debit failed last month even though I had the funds ready — it just "
            "didn't go through properly on Mal's end. Will Mal still hit me with a late "
            "payment charge for that?"
        ),
        doc="late-payment-and-charity-policy",
        kind="faq",
        must_retrieve=(
            "**My direct debit failed and it was not my fault. Will Mal still charge me?**",
        ),
        must_contain=(
            "No charge where the miss is caused by Mal or the payment rail (failed "
            "presentment, mandate cancelled in error, outage, wrong due date, clearing "
            "disruption); nightly reconciliation reverses such charges automatically, "
            "normally within 24 hours; if the rejection code shows insufficient funds the "
            "charge stands but can be disputed with evidence."
        ),
    ),
    Golden(
        question=(
            "I'm going to be right at the edge of the 5-day grace period on my next Mal "
            "instalment — if I send an international bank transfer on day 5, will it reach "
            "Mal in time, and what happens on my account before any charge kicks in?"
        ),
        doc="late-payment-and-charity-policy",
        kind="synthesis",
        must_retrieve=(
            "## Grace Period and Notification Sequence",
            "## Payment Methods and How Quickly Each Clears",
        ),
        must_contain=(
            "5 calendar day grace period with daily app reminders and no charge on days 1-5; "
            "international transfers take 2-4 working days so one sent on day 5 will not "
            "clear in time; in-app one-tap payment or debit card credit immediately; flat "
            "AED 100 applies on day 6 if still unpaid."
        ),
    ),
    # --- murabaha-everyday-finance -----------------------------------------
    Golden(
        question=(
            "I don't really get how this works without interest — if Mal isn't lending me "
            "money, what exactly happens when I finance a purchase with Murabaha, and why "
            "don't I just get the cash myself?"
        ),
        doc="murabaha-everyday-finance",
        kind="lookup",
        must_retrieve=("## What Mal Everyday Murabaha Is",),
        must_contain=(
            "It is a sale, not a loan: Mal buys the specific chosen asset, takes ownership, "
            "then resells to the customer at cost plus a declared profit fixed in dirhams; "
            "the customer never receives cash, Mal pays the supplier directly, and the "
            "customer receives the asset."
        ),
    ),
    Golden(
        question=(
            "I financed a AED 12,000 laptop over 12 months with AED 1,080 profit added, and "
            "I want to settle the whole thing right after my fifth instalment — how much "
            "would I actually owe on that day?"
        ),
        doc="murabaha-everyday-finance",
        kind="figure",
        must_retrieve=("## Worked Example: Early Settlement with Ibra'",),
        must_contain=(
            "Outstanding sale price after 5 of 12 instalments of AED 1,090 is AED 13,080 − "
            "AED 5,450 = AED 7,630; remaining profit AED 630; ibra' at the current 60% rate "
            "is AED 378; final amount AED 7,252, with no early settlement fee; ibra' is "
            "discretionary, not a right."
        ),
    ),
    Golden(
        question=(
            "I want to buy something from a supplier that isn't one of Mal's regular partner "
            "merchants — can you still finance that purchase through Murabaha, and how would "
            "that work?"
        ),
        doc="murabaha-everyday-finance",
        kind="faq",
        must_retrieve=("**Can Mal ask me to buy the item myself?**",),
        must_contain=(
            "Mal can appoint the customer as its purchasing agent under a wakala; funds go to "
            "the supplier's account not the customer's; the invoice must name Mal as buyer; "
            "invoice and proof of delivery uploaded within 7 calendar days; self-dealing "
            "prohibited; agency purchases capped at AED 40,000."
        ),
    ),
    Golden(
        question=(
            "I've taken out Takaful cover on a big item I financed through Mal Everyday "
            "Murabaha — if something happened to me, would my family be left to keep paying "
            "it off, or would the Takaful cover handle it?"
        ),
        doc="murabaha-everyday-finance",
        kind="synthesis",
        must_retrieve=(
            "## Death, Disability and Loss of Employment",
            "## Takaful Cover on Financed Goods",
        ),
        must_contain=(
            "On death Mal freezes the contract on notification, stops collections and "
            "contacts no relative for 90 days, with the outstanding sale price a claim "
            "against the estate; where the asset carried Takaful, a valid total-loss claim "
            "pays Mal directly and the contract closes; absent that, balances under "
            "AED 25,000 are written off rather than pursued against heirs."
        ),
    ),
    # --- sukuk-fractional-investing ----------------------------------------
    Golden(
        question=(
            "I've been looking at Mal Fractional Sukuk and I'm not clear on the mechanics — "
            "once I buy a unit, who actually holds it on my behalf, and what happens to my "
            "investment if Mal as a company ever ran into trouble?"
        ),
        doc="sukuk-fractional-investing",
        kind="lookup",
        must_retrieve=("## Fractional Investing, Custody and Nominee Arrangements",),
        must_contain=(
            "Units registered to Mal Nominees (ADGM) Limited, held in an omnibus account with "
            "an independent third-party custodian and reconciled daily; cash in a segregated "
            "client money account; holdings would not form part of Mal's estate — but that "
            "protects only against Mal's failure, not against the sukuk falling in value or "
            "issuer default."
        ),
    ),
    Golden(
        question=(
            "Say I put AED 10,000 into a sukuk listing just before the 11:00 batch — what "
            "fees would come out of that, and roughly how many units would I end up owning?"
        ),
        doc="sukuk-fractional-investing",
        kind="figure",
        must_retrieve=(
            "## Worked Examples: A Mal Fractional Sukuk Subscription and Distribution",
        ),
        must_contain=(
            "0.35% platform fee = AED 35 charged in addition to order value, so AED 10,035 is "
            "ring-fenced at placement; the batch executes at 101.42 per 100 nominal giving "
            "9,859.99 nominal, rounded down to 9,859 units, with the AED 1.00 residual "
            "returned the same day."
        ),
    ),
    Golden(
        question=(
            "I might need to get some of my money back before my sukuk investment reaches "
            "maturity — what would selling early actually involve for me?"
        ),
        doc="sukuk-fractional-investing",
        kind="faq",
        must_retrieve=("**Can I sell my Mal Fractional Sukuk holding before maturity?**",),
        must_contain=(
            "Eligible listings can be sold in the in-app secondary window on UAE business "
            "days 10:00-15:00 GST, settling T+2; liquidity is best-efforts and may be "
            "suspended in stressed markets; Murabaha-structured listings are generally held "
            "to maturity and flagged reduced-liquidity."
        ),
    ),
    Golden(
        question=(
            "How does the way a sukuk is structured change what I'd actually get back if the "
            "company behind it ended up defaulting?"
        ),
        doc="sukuk-fractional-investing",
        kind="synthesis",
        must_retrieve=(
            "## Asset-Backed Versus Asset-Based Mal Fractional Sukuk",
            "## Default and Recovery",
        ),
        must_contain=(
            "Asset-backed (typically Ijara) gives recourse to the real asset, with recovery "
            "following its realisable value; asset-based (typically Wakala/Murabaha) has no "
            "true sale, so holders rank as unsecured creditors of the obligor and recovery "
            "may be materially lower or nil; either way distributions stop and value can fall "
            "to zero."
        ),
    ),
    # --- wakala-savings-deposits -------------------------------------------
    Golden(
        question=(
            "I keep hearing that my Wakala savings account isn't supposed to work like an "
            "interest-paying account. Can you explain why it isn't considered interest, and "
            "what that means for whether I'm guaranteed to get my original deposit back?"
        ),
        doc="wakala-savings-deposits",
        kind="lookup",
        must_retrieve=("## Why Mal Digital Wakala Is Not Interest",),
        must_contain=(
            "Wakala is an agency arrangement where the return comes from actual realised "
            "investment profit rather than a fixed sum owed on a loan, which would be riba; "
            "principal is not guaranteed, except where loss is caused by Mal's own negligence "
            "or breach."
        ),
    ),
    Golden(
        question=(
            "If I keep AED 40,000 in my Wakala Savings Plus account for a full 30-day month "
            "and the investment pool actually earns 4.85% per annum that month, how much "
            "profit will I see credited, and does that line up with the rate I was quoted?"
        ),
        doc="wakala-savings-deposits",
        kind="figure",
        must_retrieve=("## Worked Example: A Month of Profit on Wakala Savings",),
        must_contain=(
            "Gross pool profit AED 159.45; fixed agency fee AED 14.79 deducted; net before "
            "incentive AED 144.66; incentive fee AED 9.87 retained by Mal; AED 134.79 "
            "credited, matching the 4.10% anticipated rate rather than the higher realised "
            "return."
        ),
    ),
    Golden(
        question=(
            "I put money into a 12-month Wakala Fixed-Term deposit but now need to break it "
            "early after seven months. What rate will I actually be paid instead of the 5.05% "
            "I was originally quoted, and will I be charged extra for breaking it?"
        ),
        doc="wakala-savings-deposits",
        kind="faq",
        must_retrieve=(
            "**What do I lose if I break a 12-month Mal Digital Wakala Fixed-Term deposit early?**",
        ),
        must_contain=(
            "Settled at the next-shortest completed tenor's anticipated rate (6-month at "
            "4.60%) less the 0.50% p.a. early-termination adjustment, giving 4.10% p.a. over "
            "the period actually held; no additional breakage fee; funds returned to the "
            "Everyday Account within one business day; breaking is irreversible."
        ),
    ),
    Golden(
        question=(
            "I'm thinking about keeping around AED 30,000 in my Wakala Savings account to "
            "qualify for the higher Plus rate, but there might be a month where I need to "
            "withdraw some of it temporarily. How do the account tiers determine the profit "
            "rate I earn, and could a mid-month withdrawal cost me the higher rate?"
        ),
        doc="wakala-savings-deposits",
        kind="synthesis",
        must_retrieve=(
            "## Account Tiers",
            "## Profit Calculation, Accrual and Crediting",
        ),
        must_contain=(
            "Plus requires the average daily balance to cross AED 25,000 for a full calendar "
            "month, effective the following month, and downgrades on the same basis; profit "
            "is calculated on average daily balance, so a mid-month withdrawal that pulls the "
            "average below AED 25,000 settles that month at the Core rate."
        ),
    ),
)
