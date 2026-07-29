# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Production-ready AI assistant for **Mal** (bank) that answers Islamic finance questions grounded in a Sharia finance knowledge base, plus the customer's own account context. Assignment deliverable for a Senior AI Solutions Engineer role.

Scaffolded; modules are stubs. Everything below is the target spec.

## Layout

```
core/             config.py + llm.py + embeddings.py + rerank.py — shared, imports nothing internal
app/              FastAPI — routes, schemas, session id
kb/               documents/ + chunking + ingest.py + store.py + schema.sql   (build-time)
rag/              LangGraph graph, state, nodes/                              (request-time)
pii/              detector + masker + patterns — pure, no network or secrets
accounts/         synthetic customer records + prompt renderer — pure, like pii/
observability/    trace schema + JSON emitter — leaf, imports nothing internal
tests/            unit tests per module        (not shipped)
evals/            the 3 required tests         (not shipped)
```

Everything importable is a package. **Nothing lives at the repo root** — a root-level
module is not in `[tool.hatch.build.targets.wheel] packages`, so it is silently absent
from the wheel, and an editable install hides that by putting the root on `sys.path`.
`config.py` sat there and the built wheel was broken accordingly. Adding a package
means adding it to that list.

Boundaries follow dependency profile, not tidiness:
- `core/` is imported by `app/`, `kb/`, `rag/` and `pii/`, and imports none of them back. That direction is the entry test, not "shared-ish" — a module that needs one of those is not core.
- **`core/__init__.py` deliberately re-exports nothing.** A convenience `from core import get_settings` would make importing settings drag `openai` in behind it, and `pii/` reads settings while having to stay importable with no key and no network. Import the module: `from core.config import get_settings`.
- `pii/` must stay importable without an API key or a database, so its eval runs standalone in CI. The detector drops PERSON spans made up **entirely** of brand vocabulary: spaCy tags branded product names — "Mal Digital Wakala", "Mal Everyday Murabaha" — as PERSON, intermittently by context. Every token must be in the vocabulary, not merely one of them — a contains-"Mal" test drops "Ahmed Mal Hassan", and PERSON has no pattern recognizer behind it, so a dropped span is an unrecoverable leak rather than a lower score. Scoped to PERSON only, because account numbers are literally `MAL-nnnn-nnnn-nnnn`; pattern-matched kinds mask regardless of content.
- **`observability/` is imported by `app/` alone** — not by `rag/`, and it imports nothing internal at all, not even `core`. It was going to be a shared import until `usage_log` (see **State**) made nodes record into state rather than call a tracer, so the whole trace is now assembled at the edge from final state. `Trace.from_state` takes a `Mapping`, not `rag.state.State`: a TypedDict is a `dict` at runtime, so the schema is honoured with no import and no cycle. The price is a coupling by key name that no type checker sees, so a test ast-scans the `state.get("…")` literals against `State.__annotations__` — otherwise a renamed state key leaves a trace field silently reading its default.
- `core/llm.py` and `core/embeddings.py` are shared rather than owned by a caller because both `rag/` (request-time) and `kb/` (build-time embeddings) call OpenRouter — putting either inside one would make the other import across the boundary. They hold no prompts; those belong to the node that owns them.
- They are two modules, not one class, because the calls have different shapes — embeddings have no messages, system prompt, schema or temperature, but do have a dimension contract and batching. `core/embeddings.py` imports `Model`, `Usage` and the pooled transport from `core/llm.py` rather than duplicating them, so one trace can total tokens and cost across chat and embedding calls.
- `embedding_model` is the one setting ingest and retrieval must agree on *silently*: mismatched models put documents and queries in different vector spaces, so search degrades to noise with no error. Dimension and table name mismatches fail loudly, and chunking is not configurable at all (see **Chunking**). Both sides take their client from `embedding_client()` — a single accessor means no caller is in a position to pick a different model.
- **`encoding_format="float"` is pinned on every embedding request.** The `openai` SDK silently substitutes `base64` when the caller omits it, and OpenRouter's upstreams for `google/gemini-embedding-001` do not all accept it. The failure is **intermittent, not a clean per-provider split**: 12 identical live calls returned 5 successes and 7 replies carrying `error: "Google AI Studio embeddings do not support base64 encoding_format"`. `float` measured 12/12 across both upstreams OpenRouter attributed ("Google" and "Google AI Studio"). Untreated, an ingest dies part-way through the corpus and then succeeds on retry.
- **This is embeddings-only — `core/llm.py` is not affected, and that was checked rather than assumed.** The base64 substitution lives in the SDK's `resources/embeddings.py` and `encoding_format` appears nowhere else in it; capturing the chat request on the wire shows only `messages`, `model`, `temperature`, `usage` and `response_format` — no injected defaults. Strict `response_format` itself is the one chat param an upstream could reject, so it was sampled too: 12/12 valid strict JSON across both upstreams. Re-check this if the answering or fast model changes; the guarantee is per-model, not general.
- OpenRouter returns some upstream rejections as **HTTP 200 with an `error` object**, which the SDK does not raise on. `core/embeddings.py` surfaces that message; without it the failure reads as "no embeddings" with no cause.

## Commands

Package manager is **uv** — never pip.

```bash
uv sync                       # install
uv run uvicorn app.main:app --reload
uv run python -m kb.ingest    # build the index
uv run pytest                 # unit tests + evals (live and db deselected)
uv run pytest tests/          # unit tests only
uv run pytest evals/          # the 3 assignment deliverables
uv run pytest -m live         # opt-in: real OpenRouter calls, costs money
uv run pytest evals/test_pii_redaction.py::test_name -x   # single test

# opt-in: writes and deletes rows, so never point this at DATABASE_URL
docker run -d --rm -p 55432:5432 -e POSTGRES_PASSWORD=pg -e POSTGRES_DB=maltest \
    pgvector/pgvector:pg17
TEST_DATABASE_URL=postgresql://postgres:pg@localhost:55432/maltest uv run pytest -m db
```

`tests/` and `evals/` are separate suites: unit tests target one module with
fakes, evals assert the behaviour the brief grades. Anything hitting a real
provider is marked `live` and deselected by default via `addopts`.

## Graph

```
redact        PII → masked. Pure, always first. Everything downstream reads state["query"].
account       account_id → the customer's synthetic record from accounts/, or None.
              Pure lookup, always writes — a skipped write would let the
              checkpointer carry last turn's account into a turn that sent no id.
router        ONE structured call, routes only — never answers:
              scope check + retrieval decision + history-resolved query
              → retrieve | refuse | answer
retrieve      pgvector similarity → `retrieve_candidates` (20), not top_k
rerank        cohere/rerank-v3.5 via OpenRouter → reorder, keep top_k (4)
grade         LLM grader → relevant | not, with a note explaining why
reformulate   retry-only rewrite using the grader note; owns `attempts` (max 2 retries)
generate      answers with or without context; `chunks` empty ⇒ no-retrieval path
refuse        terminal: "I can only answer Islamic finance / account questions"
no_answer     terminal: not in the KB → point at support (obviously-fake contact details)
```

Edges: `redact` → `account` → `router` · `router` → retrieve | refuse | generate · `retrieve` → rerank → grade · `grade` → generate (relevant) | reformulate (retries left) | no_answer (exhausted) · `reformulate` → retrieve.

- **Account context is a node, not app-side injection.** `POST /chat` takes an optional `account_id` (`MAL-nnnn-nnnn-nnnn`, validated at the edge); the app puts it in the invoke payload — always, `""` when absent — and the `account` node resolves it against `accounts/`, a pure synthetic store. The record holds *customer* facts only (contract figures, balances, arrears), never product rules — those live in the KB, and duplicating them is how a record contradicts the passage beside it in one prompt. Figures reuse the KB documents' own worked examples so combined answers stay arithmetically consistent. The full account number exists only as the store's dict key: records carry `masked_id`, so prompts, traces and responses cannot leak the number by construction. `generate` renders the whole record (facts stated without `[n]` — citations are for passages); the router gets one line of product names to resolve "my lease"; an unknown id is `account: None`, not a 404 — an unauthenticated endpoint should not confirm an account exists.

- **The router routes, it does not answer.** Folding generation in would save a call only on the cheapest path while making the hottest call carry generation instructions every request.
- **No separate `direct` node.** `generate` takes retrieved context as a parameter and receives an empty list on the no-retrieval path. When `chunks` is empty the prompt must forbid substantive finance claims — that path is the main hallucination surface, since nothing grounds it.
- Router bias: when uncertain prefer `retrieve` over `refuse` — a false refusal stonewalls a real customer question, a false retrieve is caught by the grader. **`refuse` is narrower than it first reads, and this was measured rather than assumed:** the first live run refused "how do I dispute a fraudulent card transaction on my Mal account?", "can I get a home mortgage from Mal?" and an FX-rate question — all Mal banking, none of them one of the six KB topics. A customer with a real problem was told it was not this assistant's business. Anything about the customer's account or Mal's services routes to `retrieve`; if the KB does not cover it the grader says so and `no_answer` hands over to support, which is a real answer. `refuse` is for turns with nothing to do with Islamic finance or Mal at all — trivia, chit-chat, "write me a Python script". Pinned by live tests.
- Citations are inline `[n]` markers plus a structured `references` list (`{n, doc, chunk_id}`). No markdown anchors — there is no frontend to render them. Keep the display-index → chunk_id mapping in state so traces log real chunk IDs.
- **History stores answers with the `[n]` markers stripped** (the response keeps them). Markers number *this* turn's passages; replayed into the next turn's prompt they sit beside renumbered passages, and a restated fact carrying its stale marker gets resolved against the new chunk list — measured live: a Sukuk figure cited to a Wakala chunk containing neither the number nor the product. Nothing raises, and the trace logs the wrong chunk as genuinely cited.
- **No follow-up questions** in answers (dropped: ungrounded text inside a grounded answer complicates the grounding eval).
- **Relevance is decided by the LLM grader alone — there is no cosine threshold and no score gate.** A numeric cutoff was considered and rejected: cosine thresholds don't transfer across embedding models and would need recalibrating whenever one changes. There is deliberately no `relevance_threshold` setting.
- Cosine similarity is still computed and carried on each chunk, because "relevance score" is a required **trace** field. It is observability only and must never influence control flow. Log top-1 similarity as the numeric score alongside the grader's verdict.
- With 202 chunks over five products, retrieval almost always returns *something* mediocre. Out-of-scope rejection comes from the **router**; the grader's real job is catching in-scope-but-unanswerable questions.
- **The grader asks "can an answer be written from these?", not "do these contain every fact asked about?"** The strict reading was written first and failed live: for "what is Murabaha and how is the markup decided", against four passages titled *What Mal Everyday Murabaha Is* and *Fixed Total Price* (rerank 0.74, all from the right document), it returned not-relevant three times and dead-ended a plainly answerable question at `no_answer`. A multi-part question fails whenever any part lacks an explicit rule. `generate` is already instructed to answer only from the passages and to say what it could not confirm, so a missing sub-detail is its job to disclose — and it does: the live two-turn run answers "whether Murabaha is available to self-employed customers could not be confirmed from Mal's guides" and then cites the eligibility rules that are present. **Loosening it did not make it a rubber stamp** — it still rejects a home-mortgage question against the goods-and-travel Murabaha passages retrieval returns for it. Both directions are pinned by live tests; recalibrate them together, never one alone.

## State

One state object: `rag/state.py` is a plain TypedDict with no LangGraph imports. `StateGraph(State)` consumes it directly and the checkpointer serializes it per `thread_id` — there is no second, framework-side state to mirror.

- Nodes take `State` and return a **partial dict** of changed keys only.
- No reducer annotations. Default last-write-wins is correct here: a retry must *replace* `chunks`, not append to the stale set. If accumulation is ever needed, use `operator.add` before reaching for `add_messages`.
- `attempts` is incremented in `reformulate` and nowhere else. Split increments are how these loops go infinite.
- `pii_spans` stores kind and offsets only — never the matched values.
- **`resolved_query` is the router's history-resolved standalone question, written once per turn and never touched by `reformulate`.** It is what `grade` and `reformulate` judge against while `search_query` mutates per retry. Pointing them at `search_query` instead would let the retry loop grade its own rewrites — measured live before the split: on the follow-up "can I use it for a home?" (no subject in the raw turn), rewrite 2 had silently switched product from Murabaha to Ijara. Pointing them at `query` is the original bug: the raw turn's pronoun is unresolved and neither node gets history.
- **`redact` resets the turn-scoped keys, because partial writes plus a checkpointer means turn N+1 starts from turn N's dict.** Nothing else clears them, and the two failures are silent: a conversational second turn reaches `generate` still carrying the first turn's `chunks` and is answered as if grounded — the ungrounded path is the main hallucination surface — and a turn following an exhausted retrieval starts at `attempts == 2` and gets no retries at all. It resets `search_query`, `resolved_query`, `chunks`, `candidate_log`, `relevant`, `grader_note`, `attempts`, `tried_queries`, `references`, `usage_log`, and builds those lists per call so no two turns share one. `route`, `answer` and `outcome` are excluded on purpose: every path writes them. Tested in `tests/test_graph.py`, which is the only place a two-turn property is visible.
- **`account_id` and `account` stay out of `redact`'s reset list on purpose.** The id is in every invoke payload (`""` when the request had none) and the account node writes `account` on every path, None included — always-written keys need no reset, and adding them would be a second mechanism doing the same job. `account_id` is a lookup input like `raw_query`: nothing renders it into a prompt, trace or response; those read the record's `masked_id`.
- **`history` is the one key that survives a turn** — plain `{role, content}` dicts, masked text only, capped at `history_max_messages`. It lives in state rather than beside it because the checkpointer persists exactly this dict per `thread_id`; a second store would be a second thing to keep in sync. Appended by the three terminal nodes, read by `router` (to resolve "is it halal?" into a standalone `search_query`) and `generate`.
- **`usage_log` carries the trace's cost data as plain JSON, appended by whichever node made the call.** Nodes record into state instead of importing a tracer, which keeps `rag/` from depending on `observability/` and keeps the totals correct across a retry loop — a retry appends a second `retrieve` entry rather than overwriting the first. Rerank entries carry `search_units`, since `cohere/rerank-v3.5` reports zero tokens and bills on units alone.

## Chunking

`kb/chunking.py` splits on markdown structure, never on length. **202 chunks** from
five documents: 129 whole `##` sections plus 73 FAQ answers, one per question.

- The corpus is hand-written and its `##` sections are already the right size —
  135 sections, median 298 tokens, p90 414. A character window fixes a problem
  they don't have: at 800 chars it produces 359 fragments, 102 of them under 100
  tokens, cutting through rent tables and worked examples.
- The one exception is the FAQ blocks, which run 1091–1559 tokens. Each Q&A pair
  inside them is atomic, so they split one pair per chunk. Max chunk drops from
  1559 to 482 tokens and every boundary is one the author wrote.
- The FAQ split keys on **whole-line bold** (`^\*\*…\*\*$`), the corpus's question
  marker, not on the header text — murabaha has two FAQs named "… — Servicing,
  Problems and Escalation" and "… — Products, Agency and Life Events" that a
  title match would silently miss. Two markers minimum, so a section using one
  bold line as emphasis stays whole.
- **Every chunk carries both `#` and `##` headers**, re-attached by `_render` with
  `strip_headers=True`. The splitter only ever inlines the header that opens a
  section, so the `#` title is never in the body — and all five documents have a
  section called "Frequently Asked Questions", which makes the title the only
  thing telling a 52-token Q&A pair which product it is about. Doing this by hand
  rather than via `strip_headers=False` keeps one code path instead of three
  (preamble / section / Q&A pairs 2..n each carry the `##` differently).
- **There are no chunking settings** — no `chunk_size`, no `chunk_overlap`, nothing
  splits on length. `MAX_CHUNK_CHARS` is a build-time backstop, not a knob: it
  raises at ingest on a section too big to embed and with no question markers to
  split on. Measured in characters at a pessimistic 3 chars/token because there
  is no local tokenizer for `google/gemini-embedding-001` — a token count here
  would be `tiktoken` estimating a different tokenizer.
- `chunk_id` is positional (`murabaha-everyday-finance#027`), so editing a document
  shifts every id after the edit. Ingest replaces a document's rows wholesale
  rather than upserting one at a time.

## Store

`kb/schema.sql` is the schema; `kb/store.py` is the only module that runs SQL
against it. Two tables — `sharia_documents` (5 rows) and `sharia_chunks` (202).

- **The documents table is not normalisation for its own sake.** Its only
  non-derivable column is `title`; it earns its place on `content_hash` +
  `embedding_model`, which together are the re-ingest skip decision, and on the
  `ON DELETE CASCADE` that makes wholesale replacement one statement.
- **`content_hash` is over the rendered chunk texts, not the source file.** The
  file is only one of the two inputs — change `kb/chunking.py` and every stored
  chunk is wrong while the file is byte-identical. Chunk ids are folded in too,
  so a pure reordering counts: they are the citation keys, so their meaning
  moved even if no text did.
- **`embedding_model` is stored because a hash match alone is not freshness.**
  Same text embedded by a different model is the one failure that raises
  nothing anywhere — different vector space, no error, retrieval quietly turns
  to noise. `IndexStats.foreign_models()` is what `/health` should check.
- **`embedding_dimensions` is deliberately *not* stored.** The `vector(1536)`
  column rejects a wrong-width vector at INSERT. Only record what would
  otherwise fail silently.
- **No indexes beyond the primary keys**, extending the ANN decision under
  **Models** to the ordinary ones: no btree on `sharia_chunks.doc` either. It
  would serve the FK's reverse check on parent DELETE — five sequential scans of
  a 202-row table per ingest, which is not a cost.
- **`apply_schema()` does not use the pool.** `register_vector` reads the
  `vector` type's OID out of the database, so it cannot run before
  `CREATE EXTENSION`; the pool's `configure` hook would fail on a virgin
  database and succeed on the retry. Schema application takes its own one-shot
  connection, and the pool is only ever touched afterwards.
- **There is no `kb_table` setting** (removed). A `.sql` file cannot take a
  parameter for an identifier, so the setting could only disagree with the file
  it names — a knob that changes the queries but not the schema they run
  against. The names are constants in `kb/store.py` and a test asserts they
  match `schema.sql`.
- Schema is create-only (`IF NOT EXISTS` throughout), not a migration chain.
  Alembic starts earning its keep the first time a column has to change under
  data worth keeping; until then this is one file you can read end to end.
  LangGraph's checkpointer creates its own tables via its `setup()` and is
  deliberately not managed here.
- **Two pools in code, one per process kind.** The server opens `async_pool()`
  alone and hands it to the LangGraph checkpointer — `AsyncPostgresSaver`
  requires its connections `autocommit` + `dict_row` + `prepare_threshold=0`,
  so those are pool kwargs. Ingest, the CLI and the `-m db` tests open the sync
  `pool()`. The async pool serves reads only (`asearch`, `astats`):
  `replace_document`/`prune` rely on the sync pool's implicit transaction for
  atomicity, which pool-wide autocommit would silently remove.
  `db_pool_max_size` (4) bounds whichever pool the process opens.
- Tests that write are marked `db` and take `TEST_DATABASE_URL`, never
  `DATABASE_URL`: they call `prune()` and `replace_document()`, and the deployed
  index is one env var away.

## Models

All LLM and embedding calls go through **OpenRouter**, which is OpenAI-compatible — use the `openai` SDK pointed at `openrouter_base_url`. One provider, one key.

| Purpose | Model |
|---|---|
| Answering | `google/gemini-3.6-flash` |
| Routing + grading | `google/gemini-3.5-flash-lite` |
| Embeddings | `google/gemini-embedding-001` @ 1536 dims |
| Reranking | `cohere/rerank-v3.5` |

- **Two tiers is deliberate.** Router and grader run on every request and emit short structured output, so they take the cheap model. Only `generate` gets the better one. Preserve this split — it's the main lever on the "keep LLM costs minimal" constraint.
- Nodes get their client from `llm.fast_llm()` or `llm.answer_llm()` — never by constructing one with a hardcoded model id, which is what would quietly collapse the two tiers into one. `LLMClient.astructured(prompt, Schema)` is the call for the router and grader: it sends a strict JSON schema and returns a validated Pydantic model or raises. Every call has a sync twin (`structured`/`complete`) for build-time work and plain tests; nodes use the async side. Both helpers are cached, and clients on the same endpoint share one connection pool per transport.
- Embeddings default to 3072 dims; pass `dimensions=1536` (OpenAI-compatible parameter, forwarded by OpenRouter). 1536 is chosen because pgvector can only index `vector` columns up to 2000 — `halfvec` reaches 4000, but staying under 2000 keeps the ordinary type.
- Truncation is safe: the model is Matryoshka-trained, so any prefix is a valid embedding. Google notes vectors below 3072 aren't re-normalised, which **does not matter here** because retrieval uses cosine (`<=>`), which is magnitude-invariant. It would matter for inner product (`<#>`) or L2.
- Assert the returned vector length equals `embedding_dimensions` at ingest, so a silent provider-side change fails loudly rather than corrupting the index.
- At 202 chunks, **don't add an ANN index** — exact scan over 202×1536 floats is sub-millisecond and strictly more accurate. The dimension choice just keeps the option open.

## Reranking

`POST https://openrouter.ai/api/v1/rerank` — separate endpoint from chat and embeddings, same key.

```
request   { model, query, documents[], top_n }
response  { id, model, provider,
            results[{ index, relevance_score, document{text} }],
            usage{ search_units, total_tokens, cost } }
```

`core/rerank.py` is the client — `RerankClient(provider, name)`, same shape as its
two siblings. Two things about it are load-bearing:

- **The echoed `document` is ignored; only `index` is read.** Results come back
  sorted by score, each carrying its position in the input. Reranking a `Chunk`
  and reading the echo back would hand you *text* and lose `chunk_id` and `doc`
  — the two fields citations are built from. `result.apply(chunks)` reorders the
  caller's own objects instead, and `scored(chunks)` pairs each with its score
  for the trace. Out-of-range and duplicate indices are fatal: unchecked, a bad
  index is an `IndexError` at best and a citation pointing at the wrong chunk at
  worst, which raises nothing and reads as plausible.
- **It goes through the `openai` SDK, not a second HTTP stack.** `/rerank` is not
  an OpenAI endpoint, so there is no SDK method for it — but `client.post(path,
  body=..., cast_to=object)` reaches it while keeping the pooled connections, the
  bearer auth, the timeout and the SDK's 429/5xx backoff. `cast_to=object` rather
  than a Pydantic model, so the 200-with-`error` replies stay readable instead of
  raising a validation error that loses the upstream's message.

Measured live: `cohere/rerank-v3.5` bills `search_units: 1, total_tokens: 0` —
**tokens are always zero and cost rides on search units**, which is why
`RerankUsage` is its own type with `as_usage()` to fold into the shared `Usage`.
A trace that only logged tokens would report reranking as free.

- **Two-stage retrieval is the whole point.** `retrieve` returns `retrieve_candidates` (20), `rerank` cuts to `top_k` (4). If those two numbers are ever equal, reranking does nothing but permute — that's the failure mode to watch for.
- Rerank's `relevance_score` is a better value for the trace's required relevance score than raw cosine, since a cross-encoder actually reads the query against the document. Log both, and record `usage.cost` next to token usage.
- Reranking **must not gate control flow** — same rule as cosine. Relevance stays the LLM grader's decision.
- Toggle with `rerank_enabled`; when off, `retrieve` should return `top_k` directly and the node passes through.
- Honest expectation: with a 202-chunk KB, pulling 20 candidates means reranking ~10% of the corpus. That is a real first stage rather than a formality, but the ceiling is still low — the corpus is five documents on five products, so the candidate set for a given question is rarely contested. It's kept because it's cheap, it's the correct production shape, and it improves the trace.

## Tracing

One JSON line per request, on stdout. `observability/trace.py` is the record,
`observability/logger.py` writes it, and `app/` is the only caller: `Stopwatch()`
before `ainvoke`, `Trace.from_state(final, latency_ms=watch.ms)` after, `emit()`
— in a `finally`, so the request that crashed is the one guaranteed a record.

- **Deliberately not the `logging` module.** With no handler installed
  `logger.info(...)` is dropped, because `logging.lastResort` only handles
  WARNING and above — and the failure is invisible, since the app serves fine
  and the traces simply never appear. A trace is a required deliverable, not
  diagnostics, so delivery must not depend on configuration living elsewhere.
  `log_level` governs the app's ordinary logging, not this. The line is built
  first and written in one `write` call: emits run on the event-loop thread,
  where writes cannot interleave, but stdout is still shared with worker
  threads (the sync nodes under `ainvoke`, uvicorn's own logging) and the
  discipline costs nothing.
- **Every field defaults and `from_state` reads with `.get` throughout.** The
  record has to come out for a turn that refused, exhausted its retries, or died
  before `redact` — a tracer that raises on a half-finished request destroys the
  evidence for exactly the requests worth investigating.
- The four the brief names are `latency_ms`, `chunk_ids`, `total_tokens` and
  `relevance_score`; a test asserts each by name, since a rename is how a
  deliverable quietly disappears. The rest — route and reason, attempts,
  outcome, the grader's verdict — is what makes a bad answer explicable.
- **`relevance_score` is the top-ranked chunk's cross-encoder score, falling
  back to its cosine.** Both underlying values are logged separately:
  `rerank_score` of `null` is how the trace shows reranking was off or failed
  open, which a single conflated number would hide.
- **`calls` is `usage_log` verbatim, next to the totals.** Per-node cost is the
  only thing that shows a retrieval loop ran twice; and rerank's entry carries
  `search_units` because a total in tokens alone prices it at zero.
- **No PII by construction, not by filtering.** The record takes `query`, never
  `raw_query`, and `pii_kinds` holds entity kinds with no offsets and no values.
  The eval builds a trace through the real redactor and greps the emitted line
  for the identifier.

## LangGraph reference

Do not implement LangGraph APIs from memory — the API moves and this repo pins `langgraph==1.2.9`. When unsure about state schemas, reducers, conditional edges, checkpointer setup, or streaming:

1. Fetch `https://docs.langchain.com/llms.txt`
2. Find the section matching the question and follow its link to the real page

Verify against the docs before writing wiring code, and re-verify if a compile or runtime error looks like an API mismatch.

## Open items

See `TASKS.md` for the session-by-session build plan and dependencies. Account
context is resolved (see **Graph**): a dedicated `account` node between
`redact` and `router`, fed by an optional `account_id` request field.

## Decisions

- **Postgres + pgvector** for the vector store, chosen because stateful `/chat` needs a database anyway — one instance backs both retrieval and the LangGraph checkpointer.
- **LangGraph is wiring only.** `rag/graph.py` is the single file that imports it; nodes and state stay framework-free so evals never compile a graph and the framework could be swapped in one file. It earns its place through the `reformulate → retrieve → grade` retry *cycle* plus the router's three-way branch — and through the Postgres checkpointer, which is the real purchase. A hand-rolled pipeline class was considered; it loses on session persistence, which is the tedious part to build correctly.
- **The request path is async end to end.** `/chat` awaits `graph.ainvoke`; the six IO nodes are `async def` and call the clients' `a*` twins (`astructured`, `acomplete`, `aembed_query`, `arerank`); the checkpointer is `AsyncPostgresSaver` on the shared async pool, built in the app's lifespan because its constructor grabs the running loop. The three pure nodes (redact, refuse, no_answer) stay sync — langgraph runs them in a worker thread under `ainvoke`, so nothing blocks the loop either way. Sync client methods and the sync pool remain for build-time work (`kb/ingest`) and plain tests. `/rerank` goes through `AsyncOpenAI`'s generic `post` — same SDK, no second HTTP stack.
- Session history lives in the LangGraph Postgres checkpointer, not in `app/`.
- Dependencies are pinned exactly in `pyproject.toml`.
- Models are settled and verified live against OpenRouter (`uv run pytest -m live`). Changing one is a `core/config.py` edit — `LLMClient` takes provider and name, so any OpenRouter model works without touching a node.

## Requirements

### API
- REST API, any stack (FastAPI preferred).
- `POST /chat` — stateful conversation (session/conversation id carries history across turns).
- `GET /health` — liveness/readiness.
- No frontend.

### RAG
- Knowledge base of **at least 5 synthetic Sharia finance documents** (e.g. Murabaha, Sukuk, Ijara, Takaful, Wakala).
- Any vector store (Chroma, pgvector, Pinecone).
- Answers must be grounded in retrieved chunks; cite/return the chunk IDs used.

### PII
- Detect and **redact before anything reaches the LLM**: customer names, account numbers, Emirates ID, and similar identifiers.
- Redaction happens on the request path — the raw PII must never appear in prompts, logs, or traces.

### Observability
- Every request emits one **structured JSON trace log** containing at minimum:
  - end-to-end latency
  - retrieved chunk IDs
  - token usage
  - relevance score for retrieved context

### Evals
At least 3 automated tests:
1. **Grounding** — no hallucination beyond retrieved context.
2. **PII redaction correctness** — known PII inputs are masked.
3. **Refusal** — non-Islamic-finance queries are declined.

### Deployment
- Publicly deployed (Railway, Render, Fly.io, or similar).

## Constraints

- **Cost**: use a small model only — `gpt-4o-mini`, Claude Haiku, or Gemini Flash.
- **Secrets**: no API keys anywhere in the repo. Config via environment variables; ship a `.env.example` with names only.
- Synthetic data only — no real customer data.

## Conventions

- Keep the RAG pipeline, PII layer, and observability layer as separate, independently testable modules — the evals target each one directly.
- Prefer configuration over hardcoding for model name, top-k, and relevance threshold.
