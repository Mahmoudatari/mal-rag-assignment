# Tasks

One session each. Read `CLAUDE.md` first — it holds the settled design decisions and the reasoning behind them.

Status: tasks 1–6 done. The API serves `/chat` and `/health` on a fully async request path; next is task 7 (evals), then deploy. Account context (see Open decision) is still open and was deliberately not folded into task 6.

---

## 1. Models — **done**

OpenRouter for everything (OpenAI-compatible, `openai` SDK). `google/gemini-3.6-flash` answers, `google/gemini-3.5-flash-lite` routes and grades, `google/gemini-embedding-001` at `dimensions=1536` embeds. Settled in `core/config.py`; see CLAUDE.md → Models for the reasoning.

`core/llm.py` is the shared OpenRouter client: `LLMClient(provider, name)` plus `structured(prompt, Schema)` for the router and grader, with token usage and cost returned on every call for the trace. Nodes take theirs from `fast_llm()` / `answer_llm()`. Unit tests in `tests/test_llm.py` run against an injected fake; the `live`-marked test confirms the key and the configured model ids for real — it passes.

`core/embeddings.py` is its sibling for vectors: `EmbeddingClient(provider, name, dimensions=...)`, `embed_query()` / `embed_documents()` (batched, order-preserving), width asserted on every response, usage returned in the same `Usage` type so a trace can total both. `kb/ingest.py` and `rag/nodes/retrieve.py` both take theirs from `embedding_client()`. Unit tests in `tests/test_embeddings.py`; the `live` tests confirm 1536-dim vectors really come back and that related text out-scores unrelated — both pass.

`core/rerank.py` completes the set: `RerankClient(provider, name)`, `rerank(query, documents, top_n=...)` returning rankings best-first, with `apply(chunks)` / `scored(chunks)` reordering the caller's own objects by `index` so `chunk_id` survives. Reached via the SDK's generic `post()` so it shares the pool and retry policy. Unit tests in `tests/test_rerank.py`; live tests confirm the relevant document ranks first and that the call reports its cost — both pass. See CLAUDE.md → Reranking.

Two OpenRouter behaviours were found live and are now pinned in code, with the reasoning in CLAUDE.md → Boundaries: `encoding_format="float"` must be explicit (the SDK's silent base64 default fails intermittently — 7 of 12 identical calls — which would kill an ingest mid-corpus), and 200-with-`error` responses need their message surfaced. Both are embeddings-only: the chat path was checked on the wire and sampled 12/12 on strict `response_format`, so `core/llm.py` needs no equivalent.

## 2. PII layer — **done**

`pii/` is detector + masker + patterns, wired into `rag/nodes/redact.py`, which
is the graph's first node and writes only `query` and `pii_spans` — so there is
no path by which raw text reaches the router, a prompt or a trace. Pure as
required: no network, no key, no database, and spans carry kind and offsets with
no matched values. Covered by `evals/test_pii_redaction.py` (40 tests), which is
one of the three assignment deliverables.

## 3. Knowledge base — **done**

Five documents are in `kb/documents/` (Murabaha, Sukuk, Ijara, Wakala, late payment/charity), written with answerable specifics so retrieval and the grounding eval have something to be right or wrong about.

`kb/chunking.py` splits on markdown structure, never on length: **202 chunks** — 129 whole `##` sections plus 73 FAQ answers, one per question. Both the `#` and `##` headers are re-attached to every chunk. No `chunk_size` / `chunk_overlap` anywhere. Tested in `tests/test_chunking.py`; see CLAUDE.md → Chunking for the measurements.

`kb/schema.sql` + `kb/store.py` are the pgvector layer: `sharia_documents` (doc, title, content_hash, embedding_model, ingested_at) and `sharia_chunks` (chunk_id PK, doc FK cascade, section, text, `vector(1536)`). Writes are wholesale per document because `chunk_id` is positional; freshness is `content_hash` over the rendered chunks **plus** the embedding model. `search()` returns cosine similarity, `stats()` feeds `/health`. Applied and round-tripped against the Railway Postgres (PG 18.4, pgvector 0.8.5). Tested in `tests/test_store.py` — pure tests run by default, writing tests are marked `db` and take `TEST_DATABASE_URL`. See CLAUDE.md → Store.

`kb/ingest.py` is the build-time entrypoint (`uv run python -m kb.ingest`, `--documents DIR`, `--force`): apply the schema, chunk, skip documents whose hash **and** embedding model both match, embed the rest via `embedding_client()`, `replace_document()` per document, `prune()` what is gone, and report chunks written, tokens and USD cost. Two properties are load-bearing and tested — the model id is resolved from settings rather than from a constructed client, so a no-op deploy makes no network call and needs no API key; and the prune keep-set is every document in the corpus, not the ones written this run, which is the line that would otherwise empty the index on the second deploy. Each document is its own transaction, so a run that dies part way leaves the finished ones correct and a re-run resumes. Tested in `tests/test_ingest.py` (23 tests, store and embedder faked).

**Run for real against the Railway Postgres.** First run embedded all five documents — 202 chunks, 50,620 tokens, $0.0076. Second run: 0 embedded, 5 unchanged, no embedding calls, so the skip holds against a live database and not just fakes. `stats()` reports 5 documents / 202 chunks / one embedding model with no foreign models. Spot-checked retrieval on three questions and each landed on the right document: murabaha late payment 0.760, selling sukuk early 0.778, ijara insurance 0.686.

The ijara case is the one to watch — the weakest score of the three, and its top hit is "Total Loss and Theft of the Vehicle" rather than a section on who carries the insurance obligation. That is the mediocre-but-plausible result the reranker and grader exist to catch, so re-check it once task 5 wires them in.

Also noted: `section` is `"Frequently Asked Questions"` for 73 of the 202 chunks, so it is not usable as a human-readable chunk label. Citations are unaffected — `chunk_id` is the key and the `#` title is inside the chunk text — but anything downstream wanting a display label needs more than `section`.

## 4. Observability — **done**

`observability/trace.py` is the record and `observability/logger.py` writes it:
`Trace` (frozen dataclass, flat primitives), `Stopwatch` for end-to-end latency,
`render()` / `emit(trace, stream=None)` for one JSON line on stdout. `app/` is the
only caller — `Stopwatch()` before `invoke`, `Trace.from_state(final,
latency_ms=watch.ms)` after, `emit()`. See CLAUDE.md → Tracing.

Task 5 inverting the dependency is what shaped this: nodes append to `usage_log`
rather than calling a tracer, so the whole record is assembled at the edge from
final state. `from_state` takes a `Mapping`, so the leaf holds — this package
imports nothing internal at all, not even `core`, and the token counts are ints
rather than a `core.llm.Usage` so the logging path never drags in the OpenAI SDK.

Beyond the four required fields: `cited_chunk_ids` (what the answer actually
referenced, a subset of what was retrieved), `cosine_score` and `rerank_score`
alongside the headline `relevance_score`, `pii_kinds`, and `calls` — `usage_log`
verbatim, because a total cannot show that a retrieval loop ran twice.

27 tests in `tests/test_observability.py`. Two are load-bearing rather than
routine: the four brief-required fields are asserted **by name**, since a rename
is how a deliverable quietly disappears; and an ast scan checks every
`state.get("…")` literal in `trace.py` against `State.__annotations__`, because
the two files agree by key name with no import between them, so a rename in
`rag/state.py` would otherwise leave a trace field silently reading its default.

Checked against a real compiled-graph final state, not just fixtures: reranking
had reordered the candidates, so `relevance_score` took the cross-encoder's 0.90
over the top chunk's 0.88 cosine, and `cited_chunk_ids` mapped [1] back to a real
`chunk_id`.

**One loose end for task 6.** `session_id` is declared in `State` and written by
no node, so `app/` must put it in the invoke payload alongside `raw_query` — pass
only `thread_id` in the config and every trace logs an empty session.

## 5. Graph nodes — **done**

All nine `run()` bodies, plus `tests/test_nodes_*.py` and `tests/test_graph.py`.

Task 4 turned out not to be a dependency after all: nodes append their provider
calls to a `usage_log` list in state as plain JSON, so the trace can be assembled
from final state without `rag/` importing `observability/`. That inverts the
order — task 4 now *reads* a shape that already exists rather than defining one
the nodes have to call into.

Two things the wiring did not survive contact with. **The checkpointer persists
one `State` per `thread_id`, and nodes write only what they change**, so turn N+1
inherited turn N's `chunks` and `attempts`; `redact` now resets the turn-scoped
keys (see **State** in `CLAUDE.md`). And **conversation history had nowhere to
live** — the router is the node that resolves "is it halal?" against prior turns,
but nothing was carrying them. It is a `history` key in state, the one key the
reset leaves alone.

## 6. API — **done**

`app/main.py` + `app/schemas.py`: `POST /chat` (stateful via `session_id`, minted as `uuid4().hex` when absent, doubling as the checkpointer's `thread_id`), `GET /health`, lifespan wiring, per-request trace emission in a `finally` so the crashed request is the one guaranteed a record. The task 4 loose end is closed: `session_id` goes into the invoke payload alongside `raw_query`, so traces log the real session. Tested in `tests/test_app.py` (12 tests, graph stubbed at the `serving_graph` seam).

`/health` answers the question it was asked: 200 only when the database answers, the index has chunks, and `stats().foreign_models()` is empty — the foreign-model check being the one failure that raises nothing at query time. 503 names the failing check.

**The request path went fully async first**, which task 6 forced: an async endpoint cannot sit on a sync graph invocation without parking every request on a thread. The conversion: `a*` twins on all three `core/` clients over shared `AsyncOpenAI` transports (rerank included — `AsyncOpenAI` has the same generic `post`, so no second HTTP stack), `async_pool()` + `asearch()`/`astats()` in `kb/store.py`, the six IO nodes now `async def`, and `acheckpointer()` returning `AsyncPostgresSaver` on the shared async pool, constructed in the lifespan because its `__init__` grabs the running loop. Sync methods and the sync pool remain for ingest and plain tests. See CLAUDE.md → Decisions ("async end to end") and → Store (two pools).

## 7. Evals — **needs 5, 6**

Three required: grounding (answer stays within retrieved chunks — scope to the answer body), PII redaction correctness, refusal of non-Islamic-finance queries. Add a couple of router cases, since everything downstream depends on it.

## 8. Deploy — **needs 6**

Railway or Render. Postgres + pgvector is already provisioned and **already holds the 202-chunk index** (see task 3), so what is left is deploying the app against it, setting env vars, and confirming the public URL answers. Re-run `uv run python -m kb.ingest` as a deploy step regardless — it is a no-op when the index is current, which is the point of the skip.

## 9. README — **last**

Setup, architecture, deployed URL, sample requests, and the design trade-offs from `CLAUDE.md`.

---

## Open decision

**Account context.** The brief requires answering against the customer's own account data; nothing in the graph does this yet. Deferred to task 6 rather than resolved before task 5 — the nodes were built with no account awareness, so picking it up means a state key, a scope line in the router's system prompt, and a block in `generate`'s grounded prompt. Still undecided: node vs. injected into state from the request payload.
