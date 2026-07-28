-- Knowledge base schema. Applied idempotently by `kb.store.apply_schema()`,
-- which `kb/ingest.py` calls before it writes anything.
--
-- This is a create-only schema, not a migration chain: every statement is
-- `IF NOT EXISTS`, so re-running it is a no-op, but it will not ALTER a table
-- that already exists in a different shape. That is the right trade for a
-- schema that ships once — a real migration story (alembic, or numbered files
-- plus a `schema_version` table) starts the first time a column has to change
-- under data worth keeping. Until then it is one file you can read end to end.
--
-- Table names are duplicated in `kb/store.py` as DOCUMENTS_TABLE / CHUNKS_TABLE,
-- because a .sql file cannot take a parameter for an identifier. A test asserts
-- the two agree, so the drift is caught rather than trusted.
--
-- LangGraph's Postgres checkpointer creates its own tables in this same
-- database via its `setup()` call. Deliberately not managed here: those are the
-- framework's, and hand-writing them would pin us to its current internals.

CREATE EXTENSION IF NOT EXISTS vector;


-- One row per markdown file in kb/documents/.
--
-- It exists for three reasons, not for tidiness. Without all three the only
-- non-derivable column would be `title` and this table would not be worth a join:
--
--   1. `content_hash` makes re-ingest a no-op when nothing changed. Ingest runs
--      on every deploy, and embedding is the one step that costs money and time.
--   2. `embedding_model` is the guard for the one failure that is otherwise
--      completely silent: documents embedded with one model and queried with
--      another land in different vector spaces, and retrieval degrades to noise
--      with no error anywhere. Recording what produced the vectors lets ingest
--      and /health compare against the configured model and fail loudly. It is
--      also what makes the hash skip safe — a hash match alone would happily
--      keep vectors from the previous model.
--   3. The chunks foreign key below.
--
-- `embedding_dimensions` is deliberately NOT recorded. The `vector(1536)`
-- column type already rejects a wrong-width vector at INSERT. Only record what
-- would otherwise fail silently.
CREATE TABLE IF NOT EXISTS sharia_documents (
    -- The filename stem, e.g. 'murabaha-everyday-finance'. Also the prefix of
    -- every chunk_id below.
    doc             text PRIMARY KEY,
    -- The `#` header. The only per-document fact not derivable from the chunks.
    title           text NOT NULL,
    -- sha256 over the rendered chunk texts, NOT over the source file: it has to
    -- cover the chunking logic too, or a change to kb/chunking.py would leave
    -- the hash equal and the stored chunks stale. See kb.store.content_hash.
    content_hash    text NOT NULL,
    embedding_model text NOT NULL,
    ingested_at     timestamptz NOT NULL DEFAULT now()
);


-- One row per retrievable chunk. 202 of them today: 129 whole `##` sections
-- plus 73 FAQ answers, one per question. See kb/chunking.py.
CREATE TABLE IF NOT EXISTS sharia_chunks (
    -- Positional and content-derived, e.g. 'murabaha-everyday-finance#027'.
    -- This is the citation key the API returns, so it is the primary key here.
    -- Because it is positional, editing a document shifts every id after the
    -- edit — which is why ingest replaces a document's rows wholesale rather
    -- than upserting one at a time, and why the FK below cascades.
    chunk_id  text PRIMARY KEY,
    doc       text NOT NULL REFERENCES sharia_documents (doc) ON DELETE CASCADE,
    -- The `##` header, or '' for text before the first one. Kept for tracing and
    -- debugging; it is already inside `text` as a rendered markdown header.
    section   text NOT NULL,
    -- Exactly the string that was embedded, and exactly the string that reaches
    -- the model as context. If those two ever differ, retrieval scores a
    -- document the answer never sees.
    text      text NOT NULL CHECK (text <> ''),
    -- 1536 rather than the model's default 3072 because pgvector's `vector`
    -- type cannot be indexed above 2000 dimensions. Safe to truncate: the model
    -- is Matryoshka-trained and retrieval uses cosine, which is
    -- magnitude-invariant. See core/config.py → embedding_dimensions.
    embedding vector(1536) NOT NULL
);


-- No index beyond the primary key, and that is a decision rather than an omission.
--
-- No ANN index (ivfflat/hnsw): at 202 rows an exact scan over 202x1536 floats
-- is sub-millisecond and strictly more accurate than any approximation. An ANN
-- index here would trade correctness for a speedup too small to measure. The
-- 1536-dimension choice above is what keeps the option open for later.
--
-- No btree on `doc` either, for the same reason. It would serve the FK's
-- reverse check on parent DELETE — five sequential scans of a 202-row table
-- during ingest, which is not a cost. Add both the day the corpus grows an
-- order of magnitude, with a measurement in hand.
