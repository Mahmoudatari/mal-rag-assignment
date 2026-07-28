"""Central configuration. Everything tunable lives here — no magic numbers in modules."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM (via OpenRouter, OpenAI-compatible) ------------------------
    openrouter_api_key: str = Field(default="", repr=False)
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Two tiers on purpose: the cheap model runs on every request (router and
    # grader, both short structured calls), the better one only writes answers.
    answer_model: str = "google/gemini-3.6-flash"
    fast_model: str = "google/gemini-3.5-flash-lite"  # router + grader

    # --- Conversation --------------------------------------------------
    # Prior turns are carried in graph state, which the checkpointer persists
    # per session forever — uncapped, a long session grows every checkpoint and
    # every router prompt with it. 20 messages is 10 turns of context.
    history_max_messages: int = 20

    # --- Retrieval -----------------------------------------------------
    # Two-stage: pull `retrieve_candidates` by vector similarity, rerank, keep
    # `top_k`. Reranking is pointless unless candidates > top_k — otherwise it
    # only reorders the same set.
    retrieve_candidates: int = 20
    top_k: int = 4

    # --- Reranking -----------------------------------------------------
    rerank_enabled: bool = True
    rerank_model: str = "cohere/rerank-v3.5"  # OpenRouter POST /api/v1/rerank

    # No relevance threshold by design: relevance is decided by the LLM grader,
    # never by a cosine cutoff. Similarity scores are recorded for tracing only.

    # Retries after a failed retrieval, i.e. 3 attempts total.
    max_retrieval_attempts: int = 2

    # No chunk_size / chunk_overlap: `kb/chunking.py` splits on markdown
    # structure — `##` sections, and FAQ blocks one Q&A pair per chunk — so
    # nothing anywhere splits on length. See that module for the measurements.

    # Used by BOTH ingest and retrieval, and the only setting that fails
    # silently: embedding documents and queries with different models puts them
    # in different vector spaces, so results become noise with no error raised.
    embedding_model: str = "google/gemini-embedding-001"

    # Passed through as the OpenAI-compatible `dimensions` parameter. The model
    # defaults to 3072; pgvector can only index `vector` columns up to 2000, so
    # we ask for 1536. Safe to truncate because the model is Matryoshka-trained.
    # Normalisation is a non-issue as long as retrieval uses cosine (`<=>`),
    # which is magnitude-invariant. Asserted on every embedding response.
    embedding_dimensions: int = 1536

    # Ingest-only in practice: retrieval embeds one query at a time. Keeps a
    # whole-corpus ingest from going out as a single oversized request.
    embedding_batch_size: int = 64

    # --- PII -----------------------------------------------------------
    # Pinned in pyproject.toml so `uv sync` fetches it. Presidio will otherwise
    # try to pip-install a model on first use, i.e. a network call on the
    # request path.
    # `lg` rather than `sm` because nothing tells us the customer's name — NER
    # is the only thing standing between a name and the prompt, so its recall is
    # the whole PERSON defence. Costs ~400MB in the deployment image.
    pii_spacy_model: str = "en_core_web_lg"

    # Allowlist. Presidio's full recognizer set fires low-confidence guesses at
    # formats irrelevant to a UAE bank (US driver's licence, US bank number),
    # which are noise rather than protection.
    pii_entities: tuple[str, ...] = (
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "EMIRATES_ID",
        "ACCOUNT_NUMBER",
    )

    # Low on purpose. Over-redaction costs a little answer quality; a missed
    # identifier is a leak into a prompt, a log and a trace at once.
    pii_score_threshold: float = 0.35

    # --- Postgres ------------------------------------------------------
    # Backs both pgvector retrieval and the LangGraph checkpointer (session state).
    database_url: str = Field(default="", repr=False)

    # No kb_table setting: table names live in kb/schema.sql, which cannot take
    # a parameter for an identifier. A setting here could only disagree with the
    # file — a knob that changes the queries but not the schema they run against.
    # kb/store.py holds the names as constants and a test asserts they match.

    # Small on purpose, and the budget for whichever pool the process opens:
    # the server's async pool (shared with the LangGraph checkpointer — a burst
    # of requests awaits a free connection rather than each holding one in a
    # thread) or ingest's sync pool. See kb/store.py on the two-pool split.
    db_pool_max_size: int = 4
    db_pool_timeout: float = 15.0  # seconds to wait for a free connection

    # --- Observability -------------------------------------------------
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
