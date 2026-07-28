"""Shared foundation: settings and the OpenRouter clients.

What belongs here is narrow, and the test is dependency direction rather than
vagueness of purpose: `core` is imported by `app/`, `kb/`, `rag/` and `pii/`,
and imports none of them back. A module that needs one of those is not core.

**This file stays empty of imports on purpose.** Re-exporting `core.llm` here
would make `from core.config import get_settings` drag `openai` in behind it —
and `pii/` imports settings, but has to stay importable with no API key and no
network so its eval runs standalone. `evals/test_pii_redaction.py` asserts
exactly that, so a convenience re-export added here fails a test rather than
quietly widening the dependency. Import the module you want:

    from core.config import get_settings
    from core.llm import fast_llm, answer_llm
    from core.embeddings import embedding_client
"""
