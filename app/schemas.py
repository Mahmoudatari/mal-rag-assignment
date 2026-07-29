"""Pydantic request/response models for the API surface.

Kept separate from `main.py` so the shapes can be imported by tests without
dragging the lifespan machinery in. `outcome` reuses `rag.state.Outcome`
rather than redeclaring the literal — the API promising a value the graph
cannot produce (or missing one it can) is exactly the drift a shared alias
prevents.
"""

from typing import Literal

from pydantic import BaseModel, Field

from rag.state import Outcome

# Mirrors `MAL-nnnn-nnnn-nnnn`, the account-number format the knowledge base
# documents themselves specify. Validated at the edge so a malformed id is a
# 422 naming the field, not a silent "no account context" deep in the graph.
ACCOUNT_ID_PATTERN = r"^MAL-\d{4}-\d{4}-\d{4}$"


class ChatRequest(BaseModel):
    """One customer turn."""

    message: str = Field(min_length=1, max_length=4000)
    # Client-supplied to continue a conversation, omitted to start one — the
    # response echoes the id either way. Capped well under the checkpointer's
    # 255-char thread_id column.
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    account_id: str | None = Field(
        default=None,
        pattern=ACCOUNT_ID_PATTERN,
        description=(
            "Optional Mal account number (`MAL-nnnn-nnnn-nnnn`). When it matches "
            "a known account, the customer's own holdings are attached as context "
            "so questions like \"how much is left on my contract?\" get a "
            "specific answer. An unknown id is treated as no account context — "
            "this endpoint neither confirms nor denies that an account exists. "
            "Demo accounts: MAL-1001-2200-4417 (Murabaha + Wakala savings), "
            "MAL-2002-3300-8802 (Ijara lease + Sukuk holdings), "
            "MAL-3003-4400-1103 (Murabaha in arrears)."
        ),
    )


class ReferenceOut(BaseModel):
    """Mirror of `rag.state.Reference`: an inline [n] marker resolved to its chunk."""

    n: int
    doc: str
    chunk_id: str


class ChatResponse(BaseModel):
    """`refused` and `no_answer` are real answers, not errors — they return 200
    with the outcome named, and `answer` carries the refusal or hand-off text."""

    answer: str
    references: list[ReferenceOut]
    outcome: Outcome
    session_id: str


class HealthResponse(BaseModel):
    """Readiness detail. `detail` names the failing check when degraded."""

    status: Literal["ok", "degraded"]
    documents: int
    chunks: int
    foreign_models: list[str]
    detail: str = ""
