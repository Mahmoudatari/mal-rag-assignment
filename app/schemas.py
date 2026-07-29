"""Pydantic request/response models for the API surface.

Kept separate from `main.py` so the shapes can be imported by tests without
dragging the lifespan machinery in. `outcome` reuses `rag.state.Outcome`
rather than redeclaring the literal — the API promising a value the graph
cannot produce (or missing one it can) is exactly the drift a shared alias
prevents.

These models are also the OpenAPI contract: the descriptions and examples here
are what Swagger UI (`/docs`) renders, so the documentation is generated from
the same shapes the server validates instead of maintained beside them.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from rag.state import Outcome

# Mirrors `MAL-nnnn-nnnn-nnnn`, the account-number format the knowledge base
# documents themselves specify. Validated at the edge so a malformed id is a
# 422 naming the field, not a silent "no account context" deep in the graph.
ACCOUNT_ID_PATTERN = r"^MAL-\d{4}-\d{4}-\d{4}$"

# Exactly `uuid4().hex` — the only format the server mints. The session id is
# the checkpointer's thread_id and the *only* key to a conversation on an
# unauthenticated endpoint, so a client-invented id ("test", "s-1") is a 422:
# accepting low-entropy ids would let anyone who sends the same string read
# and extend another customer's conversation. A well-formed id an attacker
# invents is harmless — hitting a victim's thread means guessing their
# specific 122-bit random value.
SESSION_ID_PATTERN = r"^[0-9a-f]{32}$"


class ChatRequest(BaseModel):
    """One customer turn."""

    # No session_id in the example on purpose: a fixed example id is one every
    # /docs user pastes verbatim, which lands them all on the same thread —
    # the exact hijack the format guard exists to prevent. Omitting it shows
    # the intended flow: first turn without an id, continue with the minted one.
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "message": "How much is left to pay on my Murabaha contract?",
                    "account_id": "MAL-1001-2200-4417",
                }
            ]
        }
    )

    message: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "The customer's message. PII (names, account numbers, Emirates IDs, "
            "IBANs, emails, phone numbers) is detected and masked before the text "
            "reaches any model, log or trace."
        ),
    )
    # Echoed back to continue a conversation, omitted to start one. Only the
    # server-minted format is accepted — see SESSION_ID_PATTERN.
    session_id: str | None = Field(
        default=None,
        pattern=SESSION_ID_PATTERN,
        description=(
            "Conversation id. Omit to start a new conversation — the response "
            "returns a server-minted id; send that exact id back to continue. "
            "Only server-minted ids (32 lowercase hex chars) are accepted: the "
            "id is the sole key to a conversation, so a guessable, "
            "client-invented one is rejected with 422."
        ),
    )
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

    n: int = Field(description="The inline [n] citation marker this entry resolves.")
    doc: str = Field(description="Knowledge base document the cited chunk belongs to.")
    chunk_id: str = Field(
        description="Id of the cited chunk, e.g. `murabaha-everyday-finance#027`."
    )


class ChatResponse(BaseModel):
    """`refused` and `no_answer` are real answers, not errors — they return 200
    with the outcome named, and `answer` carries the refusal or hand-off text."""

    answer: str = Field(
        description=(
            "The assistant's reply. Grounded answers carry inline [n] citation "
            "markers resolved by `references`."
        )
    )
    references: list[ReferenceOut] = Field(
        description=(
            "The knowledge base chunks the answer actually cited — empty for "
            "refusals, hand-offs and conversational turns."
        )
    )
    outcome: Outcome = Field(
        description=(
            "`answered`: a real answer (grounded or conversational). `refused`: "
            "out of scope for an Islamic-finance assistant. `no_answer`: in scope "
            "but not covered by the knowledge base — the reply hands over to "
            "support."
        )
    )
    session_id: str = Field(
        description="Echoed (or minted) conversation id — send it back to continue."
    )


class HealthResponse(BaseModel):
    """Readiness detail. `detail` names the failing check when degraded."""

    status: Literal["ok", "degraded"] = Field(
        description="`ok` means a question could be answered right now."
    )
    documents: int = Field(description="Knowledge base documents in the index.")
    chunks: int = Field(description="Embedded chunks in the index.")
    foreign_models: list[str] = Field(
        description=(
            "Embedding models found in the index other than the configured one. "
            "Must be empty: chunks embedded by another model occupy a different "
            "vector space and retrieve as noise without raising."
        )
    )
    detail: str = Field(default="", description="Names the failing check when degraded.")
