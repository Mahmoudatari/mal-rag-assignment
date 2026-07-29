"""Single structured LLM call: scope check + retrieval decision + query contextualization.

Routes only, never answers. Reads conversation history to resolve pronouns and
ellipsis ("is it halal?") into a standalone search query.

Bias: when uncertain prefer "retrieve" over "refuse" — a false refusal stonewalls
a real customer question, a false retrieve is caught by the grader.
"""

from typing import Literal

from pydantic import BaseModel, Field

from accounts import product_names
from core.llm import StructuredOutputError, fast_llm
from rag.nodes._common import logged, usage_entry
from rag.state import State

# Names the five products and the one cross-cutting policy so "in scope" is a
# concrete list rather than a vibe the model has to infer. Kept short because
# this call runs on every single request on the cheap model.
SYSTEM = """You are the router for Mal, an Islamic finance bank's customer assistant.
You decide what happens next. You never answer the customer's question.

Mal's knowledge base covers exactly these topics:
- Murabaha everyday finance (cost-plus purchase financing)
- Ijara auto lease-to-own
- Fractional Sukuk investing
- Wakala savings
- Takaful cover (Sharia-compliant insurance)
- A cross-cutting late-payment / charity policy

Decide one of three routes:
- "retrieve": the customer is asking about Mal's products, Sharia finance
  concepts, account terms, or policies — anything needing facts from the
  knowledge base. When you are unsure whether a question is in scope, choose
  this over "refuse": a wrongly skipped retrieval is caught by a grader
  downstream, but a wrongly refused customer gets no help at all.
- "answer": the turn needs no product facts at all — a greeting, thanks, or a
  meta question like "what can you do?".
- "refuse": the turn has nothing to do with Islamic finance or with Mal at all
  — general chit-chat, trivia, or a request to do something outside a banking
  assistant's job, like writing code or booking a flight.

"refuse" is narrow. A question about the customer's own Mal account, or about
Mal's banking services, is in scope even when its topic is not obviously one of
the six above — a disputed transaction, a payment problem, a fee query, a
product Mal might or might not offer. Route those to "retrieve": if the
knowledge base turns out not to cover it, the customer is handed to support,
which is still a real answer. Refusing tells a genuine customer their genuine
question is not this assistant's business, and that is the outcome worth going
out of your way to avoid.

If retrieving, rewrite the customer's current turn into a standalone search
query using the conversation history to resolve pronouns and ellipsis (e.g.
"is it halal?" after a Murabaha question becomes "is Murabaha financing
permissible under Sharia"). A line before the message may list the products
the customer's own account holds — use it the same way, to resolve references
like "my lease" or "my savings" into the right product's name in the search
query. Leave search_query empty for every other route."""


class RouteDecision(BaseModel):
    route: Literal["retrieve", "refuse", "answer"] = Field(
        description=(
            "'retrieve' for anything needing Mal product or Sharia-finance facts "
            "(prefer this when unsure — a bad retrieve is caught by a grader, a "
            "bad refusal is not); 'answer' for a greeting, thanks, or meta "
            "question needing no product facts; 'refuse' only for turns wholly "
            "unrelated to Islamic finance or Mal banking."
        )
    )
    reason: str = Field(description="One short sentence: why this route, for the trace.")
    search_query: str = Field(
        description=(
            "Only when route is 'retrieve': the customer's current turn rewritten "
            "as a standalone question, with pronouns and ellipsis resolved against "
            "the conversation history. Empty string for every other route."
        )
    )


async def run(state: State) -> dict:
    """query + history → route, route_reason, search_query, tried_queries, usage_log."""
    query = state.get("query", "")
    if not query.strip():
        # A turn that was entirely PII masks down to nothing left to route or
        # retrieve against — short-circuit rather than spend a call routing an
        # empty string, and rather than seed tried_queries with blank text.
        return {
            "route": "answer",
            "route_reason": "empty query after redaction — nothing to route",
            "search_query": "",
            "tried_queries": [],
        }

    history = state.get("history", [])

    # One line of account context, not the full record: this call runs on every
    # request, and product names are all the rewrite needs to resolve "my lease"
    # — the record's figures belong to `generate`, the only node that answers.
    prompt = query
    account = state.get("account")
    products = ", ".join(product_names(account)) if account else ""
    if products:
        prompt = f"(The customer's account holds: {products}.)\n\n{query}"

    try:
        result = await fast_llm().astructured(prompt, RouteDecision, system=SYSTEM, history=history)
    except StructuredOutputError:
        # The router is on every request, so a raised exception here kills the
        # turn outright. Failing open to retrieve costs at worst one wasted
        # retrieval that the grader catches; refusing or crashing both stonewall
        # a real customer question instead. This is the only node in the graph
        # that fails open on a structured-output failure.
        return {
            "route": "retrieve",
            "route_reason": "router unavailable — defaulting to retrieve",
            "search_query": query,
            "tried_queries": [query],
        }

    decision = result.data
    entry = usage_entry("router", result.model, result.usage)

    if decision.route == "retrieve":
        # A blank rewrite from the model must not become `retrieve`'s embedded
        # text: embed_query raises on empty input, so the raw query is the floor.
        search_query = decision.search_query.strip() or query
        return {
            "route": "retrieve",
            "route_reason": decision.reason,
            "search_query": search_query,
            "tried_queries": [search_query],
            "usage_log": logged(state, entry),
        }

    # "answer" and "refuse" never carry a search query or a retry list — only
    # "retrieve" seeds tried_queries, so a later reformulate has something to
    # append to.
    return {
        "route": decision.route,
        "route_reason": decision.reason,
        "search_query": "",
        "tried_queries": [],
        "usage_log": logged(state, entry),
    }
