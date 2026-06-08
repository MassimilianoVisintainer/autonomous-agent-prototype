"""Response generation and grounding layer — §3.3 of the thesis.

Implements the prompt-level cite-or-silent discipline: the LLM is instructed to
ground every claim in the retrieved knowledge-base chunks and to cite the chunks'
source_doc strings inline using parenthesised notation. Programmatic enforcement
(NLI critics, confidence-floor rejection, chain-of-verification) is deferred to
a later slice; this slice instils the discipline at the prompt level and extracts
citations post-hoc by substring matching.

Two intents are handled by deterministic templates and never call the LLM:
  - AMBIGUOUS_QUERY  → clarification request
  - OUT_OF_SCOPE     → polite refusal with redirection
All other intents use a single Gemini call with an intent-aware system prompt
that now includes the structured output of any tool that was called.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.intents import Intent
from src.nlu import ClassificationResult
from src.retrieval import RetrievedChunk
from src import llm_client
from src.llm_client import LLMClientError

if TYPE_CHECKING:
    from src.tools import ToolResult


@dataclass(frozen=True)
class GenerationResult:
    text: str
    citations: list[str]
    method: str


CLARIFICATION_TEMPLATE = (
    "I'd like to help, but I need a bit more information first. "
    "Could you tell me what you're trying to do, or share an order number "
    "if your question is about a specific order?"
)

REFUSAL_TEMPLATE = (
    "I'm focused on customer support questions about your orders, products, "
    "account, and our policies. Is there something along those lines I can "
    "help with today?"
)

GENERATION_SYSTEM_PROMPT_TEMPLATE = """\
You are a customer support agent for an EU-based electronics-accessories retailer.

Classified intent: {intent} (confidence: {confidence})

INSTRUCTIONS
1. Answer the customer's question using the tool result (when present) and the
   knowledge-base chunks below. Prefer tool result data for transactional facts
   (order status, amounts, dates); prefer chunks for policy and product information.
2. Cite every policy or product claim by including the source_doc string in
   parentheses, e.g. "(Returns Policy v2.3, §4.1)".
3. If neither the tool result nor the chunks contain the information needed,
   say "I don't have that information available" rather than guessing.
4. Keep your response concise: two or three short paragraphs at most.

TONE CALIBRATION
- Routine queries (product info, policies, shipping, account): be helpful and direct.
- Affective queries (complaint, multi_issue_dispute): begin with a brief empathetic
  acknowledgement of the customer's frustration before addressing the substance.
- Transactional queries: use the tool result to give a specific, grounded answer.

TOOL RESULT HANDLING — respond according to the status value:
- ok: Report the action taken or information found. Be specific — use order IDs,
  amounts, statuses, and dates from the data. Do NOT use vague placeholders.
- not_found: Explain that no order with that identifier was found, and ask the
  customer to double-check the order number.
- missing_identifier: Ask the customer for the order number, as their query did
  not include one.
- exceeded_authority: Acknowledge the request. Explain clearly that the operation
  requires senior authorization beyond what you can process directly, and that a
  human agent will need to handle it. Do NOT claim the action has been processed.
- out_of_window: Explain the time or state constraint that prevents the action
  (cancellation window elapsed, order already dispatched, etc.), referencing the
  relevant policy chunks where applicable.
- error: Apologise for a technical issue and offer to connect the customer with
  a human agent.

TOOL EXECUTION RESULT
{tool_result_block}

KNOWLEDGE BASE CHUNKS
{chunks_formatted}"""


def _format_tool_result(tool_result: ToolResult | None) -> str:
    if tool_result is None:
        return "No tool was called for this query."
    lines = [
        f"Tool: {tool_result.tool}",
        f"Status: {tool_result.status}",
    ]
    if tool_result.reason:
        lines.append(f"Reason: {tool_result.reason}")
    if tool_result.data:
        lines.append("Data:")
        for k, v in tool_result.data.items():
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _format_chunks(retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "No knowledge base chunks were retrieved for this query."
    parts = []
    for i, rc in enumerate(retrieved_chunks, start=1):
        c = rc.chunk
        parts.append(
            f"[Chunk {i} — kb_id: {c.kb_id}, source: {c.source_doc}, "
            f"relevance: {rc.score:.2f}]\n{c.content}"
        )
    return "\n\n".join(parts)


def _extract_citations(
    response_text: str, retrieved_chunks: list[RetrievedChunk]
) -> list[str]:
    seen: set[str] = set()
    citations: list[str] = []
    for rc in retrieved_chunks:
        src = rc.chunk.source_doc
        if src not in seen and src in response_text:
            citations.append(src)
            seen.add(src)
    return citations


def generate_response(
    query: str,
    classification: ClassificationResult,
    retrieved_chunks: list[RetrievedChunk],
    tool_result: ToolResult | None = None,
) -> GenerationResult:
    """Generate a grounded response appropriate to the classified intent."""
    if classification.intent == Intent.AMBIGUOUS_QUERY:
        return GenerationResult(
            text=CLARIFICATION_TEMPLATE, citations=[], method="clarification_template"
        )
    if classification.intent == Intent.OUT_OF_SCOPE:
        return GenerationResult(
            text=REFUSAL_TEMPLATE, citations=[], method="refusal_template"
        )

    system_prompt = GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        intent=classification.intent.value,
        confidence=f"{classification.confidence:.2f}",
        tool_result_block=_format_tool_result(tool_result),
        chunks_formatted=_format_chunks(retrieved_chunks),
    )

    try:
        response_text = llm_client.complete(
            system=system_prompt, user=query, json_mode=False
        )
    except LLMClientError:
        return GenerationResult(
            text=(
                "I'm having trouble producing a response right now. "
                "Let me connect you with a human agent who can help."
            ),
            citations=[],
            method="llm_error",
        )

    citations = _extract_citations(response_text, retrieved_chunks)
    return GenerationResult(text=response_text, citations=citations, method="llm")
