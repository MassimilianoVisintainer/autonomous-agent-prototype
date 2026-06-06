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
All other intents use a single Gemini call with an intent-aware system prompt.
"""

from dataclasses import dataclass

from src.intents import Intent
from src.nlu import ClassificationResult
from src.retrieval import RetrievedChunk
from src import llm_client
from src.llm_client import LLMClientError


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
1. Answer the customer's question using ONLY the information in the knowledge-base
   chunks provided below. Do not use any other knowledge.
2. Cite every claim you make by including the source_doc string in parentheses,
   e.g. "(Returns Policy v2.3, §4.1)". Use the exact source_doc strings as they
   appear in the chunks.
3. If the retrieved chunks do not contain the information needed to answer the
   question, say "I don't have that information available" rather than guessing.
4. Keep your response concise: two or three short paragraphs at most.

TONE CALIBRATION
- Routine queries (product info, policies, shipping, account): be helpful and direct.
- Affective queries (complaint, multi_issue_dispute): begin with a brief empathetic
  acknowledgement of the customer's frustration before addressing the substance.
- Transactional queries that require a system action (order_status, order_modify,
  order_cancel, refund_request): acknowledge the request clearly and describe what
  action would be taken, but state honestly that the tool integration is not yet
  active and that a human agent can assist in the meantime.

KNOWLEDGE BASE CHUNKS
{chunks_formatted}"""


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

    chunks_formatted = _format_chunks(retrieved_chunks)
    system_prompt = GENERATION_SYSTEM_PROMPT_TEMPLATE.format(
        intent=classification.intent.value,
        confidence=f"{classification.confidence:.2f}",
        chunks_formatted=chunks_formatted,
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
