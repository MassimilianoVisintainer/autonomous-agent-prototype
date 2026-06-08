"""Orchestration layer for the customer-support agent.

Slice 5: the orchestrator performs classification (src.nlu), dense-embedding
retrieval (src.retrieval), tool dispatch for transactional intents (src.tools),
and grounded response generation (src.grounding). Escalation routing is the
remaining major architectural layer, deferred to Slice 6.
"""

from dataclasses import dataclass

from src.nlu import ClassificationResult, classify
from src import retrieval, grounding, tools
from src.retrieval import RetrievedChunk
from src.grounding import GenerationResult
from src.tools import ToolResult


@dataclass(frozen=True)
class AgentResponse:
    classification: ClassificationResult
    retrieved_chunks: list[RetrievedChunk]
    tool_result: ToolResult | None
    generation: GenerationResult


def process_query(query: str) -> AgentResponse:
    """Classify, retrieve, dispatch tools, and generate a grounded response."""
    classification = classify(query)
    retrieved_chunks = retrieval.retrieve(query, k=5)
    tool_result = tools.dispatch(classification, query)
    generation = grounding.generate_response(
        query, classification, retrieved_chunks, tool_result=tool_result
    )
    return AgentResponse(
        classification=classification,
        retrieved_chunks=retrieved_chunks,
        tool_result=tool_result,
        generation=generation,
    )


def classify_query(query: str) -> ClassificationResult:
    """Backward-compatible entry point that returns only the classification."""
    return classify(query)
