"""Orchestration layer for the customer-support agent.

Slice 4: the orchestrator performs classification (src.nlu), dense-embedding
retrieval (src.retrieval), and grounded response generation (src.grounding).
Tool calls and escalation logic are deferred to subsequent slices.
"""

from dataclasses import dataclass

from src.nlu import ClassificationResult, classify
from src import retrieval, grounding
from src.retrieval import RetrievedChunk
from src.grounding import GenerationResult


@dataclass(frozen=True)
class AgentResponse:
    classification: ClassificationResult
    retrieved_chunks: list[RetrievedChunk]
    generation: GenerationResult


def process_query(query: str) -> AgentResponse:
    """Classify, retrieve, and generate a grounded response for a customer query."""
    classification = classify(query)
    retrieved_chunks = retrieval.retrieve(query, k=5)
    generation = grounding.generate_response(query, classification, retrieved_chunks)
    return AgentResponse(
        classification=classification,
        retrieved_chunks=retrieved_chunks,
        generation=generation,
    )


def classify_query(query: str) -> ClassificationResult:
    """Backward-compatible entry point that returns only the classification."""
    return classify(query)
