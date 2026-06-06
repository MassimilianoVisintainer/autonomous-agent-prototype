"""Orchestration layer for the customer-support agent.

Slice 3: the orchestrator now performs classification (via src.nlu) followed by
dense-embedding retrieval (via src.retrieval). Response generation will be added
in Slice 4.
"""

from dataclasses import dataclass

from src.nlu import ClassificationResult, classify
from src import retrieval
from src.retrieval import RetrievedChunk


@dataclass(frozen=True)
class AgentResponse:
    classification: ClassificationResult
    retrieved_chunks: list[RetrievedChunk]


def process_query(query: str) -> AgentResponse:
    """Classify the query and retrieve the top-5 relevant KB chunks."""
    classification = classify(query)
    chunks = retrieval.retrieve(query, k=5)
    return AgentResponse(classification=classification, retrieved_chunks=chunks)


def classify_query(query: str) -> ClassificationResult:
    """Backward-compatible entry point that returns only the classification."""
    return classify(query)
