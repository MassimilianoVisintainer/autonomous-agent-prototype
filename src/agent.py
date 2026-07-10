"""Orchestration layer for the customer-support agent.

Performs the full pipeline from §3.1.1 through §3.2 of the thesis —
classification (src.nlu), dense-embedding retrieval (src.retrieval), tool
dispatch for transactional intents (src.tools), escalation evaluation
(src.escalation), and grounded response generation (src.grounding).
"""

from dataclasses import dataclass

from src.nlu import ClassificationResult, classify
from src import retrieval, grounding, tools, escalation
from src.retrieval import RetrievedChunk
from src.grounding import GenerationResult
from src.tools import ToolResult
from src.escalation import EscalationResult


@dataclass(frozen=True)
class AgentResponse:
    classification: ClassificationResult
    retrieved_chunks: list[RetrievedChunk]
    tool_result: ToolResult | None
    escalation: EscalationResult
    generation: GenerationResult


def process_query(query: str) -> AgentResponse:
    """Run the full pipeline: classify → retrieve → tool → escalate → generate."""
    classification = classify(query)
    retrieved_chunks = retrieval.retrieve(query, k=5)
    tool_result = tools.dispatch(classification, query)
    escalation_result = escalation.evaluate_escalation(classification, tool_result, query)
    generation = grounding.generate_response(
        query,
        classification,
        retrieved_chunks,
        tool_result=tool_result,
        escalation_result=escalation_result,
    )
    return AgentResponse(
        classification=classification,
        retrieved_chunks=retrieved_chunks,
        tool_result=tool_result,
        escalation=escalation_result,
        generation=generation,
    )


def classify_query(query: str) -> ClassificationResult:
    """Backward-compatible entry point that returns only the classification."""
    return classify(query)
