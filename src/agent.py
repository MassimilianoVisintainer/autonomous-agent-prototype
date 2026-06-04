"""Orchestration layer for the customer-support agent.

In Slice 1 the orchestrator only performs intent classification. Later slices
will extend this module with retrieval, tool routing, response generation,
and escalation — all sequenced through this entry point.
"""

from src.nlu import ClassificationResult, classify


def classify_query(query: str) -> ClassificationResult:
    """Classify a customer query and return a ClassificationResult.

    Acts as the single entry point so that app.py and tests never reach
    directly into the NLU module.
    """
    return classify(query)
