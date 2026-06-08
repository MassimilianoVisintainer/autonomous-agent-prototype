"""Tests for src/grounding.py.

All LLM calls are mocked — no real API key is required.
Tests verify the plumbing: intent-conditional logic, template selection,
citation extraction, and error handling.
"""

import datetime
from unittest.mock import patch

import pytest

from src import llm_client
from src.data_loaders import KnowledgeBaseChunk
from src.grounding import (
    CLARIFICATION_TEMPLATE,
    REFUSAL_TEMPLATE,
    GenerationResult,
    _format_chunks,
    generate_response,
)
from src.intents import Intent
from src.llm_client import LLMClientError
from src.nlu import ClassificationResult
from src.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _classification(intent: Intent, confidence: float = 0.9) -> ClassificationResult:
    return ClassificationResult(
        intent=intent,
        confidence=confidence,
        reasoning="test reasoning",
        method="llm",
    )


def _chunk(kb_id: str, source_doc: str, content: str = "Some chunk content.") -> KnowledgeBaseChunk:
    return KnowledgeBaseChunk(
        kb_id=kb_id,
        category="test",
        topic=kb_id.lower(),
        content=content,
        source_doc=source_doc,
        last_updated=datetime.date(2025, 1, 1),
        applies_to_products=[],
        requires_authority_check=False,
    )


def _retrieved(kb_id: str, source_doc: str, content: str = "Some chunk content.", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(chunk=_chunk(kb_id, source_doc, content), score=score)


SAMPLE_CHUNKS = [
    _retrieved("RT-001", "Returns Policy v2.3, §4.1", "Items may be returned within 30 days.", 0.9),
    _retrieved("RT-002", "Returns Policy v2.3, §4.2", "Faulty items can be returned within 12 months.", 0.8),
]


# ---------------------------------------------------------------------------
# Template tests
# ---------------------------------------------------------------------------

def test_ambiguous_query_uses_clarification_template():
    result = generate_response("Help.", _classification(Intent.AMBIGUOUS_QUERY), SAMPLE_CHUNKS)
    assert result.method == "clarification_template"
    assert result.text == CLARIFICATION_TEMPLATE
    assert result.citations == []


def test_out_of_scope_uses_refusal_template():
    result = generate_response(
        "What is the weather?", _classification(Intent.OUT_OF_SCOPE), SAMPLE_CHUNKS
    )
    assert result.method == "refusal_template"
    assert result.text == REFUSAL_TEMPLATE
    assert result.citations == []


# ---------------------------------------------------------------------------
# LLM path tests
# ---------------------------------------------------------------------------

def test_informational_intent_calls_llm():
    expected_text = "The laptop stand supports screens from 11 to 17 inches."
    with patch.object(llm_client, "complete", return_value=expected_text) as mock_complete:
        result = generate_response(
            "Does the stand support 17-inch screens?",
            _classification(Intent.PRODUCT_INFO),
            SAMPLE_CHUNKS,
        )
    assert mock_complete.call_count == 1
    assert result.method == "llm"
    assert result.text == expected_text


def test_llm_error_returns_graceful_fallback():
    with patch.object(llm_client, "complete", side_effect=LLMClientError("API down")):
        result = generate_response(
            "What is your return policy?",
            _classification(Intent.RETURN_POLICY),
            SAMPLE_CHUNKS,
        )
    assert result.method == "llm_error"
    assert len(result.text) > 0
    assert result.citations == []


# ---------------------------------------------------------------------------
# Citation extraction tests
# ---------------------------------------------------------------------------

def test_citations_extracted_from_response_text():
    response_with_citation = (
        "According to our policy (Returns Policy v2.3, §4.1), "
        "you have 30 days to return the item."
    )
    with patch.object(llm_client, "complete", return_value=response_with_citation):
        result = generate_response(
            "Can I return this?",
            _classification(Intent.RETURN_POLICY),
            SAMPLE_CHUNKS,
        )
    assert "Returns Policy v2.3, §4.1" in result.citations
    assert "Returns Policy v2.3, §4.2" not in result.citations


def test_citations_empty_when_response_has_no_source_docs():
    response_no_citation = "You can return the item if it meets the requirements."
    with patch.object(llm_client, "complete", return_value=response_no_citation):
        result = generate_response(
            "Can I return this?",
            _classification(Intent.RETURN_POLICY),
            SAMPLE_CHUNKS,
        )
    assert result.citations == []


# ---------------------------------------------------------------------------
# _format_chunks tests
# ---------------------------------------------------------------------------

def test_format_chunks_handles_empty_list():
    result = _format_chunks([])
    assert "No knowledge base chunks" in result


def test_format_chunks_renders_chunks_in_order():
    chunks = [
        _retrieved("RT-001", "Returns Policy v2.3, §4.1", score=0.9),
        _retrieved("SH-001", "Shipping Policy v1.0, §2.1", score=0.7),
    ]
    result = _format_chunks(chunks)
    assert "Returns Policy v2.3, §4.1" in result
    assert "Shipping Policy v1.0, §2.1" in result
    # First chunk's source appears before the second's
    assert result.index("Returns Policy v2.3, §4.1") < result.index("Shipping Policy v1.0, §2.1")


# ---------------------------------------------------------------------------
# Tool result integration tests
# ---------------------------------------------------------------------------

def test_generation_with_ok_tool_result_includes_data_in_prompt():
    from src.tools import ToolResult
    tool_result = ToolResult(
        status="ok",
        tool="lookup_order",
        data={"order_id": "ORD-1024", "status": "in_transit", "total_amount": 145.0},
        reason=None,
    )
    captured_prompts = []

    def capture_complete(system, user, json_mode=False, model=None):
        captured_prompts.append(system)
        return "Your order ORD-1024 is currently in transit."

    with patch.object(llm_client, "complete", side_effect=capture_complete):
        result = generate_response(
            "Where is my order ORD-1024?",
            _classification(Intent.ORDER_STATUS),
            SAMPLE_CHUNKS,
            tool_result=tool_result,
        )

    assert result.method == "llm"
    assert len(captured_prompts) == 1
    system_prompt = captured_prompts[0]
    assert "ORD-1024" in system_prompt
    assert "in_transit" in system_prompt


def test_generation_with_exceeded_authority_tool_result_handled_gracefully():
    from src.tools import ToolResult
    tool_result = ToolResult(
        status="exceeded_authority",
        tool="process_refund",
        data={"order_id": "ORD-1003", "total_amount": 245.0, "authority_limit": 100.0},
        reason="Senior authorization required.",
    )
    with patch.object(llm_client, "complete", return_value="I'm sorry, this refund requires senior authorization.") as mock_complete:
        result = generate_response(
            "I want a refund on ORD-1003",
            _classification(Intent.REFUND_REQUEST),
            SAMPLE_CHUNKS,
            tool_result=tool_result,
        )
    assert result.method == "llm"
    assert mock_complete.call_count == 1
