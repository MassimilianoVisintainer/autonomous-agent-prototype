"""Tests for scripts/run_evaluation.py.

All tests use synthetic data and mocked agents — no real API calls.
"""

import datetime
import json
import pathlib

import pytest

from scripts.run_evaluation import (
    _build_failure_row,
    _build_output_row,
    _is_rate_limit_error,
    _is_row_complete,
    _load_completed_rows,
)
from src.agent import AgentResponse
from src.data_loaders import KnowledgeBaseChunk, TestQuery
from src.escalation import EscalationResult
from src.grounding import GenerationResult
from src.intents import Intent
from src.llm_client import LLMClientError
from src.nlu import ClassificationResult
from src.retrieval import RetrievedChunk
from src.tools import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_test_query(**overrides) -> TestQuery:
    defaults = dict(
        query_id="Q-001",
        query_text="Where is my order ORD-1024?",
        intent_label="order_status",
        test_category="canonical",
        expected_handling="contained",
        expected_escalation_reason=None,
        expected_tool_calls="tier1_order_lookup:ORD-1024",
        expected_kb_chunks="",
        emotional_intensity_band="none",
        threshold_test_flag=None,
        linked_query_id=None,
    )
    defaults.update(overrides)
    return TestQuery(**defaults)


def _make_chunk(kb_id: str = "RT-001") -> KnowledgeBaseChunk:
    return KnowledgeBaseChunk(
        kb_id=kb_id, category="returns", topic="standard_window",
        content="Return within 30 days.",
        source_doc="Returns Policy v2.3, §4.1",
        last_updated=datetime.date(2025, 9, 15),
        applies_to_products=[], requires_authority_check=False,
    )


def _make_agent_response(tool_result=None) -> AgentResponse:
    return AgentResponse(
        classification=ClassificationResult(
            intent=Intent.ORDER_STATUS, confidence=0.95,
            reasoning="Asking about order location.", method="llm",
        ),
        retrieved_chunks=[RetrievedChunk(chunk=_make_chunk(), score=0.87)],
        tool_result=tool_result,
        escalation=EscalationResult(
            should_escalate=False, triggers=[], emotion_score=0.0,
            reason="All escalation triggers cleared; agent handles autonomously.",
        ),
        generation=GenerationResult(
            text="Your order is in transit.", citations=[], method="llm",
        ),
    )


# ---------------------------------------------------------------------------
# _build_output_row
# ---------------------------------------------------------------------------

def test_build_output_row_includes_all_ground_truth_fields():
    tq = _make_test_query()
    row = _build_output_row(tq, _make_agent_response(), 450, "ok")
    assert row["query_id"] == "Q-001"
    assert row["query_text"] == "Where is my order ORD-1024?"
    assert row["expected_intent"] == "order_status"
    assert row["test_category"] == "canonical"
    assert row["expected_handling"] == "contained"
    assert row["expected_escalation_reason"] is None
    assert row["expected_tool_calls"] == "tier1_order_lookup:ORD-1024"
    assert row["expected_kb_chunks"] == ""
    assert row["emotional_intensity_band"] == "none"
    assert row["threshold_test_flag"] is None
    assert row["linked_query_id"] is None


def test_build_output_row_includes_all_observed_fields():
    row = _build_output_row(_make_test_query(), _make_agent_response(), 450, "ok")
    assert row["predicted_intent"] == "order_status"
    assert row["classification_confidence"] == 0.95
    assert row["classification_method"] == "llm"
    assert row["classification_reasoning"] == "Asking about order location."
    assert row["retrieved_chunk_ids"] == ["RT-001"]
    assert len(row["retrieved_chunk_scores"]) == 1
    assert row["tool_called"] is None
    assert row["tool_status"] is None
    assert row["escalation_decision"] is False
    assert row["escalation_triggers"] == []
    assert row["emotion_score"] == 0.0
    assert row["generation_text"] == "Your order is in transit."
    assert row["generation_method"] == "llm"
    assert row["citations"] == []
    assert row["harness_status"] == "ok"
    assert row["elapsed_ms"] == 450
    assert "timestamp" in row


def test_build_output_row_handles_null_tool_result():
    row = _build_output_row(_make_test_query(), _make_agent_response(tool_result=None), 100, "ok")
    assert row["tool_called"] is None
    assert row["tool_status"] is None
    assert row["tool_result_summary"] is None


def test_build_output_row_populates_tool_fields_when_present():
    tr = ToolResult(
        status="ok", tool="lookup_order",
        data={"order_id": "ORD-1024", "status": "in_transit"},
        reason=None,
    )
    row = _build_output_row(_make_test_query(), _make_agent_response(tool_result=tr), 200, "ok")
    assert row["tool_called"] == "lookup_order"
    assert row["tool_status"] == "ok"
    assert row["tool_result_summary"] is not None
    assert "ORD-1024" in row["tool_result_summary"]


# ---------------------------------------------------------------------------
# _build_failure_row
# ---------------------------------------------------------------------------

def test_build_failure_row_marks_harness_status_correctly():
    tq = _make_test_query()
    exc = RuntimeError("something exploded")
    row = _build_failure_row(tq, exc, 300, "error")
    assert row["harness_status"] == "error"
    assert row["harness_error_type"] == "RuntimeError"
    assert "something exploded" in row["harness_error_message"]
    assert row["predicted_intent"] is None
    assert row["escalation_decision"] is None
    assert row["retrieved_chunk_ids"] == []


# ---------------------------------------------------------------------------
# _load_completed_rows
# ---------------------------------------------------------------------------

def test_load_completed_rows_returns_empty_for_missing_file(tmp_path):
    result = _load_completed_rows(tmp_path / "nonexistent.jsonl")
    assert result == {}


def test_load_completed_rows_parses_valid_jsonl(tmp_path):
    p = tmp_path / "out.jsonl"
    rows = [{"query_id": f"Q-{i:03d}", "classification_method": "llm", "generation_method": "llm"} for i in range(1, 4)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    result = _load_completed_rows(p)
    assert set(result.keys()) == {"Q-001", "Q-002", "Q-003"}
    assert result["Q-001"]["classification_method"] == "llm"


def test_load_completed_rows_tolerates_malformed_trailing_line(tmp_path):
    p = tmp_path / "out.jsonl"
    content = (
        json.dumps({"query_id": "Q-001", "classification_method": "llm"}) + "\n"
        + json.dumps({"query_id": "Q-002", "classification_method": "llm"}) + "\n"
        + '{"query_id": "Q-003'  # half-written line
    )
    p.write_text(content, encoding="utf-8")
    result = _load_completed_rows(p)
    assert "Q-001" in result
    assert "Q-002" in result
    assert "Q-003" not in result


# ---------------------------------------------------------------------------
# _is_row_complete
# ---------------------------------------------------------------------------

def test_is_row_complete_returns_true_for_clean_row():
    row = {"classification_method": "llm", "generation_method": "llm"}
    assert _is_row_complete(row) is True


def test_is_row_complete_returns_false_for_classification_failure():
    row = {"classification_method": "llm_error", "generation_method": "llm"}
    assert _is_row_complete(row) is False


def test_is_row_complete_returns_false_for_generation_failure():
    row = {"classification_method": "llm", "generation_method": "llm_error"}
    assert _is_row_complete(row) is False


# ---------------------------------------------------------------------------
# _is_rate_limit_error
# ---------------------------------------------------------------------------

def test_is_rate_limit_error_detects_keywords():
    assert _is_rate_limit_error(LLMClientError("rate limit exceeded"))
    assert _is_rate_limit_error(LLMClientError("quota exhausted"))
    assert _is_rate_limit_error(LLMClientError("Gemini API call failed: 429"))
    assert not _is_rate_limit_error(LLMClientError("invalid argument"))
    assert not _is_rate_limit_error(RuntimeError("rate limit"))  # not LLMClientError
