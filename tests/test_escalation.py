"""Tests for src/escalation.py.

All three triggers are tested independently, plus multi-trigger and boundary-intent
exclusion cases. VADER is called directly (it is fast, deterministic, and offline).
"""

import pytest

from src.escalation import (
    CONFIDENCE_FLOOR,
    EMOTION_FLOOR,
    ESCALATION_TRIGGER_AUTHORITY,
    ESCALATION_TRIGGER_CONFIDENCE,
    ESCALATION_TRIGGER_EMOTION,
    detect_emotion,
    evaluate_escalation,
)
from src.intents import Intent
from src.nlu import ClassificationResult
from src.tools import ToolResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cls(intent: Intent = Intent.PRODUCT_INFO, confidence: float = 0.92) -> ClassificationResult:
    return ClassificationResult(
        intent=intent, confidence=confidence, reasoning="test", method="llm"
    )


def _tool(status: str) -> ToolResult:
    return ToolResult(status=status, tool="process_refund", data={}, reason=None)


# ---------------------------------------------------------------------------
# detect_emotion
# ---------------------------------------------------------------------------

def test_detect_emotion_returns_negative_for_angry_text():
    score = detect_emotion("This is absolutely unacceptable!!! I'm furious!")
    assert score < EMOTION_FLOOR


def test_detect_emotion_returns_near_neutral_for_factual_text():
    score = detect_emotion("Where is my order ORD-1024?")
    assert abs(score) < 0.3


def test_detect_emotion_returns_zero_for_empty_text():
    assert detect_emotion("") == 0.0


# ---------------------------------------------------------------------------
# evaluate_escalation — no escalation
# ---------------------------------------------------------------------------

def test_no_escalation_for_confident_neutral_no_tool_breach():
    result = evaluate_escalation(_cls(), None, "Where is my order ORD-1024?")
    assert result.should_escalate is False
    assert result.triggers == []


# ---------------------------------------------------------------------------
# evaluate_escalation — single triggers
# ---------------------------------------------------------------------------

def test_escalation_by_exceeded_authority_alone():
    result = evaluate_escalation(_cls(), _tool("exceeded_authority"), "I want a refund.")
    assert result.should_escalate is True
    assert result.triggers == [ESCALATION_TRIGGER_AUTHORITY]


def test_escalation_by_out_of_window_alone():
    result = evaluate_escalation(_cls(), _tool("out_of_window"), "Cancel my order.")
    assert result.should_escalate is True
    assert ESCALATION_TRIGGER_AUTHORITY in result.triggers


def test_escalation_by_high_emotion_alone():
    result = evaluate_escalation(_cls(), None, "I am absolutely livid with this service!!!")
    assert result.should_escalate is True
    assert ESCALATION_TRIGGER_EMOTION in result.triggers


def test_escalation_by_low_confidence_alone():
    result = evaluate_escalation(
        _cls(intent=Intent.ORDER_STATUS, confidence=0.4),
        None,
        "Where is my order ORD-1024?",
    )
    assert result.should_escalate is True
    assert ESCALATION_TRIGGER_CONFIDENCE in result.triggers


# ---------------------------------------------------------------------------
# Boundary-intent exclusion
# ---------------------------------------------------------------------------

def test_low_confidence_on_ambiguous_intent_does_not_trigger():
    result = evaluate_escalation(
        _cls(intent=Intent.AMBIGUOUS_QUERY, confidence=0.3),
        None,
        "Help.",
    )
    assert ESCALATION_TRIGGER_CONFIDENCE not in result.triggers


def test_low_confidence_on_out_of_scope_does_not_trigger():
    result = evaluate_escalation(
        _cls(intent=Intent.OUT_OF_SCOPE, confidence=0.3),
        None,
        "What is the weather?",
    )
    assert ESCALATION_TRIGGER_CONFIDENCE not in result.triggers


# ---------------------------------------------------------------------------
# Multi-trigger
# ---------------------------------------------------------------------------

def test_multiple_triggers_recorded():
    result = evaluate_escalation(
        _cls(),
        _tool("exceeded_authority"),
        "This is absolutely unacceptable!!! I want a refund NOW!!!",
    )
    assert result.should_escalate is True
    assert ESCALATION_TRIGGER_AUTHORITY in result.triggers
    assert ESCALATION_TRIGGER_EMOTION in result.triggers


def test_reason_string_names_firing_triggers():
    result = evaluate_escalation(
        _cls(),
        _tool("exceeded_authority"),
        "I am furious, refund me now!!!",
    )
    assert ESCALATION_TRIGGER_AUTHORITY in result.reason
