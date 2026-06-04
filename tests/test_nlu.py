"""Tests for the LLM-based intent classifier in src/nlu.py.

All LLM calls are mocked — the test suite runs without a GOOGLE_API_KEY.
"""

import json
from unittest.mock import patch

import pytest

from src import llm_client
from src.intents import Intent
from src.llm_client import LLMClientError
from src.nlu import classify


def _mock_llm(response_dict: dict):
    """Return a context manager that patches llm_client.complete to return JSON."""
    return patch.object(
        llm_client,
        "complete",
        return_value=json.dumps(response_dict),
    )


def test_empty_query_returns_ambiguous():
    result = classify("")
    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.method == "empty_input"


def test_whitespace_only_query_returns_ambiguous():
    result = classify("   \n\t  ")
    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.method == "empty_input"


def test_llm_returns_valid_intent():
    payload = {
        "intent": "order_status",
        "confidence": 0.92,
        "reasoning": "The user is asking about an order's location.",
    }
    with _mock_llm(payload):
        result = classify("Where is my order ORD-1024?")

    assert result.intent == Intent.ORDER_STATUS
    assert result.confidence == pytest.approx(0.92)
    assert result.reasoning == "The user is asking about an order's location."
    assert result.method == "llm"


def test_llm_returns_invalid_intent_code_falls_back_to_ambiguous():
    payload = {
        "intent": "not_a_real_intent",
        "confidence": 0.8,
        "reasoning": "Some reasoning.",
    }
    with _mock_llm(payload):
        result = classify("some query")

    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.method == "llm_error"
    assert result.reasoning == "LLM returned unknown intent code"


def test_llm_returns_malformed_json_falls_back_to_ambiguous():
    with patch.object(llm_client, "complete", return_value="this is not json"):
        result = classify("some query")

    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.method == "llm_error"
    assert result.reasoning == "Failed to parse LLM response"


def test_llm_client_error_falls_back_to_ambiguous():
    with patch.object(llm_client, "complete", side_effect=LLMClientError("API down")):
        result = classify("some query")

    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.method == "llm_error"
    assert result.reasoning == "LLM call failed"
