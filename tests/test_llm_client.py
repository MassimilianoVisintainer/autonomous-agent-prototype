"""Tests for src/llm_client.py using mocked Gemini responses.

No real API calls are made; all network interaction is replaced with
unittest.mock objects. The test suite runs without GOOGLE_API_KEY.
"""

from unittest.mock import MagicMock, patch

import pytest

from src import llm_client
from src.llm_client import LLMClientError, complete


def _make_mock_client(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def test_complete_raises_when_api_key_missing(monkeypatch):
    """complete() raises LLMClientError when no API key is configured."""
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "_client", None)
    with pytest.raises(LLMClientError, match="GOOGLE_API_KEY is not configured"):
        complete(system="sys", user="hello")


def test_complete_uses_cache_when_available(monkeypatch, tmp_path):
    """The second call with identical inputs hits the cache, not the API."""
    mock_client = _make_mock_client(
        '{"intent": "order_status", "confidence": 0.9, "reasoning": "test"}'
    )
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "0")
    monkeypatch.setattr(llm_client, "CACHE_PATH", tmp_path / "llm_responses.json")

    first = complete(system="sys", user="where is my order?", json_mode=True)
    second = complete(system="sys", user="where is my order?", json_mode=True)

    assert first == second
    # generate_content should only be called once — second call is served from cache
    assert mock_client.models.generate_content.call_count == 1


def test_cache_disabled_via_env_var(monkeypatch, tmp_path):
    """With DISABLE_LLM_CACHE=1, every call reaches the API."""
    mock_client = _make_mock_client(
        '{"intent": "order_status", "confidence": 0.9, "reasoning": "test"}'
    )
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")
    monkeypatch.setattr(llm_client, "CACHE_PATH", tmp_path / "llm_responses.json")

    complete(system="sys", user="where is my order?", json_mode=True)
    complete(system="sys", user="where is my order?", json_mode=True)

    assert mock_client.models.generate_content.call_count == 2


def test_complete_raises_on_api_error(monkeypatch):
    """complete() wraps Gemini exceptions in LLMClientError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("network timeout")
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")

    with pytest.raises(LLMClientError, match="Gemini API call failed"):
        complete(system="sys", user="hello")
