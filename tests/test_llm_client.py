"""Tests for src/llm_client.py using mocked Gemini responses.

No real API calls are made; all network interaction is replaced with
unittest.mock objects. The test suite runs without GOOGLE_API_KEY.
"""

from unittest.mock import MagicMock, patch

import pytest

from src import llm_client
from src.llm_client import LLMClientError, RETRY_DELAYS, complete


def _make_mock_client(response_text: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.text = response_text
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response
    return mock_client


def _rate_limit_exc() -> Exception:
    return RuntimeError("429 RESOURCE_EXHAUSTED rate limit exceeded")


# ---------------------------------------------------------------------------
# Core tests
# ---------------------------------------------------------------------------

def test_complete_raises_when_api_key_missing(monkeypatch):
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", False)
    monkeypatch.setattr(llm_client, "_client", None)
    with pytest.raises(LLMClientError, match="GOOGLE_API_KEY is not configured"):
        complete(system="sys", user="hello")


def test_complete_uses_cache_when_available(monkeypatch, tmp_path):
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
    assert mock_client.models.generate_content.call_count == 1


def test_cache_disabled_via_env_var(monkeypatch, tmp_path):
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
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("network timeout")
    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")

    with pytest.raises(LLMClientError, match="Gemini API call failed"):
        complete(system="sys", user="hello")


# ---------------------------------------------------------------------------
# Retry tests
# ---------------------------------------------------------------------------

def test_complete_retries_on_rate_limit_error(monkeypatch, tmp_path):
    """One rate-limit failure followed by success — generate_content called twice."""
    mock_response = MagicMock()
    mock_response.text = "Hello!"
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [_rate_limit_exc(), mock_response]

    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    result = complete(system="sys", user="hi")

    assert result == "Hello!"
    assert mock_client.models.generate_content.call_count == 2


def test_complete_retries_up_to_three_times_then_raises(monkeypatch):
    """Four rate-limit failures exhaust all retries → LLMClientError."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = [_rate_limit_exc()] * (len(RETRY_DELAYS) + 1)

    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    with pytest.raises(LLMClientError, match="Rate-limited after"):
        complete(system="sys", user="hi")

    assert mock_client.models.generate_content.call_count == len(RETRY_DELAYS) + 1


def test_complete_does_not_retry_on_non_rate_limit_error(monkeypatch):
    """A non-rate-limit error raises immediately without retrying."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = RuntimeError("Invalid argument")

    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "1")
    monkeypatch.setattr(llm_client.time, "sleep", lambda _: None)

    with pytest.raises(LLMClientError, match="Gemini API call failed"):
        complete(system="sys", user="hi")

    assert mock_client.models.generate_content.call_count == 1


def test_complete_does_not_retry_on_cache_hit(monkeypatch, tmp_path):
    """A cache hit returns immediately without touching generate_content."""
    import json, hashlib
    cache_path = tmp_path / "llm_responses.json"
    raw = "sys\x00hi\x00False"
    key = hashlib.sha256(raw.encode()).hexdigest()
    cache_path.write_text(json.dumps({key: "cached response"}), encoding="utf-8")

    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = _rate_limit_exc()

    monkeypatch.setattr(llm_client, "LLM_AVAILABLE", True)
    monkeypatch.setattr(llm_client, "_client", mock_client)
    monkeypatch.setenv("DISABLE_LLM_CACHE", "0")
    monkeypatch.setattr(llm_client, "CACHE_PATH", cache_path)

    result = complete(system="sys", user="hi")

    assert result == "cached response"
    assert mock_client.models.generate_content.call_count == 0
