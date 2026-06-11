"""Tests for scripts/rubric_scorer.py helper functions.

All tests use synthetic inline data and tmp_path fixtures.
Streamlit's UI is NOT invoked; only the pure-Python helper functions are tested.
"""

import json
import pathlib

import pytest

from scripts.rubric_scorer import (
    ALL_DIMENSIONS,
    LIKERT_DIMENSIONS,
    _compute_progress,
    _load_eval_rows,
    _load_existing_scores,
    _save_score,
)


# ── Fixtures and helpers ───────────────────────────────────────────────────────

def _write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _eval_row(query_id: str = "Q-001", **kwargs) -> dict:
    base = {
        "query_id": query_id,
        "query_text": f"Sample query {query_id}",
        "predicted_intent": "order_status",
        "classification_confidence": 0.95,
        "tool_called": None,
        "tool_status": None,
        "escalation_decision": False,
        "escalation_triggers": [],
        "generation_text": "Your order is in transit.",
        "generation_method": "llm",
    }
    base.update(kwargs)
    return base


def _score_row(query_id: str, dimension: str, score, notes: str = "") -> dict:
    return {
        "query_id": query_id,
        "dimension": dimension,
        "score": score,
        "notes": notes,
        "scored_at": "2026-06-11T08:00:00Z",
    }


# ── _load_eval_rows ────────────────────────────────────────────────────────────

def test_load_eval_rows_returns_list(tmp_path):
    p = tmp_path / "eval.jsonl"
    rows = [_eval_row(f"Q-{i:03d}") for i in range(1, 4)]
    _write_jsonl(p, rows)
    result = _load_eval_rows(p)
    assert isinstance(result, list)
    assert len(result) == 3
    assert result[0]["query_id"] == "Q-001"


def test_load_eval_rows_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_eval_rows(tmp_path / "nonexistent.jsonl")


def test_load_eval_rows_raises_on_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        _load_eval_rows(p)


# ── _load_existing_scores ──────────────────────────────────────────────────────

def test_load_existing_scores_returns_empty_for_missing_file(tmp_path):
    result = _load_existing_scores(tmp_path / "nonexistent.jsonl")
    assert result == {}


def test_load_existing_scores_parses_jsonl(tmp_path):
    p = tmp_path / "scores.jsonl"
    records = [
        _score_row("Q-001", "factual_accuracy", 4),
        _score_row("Q-001", "completeness", 5),
        _score_row("Q-002", "factual_accuracy", 3),
    ]
    _write_jsonl(p, records)
    result = _load_existing_scores(p)
    assert ("Q-001", "factual_accuracy") in result
    assert ("Q-001", "completeness") in result
    assert ("Q-002", "factual_accuracy") in result
    assert result[("Q-001", "factual_accuracy")]["score"] == 4
    assert result[("Q-001", "completeness")]["score"] == 5


def test_load_existing_scores_tolerates_malformed_lines(tmp_path):
    p = tmp_path / "scores.jsonl"
    p.write_text(
        json.dumps(_score_row("Q-001", "factual_accuracy", 4)) + "\n"
        + '{"broken json\n'
        + json.dumps(_score_row("Q-002", "completeness", 3)) + "\n",
        encoding="utf-8",
    )
    result = _load_existing_scores(p)
    assert ("Q-001", "factual_accuracy") in result
    assert ("Q-002", "completeness") in result
    assert len(result) == 2


# ── _save_score ────────────────────────────────────────────────────────────────

def test_save_score_writes_new_record(tmp_path):
    out = tmp_path / "scores.jsonl"
    hist = tmp_path / "history.jsonl"
    _save_score("Q-001", "factual_accuracy", 4, "", out, hist)
    result = _load_existing_scores(out)
    assert ("Q-001", "factual_accuracy") in result
    assert result[("Q-001", "factual_accuracy")]["score"] == 4
    assert not hist.exists()


def test_save_score_overwrites_existing_record(tmp_path):
    out = tmp_path / "scores.jsonl"
    hist = tmp_path / "history.jsonl"
    _save_score("Q-001", "factual_accuracy", 3, "", out, hist)
    _save_score("Q-001", "factual_accuracy", 5, "", out, hist)
    result = _load_existing_scores(out)
    assert result[("Q-001", "factual_accuracy")]["score"] == 5


def test_save_score_archives_to_history(tmp_path):
    out = tmp_path / "scores.jsonl"
    hist = tmp_path / "history.jsonl"
    _save_score("Q-001", "factual_accuracy", 3, "original", out, hist)
    _save_score("Q-001", "factual_accuracy", 5, "updated", out, hist)
    assert hist.exists()
    history_lines = [json.loads(l) for l in hist.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(history_lines) == 1
    assert history_lines[0]["score"] == 3
    assert history_lines[0]["notes"] == "original"
    assert "archived_at" in history_lines[0]


def test_save_score_preserves_other_records(tmp_path):
    out = tmp_path / "scores.jsonl"
    hist = tmp_path / "history.jsonl"
    _save_score("Q-001", "factual_accuracy", 4, "", out, hist)
    _save_score("Q-001", "completeness", 3, "", out, hist)
    _save_score("Q-002", "factual_accuracy", 5, "", out, hist)
    result = _load_existing_scores(out)
    assert len(result) == 3
    assert result[("Q-001", "completeness")]["score"] == 3
    assert result[("Q-002", "factual_accuracy")]["score"] == 5


def test_save_score_persists_hallucination_bool_and_notes(tmp_path):
    out = tmp_path / "scores.jsonl"
    hist = tmp_path / "history.jsonl"
    _save_score("Q-001", "hallucination_present", True, "Invented tracking number", out, hist)
    result = _load_existing_scores(out)
    rec = result[("Q-001", "hallucination_present")]
    assert rec["score"] is True
    assert rec["notes"] == "Invented tracking number"


# ── _compute_progress ──────────────────────────────────────────────────────────

def test_compute_progress_handles_empty():
    result = _compute_progress({}, total_queries=130)
    assert result["queries_fully_scored"] == 0
    assert result["total_judgments"] == 0
    assert result["target_judgments"] == 130 * len(ALL_DIMENSIONS)
    for dim in ALL_DIMENSIONS:
        assert result["per_dimension_completion"][dim] == 0


def test_compute_progress_counts_correctly():
    # Q-001: all 5 dimensions scored → fully scored
    # Q-002: 4 of 5 dimensions → not fully scored
    scores = {}
    for dim in ALL_DIMENSIONS:
        scores[("Q-001", dim)] = _score_row("Q-001", dim, 4)
    for dim in LIKERT_DIMENSIONS:  # 4 out of 5
        scores[("Q-002", dim)] = _score_row("Q-002", dim, 3)

    result = _compute_progress(scores, total_queries=5)
    assert result["queries_fully_scored"] == 1
    assert result["total_judgments"] == len(ALL_DIMENSIONS) + len(LIKERT_DIMENSIONS)
    assert result["target_judgments"] == 5 * len(ALL_DIMENSIONS)
    assert result["per_dimension_completion"]["factual_accuracy"] == 2
    assert result["per_dimension_completion"]["hallucination_present"] == 1


def test_compute_progress_fully_scored_requires_all_dimensions():
    scores = {("Q-001", dim): _score_row("Q-001", dim, 4) for dim in LIKERT_DIMENSIONS}
    result = _compute_progress(scores, total_queries=5)
    assert result["queries_fully_scored"] == 0


def test_compute_progress_multiple_fully_scored():
    scores = {}
    for qid in ["Q-001", "Q-002", "Q-003"]:
        for dim in ALL_DIMENSIONS:
            scores[(qid, dim)] = _score_row(qid, dim, 4)
    result = _compute_progress(scores, total_queries=10)
    assert result["queries_fully_scored"] == 3
    assert result["total_judgments"] == 3 * len(ALL_DIMENSIONS)
