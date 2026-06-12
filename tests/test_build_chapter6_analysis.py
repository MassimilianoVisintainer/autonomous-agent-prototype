"""Tests for scripts/build_chapter6_analysis.py."""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from scripts.build_chapter6_analysis import (
    _catalogue_hallucinations,
    _compute_escalation_outcomes_with_rubric,
    _compute_rubric_by_handling,
    _compute_rubric_by_intent,
    _integrate_data,
    _load_eval_rows,
    _load_metrics,
    _load_rubric_scores,
    build_synthesis,
    build_tables,
)


# ── Fixtures ────────────────────────────────────────────────────────────────────

def _make_metrics() -> dict:
    return {
        "intent_accuracy": {
            "correct": 9,
            "total": 10,
            "overall_accuracy": 0.9,
            "per_intent": {
                "order_status": {"correct": 4, "total": 4, "accuracy": 1.0},
                "refund_request": {"correct": 3, "total": 4, "accuracy": 0.75},
                "complaint": {"correct": 2, "total": 2, "accuracy": 1.0},
            },
            "confusion": {"refund_request": {"complaint": 1}},
        },
        "retrieval": {
            "mean_recall_at_5": 0.75,
            "mean_precision_at_5": 0.20,
            "rows_with_gold": 8,
        },
        "tool_calls": {
            "tier_match_count": 7,
            "rows_with_expected": 10,
            "tier_match_rate": 0.7,
            "order_id_match_count": 5,
            "order_id_rows": 6,
            "authority_appropriate": 3,
            "authority_rows": 4,
            "authority_appropriate_rate": 0.75,
            "per_tier": {
                "1": {"expected": 5, "correct": 4, "rate": 0.8},
                "2": {"expected": 3, "correct": 2, "rate": 0.667},
                "3": {"expected": 2, "correct": 1, "rate": 0.5},
            },
        },
        "escalation": {
            "true_positives": 3,
            "false_positives": 1,
            "false_negatives": 2,
            "true_negatives": 4,
            "precision": 0.75,
            "recall": 0.6,
            "f1": 0.667,
            "per_reason": {
                "high_emotion": {"gt_count": 3, "agent_triggered_count": 3, "tp": 2, "fn": 1,
                                 "precision": 0.9, "recall": 0.6},
                "exceeded_authority": {"gt_count": 2, "agent_triggered_count": 2, "tp": 1, "fn": 1,
                                       "precision": 1.0, "recall": 0.5},
                "out_of_scope": {"gt_count": 2, "agent_triggered_count": 0, "tp": 0, "fn": 2,
                                 "precision": 0.0, "recall": 0.0},
            },
        },
        "containment": {
            "gross_contained": 7,
            "total_contained": 8,
            "gross_containment_rate": 0.875,
            "net_contained": 6,
            "net_containment_rate": 0.75,
            "clarification_correct": 2,
            "total_clarified": 3,
            "clarification_rate": 0.667,
        },
        "boundary_pairs": {
            "transition_pairs": 5,
            "correct_transitions": 4,
            "no_transition_pairs": 3,
            "consistent_no_transition": 3,
            "consistency_rate": 1.0,
            "per_threshold_flag": {
                "high_emotion_transactional": {"total": 2, "correct": 2, "rate": 1.0},
                "ambiguity_resolved_by_id": {"total": 2, "correct": 1, "rate": 0.5},
            },
        },
        "latency": {
            "median_ms": 6000,
            "mean_ms": 7500,
            "p90_ms": 18000,
            "p95_ms": 25000,
            "per_intent_median": {"order_status": 5000, "refund_request": 8000, "complaint": 6000},
        },
        "summary": {"total_rows": 10, "harness_ok": 10},
    }


def _make_eval_rows() -> list[dict]:
    return [
        {"query_id": "Q-001", "query_text": "Where is my order ORD-1001?",
         "expected_intent": "order_status", "predicted_intent": "order_status",
         "expected_handling": "contained", "escalation_decision": False,
         "generation_method": "llm", "tool_called": "lookup_order",
         "tool_status": "ok", "elapsed_ms": 4500},
        {"query_id": "Q-002", "query_text": "I want a refund for ORD-1002",
         "expected_intent": "refund_request", "predicted_intent": "complaint",
         "expected_handling": "contained", "escalation_decision": False,
         "generation_method": "llm", "tool_called": None, "tool_status": None,
         "elapsed_ms": 6000},
        {"query_id": "Q-003", "query_text": "I am so angry, I need help NOW",
         "expected_intent": "complaint", "predicted_intent": "complaint",
         "expected_handling": "escalated", "escalation_decision": True,
         "generation_method": "llm_handoff", "tool_called": None, "tool_status": None,
         "elapsed_ms": 9000},
        {"query_id": "Q-004", "query_text": "Can you help me with my order ORD-1004?",
         "expected_intent": "order_status", "predicted_intent": "order_status",
         "expected_handling": "contained", "escalation_decision": False,
         "generation_method": "llm", "tool_called": "lookup_order",
         "tool_status": "ok", "elapsed_ms": 5200},
        {"query_id": "Q-005", "query_text": "What is your return policy?",
         "expected_intent": "refund_request", "predicted_intent": "refund_request",
         "expected_handling": "escalated", "escalation_decision": False,
         "generation_method": "llm", "tool_called": None, "tool_status": None,
         "elapsed_ms": 3800},
    ]


def _make_rubric_scores() -> dict:
    """Return {query_id: {dimension: record}}."""
    scores = {
        "Q-001": {
            "factual_accuracy": {"score": 5, "notes": ""},
            "completeness": {"score": 4, "notes": ""},
            "tone_appropriateness": {"score": 4, "notes": ""},
            "structural_quality": {"score": 4, "notes": ""},
            "hallucination_present": {"score": False, "notes": ""},
        },
        "Q-002": {
            "factual_accuracy": {"score": 4, "notes": ""},
            "completeness": {"score": 3, "notes": ""},
            "tone_appropriateness": {"score": 3, "notes": ""},
            "structural_quality": {"score": 3, "notes": ""},
            "hallucination_present": {"score": False, "notes": ""},
        },
        "Q-003": {
            "factual_accuracy": {"score": 3, "notes": ""},
            "completeness": {"score": 3, "notes": ""},
            "tone_appropriateness": {"score": 5, "notes": ""},
            "structural_quality": {"score": 3, "notes": ""},
            "hallucination_present": {"score": False, "notes": ""},
        },
        "Q-004": {
            "factual_accuracy": {"score": 5, "notes": ""},
            "completeness": {"score": 5, "notes": ""},
            "tone_appropriateness": {"score": 4, "notes": ""},
            "structural_quality": {"score": 4, "notes": ""},
            "hallucination_present": {"score": True, "notes": "asks for order id again when provided"},
        },
        "Q-005": {
            "factual_accuracy": {"score": 4, "notes": ""},
            "completeness": {"score": 4, "notes": ""},
            "tone_appropriateness": {"score": 4, "notes": ""},
            "structural_quality": {"score": 3, "notes": ""},
            "hallucination_present": {"score": False, "notes": ""},
        },
    }
    return scores


@pytest.fixture
def integrated_data():
    metrics = _make_metrics()
    eval_rows = _make_eval_rows()
    rubric = _make_rubric_scores()
    return _integrate_data(metrics, rubric, eval_rows)


# ── Unit tests for data loading ─────────────────────────────────────────────────

def test_load_metrics(tmp_path: pathlib.Path) -> None:
    m = _make_metrics()
    p = tmp_path / "metrics_report.json"
    p.write_text(json.dumps(m), encoding="utf-8")
    result = _load_metrics(p)
    assert result["intent_accuracy"]["total"] == 10
    assert result["escalation"]["precision"] == 0.75


def test_load_rubric_scores(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "rubric_scores.jsonl"
    lines = [
        json.dumps({"query_id": "Q-001", "dimension": "factual_accuracy", "score": 5, "notes": ""}),
        json.dumps({"query_id": "Q-001", "dimension": "completeness", "score": 4, "notes": ""}),
        json.dumps({"query_id": "Q-002", "dimension": "factual_accuracy", "score": 3, "notes": "x"}),
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    result = _load_rubric_scores(p)
    assert result["Q-001"]["factual_accuracy"]["score"] == 5
    assert result["Q-001"]["completeness"]["score"] == 4
    assert result["Q-002"]["factual_accuracy"]["notes"] == "x"


def test_load_rubric_scores_missing_file(tmp_path: pathlib.Path) -> None:
    result = _load_rubric_scores(tmp_path / "nonexistent.jsonl")
    assert result == {}


def test_load_eval_rows(tmp_path: pathlib.Path) -> None:
    rows = _make_eval_rows()
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    loaded = _load_eval_rows(p)
    assert len(loaded) == len(rows)
    assert loaded[0]["query_id"] == "Q-001"


def test_integrate_data_merges_rubric(integrated_data: dict) -> None:
    per_query = integrated_data["per_query"]
    assert "Q-001" in per_query
    assert per_query["Q-001"]["rubric"]["factual_accuracy"]["score"] == 5
    assert per_query["Q-001"]["query_text"] == "Where is my order ORD-1001?"


# ── Unit tests for cross-tab functions ──────────────────────────────────────────

def test_compute_rubric_by_handling_means(integrated_data: dict) -> None:
    rh = _compute_rubric_by_handling(integrated_data)
    # Q-001, Q-002, Q-004 are contained; Q-003 is escalated; Q-005 is escalated (FN)
    assert "contained" in rh
    assert "escalated" in rh
    # factual_accuracy: Q-001=5, Q-002=4, Q-004=5 → mean=4.667 for contained
    assert abs(rh["contained"]["factual_accuracy"]["mean"] - (5 + 4 + 5) / 3) < 0.01


def test_compute_rubric_by_handling_counts(integrated_data: dict) -> None:
    rh = _compute_rubric_by_handling(integrated_data)
    assert rh["contained"]["factual_accuracy"]["n"] == 3
    assert rh["escalated"]["factual_accuracy"]["n"] == 2


def test_compute_rubric_by_intent_keys(integrated_data: dict) -> None:
    ri = _compute_rubric_by_intent(integrated_data)
    assert "order_status" in ri
    assert "factual_accuracy" in ri["order_status"]


def test_compute_escalation_outcomes_categories(integrated_data: dict) -> None:
    cats = _compute_escalation_outcomes_with_rubric(integrated_data)
    # Q-003: escalated + expected escalated → TP
    assert "true_positive_escalation" in cats
    assert cats["true_positive_escalation"]["count"] == 1
    # Q-005: not escalated + expected escalated → FN
    assert "false_negative_escalation" in cats


def test_catalogue_hallucinations_finds_record(integrated_data: dict) -> None:
    hal = _catalogue_hallucinations(integrated_data)
    # Q-004 has hallucination_present=True with "asks for order id again when provided"
    assert len(hal) == 1
    assert hal[0]["query_id"] == "Q-004"
    assert hal[0]["failure_mode_category"] == "order_id_ignored"


def test_catalogue_hallucinations_empty_when_none(integrated_data: dict) -> None:
    # Remove hallucination flag from Q-004
    integrated_data["per_query"]["Q-004"]["rubric"]["hallucination_present"]["score"] = False
    hal = _catalogue_hallucinations(integrated_data)
    assert hal == []


# ── Integration test: build_tables and build_synthesis ─────────────────────────

def test_build_tables_creates_file(integrated_data: dict, tmp_path: pathlib.Path) -> None:
    build_tables(integrated_data, tmp_path)
    tables_file = tmp_path / "tables.md"
    assert tables_file.exists()
    content = tables_file.read_text(encoding="utf-8")
    table_count = content.count("## Table")
    assert table_count == 7


def test_build_tables_contains_all_headers(integrated_data: dict, tmp_path: pathlib.Path) -> None:
    build_tables(integrated_data, tmp_path)
    content = (tmp_path / "tables.md").read_text(encoding="utf-8")
    for title in [
        "Table 6.1", "Table 6.2", "Table 6.3", "Table 6.4",
        "Table 6.5", "Table 6.6", "Table 6.7",
    ]:
        assert title in content, f"{title} missing from tables.md"


def test_build_synthesis_creates_file(integrated_data: dict, tmp_path: pathlib.Path) -> None:
    build_synthesis(integrated_data, tmp_path, {
        "metrics": "metrics_report.json",
        "rubric": "rubric_scores.jsonl",
        "eval": "eval_results.jsonl",
    })
    synth_file = tmp_path / "chapter6_synthesis.md"
    assert synth_file.exists()
    content = synth_file.read_text(encoding="utf-8")
    for section in ["RQ1", "RQ2", "RQ3", "RQ4"]:
        assert section in content, f"{section} missing from synthesis"


def test_build_synthesis_embeds_computed_values(
    integrated_data: dict, tmp_path: pathlib.Path
) -> None:
    build_synthesis(integrated_data, tmp_path, {
        "metrics": "metrics_report.json",
        "rubric": "rubric_scores.jsonl",
        "eval": "eval_results.jsonl",
    })
    content = (tmp_path / "chapter6_synthesis.md").read_text(encoding="utf-8")
    # Gross containment 7/8 = 87.5%
    assert "87.5%" in content
    # Hallucination 1/5 = 20%
    assert "20.0%" in content
