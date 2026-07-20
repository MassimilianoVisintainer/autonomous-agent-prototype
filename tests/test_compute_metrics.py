"""Tests for scripts/compute_metrics.py.

All tests use synthetic inline data — no file I/O and no real API calls.
The main() function is NOT invoked; each metric computation function is tested
independently with controlled inputs.
"""

import pytest

from scripts.compute_metrics import (
    _observed_tool_to_tier,
    _parse_escalation_reasons,
    _parse_kb_chunks,
    _parse_tool_calls,
    compute_boundary_pair_metrics,
    compute_containment_metrics,
    compute_escalation_metrics,
    compute_intent_accuracy,
    compute_latency_metrics,
    compute_retrieval_metrics,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(**kwargs) -> dict:
    """Build a minimal row dict with sensible defaults."""
    defaults = {
        "query_id": "Q-001",
        "expected_intent": "order_status",
        "predicted_intent": "order_status",
        "expected_handling": "contained",
        "expected_escalation_reason": None,
        "expected_tool_calls": "nan",
        "expected_kb_chunks": "nan",
        "retrieved_chunk_ids": [],
        "tool_called": None,
        "tool_status": None,
        "tool_result_summary": None,
        "escalation_decision": False,
        "escalation_triggers": [],
        "generation_method": "llm",
        "linked_query_id": None,
        "threshold_test_flag": None,
        "elapsed_ms": 5000,
    }
    defaults.update(kwargs)
    return defaults


# ── _parse_tool_calls ─────────────────────────────────────────────────────────

def test_parse_tool_calls_handles_nan():
    assert _parse_tool_calls("nan") == []


def test_parse_tool_calls_handles_empty():
    assert _parse_tool_calls("") == []
    assert _parse_tool_calls(None) == []


def test_parse_tool_calls_handles_single_call():
    result = _parse_tool_calls("tier3_refund:ORD-1018")
    assert len(result) == 1
    assert result[0]["tool_key"] == "tier3_refund"
    assert result[0]["tier"] == 3
    assert result[0]["identifier"] == "ORD-1018"


def test_parse_tool_calls_handles_multiple_calls():
    result = _parse_tool_calls("tier1_customer_lookup:CUS-0005,tier1_order_lookup:ORD-1006")
    assert len(result) == 2
    assert result[0]["tool_key"] == "tier1_customer_lookup"
    assert result[0]["tier"] == 1
    assert result[0]["identifier"] == "CUS-0005"
    assert result[1]["tool_key"] == "tier1_order_lookup"
    assert result[1]["tier"] == 1
    assert result[1]["identifier"] == "ORD-1006"


# ── _parse_kb_chunks ──────────────────────────────────────────────────────────

def test_parse_kb_chunks_handles_nan():
    assert _parse_kb_chunks("nan") == []


def test_parse_kb_chunks_handles_empty():
    assert _parse_kb_chunks("") == []
    assert _parse_kb_chunks(None) == []


def test_parse_kb_chunks_handles_comma_separated():
    result = _parse_kb_chunks("BL-001,BL-005")
    assert result == ["BL-001", "BL-005"]


def test_parse_kb_chunks_strips_whitespace():
    assert _parse_kb_chunks("BL-001, BL-005 , BL-010") == ["BL-001", "BL-005", "BL-010"]


# ── _parse_escalation_reasons ─────────────────────────────────────────────────

def test_parse_escalation_reasons_handles_none():
    assert _parse_escalation_reasons(None) == []


def test_parse_escalation_reasons_handles_empty_string():
    assert _parse_escalation_reasons("") == []


def test_parse_escalation_reasons_handles_single():
    assert _parse_escalation_reasons("high_emotion") == ["high_emotion"]


def test_parse_escalation_reasons_handles_multi_trigger():
    result = _parse_escalation_reasons("high_emotion,exceeded_authority")
    assert result == ["high_emotion", "exceeded_authority"]


# ── compute_intent_accuracy ───────────────────────────────────────────────────

def _intent_rows(pairs: list[tuple[str, str]]) -> list[dict]:
    return [
        _row(query_id=f"Q-{i:03d}", expected_intent=exp, predicted_intent=pred)
        for i, (exp, pred) in enumerate(pairs, 1)
    ]


def test_compute_intent_accuracy_perfect():
    rows = _intent_rows([
        ("order_status", "order_status"),
        ("refund_request", "refund_request"),
        ("complaint", "complaint"),
        ("shipping_info", "shipping_info"),
        ("return_policy", "return_policy"),
    ])
    result = compute_intent_accuracy(rows)
    assert result.overall_accuracy == 1.0
    assert result.correct == 5
    assert result.total == 5
    assert result.confusion == {}


def test_compute_intent_accuracy_with_misclassifications():
    rows = _intent_rows([
        ("order_status", "order_status"),
        ("order_status", "order_status"),
        ("order_status", "order_status"),
        ("refund_request", "complaint"),   # wrong
        ("shipping_info", "order_status"), # wrong
    ])
    result = compute_intent_accuracy(rows)
    assert result.overall_accuracy == 0.6
    assert result.correct == 3
    assert result.confusion["refund_request"]["complaint"] == 1
    assert result.confusion["shipping_info"]["order_status"] == 1


# ── compute_retrieval_metrics ─────────────────────────────────────────────────

def test_compute_retrieval_metrics_perfect_recall():
    row = _row(
        expected_kb_chunks="RT-001",
        retrieved_chunk_ids=["RT-001", "BL-002", "BL-003", "GN-001", "GN-002"],
    )
    result = compute_retrieval_metrics([row])
    assert result.rows_with_gold == 1
    assert result.mean_recall_at_5 == 1.0
    assert result.mean_precision_at_5 == pytest.approx(0.2, abs=1e-4)


def test_compute_retrieval_metrics_partial_recall():
    row = _row(
        expected_kb_chunks="RT-001,RT-002",
        retrieved_chunk_ids=["RT-001", "BL-002", "BL-003", "GN-001", "GN-002"],
    )
    result = compute_retrieval_metrics([row])
    assert result.mean_recall_at_5 == pytest.approx(0.5, abs=1e-4)


def test_compute_retrieval_metrics_skips_no_gold_rows():
    rows = [
        _row(expected_kb_chunks="nan"),
        _row(expected_kb_chunks="RT-001", retrieved_chunk_ids=["RT-001", "A", "B", "C", "D"]),
    ]
    result = compute_retrieval_metrics(rows)
    assert result.rows_with_gold == 1


# ── compute_escalation_metrics ────────────────────────────────────────────────

def _esc_row(query_id, expected_handling, escalation_decision, escalation_triggers=None, expected_escalation_reason=None):
    return _row(
        query_id=query_id,
        expected_handling=expected_handling,
        escalation_decision=escalation_decision,
        escalation_triggers=escalation_triggers or [],
        expected_escalation_reason=expected_escalation_reason,
    )


def test_compute_escalation_metrics_basic():
    rows = [
        _esc_row("Q-001", "escalated", True, ["high_emotion"], "high_emotion"),   # TP
        _esc_row("Q-002", "escalated", False, [], "high_emotion"),                 # FN
        _esc_row("Q-003", "contained", True, ["high_emotion"], None),              # FP
        _esc_row("Q-004", "contained", False, [], None),                           # TN
        _esc_row("Q-005", "contained", False, [], None),                           # TN
    ]
    result = compute_escalation_metrics(rows)
    assert result.true_positives == 1
    assert result.false_positives == 1
    assert result.false_negatives == 1
    assert result.true_negatives == 2
    assert result.precision == pytest.approx(0.5, abs=1e-4)
    assert result.recall == pytest.approx(0.5, abs=1e-4)


def test_compute_escalation_metrics_per_reason():
    rows = [
        _esc_row("Q-001", "escalated", True, ["high_emotion"], "high_emotion"),
        _esc_row("Q-002", "escalated", True, ["exceeded_authority"], "high_emotion,exceeded_authority"),
        _esc_row("Q-003", "escalated", False, [], "high_emotion"),
        _esc_row("Q-004", "contained", False, [], None),
    ]
    result = compute_escalation_metrics(rows)
    # high_emotion: gt_count=3 (Q-001, Q-002, Q-003), agent triggered for Q-001 and Q-002 (has high_emotion in triggers)
    # tp for high_emotion: Q-001 has high_emotion in both gt and triggers
    assert result.per_reason["high_emotion"]["gt_count"] == 3
    assert result.per_reason["exceeded_authority"]["gt_count"] == 1
    # Recall for exceeded_authority: Q-002 has it in gt and triggers → tp=1, recall=1.0
    assert result.per_reason["exceeded_authority"]["recall"] == 1.0


# ── compute_containment_metrics ───────────────────────────────────────────────

def test_compute_containment_metrics_basic():
    rows = [
        _row(query_id="Q-001", expected_handling="contained", escalation_decision=False, generation_method="llm"),
        _row(query_id="Q-002", expected_handling="contained", escalation_decision=False, generation_method="llm"),
        _row(query_id="Q-003", expected_handling="contained", escalation_decision=True, generation_method="llm_handoff"),   # FP escalation
        _row(query_id="Q-004", expected_handling="contained", escalation_decision=False, generation_method="clarification_template"),  # template on contained
        _row(query_id="Q-005", expected_handling="escalated", escalation_decision=True, generation_method="llm_handoff"),
        _row(query_id="Q-006", expected_handling="clarified", escalation_decision=False, generation_method="clarification_template"),
        _row(query_id="Q-007", expected_handling="clarified", escalation_decision=False, generation_method="llm", tool_status="missing_identifier"),
        _row(query_id="Q-008", expected_handling="clarified", escalation_decision=False, generation_method="llm"),  # missed clarification
    ]
    result = compute_containment_metrics(rows)
    assert result.total_contained == 4
    assert result.gross_contained == 3     # Q-001, Q-002, Q-004 (not FP-escalated)
    assert result.gross_containment_rate == pytest.approx(3 / 4, abs=1e-4)
    assert result.net_contained == 2       # Q-001, Q-002 (not escalated AND not template)
    assert result.net_containment_rate == pytest.approx(2 / 4, abs=1e-4)
    assert result.total_clarified == 3
    assert result.clarification_correct == 2  # Q-006 (template), Q-007 (missing_identifier)
    assert result.clarification_rate == pytest.approx(2 / 3, abs=1e-4)


# ── compute_boundary_pair_metrics ─────────────────────────────────────────────

def test_compute_boundary_pair_transition_correct():
    """Canonical (contained) + boundary (escalated): agent correctly transitions."""
    rows = [
        _row(query_id="Q-001", expected_handling="contained", escalation_decision=False, generation_method="llm"),
        _row(query_id="Q-010", expected_handling="escalated", escalation_decision=True, generation_method="llm_handoff",
             linked_query_id="Q-001", threshold_test_flag="emotion_overlay"),
    ]
    result = compute_boundary_pair_metrics(rows)
    assert result.transition_pairs == 1
    assert result.correct_transitions == 1
    assert result.transition_rate == 1.0


def test_compute_boundary_pair_transition_incorrect():
    """Canonical (contained) + boundary (escalated): agent fails to escalate."""
    rows = [
        _row(query_id="Q-001", expected_handling="contained", escalation_decision=False, generation_method="llm"),
        _row(query_id="Q-010", expected_handling="escalated", escalation_decision=False, generation_method="llm",
             linked_query_id="Q-001", threshold_test_flag="emotion_overlay"),
    ]
    result = compute_boundary_pair_metrics(rows)
    assert result.transition_pairs == 1
    assert result.correct_transitions == 0
    assert result.transition_rate == 0.0


def test_compute_boundary_pair_no_transition_consistency():
    """Two contained rows linked: agent behaves consistently."""
    rows = [
        _row(query_id="Q-001", expected_handling="contained", escalation_decision=False),
        _row(query_id="Q-007", expected_handling="contained", escalation_decision=False, linked_query_id="Q-001"),
    ]
    result = compute_boundary_pair_metrics(rows)
    assert result.no_transition_pairs == 1
    assert result.consistent_no_transition == 1
    assert result.consistency_rate == 1.0


# ── compute_latency_metrics ───────────────────────────────────────────────────

def test_compute_latency_metrics_filters_outliers():
    """A row with elapsed_ms > 600_000 ms is excluded from statistics but counted."""
    rows = [
        _row(query_id=f"Q-{i:03d}", elapsed_ms=5000, predicted_intent="order_status")
        for i in range(1, 10)
    ]
    # Add one obvious outlier
    rows.append(_row(query_id="Q-099", elapsed_ms=16_940_375, predicted_intent="order_status"))
    result = compute_latency_metrics(rows)
    assert result.count_outliers == 1
    assert result.count_included == 9
    assert result.median_ms == pytest.approx(5000, abs=1.0)
    assert result.mean_ms == pytest.approx(5000, abs=1.0)


def test_compute_latency_metrics_per_intent():
    rows = [
        _row(query_id="Q-001", elapsed_ms=3000, predicted_intent="order_status"),
        _row(query_id="Q-002", elapsed_ms=7000, predicted_intent="order_status"),
        _row(query_id="Q-003", elapsed_ms=2000, predicted_intent="refund_request"),
    ]
    result = compute_latency_metrics(rows)
    assert result.per_intent_median["order_status"] == pytest.approx(5000, abs=1.0)
    assert result.per_intent_median["refund_request"] == pytest.approx(2000, abs=1.0)


# ── Relabel regression: out_of_scope refusals must be TN, not FN ────────────────

def test_refused_rows_count_as_true_negatives():
    """A 'refused'-expected row where the agent did not escalate is a true negative,
    not a false negative — this is the §4.5 construct-alignment fix."""
    from scripts.compute_metrics import compute_escalation_metrics
    rows = [
        # refused-expected, agent refused (no escalation) → TN
        {"expected_handling": "refused", "escalation_decision": False,
         "expected_escalation_reason": None, "escalation_triggers": []},
        # escalated-expected, agent escalated → TP
        {"expected_handling": "escalated", "escalation_decision": True,
         "expected_escalation_reason": "high_emotion", "escalation_triggers": ["high_emotion"]},
        # escalated-expected, agent missed → FN
        {"expected_handling": "escalated", "escalation_decision": False,
         "expected_escalation_reason": "exceeded_authority", "escalation_triggers": []},
    ]
    m = compute_escalation_metrics(rows)
    assert m.true_negatives == 1
    assert m.false_negatives == 1
    assert m.true_positives == 1
    assert m.false_positives == 0
    # out_of_scope reason must no longer appear as ground truth
    assert m.per_reason["out_of_scope"]["gt_count"] == 0