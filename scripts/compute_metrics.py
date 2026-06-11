#!/usr/bin/env python3
"""Compute the eight §4.4 metrics from an evaluation JSONL file.

Metrics:
  1. Intent classification accuracy  — overall, per-intent, confusion matrix
  2. Retrieval precision@5 / recall@5 — over rows that have retrieval gold
  3. Tool-call correctness            — tier-level match, order-ID match, status appropriateness
  4. Escalation precision/recall/F1   — overall and per ground-truth reason
  5. Outcome containment              — gross (no FP escalations) and net (substantive response)
  6. Boundary-pair transition rate    — paired variant/canonical queries
  7. Latency                          — median, mean, p90, p95, per-intent median
  8. Summary statistics

Parsing applied to harness output:
  expected_kb_chunks      — "nan" or "" → []; else comma-split into chunk-ID list
  expected_tool_calls     — "nan" or "" → []; else comma-split; tier from "tierN_" prefix
  expected_escalation_reason — None or "" → []; else comma-split

Outputs (written to --output-dir, default evaluation_results/):
  metrics_report.md   — Markdown report suitable for Chapter 6 quotation
  metrics_report.json — machine-readable with identical values
"""

import argparse
import json
import pathlib
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, median, quantiles

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

OUTCOME_CATEGORIES = ("contained", "escalated", "clarified")

INTENT_NAMES = (
    "order_status", "order_modify", "order_cancel", "refund_request",
    "product_info", "return_policy", "shipping_info", "account_help",
    "complaint", "multi_issue_dispute", "out_of_scope", "ambiguous_query",
)

ESCALATION_REASONS = ("high_emotion", "exceeded_authority", "out_of_scope", "low_confidence")

# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IntentAccuracy:
    overall_accuracy: float
    correct: int
    total: int
    per_intent: dict
    confusion: dict


@dataclass(frozen=True)
class RetrievalMetrics:
    rows_with_gold: int
    mean_precision_at_5: float
    mean_recall_at_5: float
    recall_distribution: dict


@dataclass(frozen=True)
class ToolCallMetrics:
    rows_with_expected: int
    tier_match_count: int
    tier_match_rate: float
    order_id_rows: int
    order_id_match_count: int
    order_id_match_rate: float
    authority_rows: int
    authority_appropriate: int
    authority_appropriate_rate: float
    per_tier: dict


@dataclass(frozen=True)
class EscalationMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    per_reason: dict


@dataclass(frozen=True)
class ContainmentMetrics:
    total_contained: int
    gross_contained: int
    gross_containment_rate: float
    net_contained: int
    net_containment_rate: float
    total_clarified: int
    clarification_correct: int
    clarification_rate: float


@dataclass(frozen=True)
class BoundaryPairMetrics:
    total_variant_rows: int
    transition_pairs: int
    correct_transitions: int
    transition_rate: float
    no_transition_pairs: int
    consistent_no_transition: int
    consistency_rate: float
    per_threshold_flag: dict


@dataclass(frozen=True)
class LatencyMetrics:
    count_included: int
    count_outliers: int
    median_ms: float
    mean_ms: float
    p90_ms: float
    p95_ms: float
    min_ms: int
    max_ms: int
    per_intent_median: dict


@dataclass(frozen=True)
class SummaryStats:
    total: int
    by_harness_status: dict
    by_classification_method: dict
    by_generation_method: dict


@dataclass(frozen=True)
class MetricsReport:
    computed_at: str
    input_file: str
    intent_accuracy: IntentAccuracy
    retrieval: RetrievalMetrics
    tool_calls: ToolCallMetrics
    escalation: EscalationMetrics
    containment: ContainmentMetrics
    boundary_pairs: BoundaryPairMetrics
    latency: LatencyMetrics
    summary: SummaryStats


# ── Parsing helpers ────────────────────────────────────────────────────────────

def _parse_tool_calls(raw) -> list[dict]:
    if not raw or raw == "nan":
        return []
    result = []
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        tool_key, identifier = (piece.split(":", 1) if ":" in piece else (piece, ""))
        tier = (
            int(tool_key[4])
            if tool_key.startswith("tier") and len(tool_key) > 4 and tool_key[4].isdigit()
            else 0
        )
        result.append({"tool_key": tool_key, "tier": tier, "identifier": identifier})
    return result


def _parse_kb_chunks(raw) -> list[str]:
    if not raw or raw == "nan":
        return []
    return [c.strip() for c in str(raw).split(",") if c.strip()]


def _parse_escalation_reasons(raw) -> list[str]:
    if not raw:
        return []
    return [r.strip() for r in str(raw).split(",") if r.strip()]


def _observed_tool_to_tier(tool_name) -> int | None:
    return {"lookup_order": 1, "modify_order": 2, "cancel_order": 2, "process_refund": 3}.get(
        tool_name
    )


def _observed_category(row: dict) -> str:
    if row.get("escalation_decision"):
        return "escalated"
    if (
        row.get("generation_method") == "clarification_template"
        or row.get("tool_status") == "missing_identifier"
    ):
        return "clarified"
    return "contained"


# ── Metric computation ─────────────────────────────────────────────────────────

def compute_intent_accuracy(rows: list[dict]) -> IntentAccuracy:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("predicted_intent") == r.get("expected_intent"))

    per_intent = {}
    for intent in INTENT_NAMES:
        subset = [r for r in rows if r.get("expected_intent") == intent]
        if not subset:
            continue
        n = sum(1 for r in subset if r.get("predicted_intent") == intent)
        per_intent[intent] = {"correct": n, "total": len(subset), "accuracy": round(n / len(subset), 4)}

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        exp, pred = r.get("expected_intent"), r.get("predicted_intent")
        if exp and pred and exp != pred:
            confusion[exp][pred] += 1

    return IntentAccuracy(
        overall_accuracy=round(correct / total, 4) if total else 0.0,
        correct=correct,
        total=total,
        per_intent=dict(per_intent),
        confusion={k: dict(v) for k, v in confusion.items()},
    )


def compute_retrieval_metrics(rows: list[dict]) -> RetrievalMetrics:
    eligible = [r for r in rows if _parse_kb_chunks(r.get("expected_kb_chunks", ""))]
    if not eligible:
        return RetrievalMetrics(0, 0.0, 0.0, {})

    precisions, recalls = [], []
    for r in eligible:
        gold = set(_parse_kb_chunks(r.get("expected_kb_chunks", "")))
        retrieved = set(r.get("retrieved_chunk_ids") or [])
        inter = len(gold & retrieved)
        precisions.append(inter / 5)
        recalls.append(inter / len(gold))

    buckets = {"[0.00,0.25)": 0, "[0.25,0.50)": 0, "[0.50,0.75)": 0, "[0.75,1.00)": 0, "[1.00,1.00]": 0}
    for v in recalls:
        if v < 0.25:
            buckets["[0.00,0.25)"] += 1
        elif v < 0.50:
            buckets["[0.25,0.50)"] += 1
        elif v < 0.75:
            buckets["[0.50,0.75)"] += 1
        elif v < 1.0:
            buckets["[0.75,1.00)"] += 1
        else:
            buckets["[1.00,1.00]"] += 1

    return RetrievalMetrics(
        rows_with_gold=len(eligible),
        mean_precision_at_5=round(mean(precisions), 4),
        mean_recall_at_5=round(mean(recalls), 4),
        recall_distribution=buckets,
    )


def compute_tool_call_metrics(rows: list[dict]) -> ToolCallMetrics:
    with_expected = [r for r in rows if _parse_tool_calls(r.get("expected_tool_calls", ""))]

    tier_match = 0
    tier_expected_counts: Counter = Counter()
    tier_correct_counts: Counter = Counter()

    for r in with_expected:
        expected = _parse_tool_calls(r.get("expected_tool_calls", ""))
        expected_tiers = {e["tier"] for e in expected}
        for e in expected:
            tier_expected_counts[e["tier"]] += 1
        obs_tier = _observed_tool_to_tier(r.get("tool_called"))
        if obs_tier and obs_tier in expected_tiers:
            tier_match += 1
            tier_correct_counts[obs_tier] += 1

    per_tier = {
        t: {
            "expected": tier_expected_counts[t],
            "correct": tier_correct_counts.get(t, 0),
            "rate": round(tier_correct_counts.get(t, 0) / tier_expected_counts[t], 4)
            if tier_expected_counts[t] else 0.0,
        }
        for t in sorted(tier_expected_counts)
    }

    order_id_rows = [
        r for r in with_expected
        if any(e["identifier"] for e in _parse_tool_calls(r.get("expected_tool_calls", "")))
    ]
    order_id_match = sum(
        1 for r in order_id_rows
        if any(
            e["identifier"] and e["identifier"] in (r.get("tool_result_summary") or "")
            for e in _parse_tool_calls(r.get("expected_tool_calls", ""))
        )
    )

    auth_rows = [
        r for r in rows
        if r.get("expected_handling") == "escalated"
        and "exceeded_authority" in _parse_escalation_reasons(r.get("expected_escalation_reason"))
    ]
    auth_appropriate = sum(
        1 for r in auth_rows
        if r.get("tool_status") in ("exceeded_authority", "out_of_window")
    )

    n = len(with_expected)
    na = len(auth_rows)
    no = len(order_id_rows)
    return ToolCallMetrics(
        rows_with_expected=n,
        tier_match_count=tier_match,
        tier_match_rate=round(tier_match / n, 4) if n else 0.0,
        order_id_rows=no,
        order_id_match_count=order_id_match,
        order_id_match_rate=round(order_id_match / no, 4) if no else 0.0,
        authority_rows=na,
        authority_appropriate=auth_appropriate,
        authority_appropriate_rate=round(auth_appropriate / na, 4) if na else 0.0,
        per_tier=per_tier,
    )


def compute_escalation_metrics(rows: list[dict]) -> EscalationMetrics:
    TP = sum(1 for r in rows if r.get("escalation_decision") and r.get("expected_handling") == "escalated")
    FP = sum(1 for r in rows if r.get("escalation_decision") and r.get("expected_handling") != "escalated")
    FN = sum(1 for r in rows if not r.get("escalation_decision") and r.get("expected_handling") == "escalated")
    TN = sum(1 for r in rows if not r.get("escalation_decision") and r.get("expected_handling") != "escalated")

    precision = round(TP / (TP + FP), 4) if (TP + FP) else 0.0
    recall = round(TP / (TP + FN), 4) if (TP + FN) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) else 0.0

    per_reason = {}
    for reason in ESCALATION_REASONS:
        gt_rows = [r for r in rows if reason in _parse_escalation_reasons(r.get("expected_escalation_reason"))]
        if not gt_rows:
            per_reason[reason] = {"gt_count": 0, "agent_triggered_count": 0, "tp": 0, "fn": 0, "precision": 0.0, "recall": 0.0}
            continue
        tp_r = sum(1 for r in gt_rows if r.get("escalation_decision") and reason in (r.get("escalation_triggers") or []))
        fn_r = sum(1 for r in gt_rows if not r.get("escalation_decision") or reason not in (r.get("escalation_triggers") or []))
        agent_rows = [r for r in rows if reason in (r.get("escalation_triggers") or [])]
        prec_r = round(tp_r / len(agent_rows), 4) if agent_rows else 0.0
        rec_r = round(tp_r / len(gt_rows), 4) if gt_rows else 0.0
        per_reason[reason] = {
            "gt_count": len(gt_rows),
            "agent_triggered_count": len(agent_rows),
            "tp": tp_r,
            "fn": fn_r,
            "precision": prec_r,
            "recall": rec_r,
        }

    return EscalationMetrics(
        true_positives=TP, false_positives=FP, false_negatives=FN, true_negatives=TN,
        precision=precision, recall=recall, f1=f1, per_reason=per_reason,
    )


def compute_containment_metrics(rows: list[dict]) -> ContainmentMetrics:
    contained = [r for r in rows if r.get("expected_handling") == "contained"]
    gross = sum(1 for r in contained if not r.get("escalation_decision"))
    net = sum(
        1 for r in contained
        if not r.get("escalation_decision")
        and r.get("generation_method") not in ("clarification_template", "handoff_template", "llm_handoff")
    )

    clarified = [r for r in rows if r.get("expected_handling") == "clarified"]
    clar_correct = sum(
        1 for r in clarified
        if r.get("generation_method") == "clarification_template"
        or r.get("tool_status") == "missing_identifier"
    )

    nc = len(contained)
    nl = len(clarified)
    return ContainmentMetrics(
        total_contained=nc,
        gross_contained=gross,
        gross_containment_rate=round(gross / nc, 4) if nc else 0.0,
        net_contained=net,
        net_containment_rate=round(net / nc, 4) if nc else 0.0,
        total_clarified=nl,
        clarification_correct=clar_correct,
        clarification_rate=round(clar_correct / nl, 4) if nl else 0.0,
    )


def compute_boundary_pair_metrics(rows: list[dict]) -> BoundaryPairMetrics:
    by_id = {r["query_id"]: r for r in rows}
    variants = [r for r in rows if r.get("linked_query_id")]

    transition = no_transition = correct_trans = consistent_no = 0
    flag_counts: dict[str, dict] = defaultdict(lambda: {"total": 0, "correct": 0})

    for r in variants:
        base = by_id.get(r["linked_query_id"])
        if base is None:
            continue
        exp_v, exp_b = r.get("expected_handling"), base.get("expected_handling")
        obs_v, obs_b = _observed_category(r), _observed_category(base)
        is_trans = exp_v != exp_b
        correct = obs_v != obs_b  # agent also behaved differently

        if is_trans:
            transition += 1
            if correct:
                correct_trans += 1
            flag = r.get("threshold_test_flag") or "none"
            flag_counts[flag]["total"] += 1
            flag_counts[flag]["correct"] += int(correct)
        else:
            no_transition += 1
            if obs_v == obs_b:
                consistent_no += 1

    per_flag = {
        flag: {
            "total": v["total"],
            "correct": v["correct"],
            "rate": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
        }
        for flag, v in sorted(flag_counts.items())
    }

    return BoundaryPairMetrics(
        total_variant_rows=len(variants),
        transition_pairs=transition,
        correct_transitions=correct_trans,
        transition_rate=round(correct_trans / transition, 4) if transition else 0.0,
        no_transition_pairs=no_transition,
        consistent_no_transition=consistent_no,
        consistency_rate=round(consistent_no / no_transition, 4) if no_transition else 0.0,
        per_threshold_flag=per_flag,
    )


def compute_latency_metrics(rows: list[dict]) -> LatencyMetrics:
    OUTLIER_MS = 600_000
    all_ms = [r.get("elapsed_ms") or 0 for r in rows]
    outliers = [v for v in all_ms if v > OUTLIER_MS]
    included = sorted(v for v in all_ms if v <= OUTLIER_MS)
    if not included:
        return LatencyMetrics(0, len(outliers), 0.0, 0.0, 0.0, 0.0, 0, 0, {})

    qs = quantiles(included, n=20)  # 5%, 10%, ..., 95%
    p90 = qs[17]  # 90th percentile (index 17 of 19 quantiles = 18/20 * len)
    p95 = qs[18]  # 95th percentile

    per_intent: dict[str, list] = defaultdict(list)
    for r in rows:
        ms = r.get("elapsed_ms") or 0
        if ms <= OUTLIER_MS and r.get("predicted_intent"):
            per_intent[r["predicted_intent"]].append(ms)

    return LatencyMetrics(
        count_included=len(included),
        count_outliers=len(outliers),
        median_ms=round(median(included), 1),
        mean_ms=round(mean(included), 1),
        p90_ms=round(p90, 1),
        p95_ms=round(p95, 1),
        min_ms=included[0],
        max_ms=included[-1],
        per_intent_median={intent: round(median(vs), 1) for intent, vs in sorted(per_intent.items()) if vs},
    )


def compute_summary(rows: list[dict]) -> SummaryStats:
    return SummaryStats(
        total=len(rows),
        by_harness_status=dict(Counter(r.get("harness_status") or "unknown" for r in rows)),
        by_classification_method=dict(Counter(r.get("classification_method") or "unknown" for r in rows)),
        by_generation_method=dict(Counter(r.get("generation_method") or "unknown" for r in rows)),
    )


# ── Markdown report ────────────────────────────────────────────────────────────

def _pct(numerator: int, denominator: int) -> str:
    if not denominator:
        return "N/A"
    return f"{numerator}/{denominator} = {100 * numerator / denominator:.1f}%"


def _fmt_float(v: float) -> str:
    return f"{v:.3f}"


def build_markdown(report: MetricsReport) -> str:
    lines = []
    a = lines.append

    a("# Evaluation Metrics Report")
    a("")
    a(f"Computed on {report.computed_at} from `{report.input_file}`.")
    a("")
    a(f"Total queries evaluated: {report.summary.total}.")
    a("")

    # 1. Intent accuracy
    ia = report.intent_accuracy
    a("## 1. Intent classification accuracy")
    a("")
    a(f"Overall: {ia.correct}/{ia.total} = {ia.overall_accuracy * 100:.1f}%.")
    a("")
    a("Per-intent breakdown:")
    a("")
    a("| Intent | Correct | Total | Accuracy |")
    a("|---|---|---|---|")
    for intent in INTENT_NAMES:
        d = ia.per_intent.get(intent)
        if d:
            a(f"| {intent} | {d['correct']} | {d['total']} | {d['accuracy'] * 100:.1f}% |")
    a("")
    if ia.confusion:
        a("Misclassifications (non-zero confusion matrix cells):")
        a("")
        for exp, preds in sorted(ia.confusion.items()):
            for pred, count in sorted(preds.items()):
                a(f"- {exp} → {pred}: {count}")
    else:
        a("No misclassifications.")
    a("")

    # 2. Retrieval
    ret = report.retrieval
    a("## 2. Retrieval precision and recall")
    a("")
    a(f"Evaluated over {ret.rows_with_gold} rows that have retrieval gold.")
    a("")
    a(f"Mean precision@5: {_fmt_float(ret.mean_precision_at_5)}")
    a(f"Mean recall@5: {_fmt_float(ret.mean_recall_at_5)}")
    a("")
    a("Recall@5 distribution:")
    a("")
    for bucket, count in ret.recall_distribution.items():
        a(f"- {bucket}: {count} rows")
    a("")

    # 3. Tool calls
    tc = report.tool_calls
    a("## 3. Tool-call correctness")
    a("")
    a(f"Tool-name (tier-level) match rate: {_pct(tc.tier_match_count, tc.rows_with_expected)}")
    a(f"Order-ID match rate: {_pct(tc.order_id_match_count, tc.order_id_rows)}")
    a(f"Status appropriateness rate (authority-exceeded cases): {_pct(tc.authority_appropriate, tc.authority_rows)}")
    a("")
    a("Per-tier breakdown:")
    a("")
    a("| Tier | Expected calls | Correctly fired | Rate |")
    a("|---|---|---|---|")
    for tier, d in tc.per_tier.items():
        a(f"| {tier} | {d['expected']} | {d['correct']} | {d['rate'] * 100:.1f}% |")
    a("")

    # 4. Escalation
    esc = report.escalation
    a("## 4. Escalation precision and recall")
    a("")
    a("Overall:")
    a("")
    a(f"- True positives: {esc.true_positives}")
    a(f"- False positives: {esc.false_positives}")
    a(f"- False negatives: {esc.false_negatives}")
    a(f"- True negatives: {esc.true_negatives}")
    a(f"- Precision: {_fmt_float(esc.precision)}")
    a(f"- Recall: {_fmt_float(esc.recall)}")
    a(f"- F1: {_fmt_float(esc.f1)}")
    a("")
    a("Per-reason breakdown:")
    a("")
    a("| Reason | GT count | Agent triggered | TP | FN | Precision | Recall |")
    a("|---|---|---|---|---|---|---|")
    for reason, d in esc.per_reason.items():
        a(f"| {reason} | {d['gt_count']} | {d['agent_triggered_count']} | {d['tp']} | {d['fn']} | {_fmt_float(d['precision'])} | {_fmt_float(d['recall'])} |")
    a("")

    # 5. Containment
    con = report.containment
    a("## 5. Outcome containment")
    a("")
    a(f"Gross containment: {_pct(con.gross_contained, con.total_contained)}")
    a(f"Net containment: {_pct(con.net_contained, con.total_contained)}")
    a(f"Clarification rate: {_pct(con.clarification_correct, con.total_clarified)}")
    a("")

    # 6. Boundary pairs
    bp = report.boundary_pairs
    a("## 6. Boundary-pair transition rate")
    a("")
    a(f"Variant rows with linked canonical query: {bp.total_variant_rows}")
    a(f"Transition pairs (different expected handling): {bp.transition_pairs}")
    a(f"Pairs where agent behaviour also transitioned: {bp.correct_transitions}")
    a(f"Transition rate: {_pct(bp.correct_transitions, bp.transition_pairs)}")
    a("")
    a(f"No-transition pairs (same expected handling): {bp.no_transition_pairs}")
    a(f"Consistent pairs: {_pct(bp.consistent_no_transition, bp.no_transition_pairs)}")
    a("")
    if bp.per_threshold_flag:
        a("Per-threshold-flag breakdown:")
        a("")
        a("| Threshold flag | Pair count | Correct transitions | Rate |")
        a("|---|---|---|---|")
        for flag, d in bp.per_threshold_flag.items():
            a(f"| {flag} | {d['total']} | {d['correct']} | {d['rate'] * 100:.1f}% |")
        a("")

    # 7. Latency
    lat = report.latency
    a("## 7. Latency")
    a("")
    a(f"Median: {lat.median_ms:.0f} ms")
    a(f"Mean: {lat.mean_ms:.0f} ms")
    a(f"p90: {lat.p90_ms:.0f} ms")
    a(f"p95: {lat.p95_ms:.0f} ms")
    a(f"Min: {lat.min_ms} ms  |  Max: {lat.max_ms} ms")
    a(f"Outliers excluded (>{600_000 // 1000}s): {lat.count_outliers}")
    a("")
    a("Per-intent median (ms):")
    a("")
    for intent, med in lat.per_intent_median.items():
        a(f"- {intent}: {med:.0f} ms")
    a("")

    # 8. Summary
    s = report.summary
    a("## 8. Summary")
    a("")
    a(f"- Total queries: {s.total}")
    status_parts = ", ".join(f"{k}={v}" for k, v in sorted(s.by_harness_status.items()))
    a(f"- Harness status: {status_parts}")
    clf_parts = ", ".join(f"{k}={v}" for k, v in sorted(s.by_classification_method.items()))
    a(f"- Classification method: {clf_parts}")
    gen_parts = ", ".join(f"{k}={v}" for k, v in sorted(s.by_generation_method.items()))
    a(f"- Generation method: {gen_parts}")
    a("")

    return "\n".join(lines)


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute §4.4 metrics from an evaluation JSONL file.")
    parser.add_argument("--input", required=True, help="Path to evaluation JSONL file")
    parser.add_argument("--output-dir", default="evaluation_results", help="Directory for output files")
    args = parser.parse_args()

    input_path = pathlib.Path(args.input)
    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    report = MetricsReport(
        computed_at=now,
        input_file=input_path.name,
        intent_accuracy=compute_intent_accuracy(rows),
        retrieval=compute_retrieval_metrics(rows),
        tool_calls=compute_tool_call_metrics(rows),
        escalation=compute_escalation_metrics(rows),
        containment=compute_containment_metrics(rows),
        boundary_pairs=compute_boundary_pair_metrics(rows),
        latency=compute_latency_metrics(rows),
        summary=compute_summary(rows),
    )

    md_path = output_dir / "metrics_report.md"
    json_path = output_dir / "metrics_report.json"

    md_path.write_text(build_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote metrics_report.md and metrics_report.json to {output_dir}")


if __name__ == "__main__":
    main()
