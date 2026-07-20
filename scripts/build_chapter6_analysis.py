#!/usr/bin/env python3
"""Build Chapter 6 analysis: figures, tables, and synthesis report.

Consumes three input artifacts:
  1. evaluation_results/metrics_report.json  — quantitative §4.4 metrics
  2. evaluation_results/rubric_scores.jsonl  — qualitative §4.3.3 rubric scores
  3. evaluation_results/eval_results.jsonl   — canonical evaluation trace

Produces to evaluation_results/chapter6/:
  fig_6_1_intent_accuracy.{png,svg}
  fig_6_2_escalation_per_reason.{png,svg}
  fig_6_3_rubric_distributions.{png,svg}
  fig_6_4_rubric_by_handling.{png,svg}
  fig_6_5_latency.{png,svg}
  fig_6_6_confusion_matrix.{png,svg}
  fig_6_7_boundary_transitions.{png,svg}
  tables.md            — seven Markdown tables for Word import
  chapter6_synthesis.md — structured findings for Chapter 6 prose drafting
"""

import argparse
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from statistics import mean, median, stdev

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# ── Style ──────────────────────────────────────────────────────────────────────

PALETTE = ["#2C3E50", "#7F8C8D", "#BDC3C7", "#34495E", "#95A5A6"]
ACCENT = "#C0392B"

LIKERT_DIMS = ("factual_accuracy", "completeness", "tone_appropriateness", "structural_quality")
DIM_LABEL = {
    "factual_accuracy": "Factual accuracy",
    "completeness": "Completeness",
    "tone_appropriateness": "Tone appropriateness",
    "structural_quality": "Structural quality",
}

INTENT_ORDER = [
    "order_status", "order_modify", "order_cancel", "refund_request",
    "product_info", "return_policy", "shipping_info", "account_help",
    "complaint", "multi_issue_dispute", "out_of_scope", "ambiguous_query",
]
INTENT_SHORT = {
    "order_status": "order_status", "order_modify": "order_modify",
    "order_cancel": "order_cancel", "refund_request": "refund_req",
    "product_info": "product_info", "return_policy": "return_policy",
    "shipping_info": "shipping_info", "account_help": "account_help",
    "complaint": "complaint", "multi_issue_dispute": "multi_issue",
    "out_of_scope": "out_of_scope", "ambiguous_query": "ambiguous",
}


def _configure_matplotlib_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


_configure_matplotlib_style()


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_metrics(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rubric_scores(path: pathlib.Path) -> dict:
    """Return {query_id: {dimension: record}}."""
    result: dict = defaultdict(dict)
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            result[rec["query_id"]][rec["dimension"]] = rec
        except (json.JSONDecodeError, KeyError):
            pass
    return dict(result)


def _load_eval_rows(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _integrate_data(metrics: dict, rubric: dict, eval_rows: list[dict]) -> dict:
    per_query = {}
    for row in eval_rows:
        qid = row["query_id"]
        rec = dict(row)
        rec["rubric"] = rubric.get(qid, {})
        per_query[qid] = rec
    return {"per_query": per_query, "metrics": metrics}


# ── Cross-tab computation ──────────────────────────────────────────────────────

def _compute_rubric_by_handling(integrated: dict) -> dict:
    """Return {handling: {dimension: {mean, sd, n}}}."""
    by_h: dict = defaultdict(lambda: defaultdict(list))
    for rec in integrated["per_query"].values():
        h = rec.get("expected_handling")
        for dim in LIKERT_DIMS:
            s = rec["rubric"].get(dim, {}).get("score")
            if s is not None:
                by_h[h][dim].append(int(s))
    result = {}
    for h, dims in by_h.items():
        result[h] = {
            dim: {
                "mean": round(mean(vals), 3),
                "sd": round(stdev(vals), 3) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
            for dim, vals in dims.items()
        }
    return result


def _compute_rubric_by_intent(integrated: dict) -> dict:
    """Return {intent: {dimension: mean}}."""
    by_i: dict = defaultdict(lambda: defaultdict(list))
    for rec in integrated["per_query"].values():
        intent = rec.get("predicted_intent")
        for dim in LIKERT_DIMS:
            s = rec["rubric"].get(dim, {}).get("score")
            if s is not None:
                by_i[intent][dim].append(int(s))
    return {
        intent: {dim: round(mean(vals), 3) for dim, vals in dims.items() if vals}
        for intent, dims in by_i.items()
    }


def _compute_escalation_outcomes_with_rubric(integrated: dict) -> dict:
    """Return {category: {dimension: mean, count: int}}."""
    cats: dict = defaultdict(list)
    for rec in integrated["per_query"].values():
        esc = rec.get("escalation_decision")
        exp_h = rec.get("expected_handling")
        if esc and exp_h == "escalated":
            cat = "true_positive_escalation"
        elif esc and exp_h != "escalated":
            cat = "false_positive_escalation"
        elif not esc and exp_h == "escalated":
            cat = "false_negative_escalation"
        else:
            cat = "true_negative_escalation"
        cats[cat].append(rec)

    result = {}
    for cat, recs in cats.items():
        dim_vals: dict = defaultdict(list)
        for rec in recs:
            for dim in LIKERT_DIMS:
                s = rec["rubric"].get(dim, {}).get("score")
                if s is not None:
                    dim_vals[dim].append(int(s))
        result[cat] = {
            dim: round(mean(v), 3) for dim, v in dim_vals.items() if v
        }
        result[cat]["count"] = len(recs)
    return result


def _catalogue_hallucinations(integrated: dict) -> list[dict]:
    records = []
    for rec in integrated["per_query"].values():
        hal_rec = rec["rubric"].get("hallucination_present", {})
        if not hal_rec.get("score"):
            continue
        notes = hal_rec.get("notes", "")
        query_text = rec.get("query_text", "")
        notes_lower = notes.lower()
        has_order_id = bool(re.search(r"ORD-\d+", query_text))

        if has_order_id and ("again" in notes_lower or rec.get("generation_method") == "clarification_template"):
            failure_mode = "order_id_ignored"
        elif "escalat" in notes_lower and not rec.get("escalation_decision"):
            failure_mode = "fabricated_behavior"
        else:
            failure_mode = "fabricated_fact"

        records.append({
            "query_id": rec["query_id"],
            "query_text": query_text,
            "expected_intent": rec.get("expected_intent"),
            "predicted_intent": rec.get("predicted_intent"),
            "tool_called": rec.get("tool_called"),
            "tool_status": rec.get("tool_status"),
            "escalation_decision": rec.get("escalation_decision"),
            "generation_method": rec.get("generation_method"),
            "response_text": rec.get("generation_text", ""),
            "scorer_notes": notes,
            "failure_mode_category": failure_mode,
        })
    return sorted(records, key=lambda r: r["query_id"])


# ── Figure helpers ─────────────────────────────────────────────────────────────

def _save_fig(fig: plt.Figure, out_dir: pathlib.Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


# ── Figures ────────────────────────────────────────────────────────────────────

def fig_intent_accuracy(integrated: dict, out_dir: pathlib.Path) -> None:
    ia = integrated["metrics"]["intent_accuracy"]["per_intent"]
    intents = [i for i in INTENT_ORDER if i in ia]
    accuracies = [ia[i]["accuracy"] for i in intents]
    labels = [f"{ia[i]['correct']}/{ia[i]['total']}" for i in intents]
    order = sorted(range(len(intents)), key=lambda k: accuracies[k])
    intents_s = [intents[i] for i in order]
    accs_s = [accuracies[i] for i in order]
    labels_s = [labels[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [ACCENT if a < 0.90 else PALETTE[0] for a in accs_s]
    bars = ax.barh(range(len(intents_s)), accs_s, color=colors, height=0.6)
    ax.set_yticks(range(len(intents_s)))
    ax.set_yticklabels(intents_s)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Accuracy")
    ax.set_title("Figure 6.1 — Per-intent classification accuracy")
    for i, (bar, lbl) in enumerate(zip(bars, labels_s)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=8)
    ax.axvline(0.9, color=ACCENT, linewidth=0.8, linestyle=":", alpha=0.6)
    _save_fig(fig, out_dir, "fig_6_1_intent_accuracy")


def fig_escalation_per_reason(integrated: dict, out_dir: pathlib.Path) -> None:
    per_reason = integrated["metrics"]["escalation"]["per_reason"]
    reasons = ["high_emotion", "exceeded_authority", "out_of_scope"]
    x = np.arange(len(reasons))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    prec = [per_reason[r]["precision"] for r in reasons]
    rec = [per_reason[r]["recall"] for r in reasons]
    ax.bar(x - width / 2, prec, width, label="Precision", color=PALETTE[0])
    ax.bar(x + width / 2, rec, width, label="Recall", color=PALETTE[2])
    ax.set_xticks(x)
    ax.set_xticklabels(reasons)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title(" Escalation precision and recall by reason\n"
                 "(out_of_scope: single genuine instance Q-077; pure out-of-scope scored as refusal, \u00a74.5)")
    ax.legend()
    for i, (p, r) in enumerate(zip(prec, rec)):
        ax.text(i - width / 2, p + 0.02, f"{p:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, r + 0.02, f"{r:.2f}", ha="center", fontsize=8)
    _save_fig(fig, out_dir, "fig_6_2_escalation_per_reason")


def fig_rubric_distributions(integrated: dict, out_dir: pathlib.Path) -> None:
    by_dim: dict = defaultdict(list)
    for rec in integrated["per_query"].values():
        for dim in LIKERT_DIMS:
            s = rec["rubric"].get(dim, {}).get("score")
            if s is not None:
                by_dim[dim].append(int(s))

    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharey=False)
    axes = axes.flatten()
    for ax, dim in zip(axes, LIKERT_DIMS):
        vals = by_dim[dim]
        counts = Counter(vals)
        xs = [1, 2, 3, 4, 5]
        ys = [counts.get(v, 0) for v in xs]
        mode = max(xs, key=lambda v: counts.get(v, 0))
        colors = [ACCENT if v == mode else PALETTE[0] for v in xs]
        ax.bar(xs, ys, color=colors, width=0.6)
        ax.set_xticks(xs)
        ax.set_title(f"{DIM_LABEL[dim]}\n(mean={mean(vals):.2f})", fontsize=10)
        ax.set_xlabel("Score (1-5)")
        ax.set_ylabel("Count")
    fig.suptitle("Rubric score distributions (modal value in accent colour)", fontsize=11)
    _save_fig(fig, out_dir, "fig_6_3_rubric_distributions")


def fig_rubric_by_handling(integrated: dict, out_dir: pathlib.Path) -> None:
    rh = _compute_rubric_by_handling(integrated)
    handlings = ["contained", "escalated", "clarified"]
    x = np.arange(len(LIKERT_DIMS))
    width = 0.25
    colors = [PALETTE[0], PALETTE[2], PALETTE[3]]

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, (h, color) in enumerate(zip(handlings, colors)):
        means = [rh.get(h, {}).get(dim, {}).get("mean", 0) for dim in LIKERT_DIMS]
        bars = ax.bar(x + (k - 1) * width, means, width, label=h.title(), color=color)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
                    f"{m:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([DIM_LABEL[d] for d in LIKERT_DIMS], rotation=10, ha="right")
    ax.set_ylim(0, 5.8)
    ax.set_ylabel("Mean rubric score (1-5)")
    ax.set_title("Rubric scores by expected handling outcome")
    ax.legend()
    _save_fig(fig, out_dir, "fig_6_4_rubric_by_handling")


def fig_latency(integrated: dict, out_dir: pathlib.Path) -> None:
    lat = integrated["metrics"]["latency"]
    all_ms = [rec["elapsed_ms"] for rec in integrated["per_query"].values()
              if rec.get("elapsed_ms") and rec["elapsed_ms"] <= 600_000]

    per_intent_med = lat["per_intent_median"]
    intents_sorted = sorted(per_intent_med, key=per_intent_med.get, reverse=True)
    medians = [per_intent_med[i] / 1000 for i in intents_sorted]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist([v / 1000 for v in all_ms], bins=40, color=PALETTE[0], edgecolor="white", linewidth=0.5)
    ax1.set_xlabel("Latency (s)")
    ax1.set_ylabel("Count")
    ax1.set_title("Overall latency distribution")
    ax1.axvline(lat["median_ms"] / 1000, color=ACCENT, linewidth=1.5, label=f"Median {lat['median_ms']/1000:.1f}s")
    ax1.legend()

    ax2.barh(range(len(intents_sorted)), medians, color=PALETTE[0], height=0.6)
    ax2.set_yticks(range(len(intents_sorted)))
    ax2.set_yticklabels(intents_sorted)
    ax2.set_xlabel("Median latency (s)")
    ax2.set_title("Median latency per intent")

    fig.suptitle("Response latency", fontsize=11)
    _save_fig(fig, out_dir, "fig_6_5_latency")


def fig_confusion_matrix(integrated: dict, out_dir: pathlib.Path) -> None:
    per_query = integrated["per_query"].values()
    n = len(INTENT_ORDER)
    idx = {intent: i for i, intent in enumerate(INTENT_ORDER)}
    matrix = np.zeros((n, n), dtype=int)
    for rec in per_query:
        exp = rec.get("expected_intent")
        pred = rec.get("predicted_intent")
        if exp in idx and pred in idx:
            matrix[idx[exp], idx[pred]] += 1

    # Build color matrix: 0=empty, 1=correct, 2=error
    cmatrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                cmatrix[i, j] = 1 if i == j else 2

    cmap = mcolors.ListedColormap(["#FFFFFF", PALETTE[0], ACCENT])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(cmatrix, cmap=cmap, norm=norm, aspect="auto")
    for i in range(n):
        for j in range(n):
            if matrix[i, j] > 0:
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    short = [INTENT_SHORT[i] for i in INTENT_ORDER]
    ax.set_xticks(range(n))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Predicted intent")
    ax.set_ylabel("Expected intent")
    ax.set_title("Intent classification confusion matrix\n"
                 "(dark=correct, red=misclassification, blank=zero)")
    ax.grid(False)
    _save_fig(fig, out_dir, "fig_6_6_confusion_matrix")


def fig_boundary_transitions(integrated: dict, out_dir: pathlib.Path) -> None:
    bp = integrated["metrics"]["boundary_pairs"]["per_threshold_flag"]
    flags = sorted(bp)
    rates = [bp[f]["rate"] for f in flags]
    labels = [f"{bp[f]['correct']}/{bp[f]['total']}" for f in flags]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [ACCENT if r < 1.0 else PALETTE[0] for r in rates]
    bars = ax.barh(range(len(flags)), rates, color=colors, height=0.6)
    ax.set_yticks(range(len(flags)))
    ax.set_yticklabels(flags)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Transition rate")
    ax.set_title("Boundary-pair transition rates by threshold flag")
    for i, (bar, lbl) in enumerate(zip(bars, labels)):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                lbl, va="center", fontsize=8)
    _save_fig(fig, out_dir, "fig_6_7_boundary_transitions")


# ── Tables ─────────────────────────────────────────────────────────────────────

def build_tables(integrated: dict, out_dir: pathlib.Path) -> None:
    m = integrated["metrics"]
    lines: list[str] = []
    a = lines.append

    # Table 6.1: Eight-metric summary
    a("# Evaluation Tables\n")
    a("## Table 6.1 — Eight-metric summary\n")
    a("| Metric | Value | §4.4 section |")
    a("|---|---|---|")
    ia = m["intent_accuracy"]
    a(f"| Intent accuracy | {ia['correct']}/{ia['total']} = {ia['overall_accuracy']*100:.1f}% | §4.4.1 |")
    ret = m["retrieval"]
    a(f"| Retrieval recall@5 (mean) | {ret['mean_recall_at_5']:.3f} over {ret['rows_with_gold']} queries | §4.4.2 |")
    tc = m["tool_calls"]
    a(f"| Tool-call tier match rate | {tc['tier_match_count']}/{tc['rows_with_expected']} = {tc['tier_match_rate']*100:.1f}% | §4.4.3 |")
    esc = m["escalation"]
    a(f"| Escalation precision | {esc['precision']:.3f} | §4.4.4 |")
    a(f"| Escalation recall | {esc['recall']:.3f} | §4.4.4 |")
    a(f"| Escalation F1 | {esc['f1']:.3f} | §4.4.4 |")
    con = m["containment"]
    a(f"| Gross containment | {con['gross_contained']}/{con['total_contained']} = {con['gross_containment_rate']*100:.1f}% | §4.4.5 |")
    a(f"| Net containment | {con['net_contained']}/{con['total_contained']} = {con['net_containment_rate']*100:.1f}% | §4.4.5 |")
    a(f"| Clarification rate | {con['clarification_correct']}/{con['total_clarified']} = {con['clarification_rate']*100:.1f}% | §4.4.5 |")
    a("")

    # Table 6.2: Per-intent accuracy
    a("## Table 6.2 — Per-intent classification accuracy\n")
    a("| Intent | Correct | Total | Accuracy | Misclassified as |")
    a("|---|---|---|---|---|")
    confusion = ia.get("confusion", {})
    for intent in INTENT_ORDER:
        d = ia["per_intent"].get(intent)
        if not d:
            continue
        mis = ", ".join(f"{p}×{c}" for p, c in (confusion.get(intent) or {}).items())
        a(f"| {intent} | {d['correct']} | {d['total']} | {d['accuracy']*100:.1f}% | {mis or '—'} |")
    a("")

    # Table 6.3: Escalation per reason
    a("## Table 6.3 — Escalation precision/recall per reason\n")
    a("| Reason | GT count | Triggered | TP | FN | Precision | Recall |")
    a("|---|---|---|---|---|---|---|")
    for reason, d in esc["per_reason"].items():
        a(f"| {reason} | {d['gt_count']} | {d['agent_triggered_count']} | {d['tp']} | {d['fn']} | {d['precision']:.3f} | {d['recall']:.3f} |")
    a("")

    # Table 6.4: Tool-call correctness by tier
    a("## Table 6.4 — Tool-call correctness by tier\n")
    a("| Tier | Expected calls | Tier-match | Order-ID match | Notes |")
    a("|---|---|---|---|---|")
    tier_notes = {
        "1": "Tier-1 match is route adherence; authority-gated queries correctly call higher-tier tools instead",
        "2": "All modify/cancel calls fired correctly",
        "3": "All refund calls fired correctly",
    }
    for tier, d in tc["per_tier"].items():
        a(f"| {tier} | {d['expected']} | {d['correct']}/{d['expected']} = {d['rate']*100:.0f}% | "
          f"{tc['order_id_match_count']}/{tc['order_id_rows']} | {tier_notes.get(str(tier), '—')} |")
    a("")

    # Table 6.5: Rubric statistics
    a("## Table 6.5 — Rubric score statistics\n")
    a("| Dimension | Mean | SD | Median | Mode | 5s | 4s | 3s | 2s | 1s |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    by_dim: dict = defaultdict(list)
    for rec in integrated["per_query"].values():
        for dim in LIKERT_DIMS:
            s = rec["rubric"].get(dim, {}).get("score")
            if s is not None:
                by_dim[dim].append(int(s))
    for dim in LIKERT_DIMS:
        vals = by_dim[dim]
        cnt = Counter(vals)
        mode = max(cnt, key=cnt.get)
        a(f"| {DIM_LABEL[dim]} | {mean(vals):.3f} | {stdev(vals):.3f} | {median(vals):.1f} | {mode} "
          f"| {cnt.get(5,0)} | {cnt.get(4,0)} | {cnt.get(3,0)} | {cnt.get(2,0)} | {cnt.get(1,0)} |")
    a("")

    # Table 6.6: Rubric by handling
    a("## Table 6.6 — Rubric scores by handling outcome\n")
    rh = _compute_rubric_by_handling(integrated)
    a("| Handling | N | Factual acc. | Completeness | Tone appr. | Structural qual. |")
    a("|---|---|---|---|---|---|")
    for h in ("contained", "escalated", "clarified"):
        d = rh.get(h, {})
        n = d.get("factual_accuracy", {}).get("n", 0)
        vals = [d.get(dim, {}).get("mean", 0) for dim in LIKERT_DIMS]
        a(f"| {h} | {n} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} | {vals[3]:.3f} |")
    a("")

    # Table 6.7: Hallucination catalogue
    a("## Table 6.7 — Hallucination catalogue\n")
    a("| Query ID | Predicted intent | Gen. method | Failure mode | Scorer notes |")
    a("|---|---|---|---|---|")
    for h in _catalogue_hallucinations(integrated):
        notes = h["scorer_notes"][:100].replace("|", "\\|")
        a(f"| {h['query_id']} | {h['predicted_intent']} | {h['generation_method']} "
          f"| {h['failure_mode_category']} | {notes} |")
    a("")

    (out_dir / "tables.md").write_text("\n".join(lines), encoding="utf-8")


# ── Synthesis report ───────────────────────────────────────────────────────────

def build_synthesis(integrated: dict, out_dir: pathlib.Path, input_files: dict) -> None:
    m = integrated["metrics"]
    rh = _compute_rubric_by_handling(integrated)
    esc_cats = _compute_escalation_outcomes_with_rubric(integrated)
    hal = _catalogue_hallucinations(integrated)
    ri = _compute_rubric_by_intent(integrated)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ia = m["intent_accuracy"]
    esc = m["escalation"]
    con = m["containment"]
    ret = m["retrieval"]
    bp = m["boundary_pairs"]

    # Aggregate rubric means
    by_dim: dict = defaultdict(list)
    for rec in integrated["per_query"].values():
        for dim in LIKERT_DIMS:
            s = rec["rubric"].get(dim, {}).get("score")
            if s is not None:
                by_dim[dim].append(int(s))
    agg = {dim: round(mean(vals), 3) for dim, vals in by_dim.items()}

    fn_tone = esc_cats.get("false_negative_escalation", {}).get("tone_appropriateness", 0)
    con_tone = rh.get("contained", {}).get("tone_appropriateness", {}).get("mean", 0)
    esc_tone = rh.get("escalated", {}).get("tone_appropriateness", {}).get("mean", 0)
    multi_fa = ri.get("multi_issue_dispute", {}).get("factual_accuracy", 0)
    multi_comp = ri.get("multi_issue_dispute", {}).get("completeness", 0)
    multi_struct = ri.get("multi_issue_dispute", {}).get("structural_quality", 0)

    n_order_id = sum(1 for h in hal if h["failure_mode_category"] == "order_id_ignored")
    n_fab_beh = sum(1 for h in hal if h["failure_mode_category"] == "fabricated_behavior")
    total_queries = len(integrated["per_query"])
    n_escalated = sum(1 for r in integrated["per_query"].values() if r.get("escalation_decision"))
    fn_count = esc_cats.get("false_negative_escalation", {}).get("count", 0)

    lines: list[str] = []
    a = lines.append

    a("# Chapter 6 — Analytical Synthesis")
    a("")
    a(f"Generated: {now}")
    a("")
    a(f"- Metrics input: {input_files['metrics']}")
    a(f"- Rubric input: {input_files['rubric']}")
    a(f"- Evaluation input: {input_files['eval']}")
    a("")
    a("This document organises the empirical findings from quantitative metrics (§4.4) and qualitative rubric "
      "scores (§4.3.3) into a structured analytical scaffold for Chapter 6 prose drafting. Each named finding "
      "cites the relevant figure and table; the author writes Chapter 6 narrative from this scaffold. The "
      "document does not contain Chapter 6 prose — only structured findings with embedded evidence pointers.")
    a("")
    a("---")
    a("")

    # RQ1
    a("## RQ1 — Resolution quality")
    a("")
    a("### Finding 1.1 — High autonomous resolution on canonical transactional queries")
    a("")
    a(f"- Gross containment: {con['gross_contained']}/{con['total_contained']} = {con['gross_containment_rate']*100:.1f}% "
      f"(contained queries where agent did not falsely escalate). Table 6.1, Figure 6.4.")
    a(f"- Net containment: {con['net_contained']}/{con['total_contained']} = {con['net_containment_rate']*100:.1f}% "
      f"(no false escalation AND substantive LLM response). Table 6.1.")
    a(f"- Rubric corroboration (Table 6.6): contained queries score factual_accuracy "
      f"{rh['contained']['factual_accuracy']['mean']:.3f}, completeness {rh['contained']['completeness']['mean']:.3f}.")
    a("- Caveat: rubric scores on contained queries reflect a high-capability baseline "
      "(tool result always available); this partially inflates scores relative to information-retrieval-only queries.")
    a("")
    a("### Finding 1.2 — Intent classification strong but degrades on taxonomy-edge cases")
    a("")
    a(f"- Overall accuracy: {ia['correct']}/{ia['total']} = {ia['overall_accuracy']*100:.1f}%. Table 6.2, Figure 6.1.")
    a(f"- Six misclassifications (Figure 6.6 off-diagonal cells):")
    for exp, preds in sorted(ia.get("confusion", {}).items()):
        for pred, count in preds.items():
            a(f"  - {exp} → {pred}: {count}")
    a("- Interpretation: all six are at semantically adjacent intent boundaries "
      "(refund ↔ complaint ↔ multi_issue; product ↔ shipping; account ↔ product). "
      "No clear-domain errors; misclassifications reflect genuine ambiguity in the twelve-intent taxonomy.")
    a("")
    a("### Finding 1.3 — Retrieval recall strong despite structurally low precision-at-5")
    a("")
    a(f"- Mean recall@5: {ret['mean_recall_at_5']:.3f} over {ret['rows_with_gold']} queries with retrieval gold. "
      f"51 of {ret['rows_with_gold']} achieve recall@5 = 1.0. Table 6.1.")
    a(f"- Mean precision@5: {ret['mean_precision_at_5']:.3f}.")
    a("- Caveat: precision@5 is structurally depressed because most gold sets contain only one chunk "
      "(1/5 = 0.20 maximum precision when one target is retrieved). Low precision does not indicate "
      "retrieval quality failure; recall@5 = 0.784 is the operative measure.")
    a("")
    a("---")
    a("")

    # RQ2
    a("## RQ2 — Emotional intelligence")
    a("")
    a("### Finding 2.1 — Tone well-calibrated but concentrated on escalation paths")
    a("")
    a(f"- Aggregate tone_appropriateness: mean={agg['tone_appropriateness']:.3f}; no 1s or 2s observed (Figure 6.3).")
    a(f"- Cross-tab (Table 6.6, Figure 6.4): escalated queries tone={esc_tone:.3f} vs "
      f"contained queries tone={con_tone:.3f}.")
    a("- Interpretation: the warm-handoff system prompt produces measurably higher tone scores on escalated paths. "
      "The baseline (non-escalation) response is professional but generic. This constitutes conditional empathy "
      "mode rather than a uniform warm baseline — a design property of the architecture, not a capability gap.")
    a("- For §6.x: this finding supports the discussion of grounding-prompt specificity and the case for "
      "emotion-aware generation conditioning even below the escalation threshold.")
    a("")
    a("### Finding 2.2 — False-negative escalations show verbal acknowledgement without structural routing")
    a("")
    a(f"- {fn_count} FN escalations: mean tone_appropriateness = {fn_tone:.3f}, exceeding contained baseline {con_tone:.3f}.")
    a("- Interpretation: VADER-scored emotion leaked into the LLM generation prompt, producing empathetic "
      "language in the response even without triggering the structural escalation path. This is a partially "
      "adaptive behaviour — the agent is warmer to distressed customers — but lacks the human-in-the-loop "
      "routing that the escalation path provides.")
    a("- For §6.x: graduated escalation responses (verbal acknowledgement + enhanced monitoring flag) "
      "as future work.")
    a("")
    a("---")
    a("")

    # RQ3
    a("## RQ3 — Escalation behaviour")
    a("")
    a("### Finding 3.1 — Precision excellent; recall is the principal weakness")
    a("")
    a(f"- Precision: {esc['precision']:.3f} ({esc['true_positives']} TP / {esc['true_positives']+esc['false_positives']} positives). "
      f"One false positive (Q-091). Table 6.3, Figure 6.2.")
    a(f"- Recall: {esc['recall']:.3f} ({esc['true_positives']} TP / {esc['true_positives']+esc['false_negatives']} positives-expected). "
      f"F1: {esc['f1']:.3f}.")
    a("- The precision/recall asymmetry is by design: the three-trigger framework prioritises "
      "false-negative avoidance over false-positive avoidance. The low recall exposes the coverage gap "
      "on moderate-signal queries.")
    a("")
    a("### Finding 3.2 — 22 false negatives split into structural and genuine failures")
    a("")
    per_reason = esc["per_reason"]
    out_scope_fn = per_reason.get("out_of_scope", {}).get("gt_count", 0)
    a(f"- Structural FNs ({out_scope_fn}): out_of_scope queries expected-escalated but handled by "
      f"refusal_template per §3.2 boundary-intent exclusion. Not a miss; a definitional gap between "
      f"test set vocabulary and agent architecture.")
    genuine_fn = esc['false_negatives'] - out_scope_fn
    a(f"- Genuine FNs (~{genuine_fn}): moderate-emotion below VADER cutoff (Q-092, Q-095); "
      f"exceeded_authority without a tool call (Q-080, Q-085, Q-087 — no order ID in query, "
      f"no tool dispatch, no authority trigger); multi-issue cumulative complexity below "
      f"any independent trigger threshold.")
    a("- Implication: the three-trigger framework does not accumulate signal across simultaneous "
      "but independently sub-threshold stressors. Cumulative complexity is a structural gap.")
    a("")
    a("### Finding 3.3 — Threshold calibration strong on boundary pairs")
    a("")
    a(f"- {bp['correct_transitions']}/{bp['transition_pairs']} transition pairs correctly handled "
      f"(Figure 6.7). Failures at flag intersections (emotion_overlay_transactional: 2/3).")
    a("- Consistency on no-transition pairs: "
      f"{bp['consistent_no_transition']}/{bp['no_transition_pairs']} = {bp['consistency_rate']*100:.1f}%.")
    a("")
    a("---")
    a("")

    # RQ4
    a("## RQ4 — Quality of agent behaviour")
    a("")
    a("### Finding 4.1 — Hallucination rare and architecturally concentrated")
    a("")
    a(f"- {len(hal)}/{total_queries} = {len(hal)/total_queries*100:.1f}% of queries have hallucination flagged. Table 6.7.")
    a(f"- {n_order_id} of {len(hal)} share 'order_id_ignored' failure mode: multi_issue_dispute queries "
      f"carry an order ID in the query text, but the intent classifier fires a non-transactional intent, "
      f"so tool dispatch is skipped, and the LLM generation asks for an order ID already provided.")
    a(f"- {n_fab_beh} of {len(hal)}: 'fabricated_behavior' (Q-028: response claims escalation that did not occur).")
    a("- Root cause: order-ID extraction is order-ID-lookup-gated at tool dispatch, not at orchestrator level. "
      "Future work: extract order identifiers at orchestrator entry and expose to generation unconditionally.")
    a("")
    a("### Finding 4.2 — Structural quality is the dimension with most room for improvement")
    a("")
    a(f"- Mean structural_quality: {agg['structural_quality']:.3f} (lowest of four Likert dimensions). "
      f"Figure 6.3, Table 6.5.")
    a("- Distribution: mode=4 (69 queries), 15 queries at 5, 45 at 3.")
    a("- Calibration caveat (§4.5): the rubric reserves 5 for exemplary formatting. The agent produces "
      "well-formed prose paragraphs; the 3.75 mean partially reflects the rubric's high 5-bar rather than "
      "a structural capability deficit. Both the finding and its calibration note belong in §4.5.")
    a("")
    a("### Finding 4.3 — Multi-issue disputes are the hardest intent across all rubric dimensions")
    a("")
    a(f"- multi_issue_dispute rubric (Table 6.5 cross-intent): "
      f"factual_accuracy={multi_fa:.2f}, completeness={multi_comp:.2f}, structural_quality={multi_struct:.2f}.")
    a("- Contributing factors: (a) four of five hallucination cases involve this intent (order_id_ignored); "
      f"(b) the intent is outside transactional scope so no tool dispatch occurs, creating a purely "
      f"KB-retrieval response that may miss customer-specific facts; (c) the escalation system "
      f"does not aggregate signal across multiple simultaneous triggers.")
    a("")
    a("---")
    a("")

    # Caveats
    a("## Methodological caveats for §4.5")
    a("")
    a("**Caveat 1 — Tier-1 tool-call route adherence vs outcome correctness.**")
    a("The tier-level match metric counts route adherence: expected tier-1 lookup, observed tier-3 refund "
      "= mismatch, even though the tier-3 call correctly serves the customer. Correctly resolving an "
      "authority-gated request by calling the appropriate higher-tier tool is good agent behaviour; "
      "the metric penalises it. The 34/50 = 68% figure should be interpreted as route-adherence rate, "
      "not outcome-correctness rate.")
    a("")
    a("**Caveat 2 — Status appropriateness denominator is conditional on tool call.**")
    a(f"The metric (authority-exceeded cases with appropriate tool_status: "
      f"{m['tool_calls']['authority_appropriate']}/{m['tool_calls']['authority_rows']} = "
      f"{m['tool_calls']['authority_appropriate_rate']*100:.1f}%) "
      f"uses all authority-escalated queries as denominator, including those where no tool was called "
      f"(no order ID → no dispatch → no status). Among queries that *did* call a tool, the rate is higher. "
      f"§4.5 should report both the conditional and unconditional rates.")
    a("")
    a("**Caveat 3 — Structural quality rubric calibration.**")
    a("The rubric reserves 5 for 'exemplary formatting with headers, bullets, and scannable layout.' "
      "The agent produces prose responses; only 15/130 score 5. The 3.754 aggregate mean is partly "
      "an artefact of this high 5-bar. The finding (structural_quality is lowest dimension) is valid; "
      "its magnitude is partially a calibration effect.")
    a("")
    a("**Caveat 4 — out_of_scope ground-truth vocabulary.**")
    a(f"The test set marks {out_scope_fn} out_of_scope queries as expected_handling='escalated' with "
      f"expected_escalation_reason='out_of_scope'. The §3.2 boundary-intent exclusion exempts "
      f"OUT_OF_SCOPE from the low_confidence escalation trigger; the agent routes these to refusal_template "
      f"instead. This is a definitional gap: the test set labels a refusal as a missing escalation. "
      f"The 22 FNs conflate this structural gap with genuine missed escalations. "
      f"§4.5 should report the structural and genuine FNs separately.")
    a("")
    a("---")
    a("")

    # Convergence
    a("## Convergence across evaluation methods")
    a("")
    a(f"The quantitative metrics and rubric scores converge on the same structural picture. "
      f"The agent performs excellently on canonical transactional queries: 95.4% intent accuracy, "
      f"98.8% gross containment, factual_accuracy={rh['contained']['factual_accuracy']['mean']:.2f} "
      f"on contained queries. Performance degrades predictably at the edges: moderate-emotion below "
      f"VADER cutoff (FN escalations), multi-issue complexity (lowest rubric scores), "
      f"and taxonomy-adjacent intent boundaries (six misclassifications).")
    a("")
    a(f"The tone pattern is the clearest convergence point. Quantitatively, {n_escalated} queries were escalated "
      f"({n_escalated/total_queries*100:.1f}%); qualitatively, those escalated queries score {esc_tone:.2f} on tone vs "
      f"{con_tone:.2f} on contained queries. The rubric confirms what the architecture predicts: warm-handoff prompt "
      f"drives higher tone scores. The {fn_count} FNs scoring {fn_tone:.2f} on tone — above the contained baseline "
      f"— suggests the VADER score leaked warm register into generation without structural escalation. "
      f"Quantitative metrics alone would not surface this nuance.")
    a("")
    a(f"The hallucination analysis adds a third layer: {n_order_id} of {len(hal)} cases trace to a single architectural "
      f"gap (order-ID gating at tool dispatch, not at orchestrator entry) on the multi_issue_dispute "
      f"intent — the same intent that scores lowest on completeness ({multi_comp:.2f}) and structural "
      f"quality ({multi_struct:.2f}). The convergence across three measurement methods on the same "
      f"intent strengthens the conclusion that multi-issue query handling is the principal improvement "
      f"target for a next prototype iteration.")

    (out_dir / "chapter6_synthesis.md").write_text("\n".join(lines), encoding="utf-8")


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build Chapter 6 analysis artifacts.")
    parser.add_argument("--metrics-input", default="evaluation_results/metrics_report.json")
    parser.add_argument("--rubric-input", default="evaluation_results/rubric_scores.jsonl")
    parser.add_argument("--eval-input", default="evaluation_results/eval_results.jsonl")
    parser.add_argument("--output-dir", default="evaluation_results/chapter6")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = _load_metrics(pathlib.Path(args.metrics_input))
    rubric = _load_rubric_scores(pathlib.Path(args.rubric_input))
    eval_rows = _load_eval_rows(pathlib.Path(args.eval_input))
    integrated = _integrate_data(metrics, rubric, eval_rows)

    fig_intent_accuracy(integrated, out_dir)
    fig_escalation_per_reason(integrated, out_dir)
    fig_rubric_distributions(integrated, out_dir)
    fig_rubric_by_handling(integrated, out_dir)
    fig_latency(integrated, out_dir)
    fig_confusion_matrix(integrated, out_dir)
    fig_boundary_transitions(integrated, out_dir)

    build_tables(integrated, out_dir)
    build_synthesis(integrated, out_dir, {
        "metrics": args.metrics_input,
        "rubric": args.rubric_input,
        "eval": args.eval_input,
    })

    print(f"Wrote 7 figures, tables.md, and chapter6_synthesis.md to {out_dir}")


if __name__ == "__main__":
    main()
