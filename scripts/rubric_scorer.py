#!/usr/bin/env python3
"""Rubric scoring tool for the §4.3.3 qualitative evaluation.

Five dimensions are scored per query:
  factual_accuracy     — 1-5 Likert: are the stated facts correct?
  completeness         — 1-5 Likert: does the response cover what was asked?
  tone_appropriateness — 1-5 Likert: is the emotional register appropriate?
  structural_quality   — 1-5 Likert: is the response well-organised?
  hallucination_present — bool: any fabricated claim not in KB or tool result?

Persistence:
  evaluation_results/rubric_scores.jsonl         — active scores, one record per (query_id, dimension)
  evaluation_results/rubric_scores_history.jsonl — append-only archive of overwritten records

Launch:
  streamlit run scripts/rubric_scorer.py
  RUBRIC_INPUT=evaluation_results/my_run.jsonl streamlit run scripts/rubric_scorer.py
"""

import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

DEFAULT_INPUT_PATH = os.environ.get("RUBRIC_INPUT", "evaluation_results/eval_results.jsonl")
OUTPUT_PATH = pathlib.Path("evaluation_results/rubric_scores.jsonl")
HISTORY_PATH = pathlib.Path("evaluation_results/rubric_scores_history.jsonl")

LIKERT_DIMENSIONS = ("factual_accuracy", "completeness", "tone_appropriateness", "structural_quality")
BINARY_DIMENSIONS = ("hallucination_present",)
ALL_DIMENSIONS = LIKERT_DIMENSIONS + BINARY_DIMENSIONS

ANCHORS = {
    "factual_accuracy": [
        "1 — Multiple incorrect factual claims; misleads the customer",
        "2 — One major factual error or several minor ones",
        "3 — Substantially correct but with a small factual slip",
        "4 — Fully correct; no errors",
        "5 — Fully correct AND grounded in retrieved chunks with proper citations",
    ],
    "completeness": [
        "1 — Largely fails to address the customer's question",
        "2 — Addresses part of the question but leaves significant gaps",
        "3 — Addresses the main question; omits useful secondary information",
        "4 — Addresses the question fully",
        "5 — Fully addresses AND anticipates likely follow-up questions",
    ],
    "tone_appropriateness": [
        "1 — Jarringly mismatched (e.g., cheerful response to angry customer)",
        "2 — Noticeably off but not actively harmful",
        "3 — Professional but generic; neither matched nor mismatched",
        "4 — Well-calibrated to the customer's affective state",
        "5 — Active empathy demonstrated; no formulaic feeling",
    ],
    "structural_quality": [
        "1 — Disorganised, contradictory, or hard to follow",
        "2 — Readable but poorly structured",
        "3 — Adequately structured; nothing notable either way",
        "4 — Well-structured with clear paragraphing and logical flow",
        "5 — Exemplary: organised, scannable, uses formatting to aid comprehension",
    ],
}


# ── Data helpers ───────────────────────────────────────────────────────────────

def _load_eval_rows(input_path: pathlib.Path) -> list[dict]:
    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {input_path}")
    rows = []
    for line in input_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Evaluation file is empty: {input_path}")
    return rows


def _load_existing_scores(output_path: pathlib.Path) -> dict:
    """Return {(query_id, dimension): record} for the most recent score per pair."""
    if not output_path.exists():
        return {}
    result = {}
    for line in output_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            result[(rec["query_id"], rec["dimension"])] = rec
        except (json.JSONDecodeError, KeyError):
            pass
    return result


def _save_score(
    query_id: str,
    dimension: str,
    score,
    notes: str,
    output_path: pathlib.Path,
    history_path: pathlib.Path,
) -> None:
    existing = _load_existing_scores(output_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    key = (query_id, dimension)

    if key in existing:
        archived = dict(existing[key])
        archived["archived_at"] = now
        with open(history_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(archived, ensure_ascii=False) + "\n")

    existing[key] = {
        "query_id": query_id,
        "dimension": dimension,
        "score": score,
        "notes": notes,
        "scored_at": now,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in existing.values():
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _compute_progress(scores: dict, total_queries: int) -> dict:
    per_dim = {dim: 0 for dim in ALL_DIMENSIONS}
    query_dims: dict[str, set] = {}
    for qid, dim in scores:
        if dim in per_dim:
            per_dim[dim] += 1
        query_dims.setdefault(qid, set()).add(dim)
    fully = sum(1 for dims in query_dims.values() if set(ALL_DIMENSIONS) <= dims)
    return {
        "queries_fully_scored": fully,
        "total_judgments": len(scores),
        "target_judgments": total_queries * len(ALL_DIMENSIONS),
        "per_dimension_completion": per_dim,
    }


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Rubric Scorer", layout="wide")

    input_path = pathlib.Path(DEFAULT_INPUT_PATH)
    try:
        rows = _load_eval_rows(input_path)
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.stop()

    scores = _load_existing_scores(OUTPUT_PATH)
    total = len(rows)
    prog = _compute_progress(scores, total)

    if "idx" not in st.session_state:
        st.session_state.idx = 0
    idx = max(0, min(int(st.session_state.idx), total - 1))
    row = rows[idx]
    qid = row["query_id"]

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Progress")
        st.metric("Fully scored queries", f"{prog['queries_fully_scored']} / {total}")
        total_j = prog["total_judgments"]
        target_j = prog["target_judgments"]
        st.progress(total_j / target_j if target_j else 0,
                    text=f"{total_j} / {target_j} judgments captured")
        st.write("**Per-dimension completion**")
        for dim in ALL_DIMENSIONS:
            st.write(f"- {dim}: {prog['per_dimension_completion'].get(dim, 0)}/{total}")

        hal_flagged = [q for (q, d), rec in scores.items()
                       if d == "hallucination_present" and rec.get("score")]
        if hal_flagged:
            st.write(f"**Hallucinations flagged ({len(hal_flagged)})**")
            for q in hal_flagged:
                if st.button(q, key=f"jmp_{q}"):
                    st.session_state.idx = next(
                        (i for i, r in enumerate(rows) if r["query_id"] == q), idx
                    )
                    st.rerun()

        if st.button("Resume from first unscored query"):
            all_dims = set(ALL_DIMENSIONS)
            for i, r in enumerate(rows):
                done = {d for (q, d) in scores if q == r["query_id"]}
                if done < all_dims:
                    st.session_state.idx = i
                    st.rerun()

    # ── Query display ─────────────────────────────────────────────────────────
    st.title(f"Query {idx + 1} of {total}: {qid}")
    st.markdown(f"**Query:** {row['query_text']}")

    with st.expander("Query details", expanded=True):
        c1, c2 = st.columns(2)
        intent = row.get("predicted_intent", "—")
        conf = row.get("classification_confidence", 0)
        c1.write(f"**Predicted intent:** {intent} ({conf:.0%})")
        if row.get("tool_called"):
            c1.write(f"**Tool:** {row['tool_called']} → {row.get('tool_status', '—')}")
        esc = row.get("escalation_decision")
        triggers = ", ".join(row.get("escalation_triggers") or [])
        c2.write(f"**Escalation:** {'Yes — ' + triggers if esc else 'No'}")
        c2.write(f"**Generation method:** {row.get('generation_method', '—')}")
        st.write("**Generated response:**")
        st.info(row.get("generation_text") or "(no response)")

    # ── Scoring form ──────────────────────────────────────────────────────────
    st.divider()
    prev = {d: scores.get((qid, d), {}) for d in ALL_DIMENSIONS}
    is_update = all(prev[d] for d in ALL_DIMENSIONS)

    with st.form(key=f"form_{qid}"):
        ratings: dict = {}
        for dim in LIKERT_DIMENSIONS:
            label = dim.replace("_", " ").title()
            prev_val = int(prev[dim].get("score", 3))
            ratings[dim] = st.radio(
                f"**{label}**",
                options=[1, 2, 3, 4, 5],
                format_func=lambda v, d=dim: ANCHORS[d][v - 1],
                index=prev_val - 1,
                horizontal=False,
            )

        hal_val = bool(prev["hallucination_present"].get("score", False))
        hal = st.checkbox("**Hallucination present**", value=hal_val)
        st.caption("Checked = at least one fabricated claim not supported by the KB or tool result")
        prev_notes = prev["hallucination_present"].get("notes", "")
        notes = st.text_area("Notes (describe what was fabricated, if checked)", value=prev_notes)

        st.divider()
        auto_adv = st.checkbox("Auto-advance after submit", value=True)
        btn_label = "Update scores for this query" if is_update else "Submit scores for this query"
        submitted = st.form_submit_button(btn_label, type="primary")

    if submitted:
        for dim in LIKERT_DIMENSIONS:
            _save_score(qid, dim, int(ratings[dim]), "", OUTPUT_PATH, HISTORY_PATH)
        _save_score(qid, "hallucination_present", hal, notes, OUTPUT_PATH, HISTORY_PATH)
        st.toast("Saved!")
        if auto_adv and idx < total - 1:
            st.session_state.idx = idx + 1
        st.rerun()

    # ── Navigation ────────────────────────────────────────────────────────────
    st.divider()
    c1, c2, c3 = st.columns([1, 1, 4])
    if c1.button("◀ Previous", disabled=idx == 0):
        st.session_state.idx = idx - 1
        st.rerun()
    if c2.button("Next ▶", disabled=idx == total - 1):
        st.session_state.idx = idx + 1
        st.rerun()
    qids = [r["query_id"] for r in rows]
    sel = c3.selectbox("Jump to query ID", qids, index=idx)
    if sel != qid:
        st.session_state.idx = qids.index(sel)
        st.rerun()

    prog2 = _compute_progress(_load_existing_scores(OUTPUT_PATH), total)
    total_j2, target_j2 = prog2["total_judgments"], prog2["target_judgments"]
    st.progress(
        total_j2 / target_j2 if target_j2 else 0,
        text=f"{total_j2} / {target_j2} judgments captured",
    )


if __name__ == "__main__":
    main()
