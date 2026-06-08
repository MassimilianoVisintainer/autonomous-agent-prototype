#!/usr/bin/env python3
"""Evaluation harness for the autonomous customer-support agent.

Loads the 130-query test set from test_queries.csv, runs each query through
agent.process_query(), and writes one JSON object per line to an output file
under evaluation_results/.

Key design decisions:
  - LLM response cache is disabled (DISABLE_LLM_CACHE=1 set before any imports)
    so natural LLM variance is captured rather than deterministic cached replies.
  - Output is rewritten in full after every row so a crashed run leaves a
    valid JSONL file that the next invocation will resume from.
  - Rows where classification_method or generation_method is "llm_error" are
    treated as incomplete and re-attempted on the next run with the same output
    path. Only rows where both methods succeeded are treated as done.
  - Rate-limit errors (429/quota) are retried with exponential backoff (5s, 15s,
    60s) by the LLM client itself; if process_query still raises, the harness
    marks the row harness_status="llm_failed" and continues.
"""

import os
import sys
import pathlib as _pathlib
# Ensure the repo root is on sys.path when the script is run directly
sys.path.insert(0, str(_pathlib.Path(__file__).parent.parent))

os.environ["DISABLE_LLM_CACHE"] = "1"

import argparse
import json
import pathlib
import time
from datetime import datetime, timezone

from src import agent
from src.data_loaders import load_test_queries
from src.llm_client import LLMClientError

DEFAULT_OUTPUT_DIR = pathlib.Path("evaluation_results")
RETRY_DELAYS = [5, 15, 60]
RATE_LIMIT_INDICATORS = ("rate limit", "quota", "429", "resourceexhausted")


def _is_rate_limit_error(exc: Exception) -> bool:
    if not isinstance(exc, LLMClientError):
        return False
    msg = str(exc).lower()
    return any(indicator in msg for indicator in RATE_LIMIT_INDICATORS)


def _is_row_complete(row: dict) -> bool:
    """A row is complete only if neither the classification nor the generation
    LLM call failed. Rows with llm_error in either field are re-attempted."""
    if row.get("classification_method") == "llm_error":
        return False
    if row.get("generation_method") == "llm_error":
        return False
    return True


def _tool_result_summary(tool_result) -> str | None:
    if tool_result is None:
        return None
    parts = [f"{k}={v}" for k, v in tool_result.data.items()]
    return ", ".join(parts) if parts else tool_result.status


def _build_output_row(test_query, agent_response, elapsed_ms: int, harness_status: str) -> dict:
    cl = agent_response.classification
    tr = agent_response.tool_result
    esc = agent_response.escalation
    gen = agent_response.generation
    chunks = agent_response.retrieved_chunks

    return {
        "query_id": test_query.query_id,
        "query_text": test_query.query_text,
        "expected_intent": test_query.intent_label,
        "test_category": test_query.test_category,
        "expected_handling": test_query.expected_handling,
        "expected_escalation_reason": test_query.expected_escalation_reason,
        "expected_tool_calls": test_query.expected_tool_calls,
        "expected_kb_chunks": test_query.expected_kb_chunks,
        "emotional_intensity_band": test_query.emotional_intensity_band,
        "threshold_test_flag": test_query.threshold_test_flag,
        "linked_query_id": test_query.linked_query_id,
        "predicted_intent": cl.intent.value,
        "classification_confidence": cl.confidence,
        "classification_method": cl.method,
        "classification_reasoning": cl.reasoning,
        "retrieved_chunk_ids": [rc.chunk.kb_id for rc in chunks],
        "retrieved_chunk_scores": [round(rc.score, 4) for rc in chunks],
        "tool_called": tr.tool if tr is not None else None,
        "tool_status": tr.status if tr is not None else None,
        "tool_result_summary": _tool_result_summary(tr),
        "escalation_decision": esc.should_escalate,
        "escalation_triggers": esc.triggers,
        "emotion_score": round(esc.emotion_score, 4),
        "generation_text": gen.text,
        "generation_method": gen.method,
        "citations": gen.citations,
        "harness_status": harness_status,
        "harness_error_type": None,
        "harness_error_message": None,
        "elapsed_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _build_failure_row(test_query, exception: Exception, elapsed_ms: int, harness_status: str) -> dict:
    return {
        "query_id": test_query.query_id,
        "query_text": test_query.query_text,
        "expected_intent": test_query.intent_label,
        "test_category": test_query.test_category,
        "expected_handling": test_query.expected_handling,
        "expected_escalation_reason": test_query.expected_escalation_reason,
        "expected_tool_calls": test_query.expected_tool_calls,
        "expected_kb_chunks": test_query.expected_kb_chunks,
        "emotional_intensity_band": test_query.emotional_intensity_band,
        "threshold_test_flag": test_query.threshold_test_flag,
        "linked_query_id": test_query.linked_query_id,
        "predicted_intent": None,
        "classification_confidence": None,
        "classification_method": None,
        "classification_reasoning": None,
        "retrieved_chunk_ids": [],
        "retrieved_chunk_scores": [],
        "tool_called": None,
        "tool_status": None,
        "tool_result_summary": None,
        "escalation_decision": None,
        "escalation_triggers": [],
        "emotion_score": None,
        "generation_text": None,
        "generation_method": None,
        "citations": [],
        "harness_status": harness_status,
        "harness_error_type": type(exception).__name__,
        "harness_error_message": str(exception),
        "elapsed_ms": elapsed_ms,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _load_completed_rows(output_path: pathlib.Path) -> dict[str, dict]:
    """Load existing output rows, keyed by query_id. Tolerates malformed lines."""
    if not output_path.exists():
        return {}
    rows: dict[str, dict] = {}
    try:
        for line in output_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                rows[row["query_id"]] = row
            except (json.JSONDecodeError, KeyError):
                pass
    except OSError:
        pass
    return rows


def _write_rows(output_path: pathlib.Path, rows_by_id: dict[str, dict], query_order: list[str]) -> None:
    """Rewrite the output file with all rows in query order."""
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in query_order:
            if qid in rows_by_id:
                f.write(json.dumps(rows_by_id[qid], ensure_ascii=False) + "\n")
        f.flush()


def run_query_with_retry(query_text: str):
    """Attempt process_query; the LLM client handles 429 retries internally."""
    try:
        return agent.process_query(query_text), "ok", None
    except LLMClientError as exc:
        return None, "llm_failed", exc
    except Exception as exc:
        return None, "error", exc


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent evaluation harness.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.output:
        output_path = pathlib.Path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"eval_{timestamp}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    queries = load_test_queries()
    if args.limit is not None:
        queries = queries[: args.limit]

    total = len(queries)
    query_order = [tq.query_id for tq in queries]

    existing_rows = _load_completed_rows(output_path)
    complete_ids = {qid for qid, row in existing_rows.items() if _is_row_complete(row)}
    incomplete_ids = {qid for qid, row in existing_rows.items() if not _is_row_complete(row)}
    rows_by_id: dict[str, dict] = dict(existing_rows)

    run_start = time.monotonic()
    n_new = n_reattempted = n_skipped = n_failed = 0

    print(f"Output: {output_path}")
    print(f"Queries: {total} | Complete: {len(complete_ids)} | Incomplete (will retry): {len(incomplete_ids)}")
    print("-" * 72)

    for idx, tq in enumerate(queries, start=1):
        if tq.query_id in complete_ids:
            print(f"[{idx}/{total}] {tq.query_id} | skipped (already complete)")
            n_skipped += 1
            continue

        is_reattempt = tq.query_id in incomplete_ids
        t0 = time.monotonic()
        response, status, exc = run_query_with_retry(tq.query_text)
        elapsed_ms = int((time.monotonic() - t0) * 1000)

        if response is not None:
            row = _build_output_row(tq, response, elapsed_ms, status)
            cl = response.classification
            tr = response.tool_result
            esc = response.escalation
            tool_str = f"{tr.tool}({tr.status})" if tr else "none"
            esc_str = "yes" if esc.should_escalate else "no"
            prefix = "re-attempting" if is_reattempt else "new"
            print(
                f"[{idx}/{total}] {tq.query_id} [{prefix}] | "
                f"predicted={cl.intent.value}({cl.confidence:.2f}) | "
                f"tool={tool_str} | escalate={esc_str} | {elapsed_ms}ms"
            )
            if is_reattempt:
                n_reattempted += 1
            else:
                n_new += 1
        else:
            row = _build_failure_row(tq, exc, elapsed_ms, status)
            print(f"[{idx}/{total}] {tq.query_id} | FAILED ({status}): {str(exc)[:80]}")
            n_failed += 1

        rows_by_id[tq.query_id] = row
        _write_rows(output_path, rows_by_id, query_order)

    total_time = time.monotonic() - run_start
    print("-" * 72)
    print(
        f"Done. {n_new} new | {n_reattempted} re-attempted | "
        f"{n_skipped} skipped | {n_failed} failed | {total_time:.1f}s total"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
