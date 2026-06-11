# Evaluation Metrics Report

Computed on 2026-06-11 08:11:55 UTC from `eval_results.jsonl`.

Total queries evaluated: 130.

## 1. Intent classification accuracy

Overall: 124/130 = 95.4%.

Per-intent breakdown:

| Intent | Correct | Total | Accuracy |
|---|---|---|---|
| order_status | 12 | 12 | 100.0% |
| order_modify | 10 | 10 | 100.0% |
| order_cancel | 11 | 11 | 100.0% |
| refund_request | 12 | 14 | 85.7% |
| product_info | 11 | 12 | 91.7% |
| return_policy | 11 | 11 | 100.0% |
| shipping_info | 11 | 12 | 91.7% |
| account_help | 11 | 12 | 91.7% |
| complaint | 10 | 10 | 100.0% |
| multi_issue_dispute | 7 | 8 | 87.5% |
| out_of_scope | 10 | 10 | 100.0% |
| ambiguous_query | 8 | 8 | 100.0% |

Misclassifications (non-zero confusion matrix cells):

- account_help → product_info: 1
- multi_issue_dispute → complaint: 1
- product_info → shipping_info: 1
- refund_request → complaint: 1
- refund_request → multi_issue_dispute: 1
- shipping_info → order_status: 1

## 2. Retrieval precision and recall

Evaluated over 74 rows that have retrieval gold.

Mean precision@5: 0.213
Mean recall@5: 0.784

Recall@5 distribution:

- [0.00,0.25): 8 rows
- [0.25,0.50): 4 rows
- [0.50,0.75): 11 rows
- [0.75,1.00): 0 rows
- [1.00,1.00]: 51 rows

## 3. Tool-call correctness

Tool-name (tier-level) match rate: 34/50 = 68.0%
Order-ID match rate: 42/50 = 84.0%
Status appropriateness rate (authority-exceeded cases): 8/20 = 40.0%

Per-tier breakdown:

| Tier | Expected calls | Correctly fired | Rate |
|---|---|---|---|
| 1 | 27 | 10 | 37.0% |
| 2 | 16 | 16 | 100.0% |
| 3 | 8 | 8 | 100.0% |

## 4. Escalation precision and recall

Overall:

- True positives: 16
- False positives: 1
- False negatives: 22
- True negatives: 91
- Precision: 0.941
- Recall: 0.421
- F1: 0.582

Per-reason breakdown:

| Reason | GT count | Agent triggered | TP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| high_emotion | 15 | 10 | 9 | 6 | 0.900 | 0.600 |
| exceeded_authority | 20 | 8 | 8 | 12 | 1.000 | 0.400 |
| out_of_scope | 11 | 0 | 0 | 11 | 0.000 | 0.000 |
| low_confidence | 0 | 0 | 0 | 0 | 0.000 | 0.000 |

## 5. Outcome containment

Gross containment: 80/81 = 98.8%
Net containment: 79/81 = 97.5%
Clarification rate: 9/11 = 81.8%

## 6. Boundary-pair transition rate

Variant rows with linked canonical query: 44
Transition pairs (different expected handling): 10
Pairs where agent behaviour also transitioned: 8
Transition rate: 8/10 = 80.0%

No-transition pairs (same expected handling): 34
Consistent pairs: 33/34 = 97.1%

Per-threshold-flag breakdown:

| Threshold flag | Pair count | Correct transitions | Rate |
|---|---|---|---|
| ambiguity_missing_id | 1 | 1 | 100.0% |
| ambiguity_resolved_by_id | 1 | 0 | 0.0% |
| cancel_outside_24h_window | 1 | 1 | 100.0% |
| emotion_above_threshold | 1 | 1 | 100.0% |
| emotion_overlay_transactional | 3 | 2 | 66.7% |
| modify_shipped_order | 1 | 1 | 100.0% |
| multi_issue_high_severity | 1 | 1 | 100.0% |
| refund_above_100 | 1 | 1 | 100.0% |

## 7. Latency

Median: 6124 ms
Mean: 7738 ms
p90: 19239 ms
p95: 26510 ms
Min: 516 ms  |  Max: 35858 ms
Outliers excluded (>600s): 0

Per-intent median (ms):

- account_help: 9530 ms
- ambiguous_query: 4688 ms
- complaint: 5164 ms
- multi_issue_dispute: 10226 ms
- order_cancel: 3844 ms
- order_modify: 6382 ms
- order_status: 2203 ms
- out_of_scope: 819 ms
- product_info: 9055 ms
- refund_request: 7312 ms
- return_policy: 6202 ms
- shipping_info: 3922 ms

## 8. Summary

- Total queries: 130
- Harness status: ok=130
- Classification method: llm=130
- Generation method: clarification_template=8, llm=95, llm_handoff=17, refusal_template=10
