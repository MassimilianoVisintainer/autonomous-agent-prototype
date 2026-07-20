# Evaluation Tables

## Table 6.1 — Eight-metric summary

| Metric | Value | §4.4 section |
|---|---|---|
| Intent accuracy | 124/130 = 95.4% | §4.4.1 |
| Retrieval recall@5 (mean) | 0.784 over 74 queries | §4.4.2 |
| Tool-call tier match rate | 34/50 = 68.0% | §4.4.3 |
| Escalation precision | 0.941 | §4.4.4 |
| Escalation recall | 0.571 | §4.4.4 |
| Escalation F1 | 0.711 | §4.4.4 |
| Gross containment | 80/81 = 98.8% | §4.4.5 |
| Net containment | 79/81 = 97.5% | §4.4.5 |
| Clarification rate | 9/11 = 81.8% | §4.4.5 |

## Table 6.2 — Per-intent classification accuracy

| Intent | Correct | Total | Accuracy | Misclassified as |
|---|---|---|---|---|
| order_status | 12 | 12 | 100.0% | — |
| order_modify | 10 | 10 | 100.0% | — |
| order_cancel | 11 | 11 | 100.0% | — |
| refund_request | 12 | 14 | 85.7% | multi_issue_dispute×1, complaint×1 |
| product_info | 11 | 12 | 91.7% | shipping_info×1 |
| return_policy | 11 | 11 | 100.0% | — |
| shipping_info | 11 | 12 | 91.7% | order_status×1 |
| account_help | 11 | 12 | 91.7% | product_info×1 |
| complaint | 10 | 10 | 100.0% | — |
| multi_issue_dispute | 7 | 8 | 87.5% | complaint×1 |
| out_of_scope | 10 | 10 | 100.0% | — |
| ambiguous_query | 8 | 8 | 100.0% | — |

## Table 6.3 — Escalation precision/recall per reason

| Reason | GT count | Triggered | TP | FN | Precision | Recall |
|---|---|---|---|---|---|---|
| high_emotion | 15 | 10 | 9 | 6 | 0.900 | 0.600 |
| exceeded_authority | 20 | 8 | 8 | 12 | 1.000 | 0.400 |
| out_of_scope | 1 | 0 | 0 | 1 | 0.000 | 0.000 |
| low_confidence | 0 | 0 | 0 | 0 | 0.000 | 0.000 |

## Table 6.4 — Tool-call correctness by tier

| Tier | Expected calls | Tier-match | Order-ID match | Notes |
|---|---|---|---|---|
| 1 | 27 | 10/27 = 37% | 42/50 | Tier-1 match is route adherence; authority-gated queries correctly call higher-tier tools instead |
| 2 | 16 | 16/16 = 100% | 42/50 | All modify/cancel calls fired correctly |
| 3 | 8 | 8/8 = 100% | 42/50 | All refund calls fired correctly |

## Table 6.5 — Rubric score statistics

| Dimension | Mean | SD | Median | Mode | 5s | 4s | 3s | 2s | 1s |
|---|---|---|---|---|---|---|---|---|---|
| Factual accuracy | 4.538 | 0.661 | 5.0 | 5 | 81 | 39 | 9 | 1 | 0 |
| Completeness | 4.269 | 0.814 | 4.0 | 5 | 59 | 52 | 15 | 3 | 1 |
| Tone appropriateness | 3.946 | 0.719 | 4.0 | 4 | 30 | 63 | 37 | 0 | 0 |
| Structural quality | 3.754 | 0.660 | 4.0 | 4 | 15 | 69 | 45 | 1 | 0 |

## Table 6.6 — Rubric scores by handling outcome

| Handling | N | Factual acc. | Completeness | Tone appr. | Structural qual. |
|---|---|---|---|---|---|
| contained | 81 | 4.864 | 4.543 | 3.802 | 3.938 |
| escalated | 28 | 3.929 | 3.607 | 4.464 | 3.429 |
| clarified | 11 | 4.182 | 4.182 | 3.909 | 3.273 |

## Table 6.7 — Hallucination catalogue

| Query ID | Predicted intent | Gen. method | Failure mode | Scorer notes |
|---|---|---|---|---|
| Q-028 | order_cancel | llm | fabricated_behavior | It states that it will be escalated to a senior agent, but the escalation field is set to No. |
| Q-044 | multi_issue_dispute | llm | order_id_ignored | The user is providing the order id. The agent is asking it again, though it acknowledge the the user |
| Q-100 | multi_issue_dispute | llm | order_id_ignored | The user provided the order id, but the agent is asking it again. |
| Q-102 | multi_issue_dispute | llm | order_id_ignored | the user is providing the order id, but the agent is aking for it again |
| Q-104 | multi_issue_dispute | llm | order_id_ignored | User is providing the order number, but the agent is asking it again |
| Q-125 | ambiguous_query | clarification_template | order_id_ignored | The user is specifically asking for help and giving their oder number. The agent do not understand t |
