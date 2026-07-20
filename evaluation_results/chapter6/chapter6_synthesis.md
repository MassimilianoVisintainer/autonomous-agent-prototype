# Chapter 6 — Analytical Synthesis

Generated: 2026-07-20 09:34 UTC

- Metrics input: evaluation_results/metrics_report.json
- Rubric input: evaluation_results/rubric_scores.jsonl
- Evaluation input: evaluation_results/eval_results.jsonl

This document organises the empirical findings from quantitative metrics (§4.4) and qualitative rubric scores (§4.3.3) into a structured analytical scaffold for Chapter 6 prose drafting. Each named finding cites the relevant figure and table; the author writes Chapter 6 narrative from this scaffold. The document does not contain Chapter 6 prose — only structured findings with embedded evidence pointers.

---

## RQ1 — Resolution quality

### Finding 1.1 — High autonomous resolution on canonical transactional queries

- Gross containment: 80/81 = 98.8% (contained queries where agent did not falsely escalate). Table 6.1, Figure 6.4.
- Net containment: 79/81 = 97.5% (no false escalation AND substantive LLM response). Table 6.1.
- Rubric corroboration (Table 6.6): contained queries score factual_accuracy 4.864, completeness 4.543.
- Caveat: rubric scores on contained queries reflect a high-capability baseline (tool result always available); this partially inflates scores relative to information-retrieval-only queries.

### Finding 1.2 — Intent classification strong but degrades on taxonomy-edge cases

- Overall accuracy: 124/130 = 95.4%. Table 6.2, Figure 6.1.
- Six misclassifications (Figure 6.6 off-diagonal cells):
  - account_help → product_info: 1
  - multi_issue_dispute → complaint: 1
  - product_info → shipping_info: 1
  - refund_request → multi_issue_dispute: 1
  - refund_request → complaint: 1
  - shipping_info → order_status: 1
- Interpretation: all six are at semantically adjacent intent boundaries (refund ↔ complaint ↔ multi_issue; product ↔ shipping; account ↔ product). No clear-domain errors; misclassifications reflect genuine ambiguity in the twelve-intent taxonomy.

### Finding 1.3 — Retrieval recall strong despite structurally low precision-at-5

- Mean recall@5: 0.784 over 74 queries with retrieval gold. 51 of 74 achieve recall@5 = 1.0. Table 6.1.
- Mean precision@5: 0.213.
- Caveat: precision@5 is structurally depressed because most gold sets contain only one chunk (1/5 = 0.20 maximum precision when one target is retrieved). Low precision does not indicate retrieval quality failure; recall@5 = 0.784 is the operative measure.

---

## RQ2 — Emotional intelligence

### Finding 2.1 — Tone well-calibrated but concentrated on escalation paths

- Aggregate tone_appropriateness: mean=3.946; no 1s or 2s observed (Figure 6.3).
- Cross-tab (Table 6.6, Figure 6.4): escalated queries tone=4.464 vs contained queries tone=3.802.
- Interpretation: the warm-handoff system prompt produces measurably higher tone scores on escalated paths. The baseline (non-escalation) response is professional but generic. This constitutes conditional empathy mode rather than a uniform warm baseline — a design property of the architecture, not a capability gap.
- For §6.x: this finding supports the discussion of grounding-prompt specificity and the case for emotion-aware generation conditioning even below the escalation threshold.

### Finding 2.2 — False-negative escalations show verbal acknowledgement without structural routing

- 12 FN escalations: mean tone_appropriateness = 4.250, exceeding contained baseline 3.802.
- Interpretation: VADER-scored emotion leaked into the LLM generation prompt, producing empathetic language in the response even without triggering the structural escalation path. This is a partially adaptive behaviour — the agent is warmer to distressed customers — but lacks the human-in-the-loop routing that the escalation path provides.
- For §6.x: graduated escalation responses (verbal acknowledgement + enhanced monitoring flag) as future work.

---

## RQ3 — Escalation behaviour

### Finding 3.1 — Precision excellent; recall is the principal weakness

- Precision: 0.941 (16 TP / 17 positives). One false positive (Q-091). Table 6.3, Figure 6.2.
- Recall: 0.571 (16 TP / 28 positives-expected). F1: 0.711.
- The precision/recall asymmetry is by design: the three-trigger framework prioritises false-negative avoidance over false-positive avoidance. The low recall exposes the coverage gap on moderate-signal queries.

### Finding 3.2 — 22 false negatives split into structural and genuine failures

- Structural FNs (1): out_of_scope queries expected-escalated but handled by refusal_template per §3.2 boundary-intent exclusion. Not a miss; a definitional gap between test set vocabulary and agent architecture.
- Genuine FNs (~11): moderate-emotion below VADER cutoff (Q-092, Q-095); exceeded_authority without a tool call (Q-080, Q-085, Q-087 — no order ID in query, no tool dispatch, no authority trigger); multi-issue cumulative complexity below any independent trigger threshold.
- Implication: the three-trigger framework does not accumulate signal across simultaneous but independently sub-threshold stressors. Cumulative complexity is a structural gap.

### Finding 3.3 — Threshold calibration strong on boundary pairs

- 8/10 transition pairs correctly handled (Figure 6.7). Failures at flag intersections (emotion_overlay_transactional: 2/3).
- Consistency on no-transition pairs: 33/34 = 97.1%.

---

## RQ4 — Quality of agent behaviour

### Finding 4.1 — Hallucination rare and architecturally concentrated

- 6/130 = 4.6% of queries have hallucination flagged. Table 6.7.
- 5 of 6 share 'order_id_ignored' failure mode: multi_issue_dispute queries carry an order ID in the query text, but the intent classifier fires a non-transactional intent, so tool dispatch is skipped, and the LLM generation asks for an order ID already provided.
- 1 of 6: 'fabricated_behavior' (Q-028: response claims escalation that did not occur).
- Root cause: order-ID extraction is order-ID-lookup-gated at tool dispatch, not at orchestrator level. Future work: extract order identifiers at orchestrator entry and expose to generation unconditionally.

### Finding 4.2 — Structural quality is the dimension with most room for improvement

- Mean structural_quality: 3.754 (lowest of four Likert dimensions). Figure 6.3, Table 6.5.
- Distribution: mode=4 (69 queries), 15 queries at 5, 45 at 3.
- Calibration caveat (§4.5): the rubric reserves 5 for exemplary formatting. The agent produces well-formed prose paragraphs; the 3.75 mean partially reflects the rubric's high 5-bar rather than a structural capability deficit. Both the finding and its calibration note belong in §4.5.

### Finding 4.3 — Multi-issue disputes are the hardest intent across all rubric dimensions

- multi_issue_dispute rubric (Table 6.5 cross-intent): factual_accuracy=3.62, completeness=3.38, structural_quality=3.00.
- Contributing factors: (a) four of five hallucination cases involve this intent (order_id_ignored); (b) the intent is outside transactional scope so no tool dispatch occurs, creating a purely KB-retrieval response that may miss customer-specific facts; (c) the escalation system does not aggregate signal across multiple simultaneous triggers.

---

## Methodological caveats for §4.5

**Caveat 1 — Tier-1 tool-call route adherence vs outcome correctness.**
The tier-level match metric counts route adherence: expected tier-1 lookup, observed tier-3 refund = mismatch, even though the tier-3 call correctly serves the customer. Correctly resolving an authority-gated request by calling the appropriate higher-tier tool is good agent behaviour; the metric penalises it. The 34/50 = 68% figure should be interpreted as route-adherence rate, not outcome-correctness rate.

**Caveat 2 — Status appropriateness denominator is conditional on tool call.**
The metric (authority-exceeded cases with appropriate tool_status: 8/20 = 40.0%) uses all authority-escalated queries as denominator, including those where no tool was called (no order ID → no dispatch → no status). Among queries that *did* call a tool, the rate is higher. §4.5 should report both the conditional and unconditional rates.

**Caveat 3 — Structural quality rubric calibration.**
The rubric reserves 5 for 'exemplary formatting with headers, bullets, and scannable layout.' The agent produces prose responses; only 15/130 score 5. The 3.754 aggregate mean is partly an artefact of this high 5-bar. The finding (structural_quality is lowest dimension) is valid; its magnitude is partially a calibration effect.

**Caveat 4 — out_of_scope ground-truth vocabulary.**
The test set marks 1 out_of_scope queries as expected_handling='escalated' with expected_escalation_reason='out_of_scope'. The §3.2 boundary-intent exclusion exempts OUT_OF_SCOPE from the low_confidence escalation trigger; the agent routes these to refusal_template instead. This is a definitional gap: the test set labels a refusal as a missing escalation. The 22 FNs conflate this structural gap with genuine missed escalations. §4.5 should report the structural and genuine FNs separately.

---

## Convergence across evaluation methods

The quantitative metrics and rubric scores converge on the same structural picture. The agent performs excellently on canonical transactional queries: 95.4% intent accuracy, 98.8% gross containment, factual_accuracy=4.86 on contained queries. Performance degrades predictably at the edges: moderate-emotion below VADER cutoff (FN escalations), multi-issue complexity (lowest rubric scores), and taxonomy-adjacent intent boundaries (six misclassifications).

The tone pattern is the clearest convergence point. Quantitatively, 17 queries were escalated (13.1%); qualitatively, those escalated queries score 4.46 on tone vs 3.80 on contained queries. The rubric confirms what the architecture predicts: warm-handoff prompt drives higher tone scores. The 12 FNs scoring 4.25 on tone — above the contained baseline — suggests the VADER score leaked warm register into generation without structural escalation. Quantitative metrics alone would not surface this nuance.

The hallucination analysis adds a third layer: 5 of 6 cases trace to a single architectural gap (order-ID gating at tool dispatch, not at orchestrator entry) on the multi_issue_dispute intent — the same intent that scores lowest on completeness (3.38) and structural quality (3.00). The convergence across three measurement methods on the same intent strengthens the conclusion that multi-issue query handling is the principal improvement target for a next prototype iteration.