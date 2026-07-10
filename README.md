# Autonomous AI Agent for Customer Support — Thesis Prototype

This repository contains the software artefact and evaluation apparatus accompanying the master's thesis:

> **Autonomous AI Agents in Customer Support: An Analysis of Resolution Quality, Emotional Intelligence, and Escalation Behaviour**
> Massimiliano Visintainer — Matriculation No. 9219959
> M.Sc. Computer Science, IU International University of Applied Sciences
> Supervisor: Prof. Dr. Rachel John Robinson

The thesis follows a **design-science** methodology (Hevner et al., 2004; Peffers et al., 2007): the prototype is not an engineering deliverable but an *instrument of inquiry* — a fully instrumented autonomous support agent whose every decision is recorded, so that its behaviour can be measured under controlled conditions. This repository is what makes the thesis's empirical claims inspectable and reproducible.

Everything in `evaluation_results/` is the **exact data** underlying the figures and tables of Chapter 6.

---

## What the artefact is

An autonomous customer-support agent — it selects and delivers its own responses, involving a human only when its escalation logic decides. It is implemented **from first principles**, without a high-level agent framework, so that each architectural concern maps to one independently testable module and each component's contribution can be measured in isolation.

The agent processes one query at a time through a strictly linear five-stage pipeline:

```
query → intent classification → retrieval → tool dispatch → escalation → generation → response
```

| Stage | Module | Implementation |
|---|---|---|
| Intent classification | `src/nlu.py` | Single Gemini call, constrained JSON output, twelve-intent taxonomy |
| Retrieval | `src/retrieval.py` | Dense-only, `all-MiniLM-L6-v2`, exact top-*k* (k=5) over 65 pre-embedded chunks |
| Tool dispatch | `src/tools.py` | Four authority-tiered tools, read-only eligibility checks |
| Escalation | `src/escalation.py` | Three independent triggers (authority / emotion / confidence) |
| Generation | `src/grounding.py` | Cite-or-silent prompt + post-hoc citation verification |
| Orchestration | `src/agent.py` | Sequences the stages; owns no domain logic |

Two stages make a model call (classification, generation); three are local and deterministic. Every stage writes to one uniformly shaped log record, which is what allows the Chapter 4 metrics to be computed mechanically.

### Design constraints (deliberate)

Declared in §4.2 of the thesis and enforced in code:

- No fine-tuning or retraining of the foundation model.
- No connection to live systems of record — the tool layer reads static CSV fixtures.
- Tools are **eligibility checks, not actions**: a successful refund result means the amount is *within authority*, not that money moved.
- The retrieval layer is intentionally below the production frontier — no BM25 hybrid, no cross-encoder re-ranking, no query rewriting — so that measured `recall@5` is a clean property of dense retrieval alone.

### Fixed parameters

| Parameter | Value | Source |
|---|---|---|
| Confidence floor | `0.75` | `src/escalation.py` |
| Emotion floor (VADER compound) | `-0.5` | `src/escalation.py` |
| Refund authority limit | `€100` | `src/tools.py` |
| Cancellation window | `24 hours` | `src/tools.py` |
| Reference timestamp | `2026-05-15 14:00 UTC` | `src/tools.py` |
| Retrieval depth *k* | `5` | `src/retrieval.py` |

The **fixed reference timestamp** is an internal-validity safeguard (§4.5): temporal thresholds evaluate identically on every run, so wall-clock drift cannot change which queries fall inside a cancellation window.

---

## Headline results

Computed from `evaluation_results/eval_results.jsonl` over the 130-query test set. Reported in Chapter 6; reproduced here for orientation only.

| Metric | Value |
|---|---|
| Intent classification accuracy | 124/130 = **95.4 %** |
| Retrieval recall@5 (74 gold-set queries) | **0.784** |
| Retrieval precision@5 | 0.213 † |
| Escalation precision / recall / F1 | 0.941 / 0.421 / **0.582** |
| Gross containment | 80/81 = **98.8 %** |
| Net containment | 79/81 = 97.5 % |
| Clarification rate | 9/11 = 81.8 % |
| Boundary-pair transition rate | 8/10 = 80.0 % |
| Hallucination rate | 6/130 = **4.6 %** |
| Rubric means (accuracy / completeness / tone / structure) | 4.538 / 4.269 / 3.946 / 3.754 |

† Precision@5 is mechanically constrained: 45 of the 74 gold sets contain a single target chunk, capping precision at 0.20 for those queries. **Recall is the operative retrieval measure** (§4.4).

---

## Evaluation apparatus

Four programs, run in sequence. Keeping the apparatus separate from the agent — and data collection separate from analysis — is what allows Chapter 6 to be written from already-collected data rather than results shaped as they were gathered.

| Program | Role | Thesis §|
|---|---|---|
| `scripts/run_evaluation.py` | Executes the 130-query test set, writes one log record per query | §5.6 |
| `scripts/compute_metrics.py` | Computes the eight quantitative metrics mechanically | §4.4 |
| `scripts/rubric_scorer.py` | Streamlit UI for the five-dimension human rubric | §4.3.3 |
| `scripts/build_chapter6_analysis.py` | Joins both strands; emits figures, tables, synthesis | §6 |

No human judgement enters `compute_metrics.py`; it is a deterministic function of the log. The rubric strand is scored by a **single rater** — the author — which is the study's principal reliability limitation (§4.5): the design supports *stability* but not inter-rater *reproducibility*, and no inter-rater agreement is claimed.

### Data files

| File | Contents |
|---|---|
| `data/knowledge_base.csv` | 65 pre-chunked passages, 6 categories, 150–300 tokens each |
| `data/customers.csv` | 18 fictional customers (tier, status) |
| `data/orders.csv` | 50 fictional orders, populated at the threshold boundaries |
| `data/test_queries.csv` | 130 queries: 68 canonical, 34 variant, 28 boundary |

The test set is annotated with ground-truth intent, expected handling, expected escalation reason, expected tool calls, expected KB chunks, emotional-intensity band, and `linked_query_id` joining **boundary pairs** — two queries identical except in one threshold-relevant property, which is what makes the boundary-pair analysis a paired statistic rather than a marginal proportion.

All fixtures are **synthetic and static**. Findings therefore speak to the architecture under controlled corpus conditions, not to heterogeneous enterprise content.

---

## Installation

```bash
git clone https://github.com/MassimilianoVisintainer/autonomous-agent-prototype.git
cd autonomous-agent-prototype
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```

A `GOOGLE_API_KEY` is required. Copy `.env.example` to `.env` and add your Google AI Studio key.

---

## Usage

### Tests

```bash
pytest
```

### Interactive agent

```bash
streamlit run app.py
```

Opens the chat interface with the full pipeline active. The reasoning-trace panel exposes every stage for the current interaction — classified intent with confidence, escalation decision with emotion score, retrieved chunks with similarity scores, and the full tool result. This is the inspection aid shown in Figure 5.13 of the thesis.

### Reproducing the evaluation

```bash
python scripts/run_evaluation.py
```

Runs all 130 queries and writes JSONL to `evaluation_results/eval_<timestamp>.jsonl`. The LLM response cache is **disabled** for the run, so natural model variance is captured — the pipeline is reproducible only to the extent the hosted model's own variability permits (§4.5).

- **Runtime** ≈ 25–45 min (2 LLM calls/query; free-tier rate limits on `gemini-3.1-flash-lite`).
- **Robustness** — rate-limit errors (HTTP 429) retry with backoff (5 s → 15 s → 60 s).
- **Resumability** — rerunning with the same `--output` path re-attempts only queries whose classification or generation call failed. A run degraded by daily quota can be continued the next day with the same command.
- **Flags** — `--limit N` (smoke test), `--output PATH`.

Then:

```bash
python scripts/compute_metrics.py --input evaluation_results/eval_results.jsonl
streamlit run scripts/rubric_scorer.py     # 130 queries × 5 dimensions = 650 judgements
python scripts/build_chapter6_analysis.py
```

The rubric scorer is resumable, shows anchor descriptions inline at every scale point, archives prior scores to `rubric_scores_history.jsonl` on re-scoring, and reports per-dimension score distributions in the sidebar — the audit-trail and drift-monitoring mitigations described in §4.3.3. Use `RUBRIC_INPUT=…` to score a different evaluation file.

`build_chapter6_analysis.py` writes to `evaluation_results/chapter6/`: seven figures (PNG 300 DPI + SVG), `tables.md`, and `chapter6_synthesis.md`. All values are computed from the data at runtime.

---

## Repository structure

```
.
├── app.py                          # Streamlit chat interface + reasoning trace
├── data/                           # static fixtures (§4.3.2)
│   ├── knowledge_base.csv          # 65 chunks
│   ├── customers.csv               # 18 customers
│   ├── orders.csv                  # 50 orders
│   └── test_queries.csv            # 130 annotated test queries (§4.3.1)
├── src/
│   ├── intents.py                  # twelve-intent taxonomy — single source of truth
│   ├── data_loaders.py             # typed, frozen-record CSV loaders
│   ├── llm_client.py               # Gemini client: cache + backoff retry
│   ├── nlu.py                      # intent classifier (§5.2)
│   ├── retrieval.py                # dense retrieval, embed-once (§5.3)
│   ├── tools.py                    # authority-tiered tool layer (§5.4)
│   ├── escalation.py               # three-trigger pipeline (§5.5)
│   ├── grounding.py                # cite-or-silent generation (§5.3)
│   └── agent.py                    # thin orchestrator (§5.1)
├── scripts/
│   ├── run_evaluation.py           # resumable harness (§5.6)
│   ├── compute_metrics.py          # eight quantitative metrics (§4.4)
│   ├── rubric_scorer.py            # five-dimension rubric UI (§4.3.3)
│   └── build_chapter6_analysis.py  # figures, tables, synthesis (§6)
├── evaluation_results/             # ← the thesis evidence base
│   ├── eval_results.jsonl          # 130 interaction records
│   ├── metrics_report.{md,json}    # computed metrics
│   ├── rubric_scores.jsonl         # 650 human judgements
│   ├── rubric_scores_history.jsonl # archived re-scores (audit trail)
│   └── chapter6/                   # figures, tables.md, synthesis
└── tests/                          # unit tests, one per module
```

---

## Scope and limitations

Stated in full in §4.5 and §8.4 of the thesis. In brief, the study establishes **how one fully instrumented autonomous agent behaved on one constructed test set under controlled conditions** — and nothing wider. It does not establish that autonomous agents outperform humans or alternative designs, that the findings generalise to production traffic, that the results are statistically significant, or that customers would be more satisfied.

- **Single architecture, single backbone.** No comparison against alternative designs or a human baseline. A different model would plausibly shift classification, tone, and hallucination behaviour.
- **Artificial evaluation.** Hand-authored, single-domain, single-language, read-only test set, built for deliberate boundary coverage rather than production realism.
- **Descriptive, not inferential.** A constructed population supports rates and per-category breakdowns — not significance testing or confidence intervals.
- **No satisfaction measurement.** The rubric's tone and completeness dimensions are **perception proxies**, never measurements of customer satisfaction.
- **Single rater.** Qualitative findings rest on one rater who is also the author of the prototype and the queries.

---

## Citation

```bibtex
@mastersthesis{visintainer2026autonomous,
  author  = {Visintainer, Massimiliano},
  title   = {Autonomous AI Agents in Customer Support: An Analysis of
             Resolution Quality, Emotional Intelligence, and Escalation Behaviour},
  school  = {IU International University of Applied Sciences},
  year    = {2026},
  type    = {M.Sc. thesis},
  note    = {Prototype and evaluation data:
             \url{https://github.com/MassimilianoVisintainer/autonomous-agent-prototype}}
}
```

## License

The synthetic data fixtures in `data/` describe fictional customers, orders, and products. Any resemblance to real records is coincidental.