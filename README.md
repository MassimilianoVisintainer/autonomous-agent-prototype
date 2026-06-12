# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 10 complete** — Chapter 6 analysis builder.

The full agent pipeline (classification → retrieval → tool dispatch → escalation → generation) is complete. The evaluation harness, quantitative metrics script, qualitative rubric scoring tool, and Chapter 6 analysis builder are all in place. Run `scripts/compute_metrics.py` for §4.4 metrics, `streamlit run scripts/rubric_scorer.py` for §4.3.3 rubric scoring, and `scripts/build_chapter6_analysis.py` to generate the Chapter 6 figures, tables, and synthesis report.

A `GOOGLE_API_KEY` must be set before running the app or evaluation. Copy `.env.example` to `.env` and fill in your Google AI Studio key.

## Installation

```bash
git clone <repo-url>
cd autonomous-agent-prototype
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Running

### Sanity tests

```bash
pytest
```

### Streamlit app

```bash
streamlit run app.py
```

Opens a chat interface with the full agent pipeline active — classification, retrieval, tool dispatch, escalation, and grounded response generation.

### Running the evaluation

```bash
python scripts/run_evaluation.py
```

Runs all 130 test queries through the agent and writes one JSON object per line to `evaluation_results/eval_<timestamp>.jsonl`. The LLM response cache is automatically disabled for the run so natural LLM variance is captured.

**Robustness:** The LLM client retries rate-limit errors (HTTP 429) automatically with backoff (5s → 15s → 60s) before failing. Rows where the classification or generation LLM call failed (`llm_error`) are treated as incomplete and re-attempted when you rerun with the same `--output` path — so a run degraded by quota exhaustion can be continued the next day by simply running the same command again.

**Flags:**

- `--limit N` — run only the first N queries (useful for smoke-testing)
- `--output PATH` — override the default timestamped output path

```bash
python scripts/run_evaluation.py --limit 5
python scripts/run_evaluation.py --output evaluation_results/my_run.jsonl
```

**Runtime:** The full 130-query run takes approximately 25–45 minutes depending on Gemini's rate limiting (2 LLM calls per query, 500 RPD free-tier limit on `gemini-3.1-flash-lite`).

**Resumability:** If a run is interrupted, rerunning with the same `--output` path will skip queries already completed and continue from where it left off.

Output files land in `evaluation_results/` and are committed to the repository as thesis artefacts.

## Scoring the rubric

After the evaluation run and metrics computation, score the qualitative rubric on each generated response:

```bash
streamlit run scripts/rubric_scorer.py
```

Opens a web UI at http://localhost:8501 where you score each of the 130 queries on the five §4.3.3 dimensions: factual accuracy, completeness, tone appropriateness, structural quality, and hallucination presence. Scores save to `evaluation_results/rubric_scores.jsonl` as each Submit is captured.

The tool is resumable across sessions. To jump to the first unscored query, use the "Resume from first unscored query" button in the sidebar. Re-scoring a query overwrites the visible score and archives the prior score to `evaluation_results/rubric_scores_history.jsonl`.

Suggested workflow: complete the first 10 queries as a calibration sample, review the score distribution in the sidebar, adjust your interpretation of the scale if needed, then proceed through the remaining 120 queries in 20-30 minute sessions.

Use `RUBRIC_INPUT=evaluation_results/my_run.jsonl streamlit run scripts/rubric_scorer.py` to score a different evaluation file.

## Computing metrics

After running the evaluation, compute the §4.4 metrics:

```bash
python scripts/compute_metrics.py --input evaluation_results/eval_results.jsonl
```

Outputs `metrics_report.md` (human-readable, suitable for Chapter 6 quotation) and `metrics_report.json` (machine-readable) in `evaluation_results/`. Use `--output-dir PATH` to override the output directory. The script is re-runnable against any evaluation output JSONL.

## Building Chapter 6 analysis

After computing metrics and scoring the rubric, generate the Chapter 6 figures, tables, and synthesis:

```bash
python scripts/build_chapter6_analysis.py
```

Reads `evaluation_results/metrics_report.json`, `evaluation_results/rubric_scores.jsonl`, and `evaluation_results/eval_results.jsonl`. Writes to `evaluation_results/chapter6/`:

- **7 figures** (PNG at 300 DPI + SVG each): per-intent accuracy, escalation precision/recall by reason, rubric score distributions, rubric scores by handling outcome, response latency, intent confusion matrix, boundary-pair transition rates.
- **`tables.md`** — seven Markdown tables (eight-metric summary, per-intent accuracy, escalation per reason, tool-call correctness by tier, rubric statistics, rubric by handling, hallucination catalogue). Paste directly into Word.
- **`chapter6_synthesis.md`** — structured analytical scaffold for Chapter 6 prose drafting. Organises findings by RQ (RQ1–RQ4) with embedded evidence pointers to figures and tables. All numerical values are computed from the actual data at runtime.

Override input/output paths with `--metrics-input`, `--rubric-input`, `--eval-input`, `--output-dir`.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── data/
│   ├── knowledge_base.csv
│   ├── customers.csv
│   ├── orders.csv
│   └── test_queries.csv
├── scripts/
│   ├── run_evaluation.py         # evaluation harness
│   ├── compute_metrics.py        # §4.4 quantitative metrics
│   ├── rubric_scorer.py          # §4.3.3 Streamlit rubric scoring UI
│   └── build_chapter6_analysis.py  # Chapter 6 figures, tables, synthesis
├── evaluation_results/     # JSONL output + metrics reports + rubric scores
│   └── chapter6/           # figures, tables.md, chapter6_synthesis.md (Slice 10 output)
├── src/
│   ├── intents.py          # twelve-intent taxonomy (enum + metadata)
│   ├── data_loaders.py     # typed CSV loaders
│   ├── llm_client.py       # Gemini 3.1 Flash Lite client with cache + retry
│   ├── nlu.py              # LLM-based intent classifier
│   ├── retrieval.py        # dense-embedding retrieval (all-MiniLM-L6-v2)
│   ├── tools.py            # transactional tool layer (order lookup, refund, etc.)
│   ├── escalation.py       # three-trigger escalation pipeline (VADER + thresholds)
│   ├── grounding.py        # grounded response generation with citation extraction
│   └── agent.py            # orchestrator sequencing all pipeline stages
└── tests/
    ├── test_data_loaders.py
    ├── test_llm_client.py
    ├── test_nlu.py
    ├── test_retrieval.py
    ├── test_grounding.py
    ├── test_tools.py
    ├── test_escalation.py
    ├── test_run_evaluation.py
    ├── test_compute_metrics.py
    ├── test_rubric_scorer.py
    └── test_build_chapter6_analysis.py
```
