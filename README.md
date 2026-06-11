# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 8 complete** — Metrics computation.

The full agent pipeline (classification → retrieval → tool dispatch → escalation → generation) is complete. The evaluation harness in `scripts/run_evaluation.py` runs all 130 test queries and writes structured JSON Lines output to `evaluation_results/`. `scripts/compute_metrics.py` computes the eight §4.4 metrics and writes `metrics_report.md` and `metrics_report.json`. Rubric scoring is pending (Slice 9).

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

## Computing metrics

After running the evaluation, compute the §4.4 metrics:

```bash
python scripts/compute_metrics.py --input evaluation_results/eval_results.jsonl
```

Outputs `metrics_report.md` (human-readable, suitable for Chapter 6 quotation) and `metrics_report.json` (machine-readable) in `evaluation_results/`. Use `--output-dir PATH` to override the output directory. The script is re-runnable against any evaluation output JSONL.

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
│   └── run_evaluation.py   # evaluation harness
├── evaluation_results/     # JSONL output files from evaluation runs
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
    └── test_run_evaluation.py
```
