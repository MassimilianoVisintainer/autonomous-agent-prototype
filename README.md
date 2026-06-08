# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 6 complete** — Escalation pipeline.

The Amri et al. (2025) three-trigger escalation framework is now active: `exceeded_authority` (tool threshold breach), `high_emotion` (VADER compound < −0.5), and `low_confidence` (classification confidence < 0.75, excluding boundary intents). When any trigger fires, the grounding layer produces a warm handoff response instead of an autonomous answer. The architectural skeleton from §3.1.1 through §3.2 of the thesis is complete.

A `GOOGLE_API_KEY` must be set before running the app. Copy `.env.example` to `.env` and fill in your Google AI Studio key.

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

All tests should pass — four data-loader sanity checks from Slice 0 and the per-intent classifier tests from Slice 1.

### Streamlit app

```bash
streamlit run app.py
```

Opens a chat interface. Type any customer-support query to see the classified intent and confidence in the response and sidebar reasoning trace.

## Repository structure

```
.
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── app.py
├── data/
│   ├── knowledge_base_outline.csv
│   ├── customers.csv
│   ├── orders.csv
│   └── test_queries.csv
├── src/
│   ├── __init__.py
│   ├── intents.py          # twelve-intent taxonomy (enum + metadata)
│   ├── data_loaders.py     # typed CSV loaders returning dataclasses
│   ├── nlu.py              # keyword classifier (slice 1); LLM classifier in slice 2
│   ├── retrieval.py        # placeholder — slice 3
│   ├── tools.py            # placeholder — slice 4
│   ├── escalation.py       # placeholder — slice 5
│   ├── grounding.py        # placeholder — slice 6
│   └── agent.py            # orchestrator (slice 1+)
└── tests/
    ├── __init__.py
    ├── test_data_loaders.py
    └── test_nlu.py
```
