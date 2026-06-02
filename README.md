# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 0 complete** — repository scaffolding and data-loading layer. No agent functionality yet.

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

All four tests should pass, verifying that each CSV loads with the expected row count.

### Streamlit app

```bash
streamlit run app.py
```

Opens a browser page showing the counts of loaded data (65 KB chunks, 18 customers, 50 orders, 130 test queries) and a list of the twelve intent categories.

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
│   ├── nlu.py              # placeholder — slice 2
│   ├── retrieval.py        # placeholder — slice 3
│   ├── tools.py            # placeholder — slice 4
│   ├── escalation.py       # placeholder — slice 5
│   ├── grounding.py        # placeholder — slice 6
│   └── agent.py            # placeholder — slice 7
└── tests/
    ├── __init__.py
    └── test_data_loaders.py
```
