# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 1 complete** — keyword intent classifier and chat interface.

The chat interface is live. A deterministic keyword classifier maps user queries to one of the twelve intents. Confidence is capped at 0.6 to reflect that this is a weak placeholder; Slice 2 will replace it with a Gemini-powered classifier. No response generation yet.

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
