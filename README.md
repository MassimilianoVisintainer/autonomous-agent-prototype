# Customer Support Agent — Thesis Prototype

A master's thesis prototype implementing an autonomous AI agent for customer support. The agent is evaluated across three research dimensions: **resolution quality** (accuracy and completeness of answers), **emotional intelligence** (appropriate tone calibration under affective load), and **escalation behaviour** (correct hand-off to human agents). The architecture is implemented from first principles without high-level agent frameworks.

## Status

**Slice 2 complete** — Gemini-powered intent classifier.

The keyword classifier has been replaced with a Gemini 1.5 Flash call. Classifications now include the model's own reasoning sentence visible in the sidebar. A file-based response cache (`.cache/`) avoids redundant API calls during development. Responses degrade gracefully to `ambiguous_query` if the API is unavailable. No response generation yet.

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
