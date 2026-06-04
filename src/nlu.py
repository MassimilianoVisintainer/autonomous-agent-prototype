"""Natural language understanding layer — §3.1.1 of the thesis.

Uses Gemini 1.5 Flash (via src/llm_client.py) to classify customer queries into
one of the twelve intents defined in src/intents.py. The model is prompted to
respond with a JSON object containing the intent code, a confidence score, and a
short reasoning sentence.

On any failure (missing API key, network error, malformed response) the
classifier degrades gracefully to AMBIGUOUS_QUERY rather than raising.
"""

import json
from dataclasses import dataclass

from src.intents import INTENT_METADATA, Intent
from src import llm_client
from src.llm_client import LLMClientError


@dataclass(frozen=True)
class ClassificationResult:
    intent: Intent
    confidence: float
    reasoning: str | None
    method: str


def _build_system_prompt() -> str:
    lines = [
        "You are an intent classifier for an e-commerce customer-support agent.",
        "Classify the customer query into exactly one of the following intent codes:",
        "",
    ]
    for intent, meta in INTENT_METADATA.items():
        lines.append(f"  {intent.value}: {meta.description}")
    lines += [
        "",
        "Respond ONLY with a JSON object containing exactly three keys:",
        '  "intent"     — one of the intent codes listed above (lowercase string)',
        '  "confidence" — a number between 0.0 and 1.0',
        '  "reasoning"  — one sentence explaining your choice',
        "",
        "Do not include any text outside the JSON object.",
        'If the query is empty or genuinely ambiguous, choose "ambiguous_query" and explain why.',
    ]
    return "\n".join(lines)


CLASSIFICATION_SYSTEM_PROMPT: str = _build_system_prompt()

_FALLBACK = ClassificationResult(
    intent=Intent.AMBIGUOUS_QUERY,
    confidence=0.0,
    reasoning="LLM call failed",
    method="llm_error",
)


def classify(query: str) -> ClassificationResult:
    """Classify a customer query into one of the twelve intents using Gemini."""
    if not query.strip():
        return ClassificationResult(
            intent=Intent.AMBIGUOUS_QUERY,
            confidence=1.0,
            reasoning=None,
            method="empty_input",
        )

    try:
        raw = llm_client.complete(
            system=CLASSIFICATION_SYSTEM_PROMPT,
            user=query,
            json_mode=True,
        )
    except LLMClientError:
        return _FALLBACK

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return ClassificationResult(
            intent=Intent.AMBIGUOUS_QUERY,
            confidence=0.0,
            reasoning="Failed to parse LLM response",
            method="llm_error",
        )

    try:
        intent = Intent(parsed["intent"])
    except (KeyError, ValueError):
        return ClassificationResult(
            intent=Intent.AMBIGUOUS_QUERY,
            confidence=0.0,
            reasoning="LLM returned unknown intent code",
            method="llm_error",
        )

    return ClassificationResult(
        intent=intent,
        confidence=float(parsed.get("confidence", 0.0)),
        reasoning=parsed.get("reasoning"),
        method="llm",
    )
