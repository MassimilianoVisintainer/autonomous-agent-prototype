"""Escalation pipeline — §3.2 of the thesis.

Implements the Amri et al. (2025) three-trigger framework for deciding when to
hand an interaction off to a human agent. The three triggers are evaluated
independently for each interaction — multiple may fire simultaneously and all
firing triggers are recorded.

Triggers:
  exceeded_authority  — tool returned exceeded_authority or out_of_window status
  high_emotion        — VADER compound sentiment score below EMOTION_FLOOR (-0.5)
  low_confidence      — classification confidence below CONFIDENCE_FLOOR (0.75),
                        excluding boundary intents that have their own paths

Emotion detection uses VADER (Valence Aware Dictionary and sEntiment Reasoner),
a lexicon-based rule-based analyser well-suited to short informal customer text.
"""

from __future__ import annotations

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from src.intents import Intent
from src.nlu import ClassificationResult
from src.tools import ToolResult

CONFIDENCE_FLOOR = 0.75
EMOTION_FLOOR = -0.5

ESCALATION_TRIGGER_AUTHORITY = "exceeded_authority"
ESCALATION_TRIGGER_EMOTION = "high_emotion"
ESCALATION_TRIGGER_CONFIDENCE = "low_confidence"

BOUNDARY_INTENTS = frozenset({Intent.AMBIGUOUS_QUERY, Intent.OUT_OF_SCOPE})
AUTHORITY_BREACH_STATUSES = frozenset({"exceeded_authority", "out_of_window"})

_analyzer = SentimentIntensityAnalyzer()


@dataclass(frozen=True)
class EscalationResult:
    should_escalate: bool
    triggers: list[str]
    emotion_score: float
    reason: str


def detect_emotion(text: str) -> float:
    """Return the VADER compound sentiment score for text (-1.0 to 1.0)."""
    if not text.strip():
        return 0.0
    return _analyzer.polarity_scores(text)["compound"]


def evaluate_escalation(
    classification: ClassificationResult,
    tool_result: ToolResult | None,
    query: str,
) -> EscalationResult:
    """Evaluate all three escalation triggers and return the combined result."""
    emotion_score = detect_emotion(query)
    triggers: list[str] = []

    # Trigger 1 — authority breach
    if tool_result is not None and tool_result.status in AUTHORITY_BREACH_STATUSES:
        triggers.append(ESCALATION_TRIGGER_AUTHORITY)

    # Trigger 2 — high emotion / customer distress
    if emotion_score < EMOTION_FLOOR:
        triggers.append(ESCALATION_TRIGGER_EMOTION)

    # Trigger 3 — low classification confidence (boundary intents excluded)
    if (
        classification.confidence < CONFIDENCE_FLOOR
        and classification.intent not in BOUNDARY_INTENTS
    ):
        triggers.append(ESCALATION_TRIGGER_CONFIDENCE)

    if not triggers:
        reason = "All escalation triggers cleared; agent handles autonomously."
    else:
        parts = []
        if ESCALATION_TRIGGER_AUTHORITY in triggers:
            status = tool_result.status if tool_result else "unknown"
            parts.append(f"{ESCALATION_TRIGGER_AUTHORITY} (tool status: {status})")
        if ESCALATION_TRIGGER_EMOTION in triggers:
            parts.append(f"{ESCALATION_TRIGGER_EMOTION} (compound {emotion_score:.2f})")
        if ESCALATION_TRIGGER_CONFIDENCE in triggers:
            parts.append(
                f"{ESCALATION_TRIGGER_CONFIDENCE} "
                f"(confidence {classification.confidence:.2f} < {CONFIDENCE_FLOOR})"
            )
        reason = "Escalation triggered: " + "; ".join(parts)

    return EscalationResult(
        should_escalate=len(triggers) > 0,
        triggers=triggers,
        emotion_score=emotion_score,
        reason=reason,
    )
