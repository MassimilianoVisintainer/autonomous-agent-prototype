"""Natural language understanding layer — §3.1.1 of the thesis.

In Slice 1 this module contains a deterministic keyword-based intent classifier.
It is intentionally weak: every match is a plain substring check against the
lowercased query, no stemming, no fuzzy logic, no regex backtracking.
The low confidence ceiling (0.6) signals to the rest of the system that this
classifier is a placeholder. Slice 2 replaces it with a Gemini-powered classifier.
"""

from dataclasses import dataclass

from src.intents import Intent

@dataclass(frozen=True)
class ClassificationResult:
    intent: Intent
    confidence: float
    matched_trigger: str | None
    method: str


INTENT_TRIGGERS: dict[Intent, list[str]] = {
    Intent.ORDER_STATUS: [
        "where is my order",
        "where's my order",
        "track my order",
        "order status",
        "has my order shipped",
        "has my package shipped",
        "when will my order arrive",
        "where is my package",
        "tracking info",
        "tracking number",
    ],
    Intent.ORDER_MODIFY: [
        "change my order",
        "change the address",
        "update the address",
        "update my order",
        "modify my order",
        "wrong address",
        "change the delivery address",
        "add an item to",
        "change the quantity",
        "update the shipping address",
    ],
    Intent.ORDER_CANCEL: [
        "cancel my order",
        "cancel order",
        "i want to cancel",
        "can i cancel",
        "please cancel",
        "stop my order",
    ],
    Intent.REFUND_REQUEST: [
        "refund",
        "money back",
        "get my money back",
        "issue a refund",
        "request a refund",
        "when will i be refunded",
    ],
    Intent.PRODUCT_INFO: [
        "specs",
        "specifications",
        "compatible with",
        "does it work with",
        "product details",
        "tell me about the",
        "what are the features",
        "battery life",
        "dimensions of",
        "does the",
    ],
    Intent.RETURN_POLICY: [
        "return policy",
        "return window",
        "how long do i have to return",
        "can i return",
        "return an item",
        "return procedure",
        "how do returns work",
    ],
    Intent.SHIPPING_INFO: [
        "shipping cost",
        "how long does shipping",
        "delivery time",
        "shipping time",
        "express delivery",
        "free shipping",
        "do you ship to",
        "shipping options",
        "how long will delivery",
    ],
    Intent.ACCOUNT_HELP: [
        "can't log in",
        "cannot log in",
        "reset my password",
        "forgot my password",
        "account suspended",
        "account locked",
        "login problem",
        "change my email",
        "delete my account",
        "two-factor",
        "2fa",
    ],
    Intent.COMPLAINT: [
        "very disappointed",
        "absolutely unacceptable",
        "terrible service",
        "completely wrong",
        "this is ridiculous",
        "i am furious",
        "i am angry",
        "worst experience",
        "disgraceful",
        "appalling",
    ],
    Intent.MULTI_ISSUE_DISPUTE: [
        "several problems",
        "multiple issues",
        "wrong item",
        "billing error",
        "everything has gone wrong",
        "missing item",
        "still no response",
        "charged twice",
    ],
    Intent.OUT_OF_SCOPE: [
        "weather",
        "recipe",
        "tell me a joke",
        "what time is it",
        "capital of",
        "who won",
        "football",
        "write me a poem",
        "what is the meaning",
        "stock price",
    ],
    # AMBIGUOUS_QUERY has no triggers — it is always the fallback
}


def classify(query: str) -> ClassificationResult:
    """Classify a customer query into one of the twelve intents using keyword matching."""
    normalised = query.strip().lower()

    if not normalised:
        return ClassificationResult(
            intent=Intent.AMBIGUOUS_QUERY,
            confidence=1.0,
            matched_trigger=None,
            method="keyword",
        )

    for intent, triggers in INTENT_TRIGGERS.items():
        for trigger in triggers:
            if trigger in normalised:
                return ClassificationResult(
                    intent=intent,
                    confidence=0.6,
                    matched_trigger=trigger,
                    method="keyword",
                )

    return ClassificationResult(
        intent=Intent.AMBIGUOUS_QUERY,
        confidence=0.3,
        matched_trigger=None,
        method="keyword",
    )
