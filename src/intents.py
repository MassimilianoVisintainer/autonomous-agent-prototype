"""Intent taxonomy for the autonomous customer-support agent prototype.

This module is the single source of truth for the twelve-intent classification
scheme used throughout the system. All other modules that need intent information
must import from here.
"""

import enum
from dataclasses import dataclass, field


class Intent(enum.Enum):
    ORDER_STATUS = "order_status"
    ORDER_MODIFY = "order_modify"
    ORDER_CANCEL = "order_cancel"
    REFUND_REQUEST = "refund_request"
    PRODUCT_INFO = "product_info"
    RETURN_POLICY = "return_policy"
    SHIPPING_INFO = "shipping_info"
    ACCOUNT_HELP = "account_help"
    COMPLAINT = "complaint"
    MULTI_ISSUE_DISPUTE = "multi_issue_dispute"
    OUT_OF_SCOPE = "out_of_scope"
    AMBIGUOUS_QUERY = "ambiguous_query"


class Cluster(enum.Enum):
    TRANSACTIONAL = "transactional"
    INFORMATIONAL = "informational"
    AFFECTIVE = "affective"
    BOUNDARY = "boundary"


@dataclass(frozen=True)
class IntentMetadata:
    intent: Intent
    cluster: Cluster
    description: str
    example_utterances: list[str]
    dimensions: list[str]


INTENT_METADATA: dict[Intent, IntentMetadata] = {
    Intent.ORDER_STATUS: IntentMetadata(
        intent=Intent.ORDER_STATUS,
        cluster=Cluster.TRANSACTIONAL,
        description="Customer asks for the current status or location of a specific order.",
        example_utterances=[
            "Where is my order ORD-1024?",
            "Has my package shipped yet?",
            "When will order ORD-1007 arrive?",
        ],
        dimensions=["resolution_quality"],
    ),
    Intent.ORDER_MODIFY: IntentMetadata(
        intent=Intent.ORDER_MODIFY,
        cluster=Cluster.TRANSACTIONAL,
        description="Customer requests a change to an existing order (address, quantity, items).",
        example_utterances=[
            "Can you change the delivery address for ORD-1029?",
            "I need to update the quantity on my order.",
            "Add an item to ORD-1015 please.",
        ],
        dimensions=["resolution_quality", "escalation_behaviour"],
    ),
    Intent.ORDER_CANCEL: IntentMetadata(
        intent=Intent.ORDER_CANCEL,
        cluster=Cluster.TRANSACTIONAL,
        description="Customer requests cancellation of an order before or after dispatch.",
        example_utterances=[
            "I want to cancel order ORD-1029.",
            "Please cancel my order, I changed my mind.",
            "Can I still cancel ORD-1015?",
        ],
        dimensions=["resolution_quality", "escalation_behaviour"],
    ),
    Intent.REFUND_REQUEST: IntentMetadata(
        intent=Intent.REFUND_REQUEST,
        cluster=Cluster.TRANSACTIONAL,
        description="Customer requests a refund for a completed order or returned item.",
        example_utterances=[
            "I'd like a refund for order ORD-1034.",
            "When will I get my money back?",
            "Please process a refund for my return.",
        ],
        dimensions=["resolution_quality", "escalation_behaviour"],
    ),
    Intent.PRODUCT_INFO: IntentMetadata(
        intent=Intent.PRODUCT_INFO,
        cluster=Cluster.INFORMATIONAL,
        description="Customer asks for specifications, compatibility, or details about a product.",
        example_utterances=[
            "What are the specs for the Bluetooth speaker?",
            "Is the USB-C cable compatible with my laptop?",
            "Does the webcam work on Linux?",
        ],
        dimensions=["resolution_quality"],
    ),
    Intent.RETURN_POLICY: IntentMetadata(
        intent=Intent.RETURN_POLICY,
        cluster=Cluster.INFORMATIONAL,
        description="Customer asks about the return window, conditions, or procedures.",
        example_utterances=[
            "What is your return policy?",
            "How long do I have to return an item?",
            "Can I return a opened product?",
        ],
        dimensions=["resolution_quality"],
    ),
    Intent.SHIPPING_INFO: IntentMetadata(
        intent=Intent.SHIPPING_INFO,
        cluster=Cluster.INFORMATIONAL,
        description="Customer asks about shipping options, timelines, costs, or carriers.",
        example_utterances=[
            "How long does shipping to Germany take?",
            "Do you offer express delivery?",
            "What are the shipping costs to the US?",
        ],
        dimensions=["resolution_quality"],
    ),
    Intent.ACCOUNT_HELP: IntentMetadata(
        intent=Intent.ACCOUNT_HELP,
        cluster=Cluster.INFORMATIONAL,
        description="Customer needs help with their account: login, password, profile, or suspension.",
        example_utterances=[
            "I can't log in to my account.",
            "How do I reset my password?",
            "My account has been suspended, can you help?",
        ],
        dimensions=["resolution_quality", "escalation_behaviour"],
    ),
    Intent.COMPLAINT: IntentMetadata(
        intent=Intent.COMPLAINT,
        cluster=Cluster.AFFECTIVE,
        description="Customer expresses dissatisfaction or frustration with a product or service experience.",
        example_utterances=[
            "This is absolutely unacceptable, my order arrived damaged!",
            "I'm very disappointed with the service I received.",
            "Your customer support has been terrible.",
        ],
        dimensions=["emotional_intelligence", "escalation_behaviour"],
    ),
    Intent.MULTI_ISSUE_DISPUTE: IntentMetadata(
        intent=Intent.MULTI_ISSUE_DISPUTE,
        cluster=Cluster.AFFECTIVE,
        description="Customer raises multiple interconnected grievances in a single interaction.",
        example_utterances=[
            "My order is late, the item is wrong, and I can't even get a refund!",
            "I have several problems: a missing item, a billing error, and no response to my emails.",
            "Everything has gone wrong with this order from start to finish.",
        ],
        dimensions=["emotional_intelligence", "escalation_behaviour", "resolution_quality"],
    ),
    Intent.OUT_OF_SCOPE: IntentMetadata(
        intent=Intent.OUT_OF_SCOPE,
        cluster=Cluster.BOUNDARY,
        description="Customer message falls outside the agent's domain (e.g. general knowledge, unrelated services).",
        example_utterances=[
            "What's the weather like in Lisbon today?",
            "Can you write me a poem?",
            "Who won the football match last night?",
        ],
        dimensions=["escalation_behaviour"],
    ),
    Intent.AMBIGUOUS_QUERY: IntentMetadata(
        intent=Intent.AMBIGUOUS_QUERY,
        cluster=Cluster.BOUNDARY,
        description="Customer message lacks sufficient information to determine intent or act without clarification.",
        example_utterances=[
            "When will my order arrive?",
            "I have a problem.",
            "Can you help me with something?",
        ],
        dimensions=["resolution_quality", "escalation_behaviour"],
    ),
}


def get_intent_metadata(intent: Intent) -> IntentMetadata:
    """Return the metadata record for the given Intent."""
    return INTENT_METADATA[intent]


def intent_from_label(label: str) -> Intent:
    """Convert an intent label string (as in test_queries.csv) to an Intent enum member."""
    try:
        return Intent(label)
    except ValueError:
        raise ValueError(f"Unknown intent label: {label!r}")
