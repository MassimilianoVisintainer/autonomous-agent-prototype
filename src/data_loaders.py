"""Data layer boundary for the autonomous customer-support agent prototype.

All modules that need customer, order, knowledge-base, or test-query data must
import from this module rather than reading CSVs directly. Each loader returns a
list of frozen dataclass instances with typed fields.
"""

import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass(frozen=True)
class KnowledgeBaseChunk:
    kb_id: str
    category: str
    topic: str
    description: str
    applies_to_products: list[str]
    requires_authority_check: bool


@dataclass(frozen=True)
class Customer:
    customer_id: str
    name: str
    email: str
    tier: str
    account_status: str
    registration_date: datetime.date
    preferred_language: str
    total_lifetime_value: float
    notes_flag: str


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    order_date: datetime.datetime
    status: str
    total_amount: float
    currency: str
    items_summary: str
    item_count: int
    shipping_country: str
    tracking_number: str | None
    estimated_delivery: datetime.date | None
    actual_delivery: datetime.date | None


@dataclass(frozen=True)
class TestQuery:
    query_id: str
    query_text: str
    intent_label: str
    test_category: str
    expected_handling: str
    expected_escalation_reason: str | None
    expected_tool_calls: str
    expected_kb_chunks: str
    emotional_intensity_band: str
    threshold_test_flag: str | None
    linked_query_id: str | None


def _none_if_blank(value: str) -> str | None:
    """Return None for empty/whitespace strings, otherwise the stripped string."""
    stripped = str(value).strip()
    return None if stripped == "" or stripped == "nan" else stripped


def load_knowledge_base(path: str | None = None) -> list[KnowledgeBaseChunk]:
    """Load knowledge_base_outline.csv and return typed KnowledgeBaseChunk instances."""
    csv_path = path if path is not None else str(DATA_DIR / "knowledge_base_outline.csv")
    df = pd.read_csv(csv_path, dtype=str)
    chunks = []
    for _, row in df.iterrows():
        raw_products = str(row["applies_to_products"]).strip()
        if raw_products == "" or raw_products == "nan":
            products: list[str] = []
        else:
            products = [p.strip() for p in raw_products.split(",") if p.strip()]
        chunks.append(
            KnowledgeBaseChunk(
                kb_id=str(row["kb_id"]).strip(),
                category=str(row["category"]).strip(),
                topic=str(row["topic"]).strip(),
                description=str(row["description"]).strip(),
                applies_to_products=products,
                requires_authority_check=str(row["requires_authority_check"]).strip().lower() == "true",
            )
        )
    return chunks


def load_customers(path: str | None = None) -> list[Customer]:
    """Load customers.csv and return typed Customer instances."""
    csv_path = path if path is not None else str(DATA_DIR / "customers.csv")
    df = pd.read_csv(csv_path, dtype=str)
    customers = []
    for _, row in df.iterrows():
        customers.append(
            Customer(
                customer_id=str(row["customer_id"]).strip(),
                name=str(row["name"]).strip(),
                email=str(row["email"]).strip(),
                tier=str(row["tier"]).strip(),
                account_status=str(row["account_status"]).strip(),
                registration_date=datetime.date.fromisoformat(str(row["registration_date"]).strip()),
                preferred_language=str(row["preferred_language"]).strip(),
                total_lifetime_value=float(row["total_lifetime_value"]),
                notes_flag=str(row["notes_flag"]).strip(),
            )
        )
    return customers


def load_orders(path: str | None = None) -> list[Order]:
    """Load orders.csv and return typed Order instances."""
    csv_path = path if path is not None else str(DATA_DIR / "orders.csv")
    df = pd.read_csv(csv_path, dtype=str)
    orders = []
    for _, row in df.iterrows():
        raw_tracking = _none_if_blank(row["tracking_number"])
        raw_est = _none_if_blank(row["estimated_delivery"])
        raw_actual = _none_if_blank(row["actual_delivery"])
        orders.append(
            Order(
                order_id=str(row["order_id"]).strip(),
                customer_id=str(row["customer_id"]).strip(),
                order_date=datetime.datetime.fromisoformat(
                    str(row["order_date"]).strip().rstrip("Z")
                ).replace(tzinfo=datetime.timezone.utc),
                status=str(row["status"]).strip(),
                total_amount=float(row["total_amount"]),
                currency=str(row["currency"]).strip(),
                items_summary=str(row["items_summary"]).strip(),
                item_count=int(row["item_count"]),
                shipping_country=str(row["shipping_country"]).strip(),
                tracking_number=raw_tracking,
                estimated_delivery=datetime.date.fromisoformat(raw_est) if raw_est else None,
                actual_delivery=datetime.date.fromisoformat(raw_actual) if raw_actual else None,
            )
        )
    return orders


def load_test_queries(path: str | None = None) -> list[TestQuery]:
    """Load test_queries.csv and return typed TestQuery instances."""
    csv_path = path if path is not None else str(DATA_DIR / "test_queries.csv")
    df = pd.read_csv(csv_path, dtype=str)
    queries = []
    for _, row in df.iterrows():
        queries.append(
            TestQuery(
                query_id=str(row["query_id"]).strip(),
                query_text=str(row["query_text"]).strip(),
                intent_label=str(row["intent_label"]).strip(),
                test_category=str(row["test_category"]).strip(),
                expected_handling=str(row["expected_handling"]).strip(),
                expected_escalation_reason=_none_if_blank(row["expected_escalation_reason"]),
                expected_tool_calls=str(row["expected_tool_calls"]).strip(),
                expected_kb_chunks=str(row["expected_kb_chunks"]).strip(),
                emotional_intensity_band=str(row["emotional_intensity_band"]).strip(),
                threshold_test_flag=_none_if_blank(row["threshold_test_flag"]),
                linked_query_id=_none_if_blank(row["linked_query_id"]),
            )
        )
    return queries
