"""Tool tier for transactional intents — §3.1.3 of the thesis.

All four tool functions are read-only at the persistence level: they read from
the orders and customers CSV files and return structured ToolResult objects but
never write to disk. This honours the §4.3.2 commitment that order state does
not advance during the evaluation run.

Authority gating thresholds match §E of the Prototype Data Specification:
  - Refund authority limit: €100 (REFUND_AUTHORITY_LIMIT)
  - Cancellation window: 24 hours from order placement (CANCELLATION_WINDOW)
  - Modifiable statuses: placed, processing (MODIFIABLE_STATUSES)

The cancellation window check uses a fixed REFERENCE_NOW timestamp of
2026-05-15 14:00:00 UTC, matching the orders.csv design, to keep evaluation
reproducible regardless of when the code is actually run.
"""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from src.intents import Intent
from src.nlu import ClassificationResult
from src.data_loaders import Customer, Order, load_customers, load_orders

ORDER_ID_PATTERN = re.compile(r"\bORD-\d{4}\b")
REFERENCE_NOW = datetime(2026, 5, 15, 14, 0, 0, tzinfo=timezone.utc)
CANCELLATION_WINDOW = timedelta(hours=24)
REFUND_AUTHORITY_LIMIT = 100.0
MODIFIABLE_STATUSES = frozenset({"placed", "processing"})

_orders_by_id: dict[str, Order] | None = None
_customers_by_id: dict[str, Customer] | None = None


@dataclass(frozen=True)
class ToolResult:
    status: str
    tool: str
    data: dict
    reason: str | None


def _get_orders_index() -> dict[str, Order]:
    global _orders_by_id
    if _orders_by_id is None:
        _orders_by_id = {o.order_id: o for o in load_orders()}
    return _orders_by_id


def _get_customers_index() -> dict[str, Customer]:
    global _customers_by_id
    if _customers_by_id is None:
        _customers_by_id = {c.customer_id: c for c in load_customers()}
    return _customers_by_id


def extract_order_id(text: str) -> str | None:
    """Return the first ORD-NNNN pattern found in text, or None."""
    match = ORDER_ID_PATTERN.search(text)
    return match.group() if match else None


def _order_data(order: Order) -> dict:
    return {
        "order_id": order.order_id,
        "customer_id": order.customer_id,
        "status": order.status,
        "total_amount": order.total_amount,
        "currency": order.currency,
        "items_summary": order.items_summary,
        "shipping_country": order.shipping_country,
        "tracking_number": order.tracking_number,
        "estimated_delivery": str(order.estimated_delivery) if order.estimated_delivery else None,
        "actual_delivery": str(order.actual_delivery) if order.actual_delivery else None,
        "order_date": order.order_date.isoformat(),
    }


def lookup_order(order_id: str) -> ToolResult:
    """Return full order details for order_id."""
    orders = _get_orders_index()
    if order_id not in orders:
        return ToolResult(
            status="not_found",
            tool="lookup_order",
            data={"order_id": order_id},
            reason="No order with this identifier exists in the system.",
        )
    return ToolResult(
        status="ok",
        tool="lookup_order",
        data=_order_data(orders[order_id]),
        reason=None,
    )


def modify_order(order_id: str, reference_now: datetime | None = None) -> ToolResult:
    """Check whether the order can be modified and return the result."""
    orders = _get_orders_index()
    if order_id not in orders:
        return ToolResult(
            status="not_found",
            tool="modify_order",
            data={"order_id": order_id},
            reason="No order with this identifier exists in the system.",
        )
    order = orders[order_id]
    if order.status not in MODIFIABLE_STATUSES:
        return ToolResult(
            status="out_of_window",
            tool="modify_order",
            data={"order_id": order_id, "current_status": order.status},
            reason="The order has already been dispatched and cannot be modified.",
        )
    return ToolResult(
        status="ok",
        tool="modify_order",
        data={"order_id": order_id, "current_status": order.status},
        reason="The order is in a modifiable state. The customer should be asked what specifically they wish to change.",
    )


def cancel_order(order_id: str, reference_now: datetime | None = None) -> ToolResult:
    """Check whether the order can be cancelled within the 24-hour window."""
    now = reference_now if reference_now is not None else REFERENCE_NOW
    orders = _get_orders_index()
    if order_id not in orders:
        return ToolResult(
            status="not_found",
            tool="cancel_order",
            data={"order_id": order_id},
            reason="No order with this identifier exists in the system.",
        )
    order = orders[order_id]
    if order.status not in MODIFIABLE_STATUSES:
        return ToolResult(
            status="out_of_window",
            tool="cancel_order",
            data={"order_id": order_id, "current_status": order.status},
            reason="The order has already been dispatched and cannot be cancelled by the agent.",
        )
    elapsed = now - order.order_date
    if elapsed > CANCELLATION_WINDOW:
        hours = elapsed.total_seconds() / 3600
        return ToolResult(
            status="out_of_window",
            tool="cancel_order",
            data={"order_id": order_id, "hours_since_placement": round(hours, 1)},
            reason="The 24-hour cancellation window has elapsed. Cancellation requires human agent review.",
        )
    return ToolResult(
        status="ok",
        tool="cancel_order",
        data={"order_id": order_id, "total_amount": order.total_amount, "currency": order.currency},
        reason=None,
    )


def process_refund(order_id: str) -> ToolResult:
    """Check refund authority and return the result."""
    orders = _get_orders_index()
    if order_id not in orders:
        return ToolResult(
            status="not_found",
            tool="process_refund",
            data={"order_id": order_id},
            reason="No order with this identifier exists in the system.",
        )
    order = orders[order_id]
    if order.total_amount > REFUND_AUTHORITY_LIMIT:
        return ToolResult(
            status="exceeded_authority",
            tool="process_refund",
            data={
                "order_id": order_id,
                "total_amount": order.total_amount,
                "currency": order.currency,
                "authority_limit": REFUND_AUTHORITY_LIMIT,
            },
            reason="The refund amount exceeds the agent's autonomous authority limit. Senior authorization is required.",
        )
    return ToolResult(
        status="ok",
        tool="process_refund",
        data={"order_id": order_id, "total_amount": order.total_amount, "currency": order.currency},
        reason=None,
    )


_TOOL_NAME_MAP = {
    Intent.ORDER_STATUS: "lookup_order",
    Intent.ORDER_MODIFY: "modify_order",
    Intent.ORDER_CANCEL: "cancel_order",
    Intent.REFUND_REQUEST: "process_refund",
}

_TRANSACTIONAL_INTENTS = frozenset(_TOOL_NAME_MAP)


def dispatch(
    classification: ClassificationResult,
    query: str,
    reference_now: datetime | None = None,
) -> ToolResult | None:
    """Route transactional intents to the appropriate tool. Returns None for others."""
    if classification.intent not in _TRANSACTIONAL_INTENTS:
        return None

    tool_name = _TOOL_NAME_MAP[classification.intent]
    order_id = extract_order_id(query)
    if order_id is None:
        return ToolResult(
            status="missing_identifier",
            tool=tool_name,
            data={},
            reason="The customer's query does not include an order identifier. Ask the customer to provide the order number.",
        )

    if classification.intent == Intent.ORDER_STATUS:
        return lookup_order(order_id)
    if classification.intent == Intent.ORDER_MODIFY:
        return modify_order(order_id, reference_now)
    if classification.intent == Intent.ORDER_CANCEL:
        return cancel_order(order_id, reference_now)
    return process_refund(order_id)
