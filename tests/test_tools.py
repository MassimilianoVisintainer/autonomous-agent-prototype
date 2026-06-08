"""Tests for src/tools.py.

All tests use synthetic Order/Customer data injected via monkeypatch.
No real CSV files are read and no real API calls are made.
"""

import datetime
from unittest.mock import patch

import pytest

from src import tools
from src.data_loaders import Customer, Order
from src.intents import Intent
from src.nlu import ClassificationResult
from src.tools import (
    REFERENCE_NOW,
    ToolResult,
    cancel_order,
    dispatch,
    extract_order_id,
    lookup_order,
    modify_order,
    process_refund,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_order(
    order_id: str = "ORD-1024",
    status: str = "placed",
    total_amount: float = 89.99,
    hours_before_now: float = 1.0,
) -> Order:
    order_date = REFERENCE_NOW - datetime.timedelta(hours=hours_before_now)
    return Order(
        order_id=order_id,
        customer_id="CUS-0001",
        order_date=order_date,
        status=status,
        total_amount=total_amount,
        currency="EUR",
        items_summary="1x Bluetooth Speaker",
        item_count=1,
        shipping_country="PT",
        tracking_number=None,
        estimated_delivery=None,
        actual_delivery=None,
    )


def _classification(intent: Intent) -> ClassificationResult:
    return ClassificationResult(
        intent=intent, confidence=0.9, reasoning="test", method="llm"
    )


@pytest.fixture(autouse=True)
def reset_caches():
    """Reset module-level caches before each test."""
    tools._orders_by_id = None
    tools._customers_by_id = None
    yield
    tools._orders_by_id = None
    tools._customers_by_id = None


def _patch_orders(monkeypatch, orders: list[Order]) -> None:
    monkeypatch.setattr(tools, "_orders_by_id", {o.order_id: o for o in orders})


# ---------------------------------------------------------------------------
# extract_order_id
# ---------------------------------------------------------------------------

def test_extract_order_id_finds_simple_match():
    assert extract_order_id("Where is my order ORD-1024?") == "ORD-1024"


def test_extract_order_id_returns_none_for_no_match():
    assert extract_order_id("Where is my order?") is None


def test_extract_order_id_finds_first_when_multiple():
    result = extract_order_id("Cancel ORD-1024 and ORD-1025")
    assert result == "ORD-1024"


# ---------------------------------------------------------------------------
# lookup_order
# ---------------------------------------------------------------------------

def test_lookup_order_returns_ok_for_existing_order(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024")])
    result = lookup_order("ORD-1024")
    assert result.status == "ok"
    assert result.tool == "lookup_order"
    assert result.data["order_id"] == "ORD-1024"
    assert "status" in result.data
    assert "total_amount" in result.data


def test_lookup_order_returns_not_found_for_missing_order(monkeypatch):
    _patch_orders(monkeypatch, [])
    result = lookup_order("ORD-9999")
    assert result.status == "not_found"
    assert result.tool == "lookup_order"


# ---------------------------------------------------------------------------
# modify_order
# ---------------------------------------------------------------------------

def test_modify_order_returns_ok_for_placed_status(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", status="placed")])
    result = modify_order("ORD-1024")
    assert result.status == "ok"
    assert result.tool == "modify_order"


def test_modify_order_returns_out_of_window_for_shipped(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", status="shipped")])
    result = modify_order("ORD-1024")
    assert result.status == "out_of_window"
    assert result.tool == "modify_order"


# ---------------------------------------------------------------------------
# cancel_order
# ---------------------------------------------------------------------------

def test_cancel_order_returns_ok_inside_window(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", status="placed", hours_before_now=1.0)])
    result = cancel_order("ORD-1024", reference_now=REFERENCE_NOW)
    assert result.status == "ok"
    assert result.tool == "cancel_order"


def test_cancel_order_returns_out_of_window_after_24h(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", status="placed", hours_before_now=48.0)])
    result = cancel_order("ORD-1024", reference_now=REFERENCE_NOW)
    assert result.status == "out_of_window"
    assert result.tool == "cancel_order"


def test_cancel_order_returns_out_of_window_for_shipped(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", status="shipped", hours_before_now=1.0)])
    result = cancel_order("ORD-1024", reference_now=REFERENCE_NOW)
    assert result.status == "out_of_window"
    assert result.tool == "cancel_order"


# ---------------------------------------------------------------------------
# process_refund
# ---------------------------------------------------------------------------

def test_process_refund_returns_ok_below_threshold(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", total_amount=89.99)])
    result = process_refund("ORD-1024")
    assert result.status == "ok"
    assert result.tool == "process_refund"


def test_process_refund_returns_exceeded_authority_above_threshold(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024", total_amount=145.00)])
    result = process_refund("ORD-1024")
    assert result.status == "exceeded_authority"
    assert result.tool == "process_refund"
    assert result.data["authority_limit"] == 100.0


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def test_dispatch_returns_none_for_non_transactional_intent(monkeypatch):
    result = dispatch(_classification(Intent.PRODUCT_INFO), "What are the specs?")
    assert result is None


def test_dispatch_returns_missing_identifier_when_no_order_id(monkeypatch):
    result = dispatch(_classification(Intent.ORDER_STATUS), "Where is my order?")
    assert result is not None
    assert result.status == "missing_identifier"
    assert result.tool == "lookup_order"


def test_dispatch_routes_order_status_to_lookup_order(monkeypatch):
    _patch_orders(monkeypatch, [_make_order("ORD-1024")])
    with patch.object(tools, "lookup_order", wraps=lookup_order) as mock_lookup:
        result = dispatch(
            _classification(Intent.ORDER_STATUS), "Where is my order ORD-1024?"
        )
    mock_lookup.assert_called_once_with("ORD-1024")
    assert result is not None
    assert result.tool == "lookup_order"
