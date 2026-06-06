"""Sanity tests for the data loading layer.

Each test verifies that the corresponding CSV loads with the expected row count
and passes at least one shallow structural check.
"""

import datetime
import re

from src.data_loaders import (
    load_customers,
    load_knowledge_base,
    load_orders,
    load_test_queries,
)


def test_knowledge_base_loads_with_expected_count():
    chunks = load_knowledge_base()
    assert len(chunks) == 65
    # Every kb_id must follow the pattern XX-NNN (two letters, hyphen, three digits)
    pattern = re.compile(r"^[A-Z]{2}-\d{3}$")
    for chunk in chunks:
        assert pattern.match(chunk.kb_id), f"Unexpected kb_id format: {chunk.kb_id}"
    # New schema: source_doc is a non-empty string; last_updated is a date
    first = chunks[0]
    assert isinstance(first.source_doc, str) and len(first.source_doc) > 0
    assert isinstance(first.last_updated, datetime.date)


def test_customers_loads_with_expected_count():
    customers = load_customers()
    assert len(customers) == 18
    # Every customer_id must follow the CUS-NNNN pattern
    pattern = re.compile(r"^CUS-\d{4}$")
    for customer in customers:
        assert pattern.match(customer.customer_id), (
            f"Unexpected customer_id format: {customer.customer_id}"
        )


def test_orders_loads_with_expected_count():
    orders = load_orders()
    assert len(orders) == 50
    # Every order's customer_id must correspond to a real customer (foreign-key check)
    customers = load_customers()
    valid_customer_ids = {c.customer_id for c in customers}
    for order in orders:
        assert order.customer_id in valid_customer_ids, (
            f"Order {order.order_id} references unknown customer {order.customer_id}"
        )


def test_test_queries_loads_with_expected_count():
    queries = load_test_queries()
    assert len(queries) == 130
    # Every query_id must follow the Q-NNN pattern
    pattern = re.compile(r"^Q-\d{3}$")
    for query in queries:
        assert pattern.match(query.query_id), f"Unexpected query_id format: {query.query_id}"
