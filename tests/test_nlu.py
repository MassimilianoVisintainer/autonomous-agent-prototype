"""Tests for the keyword-based intent classifier in src/nlu.py."""

from src.intents import Intent
from src.nlu import classify


def test_classify_order_status_canonical_query():
    result = classify("Where is my order ORD-1024?")
    assert result.intent == Intent.ORDER_STATUS


def test_classify_order_modify_canonical_query():
    result = classify("Can you change the delivery address for ORD-1029?")
    assert result.intent == Intent.ORDER_MODIFY


def test_classify_order_cancel_canonical_query():
    result = classify("I want to cancel order ORD-1029.")
    assert result.intent == Intent.ORDER_CANCEL


def test_classify_refund_request_canonical_query():
    result = classify("I'd like a refund for order ORD-1034.")
    assert result.intent == Intent.REFUND_REQUEST


def test_classify_product_info_canonical_query():
    result = classify("What are the specs of the Bluetooth speaker?")
    assert result.intent == Intent.PRODUCT_INFO


def test_classify_return_policy_canonical_query():
    result = classify("What is your return policy?")
    assert result.intent == Intent.RETURN_POLICY


def test_classify_shipping_info_canonical_query():
    result = classify("How long does shipping to Germany take?")
    assert result.intent == Intent.SHIPPING_INFO


def test_classify_account_help_canonical_query():
    result = classify("I can't log in to my account.")
    assert result.intent == Intent.ACCOUNT_HELP


def test_classify_complaint_canonical_query():
    result = classify("This is absolutely unacceptable, my order arrived damaged!")
    assert result.intent == Intent.COMPLAINT


def test_classify_multi_issue_dispute_canonical_query():
    result = classify("I have several problems: a missing item and a billing error.")
    assert result.intent == Intent.MULTI_ISSUE_DISPUTE


def test_classify_out_of_scope_canonical_query():
    result = classify("What's the weather like in Lisbon today?")
    assert result.intent == Intent.OUT_OF_SCOPE


def test_empty_query_returns_ambiguous():
    result = classify("")
    assert result.intent == Intent.AMBIGUOUS_QUERY


def test_whitespace_only_query_returns_ambiguous():
    result = classify("   \n\t  ")
    assert result.intent == Intent.AMBIGUOUS_QUERY


def test_query_with_no_triggers_returns_ambiguous():
    result = classify("xyz random gibberish text")
    assert result.intent == Intent.AMBIGUOUS_QUERY
    assert result.confidence == 0.3


def test_classification_result_includes_matched_trigger():
    result = classify("Where is my order ORD-1024?")
    assert result.matched_trigger is not None
    assert result.matched_trigger in "where is my order ORD-1024?".lower()
