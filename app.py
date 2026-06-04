"""Streamlit entry point — Slice 1: keyword intent classifier and chat interface."""

import streamlit as st

import src.agent as agent
from src.data_loaders import load_customers, load_knowledge_base, load_orders, load_test_queries

st.set_page_config(page_title="Customer Support Agent — Thesis Prototype", layout="wide")

st.title("Customer Support Agent — Thesis Prototype")
st.caption("Slice 1 — keyword intent classifier")

# --- Session state -----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("Reasoning trace")

    last_classification = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and "classification" in msg:
            last_classification = msg["classification"]
            break

    if last_classification is not None:
        st.write(f"**Intent:** {last_classification.intent.value}")
        st.write(f"**Confidence:** {last_classification.confidence:.2f}")
        trigger_display = last_classification.matched_trigger or "no trigger matched"
        st.write(f"**Matched trigger:** `{trigger_display}`")
        st.write(f"**Method:** {last_classification.method}")
    else:
        st.write("Type a query to see the reasoning trace.")

    st.caption(
        "Response generation, retrieval, tool calls, and escalation "
        "are not yet implemented."
    )

    with st.expander("Data loaded", expanded=False):
        kb = load_knowledge_base()
        customers = load_customers()
        orders = load_orders()
        queries = load_test_queries()
        col1, col2 = st.columns(2)
        col1.metric("KB Chunks", len(kb))
        col1.metric("Customers", len(customers))
        col2.metric("Orders", len(orders))
        col2.metric("Test Queries", len(queries))

# --- Chat history ------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- Chat input --------------------------------------------------------------

user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    result = agent.classify_query(user_input)

    response = (
        f"I classified this as **{result.intent.value}** "
        f"(confidence {result.confidence:.2f}). "
        "Response generation is not implemented yet — Slice 3 will add it."
    )
    st.session_state.messages.append(
        {"role": "assistant", "content": response, "classification": result}
    )

    st.rerun()
