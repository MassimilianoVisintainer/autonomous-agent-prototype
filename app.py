"""Streamlit entry point — Slice 0 placeholder confirming data loads correctly."""

import streamlit as st

from src.data_loaders import load_customers, load_knowledge_base, load_orders, load_test_queries
from src.intents import INTENT_METADATA

st.set_page_config(page_title="Customer Support Agent — Thesis Prototype")

st.title("Customer Support Agent — Thesis Prototype")

st.write(
    "This is **Slice 0** of the build. The repository scaffold and data-loading layer are in place. "
    "No agent functionality is implemented yet — that begins in Slice 1."
)

kb_chunks = load_knowledge_base()
customers = load_customers()
orders = load_orders()
test_queries = load_test_queries()

col1, col2, col3, col4 = st.columns(4)
col1.metric("KB Chunks", len(kb_chunks))
col2.metric("Customers", len(customers))
col3.metric("Orders", len(orders))
col4.metric("Test Queries", len(test_queries))

with st.expander("View loaded intents"):
    for intent, meta in INTENT_METADATA.items():
        st.markdown(
            f"**{intent.value}** &nbsp;·&nbsp; cluster: `{meta.cluster.value}`  \n{meta.description}"
        )
