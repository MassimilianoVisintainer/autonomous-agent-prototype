"""Streamlit entry point — Slice 4: Grounded response generation."""

import streamlit as st

import src.agent as agent
from src.data_loaders import load_customers, load_knowledge_base, load_orders, load_test_queries

st.set_page_config(page_title="Customer Support Agent — Thesis Prototype", layout="wide")

st.title("Customer Support Agent — Thesis Prototype")
st.caption("Slice 4 — Grounded response generation")

# --- Session state -----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.caption("Powered by Gemini 2.5 Flash")

    # Find the most recent assistant response
    last_response = None
    for msg in reversed(st.session_state.messages):
        if msg["role"] == "assistant" and "agent_response" in msg:
            last_response = msg["agent_response"]
            break

    # Reasoning trace ---------------------------------------------------------
    st.header("Reasoning trace")
    if last_response is not None:
        cl = last_response.classification
        st.write(f"**Intent:** {cl.intent.value}")
        st.write(f"**Confidence:** {cl.confidence:.2f}")
        st.write(f"**Reasoning:** {cl.reasoning or '(no reasoning provided)'}")
        st.write(f"**Method:** {cl.method}")
    else:
        st.write("Type a query to see the reasoning trace.")

    st.caption("Tool calls and escalation are not yet implemented.")

    # Citations ---------------------------------------------------------------
    st.header("Citations")
    if last_response is not None:
        gen = last_response.generation
        n_cited = len(gen.citations)
        n_retrieved = len(last_response.retrieved_chunks)
        if gen.citations:
            st.write(f"Cited {n_cited} of {n_retrieved} retrieved chunks")
            for src in gen.citations:
                st.markdown(f"- {src}")
        else:
            st.write("No citations in this response")
    else:
        st.write("Citations from the response will appear here.")

    # Retrieved chunks --------------------------------------------------------
    st.header("Retrieved chunks")
    if last_response is not None and last_response.retrieved_chunks:
        for rc in last_response.retrieved_chunks:
            c = rc.chunk
            preview = c.content[:120] + ("…" if len(c.content) > 120 else "")
            st.markdown(
                f"**{c.kb_id}** &nbsp;·&nbsp; score: `{rc.score:.2f}` &nbsp;·&nbsp; {c.topic}"
            )
            st.caption(preview)
            with st.expander(c.kb_id):
                st.write(c.content)
                st.caption(f"Source: {c.source_doc}")
    else:
        st.write("Top knowledge base matches will appear here after the first query.")

    # Data loaded -------------------------------------------------------------
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

    result = agent.process_query(user_input)
    response_text = result.generation.text

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "agent_response": result}
    )

    st.rerun()
