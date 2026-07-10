"""Streamlit entry point for the customer-support agent."""

import streamlit as st

import src.agent as agent
from src.data_loaders import load_customers, load_knowledge_base, load_orders, load_test_queries

st.set_page_config(page_title="Customer Support Agent — Thesis Prototype", layout="wide")

st.title("Customer Support Agent — Thesis Prototype")

# --- Session state -----------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.caption("Powered by Gemini 3.1 Flash Lite")

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

    # Escalation --------------------------------------------------------------
    st.header("Escalation")
    esc = last_response.escalation if last_response is not None else None
    if esc is not None:
        if esc.should_escalate:
            st.markdown("**:red[Escalate to human]**")
            for t in esc.triggers:
                st.markdown(f"- `{t}`")
        else:
            st.write("Handle autonomously")

        score = esc.emotion_score
        if score < -0.5:
            sentiment_label = "strongly negative"
        elif score < -0.05:
            sentiment_label = "negative"
        elif score > 0.05:
            sentiment_label = "positive"
        else:
            sentiment_label = "neutral"
        st.write(f"**Emotion score:** {score:.2f} ({sentiment_label})")
        st.caption(esc.reason)
    else:
        st.write("Escalation decision will appear here.")

    # Citations ---------------------------------------------------------------
    st.header("Citations")
    if last_response is not None:
        gen = last_response.generation
        if gen.citations:
            st.write(f"Cited {len(gen.citations)} of {len(last_response.retrieved_chunks)} retrieved chunks")
            for src in gen.citations:
                st.markdown(f"- {src}")
        else:
            st.write("No citations in this response")
    else:
        st.write("Citations from the response will appear here.")

    # Tool execution ----------------------------------------------------------
    st.header("Tool execution")
    tr = last_response.tool_result if last_response is not None else None
    if tr is not None:
        _STATUS_STYLE = {
            "ok": ("green", "✓"),
            "exceeded_authority": ("orange", "⚠"),
            "out_of_window": ("orange", "⚠"),
            "not_found": ("red", "✗"),
            "missing_identifier": ("red", "✗"),
            "error": ("red", "✗"),
        }
        colour, icon = _STATUS_STYLE.get(tr.status, ("grey", "·"))
        st.write(f"**Tool:** `{tr.tool}`")
        st.markdown(f"**Status:** :{colour}[{icon} {tr.status}]")
        if tr.reason:
            st.caption(tr.reason)
        if tr.data:
            st.json(tr.data)
    else:
        st.write("Tool calls for transactional queries will appear here.")

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
    st.session_state.messages.append(
        {"role": "assistant", "content": result.generation.text, "agent_response": result}
    )
    st.rerun()
