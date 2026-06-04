import os
import streamlit as st
import pandas as pd
import anthropic
from dotenv import load_dotenv

load_dotenv()
DATA_PATH = r"F:\finance-ai-course\data\trial_balance.csv"

st.set_page_config(page_title="Finance AI Analyst", page_icon="=", layout="wide")
st.title("= Finance AI Analyst")
st.caption("Upload your trial balance -> get AI-powered variance commentary")

with st.sidebar:
    st.header("Settings")
    entity = st.selectbox("Entity", ["EntityA", "EntityB"])
    period = st.selectbox("Period", ["2025-12", "2025-11"])
    model = st.selectbox("AI Model", ["claude-opus-4-5", "claude-haiku-4-5-20251001"])

uploaded = st.file_uploader("Upload Trial Balance CSV", type=["csv"])
df = pd.read_csv(uploaded) if uploaded else pd.read_csv(DATA_PATH)

filtered = df[(df["entity"] == entity) & (df["period"] == period)]
revenue = filtered[filtered["account_type"] == "Income"]["credit"].sum()
cogs    = filtered[filtered["account_code"].astype(str) == "5001"]["debit"].sum()
opex    = filtered[(filtered["account_type"] == "Expense") &
                   (filtered["account_code"].astype(str) != "5001")]["debit"].sum()
gross   = revenue - cogs
net     = gross - opex

col1, col2, col3, col4 = st.columns(4)
col1.metric("Revenue", f"${revenue:,.0f}")
col2.metric("Gross Profit", f"${gross:,.0f}", f"{gross/revenue*100:.1f}%" if revenue else "--")
col3.metric("OpEx", f"${opex:,.0f}")
col4.metric("Net Profit", f"${net:,.0f}", f"{net/revenue*100:.1f}%" if revenue else "--")

st.subheader("Trial Balance Detail")
st.dataframe(filtered, use_container_width=True)

st.subheader("AI Variance Commentary")
if st.button("Generate Commentary"):
    with st.spinner("Analysing..."):
        client = anthropic.Anthropic()
        pnl_summary = f"Entity: {entity}, Period: {period}\nRevenue: ${revenue:,.0f} | COGS: ${cogs:,.0f} | Gross: ${gross:,.0f} ({gross/revenue*100:.1f}%)\nOpEx: ${opex:,.0f} | Net: ${net:,.0f} ({net/revenue*100:.1f}% margin)"
        msg = client.messages.create(
            model=model, max_tokens=512, temperature=0.1,
            system="Senior finance analyst. Write board-ready commentary. Be concise.",
            messages=[{"role": "user", "content": f"Write a variance commentary for this P&L:\n{pnl_summary}"}]
        )
        st.success(msg.content[0].text)
