import os
import streamlit as st
import pandas as pd
import anthropic
from dotenv import load_dotenv
import io

load_dotenv()

# --- Load data — works both locally and on Streamlit Cloud ---
def get_default_data():
    csv_content = """account_code,account_name,account_type,debit,credit,period,entity
1001,Cash and Bank,Asset,45200.00,0,2025-12,EntityA
1100,Accounts Receivable,Asset,128400.00,0,2025-12,EntityA
1200,Inventory,Asset,67800.00,0,2025-12,EntityA
1500,Fixed Assets,Asset,320000.00,0,2025-12,EntityA
2001,Accounts Payable,Liability,0,89500.00,2025-12,EntityA
2100,Accrued Expenses,Liability,0,14200.00,2025-12,EntityA
3001,Share Capital,Equity,0,250000.00,2025-12,EntityA
4001,Revenue - Sales,Income,0,410000.00,2025-12,EntityA
4002,Revenue - Services,Income,0,52000.00,2025-12,EntityA
5001,Cost of Goods Sold,Expense,220000.00,0,2025-12,EntityA
6001,Salaries,Expense,68000.00,0,2025-12,EntityA
6002,Rent,Expense,24000.00,0,2025-12,EntityA
6003,Utilities,Expense,8400.00,0,2025-12,EntityA
6004,Marketing,Expense,15000.00,0,2025-12,EntityA
7001,Interest Expense,Expense,3200.00,0,2025-12,EntityA
1001,Cash and Bank,Asset,38900.00,0,2025-11,EntityA
4001,Revenue - Sales,Income,0,385000.00,2025-11,EntityA
5001,Cost of Goods Sold,Expense,198000.00,0,2025-11,EntityA
6001,Salaries,Expense,68000.00,0,2025-11,EntityA
1001,Cash and Bank,Asset,41200.00,0,2025-12,EntityB
4001,Revenue - Sales,Income,0,195000.00,2025-12,EntityB
5001,Cost of Goods Sold,Expense,102000.00,0,2025-12,EntityB
2001,Accounts Payable,Liability,0,44000.00,2025-12,EntityB"""
    return pd.read_csv(io.StringIO(csv_content))

st.set_page_config(page_title="Finance AI Analyst", page_icon="📊", layout="wide")
st.title("📊 Finance AI Analyst")
st.caption("Upload your trial balance -> get AI-powered variance commentary")

with st.sidebar:
    st.header("Settings")
    entity = st.selectbox("Entity", ["EntityA", "EntityB"])
    period = st.selectbox("Period", ["2025-12", "2025-11"])
    model = st.selectbox("AI Model", ["claude-opus-4-5", "claude-haiku-4-5-20251001"])

uploaded = st.file_uploader("Upload Trial Balance CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    df = get_default_data()
    st.info("Using sample data — upload your own CSV to analyse it")

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
