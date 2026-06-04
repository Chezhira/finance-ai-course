import os
import streamlit as st
import pandas as pd
import anthropic
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

TB_PATH  = r"F:\finance-ai-course\data\trial_balance.csv"
GL_PATH  = r"F:\finance-ai-course\data\gl_transactions.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_AI_ROWS = 200      # max rows sent to AI
CHUNK_SIZE  = 300      # rows per chunk for large files
MAX_CHUNKS  = 5        # max chunks to process

client = anthropic.Anthropic()

def call_claude(prompt, system="You are a senior finance analyst. Be concise and precise."):
    r = client.messages.create(model="claude-opus-4-5", max_tokens=1024,
        temperature=0.1, system=system,
        messages=[{"role":"user","content":prompt}])
    return r.content[0].text

def call_ollama(prompt):
    r = requests.post(OLLAMA_URL,
        json={"model":"llama3.2","prompt":prompt,"stream":False})
    return r.json()["response"]

# --- Smart GL sampler ---
def smart_gl_sample(df, max_rows=MAX_AI_ROWS):
    """Always include flagged items. Fill remainder with random sample."""
    try:
        dupes     = df[df.duplicated(subset=["date","account_code","debit","credit"], keep=False)]
        suspense  = df[df["account_code"].astype(str) == "999"]
        admin     = df[df["posted_by"] == "admin"] if "posted_by" in df.columns else pd.DataFrame()
        round_num = df[(df["debit"] >= 10000) & (df["debit"] % 1000 == 0)]
        flagged   = pd.concat([dupes, suspense, admin, round_num]).drop_duplicates()
        remaining = max_rows - len(flagged)
        if remaining > 0:
            normal = df.drop(flagged.index).sample(min(remaining, len(df.drop(flagged.index))), random_state=42)
            return pd.concat([flagged, normal]), len(flagged)
        return flagged.head(max_rows), len(flagged)
    except:
        return df.head(max_rows), 0

# --- GL stats summary ---
def gl_stats_summary(df):
    """Summarise GL stats for AI — works on any size dataset."""
    dupes    = df.duplicated(subset=["date","account_code","debit","credit"]).sum() if all(c in df.columns for c in ["date","account_code","debit","credit"]) else 0
    suspense = len(df[df["account_code"].astype(str)=="999"]) if "account_code" in df.columns else 0
    admin    = len(df[df["posted_by"]=="admin"]) if "posted_by" in df.columns else 0
    round_n  = len(df[(df["debit"]>=10000) & (df["debit"]%1000==0)]) if "debit" in df.columns else 0
    top_accs = df.groupby("account_name")["debit"].sum().nlargest(5).to_dict() if "account_name" in df.columns else {}

    return f"""
DATASET STATISTICS
Total transactions : {len(df):,}
Total debits       : ${df["debit"].sum():,.0f}
Total credits      : ${df["credit"].sum():,.0f}
Unique accounts    : {df["account_code"].nunique() if "account_code" in df.columns else "N/A"}
Unique posters     : {df["posted_by"].nunique() if "posted_by" in df.columns else "N/A"}
Date range         : {df["date"].min() if "date" in df.columns else "N/A"} to {df["date"].max() if "date" in df.columns else "N/A"}

AUTOMATED FLAGS
Duplicate entries  : {dupes:,}
Suspense entries   : {suspense:,}
Admin-posted       : {admin:,}
Round numbers >10k : {round_n:,}

TOP ACCOUNTS BY DEBIT VOLUME
{chr(10).join([f"  {k}: ${v:,.0f}" for k,v in top_accs.items()])}
"""

# --- Chunked audit for large files ---
def audit_large_gl(df, routing, progress_bar):
    """Process GL in chunks, synthesise findings at end."""
    chunks = min(MAX_CHUNKS, (len(df) // CHUNK_SIZE) + 1)
    findings = []

    for i in range(chunks):
        chunk = df.iloc[i*CHUNK_SIZE:(i+1)*CHUNK_SIZE]
        progress_bar.progress(int((i+1)/chunks * 70))
        prompt = f"""Audit these {len(chunk)} GL transactions. 
Be brief — list only genuine issues with transaction IDs.
{chunk.to_string()}"""
        if "Cloud" in routing:
            finding = call_claude(prompt, "GL auditor. List issues only. Be concise.")
        else:
            finding = call_ollama(prompt)
        findings.append(f"Chunk {i+1}/{chunks}:\n{finding}")

    # Synthesise all chunk findings
    progress_bar.progress(85)
    synthesis_prompt = f"""Synthesise these audit findings into one structured report.
Remove duplicates. Give overall risk rating Low/Medium/High.

{chr(10).join(findings)}"""

    if "Cloud" in routing:
        return call_claude(synthesis_prompt, "Senior audit manager. Write final audit report.")
    return call_ollama(synthesis_prompt)

def load_tb():
    if st.session_state.get("tb"):
        return pd.read_csv(st.session_state["tb"])
    return pd.read_csv(TB_PATH)

def load_gl():
    if st.session_state.get("gl"):
        return pd.read_csv(st.session_state["gl"])
    return pd.read_csv(GL_PATH)

# ============================================================
# LAYOUT
# ============================================================
st.set_page_config(page_title="Finance AI Command Centre", page_icon="F", layout="wide")

st.sidebar.title("Finance AI")
st.sidebar.caption("Command Centre")

tool   = st.sidebar.radio("Select Tool", [
    "P&L Analyser",
    "GL Audit Agent",
    "Policy Q&A (RAG)",
    "Month-End Commentary",
])
entity = st.sidebar.selectbox("Entity", ["EntityA","EntityB"])
period = st.sidebar.selectbox("Period", ["2025-12","2025-11"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Upload your own data**")
tb_up = st.sidebar.file_uploader("Trial Balance CSV", type=["csv"], key="tb")
gl_up = st.sidebar.file_uploader("GL Transactions CSV", type=["csv"], key="gl")

if tb_up:
    st.sidebar.success(f"TB loaded: {tb_up.name}")
if gl_up:
    st.sidebar.success(f"GL loaded: {gl_up.name}")

# ============================================================
# TOOL 1 — P&L Analyser
# ============================================================
if tool == "P&L Analyser":
    st.title("P&L Analyser")
    st.caption(f"{entity} | {period}")

    df   = load_tb()
    data = df[(df["entity"]==entity) & (df["period"]==period)]

    if data.empty:
        st.warning("No data for selected entity/period. Upload a trial balance or change the filters.")
        st.stop()

    revenue = data[data["account_type"]=="Income"]["credit"].sum()
    cogs    = data[data["account_code"].astype(str)=="5001"]["debit"].sum()
    opex    = data[(data["account_type"]=="Expense") &
                   (data["account_code"].astype(str)!="5001")]["debit"].sum()
    gross   = revenue - cogs
    net     = gross - opex

    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Revenue",      f"${revenue:,.0f}")
    col2.metric("Gross Profit", f"${gross:,.0f}",
                f"{gross/revenue*100:.1f}%" if revenue else "--")
    col3.metric("OpEx",         f"${opex:,.0f}")
    col4.metric("Net Profit",   f"${net:,.0f}",
                f"{net/revenue*100:.1f}%" if revenue else "--")

    st.subheader("Trial Balance")
    st.dataframe(data, use_container_width=True)

    if st.button("Generate AI Commentary"):
        with st.spinner("Analysing..."):
            summary = (f"Entity: {entity} | Period: {period}\n"
                       f"Revenue ${revenue:,.0f} | COGS ${cogs:,.0f} | "
                       f"Gross ${gross:,.0f} ({gross/revenue*100:.1f}%) | "
                       f"OpEx ${opex:,.0f} | Net ${net:,.0f} ({net/revenue*100:.1f}%)")
            st.success(call_claude(
                f"Write board variance commentary:\n{summary}"))

# ============================================================
# TOOL 2 — GL Audit Agent
# ============================================================
elif tool == "GL Audit Agent":
    st.title("GL Audit Agent")
    st.caption("Transaction-level audit — handles large GL files intelligently")

    routing = st.radio("Data routing",
        ["Cloud (Claude)", "Local (Ollama — data stays on machine)"],
        horizontal=True)

    if st.button("Run GL Audit"):
        df = load_gl()

        # Filter by entity
        if "entity" in df.columns:
            df = df[df["entity"]==entity]

        if df.empty:
            st.warning("No GL transactions found.")
            st.stop()

        st.subheader("Dataset Overview")
        col1,col2,col3,col4 = st.columns(4)
        col1.metric("Transactions", f"{len(df):,}")
        col2.metric("Total Debits",  f"${df['debit'].sum():,.0f}")
        col3.metric("Total Credits", f"${df['credit'].sum():,.0f}")
        bal = df["debit"].sum() - df["credit"].sum()
        col4.metric("Net Difference", f"${bal:,.0f}",
                    delta_color="inverse" if abs(bal) > 0.01 else "normal")

        # --- Automated Python-side flags (runs on full dataset) ---
        st.subheader("Automated Audit Flags")
        flags_found = False

        dupes = df[df.duplicated(
            subset=["date","account_code","debit","credit"], keep=False)] if all(
            c in df.columns for c in ["date","account_code","debit","credit"]) else pd.DataFrame()
        if not dupes.empty:
            st.error(f"DUPLICATES: {len(dupes)} rows — e.g. {dupes['description'].iloc[0][:60] if 'description' in dupes.columns else ''}")
            with st.expander("Show duplicate transactions"):
                st.dataframe(dupes, use_container_width=True)
            flags_found = True

        if "account_code" in df.columns:
            suspense = df[df["account_code"].astype(str)=="999"]
            if not suspense.empty:
                st.error(f"SUSPENSE ACCOUNT: {len(suspense)} entries — ${suspense['debit'].sum():,.0f} debit / ${suspense['credit'].sum():,.0f} credit")
                flags_found = True

        if "posted_by" in df.columns:
            admin_e = df[df["posted_by"]=="admin"]
            if not admin_e.empty:
                st.warning(f"ADMIN POSTED: {len(admin_e)} entries — review authorisation trail")
                flags_found = True

        round_e = df[(df["debit"]>=10000) & (df["debit"]%1000==0)]
        if not round_e.empty:
            st.warning(f"ROUND NUMBERS: {len(round_e)} entries over $10k with round amounts")
            flags_found = True

        if not flags_found:
            st.success("No automated flags raised")

        # --- AI Analysis ---
        st.subheader("AI Audit Report")
        progress = st.progress(0)

        is_large = len(df) > CHUNK_SIZE

        if is_large:
            st.info(f"Large dataset ({len(df):,} rows) — using chunked analysis across {min(MAX_CHUNKS, len(df)//CHUNK_SIZE +1)} chunks")
            with st.spinner("Running chunked audit..."):
                # Always include stats summary
                stats = gl_stats_summary(df)
                report = audit_large_gl(df, routing, progress)
                progress.progress(100)
                st.write(f"**Dataset Stats:**\n{stats}")
                st.write(report)
        else:
            # Small dataset — smart sample + full AI review
            sample, n_flagged = smart_gl_sample(df)
            progress.progress(30)
            st.caption(f"Sending {len(sample)} rows to AI ({n_flagged} flagged items + random sample)")

            prompt = f"""You are a forensic GL auditor. Review these transactions.
Reference specific transaction IDs. Give risk rating Low/Medium/High.

{gl_stats_summary(df)}

TRANSACTION DETAIL (sample of {len(sample)} from {len(df)} total):
{sample[["txn_id","date","account_name","description","debit","credit","posted_by"]].to_string() if "txn_id" in sample.columns else sample.to_string()}"""

            with st.spinner("AI analysing..." if "Cloud" in routing else "Running locally..."):
                analysis = call_claude(
                    prompt, "Forensic GL auditor. Cite transaction IDs. Structured report.") if "Cloud" in routing else call_ollama(prompt)
                progress.progress(100)
                st.write(analysis)

        with st.expander("Full GL transactions"):
            st.dataframe(df, use_container_width=True)

# ============================================================
# TOOL 3 — Policy Q&A (RAG)
# ============================================================
elif tool == "Policy Q&A (RAG)":
    st.title("Policy Q&A")
    st.caption("Answers grounded in IFRS standards and company policy")

    DOCS = [
        {"id":"ias16_1","text":"IAS 16 PPE: Recognised when probable future benefits flow and cost measured reliably. Initial measurement at cost including purchase price and directly attributable costs."},
        {"id":"ias16_2","text":"IAS 16 Depreciation: Significant parts depreciated separately over useful life. Residual value reviewed annually."},
        {"id":"ias2_1","text":"IAS 2 Inventories: Measured at lower of cost and net realisable value. Cost includes purchase, conversion, and other costs to bring to present location."},
        {"id":"ias21_1","text":"IAS 21 Foreign Currency: Transactions recorded at spot rate. Monetary items translated at closing rate. Exchange differences in profit or loss."},
        {"id":"policy_1","text":"Procurement Policy: Orders above $5,000 require Finance Manager and Department Head approval. Orders above $50,000 require Board approval. Emergency up to $2,000 single manager."},
        {"id":"policy_2","text":"Revenue Recognition: Product sales at delivery. Service revenue over performance period. Long-term contracts over 12 months use percentage of completion."},
    ]

    if "embedder" not in st.session_state:
        with st.spinner("Building document index..."):
            emb = SentenceTransformer("all-MiniLM-L6-v2")
            db  = chromadb.Client()
            try:
                col = db.get_collection("finance_docs")
            except:
                col = db.create_collection("finance_docs")
                texts = [d["text"] for d in DOCS]
                col.add(documents=texts,
                        embeddings=emb.encode(texts).tolist(),
                        ids=[d["id"] for d in DOCS])
            st.session_state.embedder   = emb
            st.session_state.collection = col

    question = st.text_input("Ask a question",
        placeholder="What approval is needed for a $75,000 purchase order?")

    if question and st.button("Search"):
        with st.spinner("Searching..."):
            q_emb   = st.session_state.embedder.encode([question]).tolist()
            results = st.session_state.collection.query(
                query_embeddings=q_emb, n_results=3)
            context = "\n\n".join(
                [f"[{i+1}] {doc}" for i,doc in enumerate(results["documents"][0])])
            answer  = call_claude(
                f"Context:\n{context}\n\nQuestion: {question}",
                "Finance compliance assistant. Answer ONLY from context. Cite [1][2].")
            st.write(answer)
            with st.expander("Source documents"):
                st.text(context)

# ============================================================
# TOOL 4 — Month-End Commentary
# ============================================================
elif tool == "Month-End Commentary":
    st.title("Month-End Commentary")
    st.caption("3-step automated pipeline — validate, analyse, draft")

    prior = st.selectbox("Compare against", ["2025-11"])

    if st.button("Run Month-End Close"):
        df   = load_tb()
        curr = df[(df["entity"]==entity) & (df["period"]==period)]
        prev = df[(df["entity"]==entity) & (df["period"]==prior)]

        if curr.empty:
            st.warning("No data for selected period.")
            st.stop()

        curr_rev = curr[curr["account_type"]=="Income"]["credit"].sum()
        prev_rev = prev[prev["account_type"]=="Income"]["credit"].sum()
        curr_net = curr_rev - curr[curr["account_type"]=="Expense"]["debit"].sum()
        prev_net = prev_rev - prev[prev["account_type"]=="Expense"]["debit"].sum()

        col1,col2 = st.columns(2)
        col1.metric("Revenue",    f"${curr_rev:,.0f}",
                    f"{(curr_rev-prev_rev)/prev_rev*100:+.1f}% vs {prior}" if prev_rev else "--")
        col2.metric("Net Profit", f"${curr_net:,.0f}",
                    f"{(curr_net-prev_net)/prev_net*100:+.1f}% vs {prior}" if prev_net else "--")

        tb_summary  = curr[["account_name","account_type","debit","credit"]].to_string()
        comparison  = (f"Revenue: {period} ${curr_rev:,.0f} vs {prior} ${prev_rev:,.0f} "
                       f"({(curr_rev-prev_rev)/prev_rev*100:+.1f}%)")

        progress = st.progress(0)

        with st.spinner("Step 1 — Validating trial balance..."):
            validation = call_claude(
                f"Review trial balance for month-end issues:\n{tb_summary}")
            progress.progress(33)

        with st.spinner("Step 2 — Variance analysis..."):
            variance = call_claude(
                f"Variance analysis:\nValidation: {validation[:300]}\nData: {comparison}")
            progress.progress(66)

        with st.spinner("Step 3 — Board commentary..."):
            commentary = call_claude(
                f"Write 150-word board commentary. Headline, drivers, risk, action:\n{variance}",
                "CFO writing for the board. Precise and concise.")
            progress.progress(100)

        st.subheader("Board Commentary")
        st.success(commentary)

        with st.expander("Variance analysis"):
            st.write(variance)
        with st.expander("Trial balance validation"):
            st.write(validation)
