# Finance AI Builder — FDE Pivot Track

> **20 years of finance depth. 4 years of AI build track record. Production-grade systems, not demos.**

[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://finance-ai-command-centre.streamlit.app) [![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)](https://python.org) [![Anthropic](https://img.shields.io/badge/Claude-API-191919?style=flat)](https://anthropic.com)

---

## Live Demos

| App | Description | Link |
|-----|-------------|------|
| **Finance AI Command Centre** | GL audit · P&L · Policy Q&A · Month-end automation | [Launch](https://finance-ai-command-centre.streamlit.app) |
| **Finance AI Analyst** | P&L dashboard with AI-generated variance commentary | [Launch](https://finance-ai-analyst.streamlit.app) |

---

## What this repo builds

A 10-session, 30-hour programme building finance AI systems from the ground up — each session ships a working, production-relevant artifact.

| # | Artifact | What it does | FDE signal |
|---|----------|-------------|------------|
| 1 | `day1/tb_analyser.py` | Ingests any trial balance CSV, outputs P&L with MoM variance | Finance-native Python |
| 2 | `day2/prompt_library.py` | 10 reusable IFRS-grade prompts — variance commentary, audit queries, disclosures | Prompt engineering |
| 3 | `day3/gl_audit_agent.py` | Agentic GL auditor using Claude tool use — checks, flags, reports | Agentic systems |
| 4 | `day4/app.py` | Live Streamlit dashboard — upload TB, get AI commentary | Can ship to production |
| 5 | `finops-mcp-server` | Extended to 20 tools — added intercompany reconciliation tool | MCP / protocol depth |
| 6 | `day6/local_gl_auditor.py` | Full GL audit via Ollama — zero data leaves the machine | Enterprise privacy arch |
| 7 | `day7/model_router.py` | Routes finance tasks to Groq / Claude / Ollama by complexity and sensitivity | Cost optimisation |
| 8 | `day8/finance_rag.py` | RAG over IFRS standards and policy docs — grounded, cited answers | Document AI |
| 9 | `day9/month_end_pipeline.py` | LangChain 3-step pipeline: validate, analyse, board commentary | Workflow automation |
| 10 | `day10/app.py` | Finance AI Command Centre — all tools in one deployed app | Primary FDE demo |

---

## Tech stack

| Layer | Technology |
|-------|------------|
| LLM (cloud) | Claude claude-opus-4-5 (Anthropic) |
| LLM (local) | Ollama + Llama 3.2 — offline, zero API cost |
| Fast inference | Groq — llama-3.1-8b-instant |
| Agent framework | Anthropic native tool use |
| Pipelines | LangChain LCEL |
| UI | Streamlit (deployed on Streamlit Cloud) |
| MCP server | FastMCP — 20 finance tools |
| Data | pandas |
| RAG | Keyword search over IFRS and policy corpus |

---

## Key capabilities

- **GL audit at scale** — smart sampling handles 50k+ row GL files; Python rules catch duplicates, suspense entries, admin postings, and round-number concentrations before AI runs
- **Cloud / local routing** — sensitive client data stays on-premise via Ollama; standard queries go to Claude
- **IFRS-grounded answers** — RAG system answers policy questions from your own documents, with citations
- **Month-end automation** — 3-step LangChain pipeline from raw trial balance to board-ready commentary
- **MCP-native** — all core finance tools exposed via Model Context Protocol for Claude Desktop and Claude Code

---

## Run locally

    conda activate financeai
    cd finance-ai-course
    streamlit run day10/app.py

---

## Related repos

- [finops-mcp-server](https://github.com/Chezhira/finops-mcp-server) — 20-tool MCP server for finance AI ecosystem
- [finance-accounting-ecosystem](https://github.com/Chezhira/finance-accounting-ecosystem) — 30+ agent multi-entity accounting system

---

## Author

**Zahidah Murira**
CMA (ICMA Australia) · CFA Institute Associate Member · Finance AI Systems Builder

20 years of progressive finance experience across manufacturing, agribusiness, aquaculture, carbon credits, and distribution — combined with 4 years building production AI systems for finance workflows.

[GitHub](https://github.com/Chezhira) · [Email](mailto:ziddmurira@gmail.com)
