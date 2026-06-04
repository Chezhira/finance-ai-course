# Finance AI Builder — FDE Pivot Track

A hands-on finance AI course building production-grade systems, engineered for Forward Deployed Engineer roles at AI-native companies.

## Live Demos
- [Finance AI Command Centre](https://finance-ai-command-centre.streamlit.app) — GL audit, P&L analysis, Policy Q&A, Month-End automation
- [Finance AI Analyst](https://finance-ai-analyst.streamlit.app) — P&L dashboard with AI commentary

## What this repo contains

| Session | Artifact | Description |
|---------|----------|-------------|
| Day 1 | `day1/tb_analyser.py` | Trial balance P&L analyser with MoM variance |
| Day 2 | `day2/prompt_library.py` | Finance prompt library — 10 reusable IFRS prompts |
| Day 3 | `day3/gl_audit_agent.py` | GL audit agent with Claude tool use |
| Day 4 | `day4/app.py` | Streamlit finance AI dashboard (live) |
| Day 5 | finops-mcp-server | Extended to 20 tools — intercompany reconciliation added |
| Day 6 | `day6/local_gl_auditor.py` | Offline GL audit via Ollama — zero cloud dependency |
| Day 7 | `day7/model_router.py` | Multi-model router — Groq/Claude/Ollama by task type |
| Day 8 | `day8/finance_rag.py` | RAG system over IFRS standards and finance policy docs |
| Day 9 | `day9/month_end_pipeline.py` | LangChain month-end close pipeline — 3-step automated commentary |
| Day 10 | `day10/app.py` | Finance AI Command Centre — full capstone (live) |

## Architecture

    Trial Balance / GL Transactions CSV
            |
       pandas (data layer)
            |
       Smart sampling + Python audit rules
            |
       Claude API (tool use / chains)
            |
       Structured report + AI commentary
            |
       Streamlit UI (deployed)

## Tech stack
- **LLM**: Claude claude-opus-4-5 (Anthropic)
- **Local LLM**: Ollama Llama 3.2 (offline, zero cloud)
- **Agent framework**: Anthropic native tool use
- **Pipelines**: LangChain LCEL
- **UI**: Streamlit
- **Data**: pandas
- **Search**: Keyword RAG over IFRS and policy docs
- **APIs**: Anthropic, Groq, Ollama

## Domain
Finance AI — GL audit automation, variance analysis, IFRS-grounded commentary, multi-entity group reporting, intercompany reconciliation, month-end close automation.

## Run locally
    conda activate financeai
    cd finance-ai-course
    streamlit run day10/app.py

## Author
Zahidah Murira — CMA (ICMA Australia) | CFA Institute Associate Member | Finance AI Systems Builder
20yr finance depth across manufacturing, agribusiness, aquaculture and carbon credits + AI build track record.

[GitHub](https://github.com/Chezhira) | ziddmurira@gmail.com
