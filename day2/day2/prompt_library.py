import os, anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic()

FINANCE_SYSTEM_PROMPT = """You are a senior finance analyst with deep expertise in IFRS 
accounting standards, multi-entity group reporting, and variance analysis."""

PROMPT_LIBRARY = {
    "variance_commentary": "Analyse the following P&L data and write a board-ready variance commentary. Max 150 words. Data: {data}",
    "gl_anomaly_describe": "Review this GL transaction and identify anomalies: unusual amounts, round-number risk, duplicates, timing issues. Transaction: {transaction}",
    "reconciliation_gap": "Explain this reconciliation gap for a non-finance manager. List 3 likely root causes. Gap details: {gap_details}",
    "ifrs_disclosure": "Draft an IFRS-compliant disclosure note. Cite the relevant standard. Item: {item}",
    "aged_debt_commentary": "Write a collections commentary. Identify top 3 risk accounts and recommended actions. Report: {data}",
}

def use_prompt(key: str, **kwargs) -> str:
    prompt = PROMPT_LIBRARY[key].format(**kwargs)
    response = client.messages.create(
        model="claude-opus-4-5", max_tokens=1024, temperature=0.1,
        system=FINANCE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

if __name__ == "__main__":
    print(use_prompt("variance_commentary",
        data="Revenue $462k vs $390k budget. COGS $220k. Net profit $123k (26.7%)"))
