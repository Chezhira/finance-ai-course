import os, requests, pandas as pd, json

DATA_PATH = r"F:\finance-ai-course\data\trial_balance.csv"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"

def ollama_query(prompt):
    response = requests.post(OLLAMA_URL, json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False})
    return response.json()["response"]

def local_gl_audit(entity, period):
    df = pd.read_csv(DATA_PATH)
    data = df[(df["entity"]==entity) & (df["period"]==period)]
    if data.empty:
        return {"error": f"No data for {entity}/{period}"}
    total_dr, total_cr = data["debit"].sum(), data["credit"].sum()
    diff = total_dr - total_cr
    round_flags = [{"account": row["account_name"], "amount": float(row["debit"])}
                   for _, row in data.iterrows()
                   if float(row["debit"]) >= 10000 and float(row["debit"]) % 1000 == 0]
    summary = data[["account_name","account_type","debit","credit"]].to_string()
    prompt = f"""You are a finance auditor. Review this trial balance for {entity}, {period}:
{summary}
Identify: unusual balances, round-number risks, missing accounts. Give risk rating Low/Medium/High."""
    return {"entity": entity, "period": period, "balanced": abs(diff)<0.01,
            "difference": diff, "round_number_flags": round_flags,
            "ai_analysis": ollama_query(prompt), "data_left_machine": False}

if __name__ == "__main__":
    os.chdir(r"F:\finance-ai-course")
    result = local_gl_audit("EntityA", "2025-12")
    print(json.dumps({k:v for k,v in result.items() if k != "ai_analysis"}, indent=2))
    print("\nAI Analysis:", result["ai_analysis"])
