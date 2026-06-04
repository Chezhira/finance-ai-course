import os, json, pandas as pd, anthropic
from dotenv import load_dotenv

os.chdir(r"F:\finance-ai-course")
load_dotenv()

DATA_PATH = r"F:\finance-ai-course\data\trial_balance.csv"
client = anthropic.Anthropic()

def get_account_balance(account_code, period, entity):
    df = pd.read_csv(DATA_PATH)
    row = df[(df["account_code"].astype(str)==account_code)&(df["period"]==period)&(df["entity"]==entity)]
    if row.empty: return {"error": f"No data for {account_code}/{period}/{entity}"}
    r = row.iloc[0]
    return {"account_code":account_code,"account_name":r["account_name"],
            "debit":float(r["debit"]),"credit":float(r["credit"]),"net":float(r["debit"])-float(r["credit"])}

def check_trial_balance(period, entity):
    df = pd.read_csv(DATA_PATH)
    f = df[(df["period"]==period)&(df["entity"]==entity)]
    dr, cr = f["debit"].sum(), f["credit"].sum()
    return {"period":period,"entity":entity,"total_debits":dr,"total_credits":cr,
            "difference":dr-cr,"balanced":str(abs(dr-cr)<0.01)}

def flag_round_numbers(entity, period, threshold=10000):
    df = pd.read_csv(DATA_PATH)
    f = df[(df["entity"]==entity)&(df["period"]==period)]
    flags = []
    for _, row in f.iterrows():
        for col in ["debit","credit"]:
            val = float(row[col])
            if val >= threshold and val % 1000 == 0:
                flags.append({"account":row["account_name"],"amount":val,"side":col})
    return flags

TOOL_SCHEMAS = [
    {"name":"get_account_balance","description":"Get balance for a specific account, period and entity.",
     "input_schema":{"type":"object","properties":{"account_code":{"type":"string"},
     "period":{"type":"string"},"entity":{"type":"string"}},"required":["account_code","period","entity"]}},
    {"name":"check_trial_balance","description":"Check if debits equal credits.",
     "input_schema":{"type":"object","properties":{"period":{"type":"string"},
     "entity":{"type":"string"}},"required":["period","entity"]}},
    {"name":"flag_round_numbers","description":"Flag suspiciously round transaction amounts.",
     "input_schema":{"type":"object","properties":{"entity":{"type":"string"},
     "period":{"type":"string"},"threshold":{"type":"integer"}},"required":["entity","period"]}},
]

TOOL_MAP = {"get_account_balance":get_account_balance,
            "check_trial_balance":check_trial_balance,
            "flag_round_numbers":flag_round_numbers}

def run_finance_agent(task):
    messages = [{"role":"user","content":task}]
    while True:
        response = client.messages.create(
            model="claude-opus-4-5", max_tokens=2048,
            system="You are a GL audit agent. Check trial balance first, then accounts, then anomalies.",
            tools=TOOL_SCHEMAS, messages=messages)
        if response.stop_reason == "end_turn":
            return response.content[0].text
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  -> {block.name}({block.input})")
                result = TOOL_MAP[block.name](**block.input)
                tool_results.append({"type":"tool_result","tool_use_id":block.id,
                                     "content":json.dumps(result, default=str)})
        messages.append({"role":"assistant","content":response.content})
        messages.append({"role":"user","content":tool_results})

if __name__ == "__main__":
    task = """Audit EntityA period 2025-12:
    1. Verify trial balance is balanced
    2. Check revenue accounts 4001 and 4002
    3. Flag round-number anomalies
    4. Write 3-point audit summary with risk rating Low/Medium/High"""
    print("Running GL Audit Agent...\n")
    print(run_finance_agent(task))
