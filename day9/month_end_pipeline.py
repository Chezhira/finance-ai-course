import os, pandas as pd, anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

os.chdir(r"F:\finance-ai-course")
load_dotenv()

DATA_PATH = r"F:\finance-ai-course\data\trial_balance.csv"
llm = ChatAnthropic(model="claude-opus-4-5", temperature=0.1)
parser = StrOutputParser()

validate_prompt = ChatPromptTemplate.from_template(
    "You are a finance controller. Review this trial balance and identify issues:\n{trial_balance_summary}\nList problems or say Validated.")
variance_prompt = ChatPromptTemplate.from_template(
    "Given validation:\n{validation_result}\nAnd comparison:\n{period_comparison}\nWrite variance analysis. Flag items over 10%.")
commentary_prompt = ChatPromptTemplate.from_template(
    "Based on this analysis:\n{variance_analysis}\nWrite board commentary max 150 words. Structure: headline, drivers, risk, action.")

step1 = validate_prompt | llm | parser
step2 = variance_prompt | llm | parser
step3 = commentary_prompt | llm | parser

def run_month_end_close(entity, current_period, prior_period):
    df = pd.read_csv(DATA_PATH)
    curr = df[(df["entity"]==entity) & (df["period"]==current_period)]
    prev = df[(df["entity"]==entity) & (df["period"]==prior_period)]
    tb_summary = curr[["account_name","account_type","debit","credit"]].to_string()
    curr_rev = curr[curr["account_type"]=="Income"]["credit"].sum()
    prev_rev = prev[prev["account_type"]=="Income"]["credit"].sum()
    period_comparison = f"Revenue: {current_period} ${curr_rev:,.0f} vs {prior_period} ${prev_rev:,.0f} ({(curr_rev-prev_rev)/prev_rev*100:+.1f}%)"
    validation = step1.invoke({"trial_balance_summary": tb_summary})
    variance = step2.invoke({"validation_result": validation, "period_comparison": period_comparison})
    commentary = step3.invoke({"variance_analysis": variance})
    return {"validation": validation, "variance": variance, "commentary": commentary}

if __name__ == "__main__":
    result = run_month_end_close("EntityA", "2025-12", "2025-11")
    print("=== BOARD COMMENTARY ===")
    print(result["commentary"])
