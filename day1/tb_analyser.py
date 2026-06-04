import os, pandas as pd

def analyse_trial_balance(filepath, entity, current_period, prior_period=None):
    df = pd.read_csv(filepath)
    curr = df[(df["entity"] == entity) & (df["period"] == current_period)]
    if curr.empty:
        print(f"No data found for {entity} / {current_period}")
        return
    revenue = curr[curr["account_type"] == "Income"]["credit"].sum()
    cogs    = curr[curr["account_code"].astype(str) == "5001"]["debit"].sum()
    opex    = curr[(curr["account_type"] == "Expense") & (curr["account_code"].astype(str) != "5001")]["debit"].sum()
    gross   = revenue - cogs
    net     = gross - opex
    gm_pct  = round(gross / revenue * 100, 1) if revenue else 0
    nm_pct  = round(net   / revenue * 100, 1) if revenue else 0
    print("=" * 45)
    print(f"  P&L SUMMARY — {entity}  |  {current_period}")
    print("=" * 45)
    print(f"  Revenue          ${revenue:>12,.0f}")
    print(f"  COGS            (${cogs:>11,.0f})")
    print(f"  Gross Profit     ${gross:>12,.0f}  ({gm_pct}%)")
    print(f"  OpEx            (${opex:>11,.0f})")
    print(f"  Net Profit       ${net:>12,.0f}  ({nm_pct}%)")
    print("=" * 45)
    if prior_period:
        prev_rev = df[(df["entity"]==entity)&(df["period"]==prior_period)]
        prev_rev = prev_rev[prev_rev["account_type"]=="Income"]["credit"].sum()
        if prev_rev:
            mom = revenue - prev_rev
            print(f"  Revenue vs {prior_period}: ${mom:+,.0f} ({mom/prev_rev*100:+.1f}% MoM)")
            print("=" * 45)

if __name__ == "__main__":
    os.chdir(r"F:\finance-ai-course")
    analyse_trial_balance("data/trial_balance.csv", "EntityA", "2025-12", "2025-11")
    analyse_trial_balance("data/trial_balance.csv", "EntityB", "2025-12")
