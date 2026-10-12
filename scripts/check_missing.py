"""
Reports which scenarios are missing for each evaluation condition in the
latest evaluation_results_*.json, and how many claims each condition has.

Usage (from the project root):
    python scripts/check_missing.py
"""

import glob
import json

files = sorted(glob.glob("outputs/results/evaluation_results_*.json"))
if not files:
    raise SystemExit("No evaluation_results_*.json found in outputs/results/")

path = files[-1]
rows = json.load(open(path))
print("file:", path)

all_periods = {r["time_period"] for r in rows}
print("scenarios seen across all conditions:", len(all_periods))

for condition in ["single_agent", "unverified_multiagent", "verify_fin"]:
    have = {r["time_period"] for r in rows if r["condition"] == condition}
    missing = sorted(all_periods - have)
    print(f"{condition:<24} rows={len(have):<3} missing={missing}")

print()
print("verify_fin claims per scenario (a low number can mean an analyst agent failed):")
for r in sorted((r for r in rows if r["condition"] == "verify_fin"), key=lambda r: r["time_period"]):
    print(f"  {r['time_period']:<20} total_claims={r.get('total_claims')}")
