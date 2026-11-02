"""
inspect_fn_cases.py

Prints every missed error (false negative) and every wrongly flagged clean
report (false positive) from the latest run, with the report text.

    python inspect_fn_cases.py            # both
    python inspect_fn_cases.py --only fn  # misses only
"""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--eval", default="data/real_eval_set.json")
ap.add_argument("--results", default="output/groundguard_results.json")
ap.add_argument("--only", choices=["fn", "fp"], default=None)
args = ap.parse_args()

cases = json.load(open(args.eval, encoding="utf-8"))
results = json.load(open(args.results, encoding="utf-8"))

for case, res in zip(cases, results):
    kind = None
    if case["is_corrupted"] and not res["flagged"]:
        kind = "fn"
    elif not case["is_corrupted"] and res["flagged"]:
        kind = "fp"
    if kind is None or (args.only and kind != args.only):
        continue

    print("=" * 80)
    print("MISSED ERROR" if kind == "fn" else "FALSE ALARM", "|", case["case_id"], "|", case["error_type"])
    if case.get("error_detail"):
        print("INJECTED:", case["error_detail"])
    print("STRUCTURED FINDINGS:")
    for f in case["findings"]:
        print("  ", f["label"], "| severity:", f["severity"], "| side:", f.get("laterality"))
    print("CHECKER SAID:", res["unsupported_claims"] + res["missing_findings"])
    print("REPORT:\n" + case["report"] + "\n")
