import json

with open("output/groundguard_results.json", "r", encoding="utf-8") as f:
    results = json.load(f)

false_negatives = [
    r for r in results
    if r.get("is_corrupted") is True
    and r.get("flagged") is False
]

print("FALSE NEGATIVES:", len(false_negatives))

for r in false_negatives:
    print("=" * 70)
    print("CASE:", r["case_id"])
    print("is_corrupted:", r["is_corrupted"])
    print("flagged:", r["flagged"])
    print("true_error_type:", r["true_error_type"])
    print("unsupported_claims:", r["unsupported_claims"])
    print("missing_findings:", r["missing_findings"])