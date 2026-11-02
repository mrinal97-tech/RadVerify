"""
evaluate.py

Scores groundguard_results.json:
  - confusion matrix, precision and recall with 95% Wilson intervals
  - recall per injected error type
  - what the false positives were flagged for
  - expected precision at realistic error rates
Writes the numbers to output/metrics.json as well.
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="output/groundguard_results.json")
    ap.add_argument("--out", default="output/metrics.json")
    args = ap.parse_args()

    results = json.load(open(args.results, encoding="utf-8"))

    tp = sum(1 for r in results if r["is_corrupted"] and r["flagged"])
    fn = sum(1 for r in results if r["is_corrupted"] and not r["flagged"])
    fp = sum(1 for r in results if not r["is_corrupted"] and r["flagged"])
    tn = sum(1 for r in results if not r["is_corrupted"] and not r["flagged"])

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    print("\n========== RADVERIFY EVALUATION ==========")
    print(f"Cases: {len(results)}  (corrupted {tp + fn}, clean {fp + tn})")
    print(f"TP {tp}   FN {fn}   FP {fp}   TN {tn}")
    lo, hi = wilson(tp, tp + fp)
    print(f"Precision : {precision:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    lo, hi = wilson(tp, tp + fn)
    print(f"Recall    : {recall:.3f}  (95% CI {lo:.3f}-{hi:.3f})")
    print(f"False-positive rate on clean reports: {fpr:.3f}")

    # Recall per error type
    per_type = {}
    totals, caught = Counter(), Counter()
    for r in results:
        if r["is_corrupted"]:
            totals[r["true_error_type"]] += 1
            caught[r["true_error_type"]] += int(r["flagged"])
    print("\nRecall by error type")
    for etype in sorted(totals):
        lo, hi = wilson(caught[etype], totals[etype])
        per_type[etype] = {"caught": caught[etype], "total": totals[etype]}
        print(f"  {etype:16s} {caught[etype]:3d}/{totals[etype]:<3d}  ({caught[etype] / totals[etype]:.2f}, 95% CI {lo:.2f}-{hi:.2f})")

    # Why clean reports were flagged
    fp_causes = Counter()
    for r in results:
        if not r["is_corrupted"] and r["flagged"]:
            for m in r.get("missing_findings", []):
                fp_causes[f"missing {m.get('label')}"] += 1
            for u in r.get("unsupported_claims", []):
                fp_causes[u.get("type", "UNSUPPORTED")] += 1
    if fp_causes:
        print("\nFalse positives flagged for")
        for cause, n in fp_causes.most_common():
            print(f"  {n:3d}  {cause}")

    # Precision depends on how often reports are actually wrong
    print("\nExpected precision if this share of reports had an error")
    at_prev = {}
    for prev in (0.02, 0.05, 0.10):
        denom = recall * prev + fpr * (1 - prev)
        ppv = recall * prev / denom if denom else 0.0
        at_prev[str(prev)] = round(ppv, 3)
        print(f"  {prev:4.0%}  ->  {ppv:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": round(precision, 3), "recall": round(recall, 3),
        "false_positive_rate": round(fpr, 3),
        "recall_by_error_type": per_type,
        "false_positive_causes": dict(fp_causes),
        "precision_at_prevalence": at_prev,
    }, open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
