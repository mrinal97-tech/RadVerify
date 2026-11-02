"""
evaluate_gold.py

Uses your hand-labelled data/gold_sheet.csv (see make_gold_sheet.py) to answer
two questions the main benchmark cannot:

  1. How good is the regex labeller?   (adapter findings vs your labels)
  2. How does the checker do when the truth is independent of its own parser?
     - false positives on the 50 clean reports
     - recall on ADDED_FINDING and NEGATION_FLIP errors injected into them
"""

import argparse
import csv
import random
from collections import Counter

import lexicon
from groundguard import check_heuristic
from real_data_adapter import extract_findings
from real_error_injection import inject_added_finding, inject_negation_flip


def load_gold(path):
    labels = list(lexicon.FINDING_PATTERNS)
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    gold = []
    for r in rows:
        present = [l for l in labels if str(r.get(l, "")).strip() in ("1", "y", "yes", "x")]
        gold.append({"case_id": r["case_id"], "report": r["report"], "labels": set(present)})
    return gold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold_sheet.csv")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    gold = load_gold(args.gold)
    if not any(g["labels"] for g in gold):
        print("No labels found. Fill in data/gold_sheet.csv first (put 1 where a finding is present).")
        return

    # 1. Labeller vs hand labels
    per_label = {l: Counter() for l in lexicon.FINDING_PATTERNS}
    for g in gold:
        caption = g["report"].replace("FINDINGS: ", "").replace("\nIMPRESSION: see findings.", "")
        auto = {f["label"] for f in extract_findings(caption)}
        for l in lexicon.FINDING_PATTERNS:
            if l in auto and l in g["labels"]:
                per_label[l]["tp"] += 1
            elif l in auto:
                per_label[l]["fp"] += 1
            elif l in g["labels"]:
                per_label[l]["fn"] += 1

    print("\n=== 1. Regex labeller vs your labels ===")
    tot = Counter()
    for l, c in per_label.items():
        tot.update(c)
        if c["tp"] + c["fp"] + c["fn"]:
            print(f"  {l:17s} agree {c['tp']:2d}   labeller extra {c['fp']:2d}   labeller missed {c['fn']:2d}")
    p = tot["tp"] / max(1, tot["tp"] + tot["fp"])
    r = tot["tp"] / max(1, tot["tp"] + tot["fn"])
    print(f"  Overall labeller precision {p:.2f}, recall {r:.2f}")

    # 2. Checker against independent truth
    cases = []
    for g in gold:
        cases.append({
            "case_id": g["case_id"],
            "report": g["report"],
            "findings": [{"label": l, "present": True, "severity": "unspecified", "laterality": None}
                         for l in sorted(g["labels"])],
        })

    fp = [check_heuristic(c) for c in cases]
    n_fp = sum(r["flagged"] for r in fp)
    print("\n=== 2. Checker with hand-labelled truth ===")
    print(f"  Clean reports wrongly flagged: {n_fp}/{len(cases)}")
    for res in fp:
        if res["flagged"]:
            why = [m["label"] for m in res["missing_findings"]] + [u["type"] for u in res["unsupported_claims"]]
            print(f"    {res['case_id']}: {', '.join(why)}")

    rng = random.Random(args.seed)
    caught, total = Counter(), Counter()
    for c in cases:
        for name, inj in (("ADDED_FINDING", inject_added_finding), ("NEGATION_FLIP", inject_negation_flip)):
            bad = inj(c, rng)
            if bad is None:
                continue
            total[name] += 1
            caught[name] += int(check_heuristic(bad)["flagged"])
    for name in total:
        print(f"  Recall on {name:14s} {caught[name]}/{total[name]}")


if __name__ == "__main__":
    main()
