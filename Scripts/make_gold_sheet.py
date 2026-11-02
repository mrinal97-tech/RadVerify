"""
make_gold_sheet.py

Creates data/gold_sheet.csv: 50 real reports for YOU to label by hand.

Why: real_data_adapter.py extracts findings from the same text the checker
reads, so the main benchmark partly measures two parsers agreeing with each
other. Hand labels give an independent ground truth.

How to fill it in (open in Excel / Google Sheets):
  - Read the report text.
  - For each finding column, put 1 if the report says the finding IS present.
    Leave it empty if it is absent, negated ("no effusion") or not mentioned.
  - Use 'notes' for anything ambiguous ("possible", "cannot exclude", ...).
Then run:  python evaluate_gold.py
"""

import argparse
import csv
import json
import random

import lexicon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/real_clean_reports.json")
    ap.add_argument("--out", default="data/gold_sheet.csv")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    cases = json.load(open(args.inp, encoding="utf-8"))
    rng = random.Random(args.seed)

    # Oversample reports with findings so most finding types get labelled at least once.
    with_f = [c for c in cases if c["findings"]]
    without_f = [c for c in cases if not c["findings"]]
    k_with = min(len(with_f), int(args.n * 0.6))
    sample = rng.sample(with_f, k_with) + rng.sample(without_f, min(len(without_f), args.n - k_with))
    rng.shuffle(sample)

    labels = list(lexicon.FINDING_PATTERNS)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "report"] + labels + ["notes"])
        for c in sample:
            w.writerow([c["case_id"], c["report"]] + [""] * len(labels) + [""])
    print(f"Wrote {len(sample)} reports to label -> {args.out}")


if __name__ == "__main__":
    main()
