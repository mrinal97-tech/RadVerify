"""
real_data_adapter.py

Turns real IU Chest X-Ray captions into structured findings.

All finding / negation / severity / laterality rules come from lexicon.py,
the same vocabulary the checker uses.

NOTE: findings extracted here come from the SAME text the checker later reads,
so they are not independent ground truth. Use make_gold_sheet.py and
evaluate_gold.py for an evaluation against hand-labelled findings.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import lexicon


def extract_findings(caption: str) -> list:
    findings = []
    sentences = lexicon.split_sentences(caption)
    for label in lexicon.FINDING_PATTERNS:
        first = None
        sides = set()
        for sent in sentences:
            for s, e in lexicon.positive_mentions(sent, label):
                if first is None:
                    first = (sent, s, e)
                side = lexicon.laterality_at(sent, s, e)
                if side:
                    sides.add(side)
        if first is None:
            continue
        # A finding described on different sides in different sentences is bilateral.
        if "bilateral" in sides or {"left", "right"} <= sides:
            laterality = "bilateral"
        else:
            laterality = sides.pop() if sides else None
        sent, s, e = first
        findings.append({
            "label": label,
            "present": True,
            "severity": lexicon.severity_at(sent, s, e),
            "laterality": laterality,
            "source_sentence": sent.rstrip("."),
        })
    return findings


def build_cases(csv_path: str, limit: int = None, frontal_only: bool = True) -> list:
    cases = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if frontal_only and row["projection"] != "Frontal":
                continue
            caption = row["org_caption"].strip()
            if not caption:
                continue
            cases.append(make_case(row["image_id"].replace(".png", ""), caption))
            if limit and len(cases) >= limit:
                break
    return cases


def make_case(case_id: str, caption: str) -> dict:
    return {
        "case_id": case_id,
        "findings": extract_findings(caption),
        "report": f"FINDINGS: {caption}\nIMPRESSION: see findings.",
        "source": "real_iu_chest_xray",
    }


def rebuild_from_json(json_path: str) -> list:
    """Re-extract findings from an existing real_clean_reports.json
    (useful when the lexicon changes and the source CSV is not at hand)."""
    old = json.load(open(json_path, encoding="utf-8"))
    cases = []
    for c in old:
        caption = re.sub(r"^FINDINGS:\s*", "", c["report"])
        caption = re.sub(r"\nIMPRESSION: see findings\.\s*$", "", caption)
        cases.append(make_case(c["case_id"], caption))
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="C:/Users/mrina/Downloads/cleaned_dataset.csv",
                    help="source CSV of IU captions")
    ap.add_argument("--from-json", dest="from_json", default=None,
                    help="re-extract from an existing real_clean_reports.json instead of the CSV")
    ap.add_argument("--out", default="data/real_clean_reports.json")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    if args.from_json:
        cases = rebuild_from_json(args.from_json)
    else:
        cases = build_cases(args.inp, limit=args.limit)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cases, open(args.out, "w", encoding="utf-8"), indent=2)
    with_findings = sum(1 for c in cases if c["findings"])
    print(f"Wrote {len(cases)} real cases ({with_findings} with at least one finding) -> {args.out}")
