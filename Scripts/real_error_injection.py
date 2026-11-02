"""
real_error_injection.py

Builds a BALANCED evaluation set from real IU chest X-ray reports.

Five error types, each with its own quota:
  ADDED_FINDING    - an absent finding is written into the report (IU style, mid-report)
  OMITTED_FINDING  - the sentence carrying a real finding is removed
  SEVERITY_FLIP    - e.g. "small effusion" -> "large effusion"
  NEGATION_FLIP    - e.g. "no pneumothorax" -> "there is pneumothorax"
  LATERALITY_FLIP  - e.g. "left effusion" -> "right effusion"

Each clean report is used as the source of at most one corrupted copy.
Every corruption is checked with lexicon.py so the error really exists in the text.
"""

import argparse
import copy
import json
import random
import re
from collections import Counter
from pathlib import Path

import lexicon

ALL_LABELS = list(lexicon.FINDING_PATTERNS)
ERROR_TYPES = ["SEVERITY_FLIP", "LATERALITY_FLIP", "OMITTED_FINDING", "NEGATION_FLIP", "ADDED_FINDING"]


def _body(report: str):
    """Split 'FINDINGS: ... \\nIMPRESSION: ...' into (findings_text, tail)."""
    m = re.match(r"FINDINGS:\s*(.*?)(\nIMPRESSION:.*)?$", report, flags=re.S)
    return m.group(1), (m.group(2) or "")


def _with_findings_sentences(case):
    return [f for f in case["findings"] if f.get("source_sentence")]


# ---------------------------------------------------------
# INJECTORS (each returns a corrupted copy or None)
# ---------------------------------------------------------

def inject_added_finding(case, rng):
    present = {f["label"] for f in case["findings"]}
    candidates = [l for l in ALL_LABELS
                  if l not in present and not lexicon.is_positively_mentioned(case["report"], l)]
    if not candidates:
        return None
    label = rng.choice(candidates)
    sev = rng.choice(["small", "moderate", "large", ""])
    side = rng.choice(["left", "right", ""]) if label in ("pleural_effusion", "pneumothorax", "opacity", "atelectasis", "fracture") else ""
    new_sentence = " ".join(w for w in [sev, side, lexicon.WRITE_PHRASE[label]] if w) + "."

    body, tail = _body(case["report"])
    sentences = re.split(r"(?<=\.)\s+", body.strip())
    pos = rng.randint(1, len(sentences)) if len(sentences) > 1 else len(sentences)
    sentences.insert(pos, new_sentence)

    c = copy.deepcopy(case)
    c["report"] = "FINDINGS: " + " ".join(sentences) + tail
    if not lexicon.is_positively_mentioned(c["report"], label):
        return None
    c["error_type"], c["error_detail"] = "ADDED_FINDING", new_sentence
    return c


def inject_omitted_finding(case, rng):
    candidates = _with_findings_sentences(case)
    rng.shuffle(candidates)
    for f in candidates:
        sentence = f["source_sentence"]
        if sentence not in case["report"]:
            continue
        report = case["report"].replace(sentence, "").replace("..", ".").replace("  ", " ")
        if lexicon.is_positively_mentioned(report, f["label"]):
            continue  # another sentence still states the finding: not a real omission
        c = copy.deepcopy(case)
        c["report"] = report
        c["error_type"], c["error_detail"] = "OMITTED_FINDING", sentence
        return c
    return None


def inject_severity_flip(case, rng):
    flip = {"mild": "large", "moderate": "small", "severe": "small"}
    for f in _with_findings_sentences(case):
        if f["severity"] == "unspecified":
            continue
        sentence = f["source_sentence"]
        if sentence not in case["report"]:
            continue
        cue = next((w for w in lexicon.SEVERITY_CUES[f["severity"]]
                    if re.search(rf"\b{w}\b", sentence.lower())), None)
        if cue is None:
            continue
        new_sentence = re.sub(rf"\b{cue}\b", flip[f["severity"]], sentence.lower(), count=1)
        c = copy.deepcopy(case)
        c["report"] = case["report"].replace(sentence, new_sentence)
        c["error_type"] = "SEVERITY_FLIP"
        c["error_detail"] = f"'{cue}' -> '{flip[f['severity']]}' in: {sentence}"
        return c
    return None


def inject_laterality_flip(case, rng):
    swap = {"left": "right", "right": "left"}
    for f in _with_findings_sentences(case):
        side = f.get("laterality")
        if side not in swap:
            continue
        sentence = f["source_sentence"]
        if sentence not in case["report"] or not re.search(rf"\b{side}\b", sentence.lower()):
            continue
        new_sentence = re.sub(rf"\b{side}\b", swap[side], sentence.lower())
        c = copy.deepcopy(case)
        c["report"] = case["report"].replace(sentence, new_sentence)
        c["error_type"] = "LATERALITY_FLIP"
        c["error_detail"] = f"'{side}' -> '{swap[side]}' in: {sentence}"
        return c
    return None


NEG_PREFIX = re.compile(
    r"^\s*no\s+(?:evidence of\s+)?(?:(?:acute|focal|significant|definite|visible|large|other)\s+)*",
    flags=re.I,
)


def inject_negation_flip(case, rng):
    """Turn a negated finding into a positive one: 'no pneumothorax' -> 'there is pneumothorax'."""
    present = {f["label"] for f in case["findings"]}
    sentences = lexicon.split_sentences(_body(case["report"])[0])
    rng.shuffle(sentences)
    for sent in sentences:
        if not NEG_PREFIX.match(sent):
            continue
        labels = [l for l in ALL_LABELS
                  if l not in present and lexicon.mentions(sent, l) and not lexicon.positive_mentions(sent, l)]
        if not labels:
            continue
        rest = NEG_PREFIX.sub("", sent)
        if re.search(r"\b(?:no|not|without)\b", rest.lower()):
            continue  # keep it to simple single-negation sentences
        new_sent = "there is " + rest
        c = copy.deepcopy(case)
        c["report"] = case["report"].replace(sent, new_sent, 1)
        if not any(lexicon.is_positively_mentioned(c["report"], l) for l in labels):
            continue
        c["error_type"] = "NEGATION_FLIP"
        c["error_detail"] = f"'{sent}' -> '{new_sent}'"
        return c
    return None


INJECTORS = {
    "ADDED_FINDING": inject_added_finding,
    "OMITTED_FINDING": inject_omitted_finding,
    "SEVERITY_FLIP": inject_severity_flip,
    "NEGATION_FLIP": inject_negation_flip,
    "LATERALITY_FLIP": inject_laterality_flip,
}


def build_eval_set(clean_reports: list, per_type: int = 25, seed: int = 11) -> list:
    rng = random.Random(seed)
    eval_set = []
    for case in clean_reports:
        c = copy.deepcopy(case)
        c["is_corrupted"], c["error_type"], c["error_detail"] = False, "CLEAN", None
        eval_set.append(c)

    pool = list(clean_reports)
    rng.shuffle(pool)
    used = set()
    counts = Counter()

    # Scarce error types first, so they get first pick of suitable reports.
    for etype in ERROR_TYPES:
        for case in pool:
            if counts[etype] >= per_type:
                break
            if case["case_id"] in used:
                continue
            corrupted = INJECTORS[etype](case, rng)
            if corrupted is None:
                continue
            corrupted["is_corrupted"] = True
            eval_set.append(corrupted)
            used.add(case["case_id"])
            counts[etype] += 1

    print("Corrupted cases per type:")
    for etype in ERROR_TYPES:
        note = "" if counts[etype] >= per_type else "  <- below quota: use more source reports (--limit in real_data_adapter.py)"
        print(f"  {etype:16s} {counts[etype]:3d}{note}")
    return eval_set


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/real_clean_reports.json")
    ap.add_argument("--out", default="data/real_eval_set.json")
    ap.add_argument("--per_type", type=int, default=25)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    clean = json.load(open(args.inp, encoding="utf-8"))
    eval_set = build_eval_set(clean, per_type=args.per_type, seed=args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_set, open(args.out, "w", encoding="utf-8"), indent=2)
    n_bad = sum(e["is_corrupted"] for e in eval_set)
    print(f"Wrote {len(eval_set)} cases ({n_bad} corrupted, {len(eval_set) - n_bad} clean) -> {args.out}")
