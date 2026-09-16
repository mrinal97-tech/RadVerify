
import argparse, csv, json, re
from pathlib import Path

FINDING_LABELS = {
    "cardiomegaly": [r"cardiomegaly", r"heart (is |size )?enlarged", r"enlarged (cardiac|heart)"],
    "opacity": [r"opacit(y|ies)"],
    "pleural_effusion": [r"pleural effusion", r"\beffusion(s)?\b"],
    "pneumothorax": [r"pneumothora(x|ces)"],
    "atelectasis": [r"atelectasis"],
    "pulmonary_edema": [r"pulmonary edema", r"\bedema\b"],
    "consolidation": [r"consolidation"],
    "fracture": [r"fracture"],
    "support_devices": [r"\bcatheter\b", r"\bpacemaker\b", r"\bsternotomy\b", r"\baicd\b", r"\bdrainage tube\b"],
}
NEGATION_CUES = ["no ", "no evidence of", "no focal", "without", "negative for",
                 "free of", "clear of", "absence of", "rule out", "not seen", "resolved"]
SEVERITY_CUES = {"mild": ["mild", "minimal", "small", "trace"], "moderate": ["moderate"],
                  "severe": ["severe", "large", "extensive", "complete", "significant"]}
LATERALITY_CUES = {"left": "left", "right": "right", "bilateral": "bilateral", "bilaterally": "bilateral"}


def _sentence_span(caption: str, pos: int) -> tuple:
    # Negation must be checked within the SAME sentence, not a fixed character
    # window -- a fixed window can clip a multi-word cue at its edge (this
    # actually happened with "no evidence of") and can leak negation across
    # unrelated sentences.
    start = caption.rfind(".", 0, pos)
    start = 0 if start == -1 else start + 1
    end = caption.find(".", pos)
    end = len(caption) if end == -1 else end
    return start, end


def _is_negated(caption: str, match_start: int) -> bool:
    sent_start, _ = _sentence_span(caption, match_start)
    return any(cue in caption[sent_start:match_start] for cue in NEGATION_CUES)


def _nearby_severity(caption: str, match_start: int, match_end: int) -> str:
    sent_start, sent_end = _sentence_span(caption, match_start)
    window = caption[sent_start:sent_end]
    for sev, cues in SEVERITY_CUES.items():
        if any(cue in window for cue in cues):
            return sev
    return "unspecified"


def _nearby_laterality(caption: str, match_start: int, match_end: int):
    sent_start, sent_end = _sentence_span(caption, match_start)
    window = caption[sent_start:sent_end]
    for cue, lat in LATERALITY_CUES.items():
        if cue in window:
            return lat
    return None


def extract_findings(caption: str) -> list:
    lowered = caption.lower()
    findings = []
    for label, patterns in FINDING_LABELS.items():
        # Gather ALL matches from ALL patterns for this label FIRST, then decide
        # present/absent from the whole set -- checking one pattern at a time and
        # stopping early caused a bug where a correctly-negated specific match
        # ("pleural effusion") was overridden by a looser, later match ("effusion").
        all_matches = []
        for pattern in patterns:
            all_matches.extend(re.finditer(pattern, lowered))
        if not all_matches:
            continue
        non_negated = [m for m in all_matches if not _is_negated(lowered, m.start())]
        if non_negated:
            m = non_negated[0]
            sent_start, sent_end = _sentence_span(lowered, m.start())
            findings.append({
                "label": label, "present": True,
                "severity": _nearby_severity(lowered, m.start(), m.end()),
                "laterality": _nearby_laterality(lowered, m.start(), m.end()),
                "source_sentence": caption[sent_start:sent_end].strip(),
            })
    return findings


def build_cases(csv_path: str, limit: int = None, frontal_only: bool = True) -> list:
    cases = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if frontal_only and row["projection"] != "Frontal":
                continue
            caption = row["org_caption"].strip()
            if not caption:
                continue
            cases.append({
                "case_id": row["image_id"].replace(".png", ""),
                "findings": extract_findings(caption),
                "report": f"FINDINGS: {caption}\nIMPRESSION: see findings.",
                "source": "real_iu_chest_xray",
            })
            if limit and len(cases) >= limit:
                break
    return cases


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="C:/Users/mrina/Downloads/cleaned_dataset.csv")
    ap.add_argument("--out", default="data/real_clean_reports.json")
    ap.add_argument("--limit", type=int, default=300)
    args = ap.parse_args()
    cases = build_cases(args.inp, limit=args.limit)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cases, open(args.out, "w"), indent=2)
    print(f"Wrote {len(cases)} real cases -> {args.out}")