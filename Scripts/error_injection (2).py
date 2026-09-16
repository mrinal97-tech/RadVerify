import argparse, copy, json, random
from pathlib import Path
from report_gen import LABEL_PHRASE, SEVERITY_PHRASE, _findings_to_sentence

random.seed(7)

ALL_LABELS = list(LABEL_PHRASE.keys())


def inject_added_finding(case: dict) -> dict:
    c = copy.deepcopy(case)
    present_labels = {f["label"] for f in c["findings"]}
    candidates = [l for l in ALL_LABELS if l not in present_labels]
    if not candidates:
        return None
    fake = {"label": random.choice(candidates), "present": True,
            "severity": random.choice(list(SEVERITY_PHRASE)), "laterality": None}
    fake_sentence = _findings_to_sentence(fake)
    c["report"] = c["report"].replace("FINDINGS: ", f"FINDINGS: {fake_sentence} ")
    c["error_type"] = "ADDED_FINDING"
    c["error_detail"] = fake_sentence
    return c

def inject_omitted_finding(case: dict) -> dict:
    if not case["findings"]:
        return None
    c = copy.deepcopy(case)
    dropped = random.choice(c["findings"])
    sentence = _findings_to_sentence(dropped)
    label_phrase = LABEL_PHRASE[dropped["label"]]
    impression_fragment = f"{dropped['severity']} {label_phrase}"
    if sentence not in c["report"]:
        return None
    
    report = c["report"].replace(sentence, "").replace("  ", " ")
    parts = [p.strip() for p in report.split("IMPRESSION:")]
    if len(parts) == 2:
        findings_part, impression_part = parts
        pieces = [p.strip().rstrip(".") for p in impression_part.split(";")]
        pieces = [p for p in pieces if impression_fragment not in p]
        impression_part = "; ".join(pieces) + "." if pieces else "No acute findings."
        report = f"{findings_part}\nIMPRESSION: {impression_part}"

    c["report"] = report
    c["error_type"] = "OMITTED_FINDING"
    c["error_detail"] = sentence
    return c

def inject_severity_flip(case: dict) -> dict:
    if not case["findings"]:
        return None
    c = copy.deepcopy(case)
    idx = random.randrange(len(c["findings"]))
    original = c["findings"][idx]
    old_sentence = _findings_to_sentence(original)
    if old_sentence not in c["report"]:
        return None
    flip_map = {"mild": "severe", "moderate": "severe", "severe": "mild"}
    flipped = copy.deepcopy(original)
    flipped["severity"] = flip_map[original["severity"]]
    new_sentence = _findings_to_sentence(flipped)
    c["report"] = c["report"].replace(old_sentence, new_sentence)
    c["error_type"] = "SEVERITY_FLIP"
    c["error_detail"] = f"'{old_sentence}' -> '{new_sentence}'"
    return c


INJECTORS = [inject_added_finding, inject_omitted_finding, inject_severity_flip]


def build_eval_set(clean_reports: list, error_rate: float = 0.15) -> list:
    eval_set = []
    for case in clean_reports:
        c = copy.deepcopy(case)
        c["is_corrupted"], c["error_type"], c["error_detail"] = False, "CLEAN", None
        eval_set.append(c)

    n_to_corrupt = int(len(clean_reports) * error_rate)
    targets = random.sample(clean_reports, k=min(n_to_corrupt, len(clean_reports)))
    for case in targets:
        random.shuffle(INJECTORS)
        for injector in INJECTORS:
            corrupted = injector(case)
            if corrupted is not None:
                corrupted["is_corrupted"] = True
                eval_set.append(corrupted)
                break
    return eval_set


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="data/clean_reports.json")
    ap.add_argument("--out", default="data/eval_set.json")
    ap.add_argument("--error_rate", type=float, default=0.15)
    args = ap.parse_args()
    clean = json.load(open(args.inp))
    eval_set = build_eval_set(clean, error_rate=args.error_rate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_set, open(args.out, "w"), indent=2)
    n_corrupt = sum(1 for c in eval_set if c["is_corrupted"])
    print(f"Wrote {len(eval_set)} cases ({n_corrupt} corrupted) -> {args.out}")