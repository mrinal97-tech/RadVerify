import argparse, copy, json, random, re
from pathlib import Path
from report_gen import LABEL_PHRASE, SEVERITY_PHRASE, _findings_to_sentence
from real_data_adapter import SEVERITY_CUES
from report_gen import LABEL_PHRASE, SEVERITY_PHRASE, _findings_to_sentence
from real_data_adapter import SEVERITY_CUES
from groundguard import _is_negated_mention

random.seed(11)
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
    """
    Remove a finding from a real report.

    The corruption is accepted only if the targeted finding
    is no longer positively represented after the sentence
    is removed.
    """

    candidates = c_findings_with_sentence(case)

    if not candidates:
        return None

    # Try candidates in random order until we find
    # a genuine omission.
    random.shuffle(candidates)

    for dropped in candidates:

        c = copy.deepcopy(case)

        sentence = dropped["source_sentence"]

        if sentence not in c["report"]:
            continue

        corrupted_report = (
            c["report"]
            .replace(sentence, "")
            .replace("..", ".")
            .replace("  ", " ")
        )

        # -------------------------------------------------
        # Verify that the finding was REALLY omitted.
        #
        # A source sentence can overlap with another
        # sentence that still expresses the same finding.
        # -------------------------------------------------

        label = dropped["label"]

        phrase = LABEL_PHRASE[label]
        alternative_phrase = label.replace("_", " ")

        still_present = False

        for report_sentence in re.split(
            r"(?<=[.!?])\s+",
            corrupted_report
        ):

            low = report_sentence.lower()

            positions = []

            phrase_pos = low.find(phrase)

            if phrase_pos != -1:
                positions.append(phrase_pos)

            alternative_pos = low.find(
                alternative_phrase
            )

            if (
                alternative_pos != -1
                and alternative_pos != phrase_pos
            ):
                positions.append(alternative_pos)

            for pos in positions:

                # Negative mention does not count.
                if not _is_negated_mention(
                    report_sentence,
                    pos
                ):
                    still_present = True
                    break

            if still_present:
                break

        # If another positive mention remains,
        # this is NOT a valid omission.
        if still_present:
            continue

        # -------------------------------------------------
        # VALID OMISSION
        # -------------------------------------------------

        c["report"] = corrupted_report
        c["error_type"] = "OMITTED_FINDING"
        c["error_detail"] = sentence

        return c

    # No candidate produced a genuine omission.
    return None


def inject_severity_flip(case: dict) -> dict:
    candidates = [f for f in c_findings_with_sentence(case) if f["severity"] != "unspecified"]
    if not candidates:
        return None
    c = copy.deepcopy(case)
    target = random.choice(candidates)
    sentence = target["source_sentence"]
    if sentence not in c["report"]:
        return None
    flip_map = {"mild": "severe", "moderate": "severe", "severe": "mild"}
    new_severity = flip_map[target["severity"]]
    old_cue = next((w for w in SEVERITY_CUES[target["severity"]] if w in sentence.lower()), None)
    if old_cue is None:
        return None
    new_cue = SEVERITY_CUES[new_severity][0]
    new_sentence = sentence.lower().replace(old_cue, new_cue, 1)
    c["report"] = c["report"].replace(sentence, new_sentence)
    c["error_type"] = "SEVERITY_FLIP"
    c["error_detail"] = f"'{old_cue}' -> '{new_cue}' in: {sentence}"
    return c


def c_findings_with_sentence(case):
    return [f for f in case["findings"] if f.get("source_sentence")]


INJECTORS = [inject_added_finding, inject_omitted_finding, inject_severity_flip]


def build_eval_set(clean_reports: list, error_rate: float = 0.20) -> list:
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
    ap.add_argument("--in", dest="inp", default="data/real_clean_reports.json")
    ap.add_argument("--out", default="data/real_eval_set.json")
    ap.add_argument("--error_rate", type=float, default=0.20)
    args = ap.parse_args()
    clean = json.load(open(args.inp))
    eval_set = build_eval_set(clean, error_rate=args.error_rate)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(eval_set, open(args.out, "w"), indent=2)
    print(f"Wrote {len(eval_set)} cases -> {args.out}")