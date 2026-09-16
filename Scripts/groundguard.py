
"""
groundguard.py

The verification layer for RadVerify.

Runs TWO checks:
  1. GROUNDEDNESS (precision)
     - Every claim in the report must trace back to a structured finding.

  2. COMPLETENESS (recall)
     - Every structured finding must appear in the generated report.

A case is flagged if either check fails.

Backends:
  - heuristic : deterministic rule-based checker
  - ollama    : local Ollama LLM checker
"""

import argparse
import json
import os
import re
from pathlib import Path

from report_gen import LABEL_PHRASE, SEVERITY_PHRASE


# ---------------------------------------------------------
# NEGATION HANDLING
# ---------------------------------------------------------

NEGATION_CUES = [
    "no ",
    "no evidence of",
    "no focal",
    "without",
    "negative for",
    "free of",
    "clear of",
    "absence of",
    "rule out",
    "not seen",
    "resolved",
]


def _is_negated_mention(sentence: str, phrase_pos: int) -> bool:
    """
    Check whether a finding is being negated.

    Example:
        "There is no pneumothorax."

    The word 'pneumothorax' exists in the sentence,
    but it is NOT being claimed as present.
    """

    window = sentence[:phrase_pos].lower()

    return any(
        cue in window
        for cue in NEGATION_CUES
    )


# ---------------------------------------------------------
# SENTENCE SPLITTING
# ---------------------------------------------------------

def _split_sentences(report: str) -> list:

    body = report.strip()

    body = re.sub(
        r"\bFINDINGS:\s*",
        "",
        body,
        flags=re.IGNORECASE
    )

    body = re.sub(
        r"\bIMPRESSION:\s*",
        "",
        body,
        flags=re.IGNORECASE
    )

    return [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+",
            body
        )
        if sentence.strip()
    ]
    
def _finding_is_positively_mentioned(report: str, label: str) -> bool:
    """
    Return True only if the finding is positively mentioned.

    Negated mentions such as:
        no fracture
        no pneumothorax
        without pleural effusion

    do NOT count as evidence that the finding is present.
    """

    sentences = _split_sentences(report)

    phrase = LABEL_PHRASE[label]
    alternative_phrase = label.replace("_", " ")

    for sentence in sentences:

        low = sentence.lower()

        positions = []

        phrase_pos = low.find(phrase)

        if phrase_pos != -1:
            positions.append(phrase_pos)

        alternative_pos = low.find(alternative_phrase)

        if (
            alternative_pos != -1
            and alternative_pos != phrase_pos
        ):
            positions.append(alternative_pos)

        for pos in positions:

            if not _is_negated_mention(sentence, pos):
                return True

    return False


# ---------------------------------------------------------
# HEURISTIC CHECKER
# ---------------------------------------------------------

def check_heuristic(case: dict) -> dict:

    report = case["report"]
    findings = case["findings"]

    findings_by_label = {
        f["label"]: f
        for f in findings
    }

    sentences = _split_sentences(report)

    unsupported = []
    mentioned_labels = set()

    # -----------------------------------------------------
    # GROUNDEDNESS
    # -----------------------------------------------------

    for sent in sentences:

        low = sent.lower().strip()

        for label, phrase in LABEL_PHRASE.items():

            alternatives = {
                phrase.lower(),
                label.replace("_", " ").lower(),
            }

            found_phrase = None
            found_pos = -1

            for candidate in alternatives:

                pos = low.find(candidate)

                if pos != -1:
                    found_phrase = candidate
                    found_pos = pos
                    break

            if found_pos == -1:
                continue

            # -------------------------------------------------
            # NEGATIVE / NORMAL STATEMENT
            # -------------------------------------------------

            if _is_negated_mention(
                sent,
                found_pos
            ):
                continue

            # -------------------------------------------------
            # POSITIVE FINDING
            # -------------------------------------------------

            mentioned_labels.add(label)

            gt = findings_by_label.get(label)

            # -------------------------------------------------
            # ADDED FINDING
            # -------------------------------------------------

            if gt is None:

                unsupported.append({
                    "sentence": sent,
                    "reason": (
                        f"'{label}' not in structured findings"
                    ),
                    "type": "ADDED_FINDING",
                })

                continue

            # -------------------------------------------------
            # SEVERITY
            # -------------------------------------------------

            sev_in_sentence = None

            for severity in SEVERITY_PHRASE:

                if re.search(
                    rf"\b{re.escape(severity)}\b",
                    low
                ):

                    sev_in_sentence = severity
                    break

            if sev_in_sentence:

                gt_severity = gt.get(
                    "severity"
                )

                # Ignore unspecified ground truth severity
                if (
                    gt_severity
                    and gt_severity != "unspecified"
                    and sev_in_sentence != gt_severity
                ):

                    unsupported.append({
                        "sentence": sent,
                        "reason": (
                            f"severity '{sev_in_sentence}' "
                            f"contradicts finding severity "
                            f"'{gt_severity}'"
                        ),
                        "type": "SEVERITY_FLIP",
                    })

    # -----------------------------------------------------
    # COMPLETENESS
    # -----------------------------------------------------

    missing = []

    for finding in findings:

      label = finding["label"]

      if not _finding_is_positively_mentioned(
        report,
        label
    ):

        missing.append({
            "label": label,
            "reason": (
                "structured finding not positively "
                "mentioned in report"
            ),
            "type": "OMITTED_FINDING",
        })
    # -----------------------------------------------------
    # FINAL VERDICT
    # -----------------------------------------------------

    return {
        "case_id": case["case_id"],
        "flagged": bool(
            unsupported or missing
        ),
        "unsupported_claims": unsupported,
        "missing_findings": missing,
    }

# ---------------------------------------------------------
# OLLAMA CHECKER
# ---------------------------------------------------------

def check_ollama(case: dict, client, model: str) -> dict:
    """
    Use a LOCAL Ollama model to verify the report.

    No API key.
    No Claude.
    No OpenAI.
    No Gemini.
    No API quota.
    """

    findings_json = json.dumps(
        case["findings"],
        indent=2
    )

    report = case["report"]


    prompt = f"""
You are GroundGuard, a strict verification system for AI-generated
chest X-ray reports.

Your task is to compare a generated report against STRUCTURED FINDINGS.

The structured findings are the ONLY source of truth.

====================================================
VALID FINDING LABELS
====================================================

The only valid finding labels are:

- cardiomegaly
- opacity
- pleural_effusion
- pneumothorax
- atelectasis
- pulmonary_edema
- consolidation
- fracture
- support_devices

These labels may appear in the report using natural language.

Use the following equivalences:

cardiomegaly
= enlarged cardiac silhouette
= enlarged heart
= severe/moderate/mild cardiomegaly

opacity
= pulmonary opacity
= increased density in the lung
= ill-defined lung opacity

pleural_effusion
= pleural fluid
= blunting of the costophrenic angle
= fluid in the pleural space

pneumothorax
= pleural line with absent peripheral lung markings
= pneumothorax

atelectasis
= volume loss
= atelectatic change

pulmonary_edema
= pulmonary edema
= interstitial edema
= increased interstitial markings suggesting fluid overload

consolidation
= airspace consolidation
= dense airspace opacity
= airspace disease

fracture
= fracture
= cortical discontinuity

support_devices
= line/tube/device
= radiopaque support device

====================================================
NEGATIVE STATEMENTS
====================================================

These are NOT positive findings and must NOT be flagged:

- No pneumothorax.
- No pleural effusion.
- No acute findings.
- No acute cardiopulmonary abnormality.
- No significant abnormalities.
- No other significant abnormalities are identified.
- No other abnormalities are seen.
- Lungs are clear.
- Heart size is normal.
- No evidence of disease.
- Absence of pneumothorax.
- Without pleural effusion.

A negative statement does NOT add a finding.

====================================================
CHECK 1 — GROUNDEDNESS
====================================================

Look only for POSITIVE claims.

A positive finding is an ADDED_FINDING only when:

1. The report positively claims a finding, AND
2. That finding does not exist in STRUCTURED FINDINGS.

Example:

STRUCTURED FINDINGS:
[
  {{"label": "cardiomegaly", "severity": "severe"}}
]

REPORT:
"The chest X-ray shows severe cardiomegaly."

Result:
NO ERROR.

It is the same finding.

Example:

STRUCTURED FINDINGS:
[
  {{"label": "cardiomegaly", "severity": "severe"}}
]

REPORT:
"Moderate pneumothorax is present."

Result:
ADDED_FINDING.

====================================================
CHECK 2 — SEVERITY
====================================================

Compare severity only when the finding exists.

Valid severities:

- mild
- moderate
- severe

Example:

STRUCTURED:
cardiomegaly = mild

REPORT:
"Severe cardiomegaly."

Result:
SEVERITY_FLIP.

Do NOT call a severity flip when the finding is absent.
That situation is ADDED_FINDING.

====================================================
CHECK 3 — COMPLETENESS
====================================================

Every structured finding must be represented somewhere
in the report.

Natural-language equivalents count.

Example:

STRUCTURED:
cardiomegaly

REPORT:
"The cardiac silhouette is enlarged."

This is COMPLETE.

Do NOT require the exact word "cardiomegaly".

====================================================
IMPORTANT
====================================================

Statements such as:

"No significant abnormalities are seen."

are NOT findings.

They should NEVER appear in missing_findings.

Statements such as:

"No other significant abnormalities are identified."

are NOT findings.

They should NEVER appear in unsupported_claims.

The section heading "FINDINGS" is not a finding.

The section heading "IMPRESSION" is not a finding.

Do not flag ordinary report structure.

====================================================
OUTPUT
====================================================

Return ONLY valid JSON.

Use exactly this structure:

{{
  "flagged": false,
  "unsupported_claims": [],
  "missing_findings": []
}}

For an added finding:

{{
  "flagged": true,
  "unsupported_claims": [
    {{
      "sentence": "...",
      "reason": "...",
      "type": "ADDED_FINDING"
    }}
  ],
  "missing_findings": []
}}

For a severity error:

{{
  "flagged": true,
  "unsupported_claims": [
    {{
      "sentence": "...",
      "reason": "...",
      "type": "SEVERITY_FLIP"
    }}
  ],
  "missing_findings": []
}}

For an omitted finding:

{{
  "flagged": true,
  "unsupported_claims": [],
  "missing_findings": [
    {{
      "label": "...",
      "reason": "...",
      "type": "OMITTED_FINDING"
    }}
  ]
}}

If the report is correct:

{{
  "flagged": false,
  "unsupported_claims": [],
  "missing_findings": []
}}

====================================================
STRUCTURED FINDINGS
====================================================

{findings_json}

====================================================
GENERATED REPORT
====================================================

{report}
"""



    # -----------------------------------------------------
    # JSON SCHEMA
    # -----------------------------------------------------

    output_schema = {
        "type": "object",
        "properties": {
            "flagged": {
                "type": "boolean"
            },

            "unsupported_claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {
                            "type": "string"
                        },
                        "reason": {
                            "type": "string"
                        },
                        "type": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "sentence",
                        "reason",
                        "type"
                    ]
                }
            },

            "missing_findings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string"
                        },
                        "reason": {
                            "type": "string"
                        },
                        "type": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "label",
                        "reason",
                        "type"
                    ]
                }
            }
        },

        "required": [
            "flagged",
            "unsupported_claims",
            "missing_findings"
        ]
    }

    # -----------------------------------------------------
    # CALL LOCAL OLLAMA
    # -----------------------------------------------------

    response = client.chat(
        model=model,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        format=output_schema,

        options={
            "temperature": 0
        }
    )

    # -----------------------------------------------------
    # GET MODEL RESPONSE
    # -----------------------------------------------------

    text = response["message"]["content"].strip()

    # -----------------------------------------------------
    # PARSE JSON
    # -----------------------------------------------------

    try:

        parsed = json.loads(text)

    except json.JSONDecodeError as e:

        print("\n========== OLLAMA RAW RESPONSE ==========")
        print(text)
        print("=========================================\n")

        raise RuntimeError(
            "Ollama returned invalid JSON.\n"
            f"JSON error: {e}"
        )

    # -----------------------------------------------------
    # VALIDATE OUTPUT TYPES
    # -----------------------------------------------------

    unsupported = parsed.get(
        "unsupported_claims",
        []
    )

    missing = parsed.get(
        "missing_findings",
        []
    )

    if not isinstance(unsupported, list):
        unsupported = []

    if not isinstance(missing, list):
        missing = []

    parsed["unsupported_claims"] = unsupported
    parsed["missing_findings"] = missing

    # -----------------------------------------------------
    # IMPORTANT:
    # PYTHON MAKES THE FINAL SAFETY DECISION
    # -----------------------------------------------------

    # Do NOT trust the LLM's "flagged" field.
    #
    # If there is evidence of an error:
    #     flagged = True
    #
    # Otherwise:
    #     flagged = False

    parsed["flagged"] = bool(
        unsupported or missing
    )

    # -----------------------------------------------------
    # ADD CASE ID
    # -----------------------------------------------------

    parsed["case_id"] = case["case_id"]

    return parsed



# ---------------------------------------------------------
# RUN GROUNDGUARD
# ---------------------------------------------------------

def run(
    eval_set: list,
    backend: str = "heuristic",
    model: str = "llama3.2",
) -> list:

    client = None

    # -----------------------------------------------------
    # INITIALIZE OLLAMA
    # -----------------------------------------------------

    if backend == "ollama":

        import ollama

        client = ollama

        print(
            f"Using local Ollama model: {model}"
        )

    results = []

    # -----------------------------------------------------
    # PROCESS EVERY CASE
    # -----------------------------------------------------

    for index, case in enumerate(
        eval_set,
        start=1
    ):

        print(
            f"Checking case "
            f"{index}/{len(eval_set)}..."
        )

        if backend == "ollama":

            result = check_ollama(
                case,
                client,
                model,
            )

        else:

            result = check_heuristic(
                case
            )

        # -------------------------------------------------
        # ATTACH GROUND-TRUTH ANSWER KEY
        # -------------------------------------------------

        result["is_corrupted"] = (
            case["is_corrupted"]
        )

        result["true_error_type"] = (
            case["error_type"]
        )

        results.append(result)

    return results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    ap = argparse.ArgumentParser(
        description="RadVerify GroundGuard"
    )

    ap.add_argument(
        "--backend",
        choices=[
            "heuristic",
            "ollama",
        ],
        default="heuristic",
    )

    ap.add_argument(
        "--model",
        default="llama3.2",
        help="Ollama model name",
    )

    ap.add_argument(
        "--in",
        dest="inp",
        default="data/eval_set.json",
    )

    ap.add_argument(
        "--out",
        default="output/groundguard_results.json",
    )

    args = ap.parse_args()

    # -----------------------------------------------------
    # LOAD EVALUATION DATA
    # -----------------------------------------------------

    with open(
        args.inp,
        "r",
        encoding="utf-8",
    ) as f:

        eval_set = json.load(f)

    # -----------------------------------------------------
    # RUN GROUNGuard
    # -----------------------------------------------------

    results = run(
        eval_set,
        backend=args.backend,
        model=args.model,
    )

    # -----------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # -----------------------------------------------------

    Path(
        args.out
    ).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # SAVE RESULTS
    # -----------------------------------------------------

    with open(
        args.out,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    # -----------------------------------------------------
    # FINAL MESSAGE
    # -----------------------------------------------------

    flagged = sum(
        1
        for result in results
        if result["flagged"]
    )

    print()
    print(
        f"Wrote {len(results)} verdicts "
        f"({args.backend} backend)"
    )

    print(
        f"Flagged: {flagged}"
    )

    print(
        f"Safe: {len(results) - flagged}"
    )

    print(
        f"Output: {args.out}"
    )
