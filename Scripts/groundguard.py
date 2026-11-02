"""
groundguard.py

The verification layer for RadVerify.

Runs TWO checks:
  1. GROUNDEDNESS (precision)
     - Every positive claim in the report must trace back to a structured
       finding, with matching severity and laterality.

  2. COMPLETENESS (recall)
     - Every structured finding must be positively mentioned in the report.

A case is flagged if either check fails.

Vocabulary, negation, severity and laterality rules all come from lexicon.py,
shared with the labeller and the error injector.

Backends:
  - heuristic : deterministic rule-based checker
  - ollama    : local Ollama LLM checker
"""

import argparse
import json
from pathlib import Path

import lexicon


# ---------------------------------------------------------
# HEURISTIC CHECKER
# ---------------------------------------------------------

def check_heuristic(case: dict) -> dict:

    report = case["report"]
    findings_by_label = {f["label"]: f for f in case["findings"]}

    unsupported = []
    mentioned_labels = set()

    # -----------------------------------------------------
    # GROUNDEDNESS: every positive claim must be supported
    # -----------------------------------------------------

    for sent in lexicon.split_sentences(report):

        for label in lexicon.FINDING_PATTERNS:

            spans = lexicon.positive_mentions(sent, label)
            if not spans:
                continue

            start, end = spans[0]
            mentioned_labels.add(label)
            gt = findings_by_label.get(label)

            # Finding claimed but not in structured findings
            if gt is None:
                unsupported.append({
                    "sentence": sent,
                    "reason": f"'{label}' not in structured findings",
                    "type": "ADDED_FINDING",
                })
                continue

            # Severity must match when both sides specify it
            sev = lexicon.severity_at(sent, start, end)
            gt_sev = gt.get("severity")
            if sev != "unspecified" and gt_sev not in (None, "unspecified") and sev != gt_sev:
                unsupported.append({
                    "sentence": sent,
                    "reason": f"severity '{sev}' contradicts finding severity '{gt_sev}'",
                    "type": "SEVERITY_FLIP",
                })

            # Laterality: flag only a clear left/right contradiction.
            # A bilateral finding is consistent with either side being mentioned.
            lat = lexicon.laterality_at(sent, start, end)
            gt_lat = gt.get("laterality")
            if lat in ("left", "right") and gt_lat in ("left", "right") and lat != gt_lat:
                unsupported.append({
                    "sentence": sent,
                    "reason": f"laterality '{lat}' contradicts finding laterality '{gt_lat}'",
                    "type": "LATERALITY_FLIP",
                })

    # -----------------------------------------------------
    # COMPLETENESS: every structured finding must appear
    # -----------------------------------------------------

    missing = [
        {
            "label": label,
            "reason": "structured finding not positively mentioned in report",
            "type": "OMITTED_FINDING",
        }
        for label in findings_by_label
        if label not in mentioned_labels
    ]

    return {
        "case_id": case["case_id"],
        "flagged": bool(unsupported or missing),
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
= catheter
= pacemaker / AICD
= sternotomy wires
= drainage tube

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
CHECK 2b — LATERALITY
====================================================

If a structured finding has a laterality (left / right / bilateral)
and the report states a DIFFERENT side for that finding,
report it in unsupported_claims with type "LATERALITY_FLIP".

Example:

STRUCTURED:
pleural_effusion, laterality = left

REPORT:
"Small right pleural effusion."

Result:
LATERALITY_FLIP.

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
