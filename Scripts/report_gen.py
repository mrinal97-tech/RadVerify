import argparse, json,os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEVERITY_PHRASE = {"mild": "mild", "moderate": "moderate", "severe": "severe"}

LABEL_PHRASE = {
    "cardiomegaly": "cardiomegaly", "opacity": "pulmonary opacity",
    "pleural_effusion": "pleural effusion", "pneumothorax": "pneumothorax",
    "atelectasis": "atelectasis", "pulmonary_edema": "pulmonary edema",
    "consolidation": "consolidation", "fracture": "a fracture",
    "support_devices": "support devices",
}

def _findings_to_sentence(f: dict) -> str:
    label = LABEL_PHRASE[f["label"]]
    sev = SEVERITY_PHRASE[f["severity"]]
    lat = f.get("laterality")
    lat_str = f"{lat} " if lat and lat != "bilateral" else ("bilateral " if lat == "bilateral" else "")
    return f"There is {sev} {lat_str}{label}."

PARAPHRASES = {
    "cardiomegaly": "The cardiac silhouette is enlarged.",
    "opacity": "There is an ill-defined area of increased density in the lung field.",
    "pleural_effusion": "Blunting of the costophrenic angle is noted, consistent with fluid.",
    "pneumothorax": "A visceral pleural line is seen with absent lung markings peripherally.",
    "atelectasis": "There is volume loss with associated linear scarring.",
    "pulmonary_edema": "Increased interstitial markings are present, suggestive of fluid overload.",
    "consolidation": "A dense area of airspace disease is identified.",
    "fracture": "A cortical discontinuity is seen.",
    "support_devices": "A radiopaque line/tube is present in the expected position.",
}

def write_template_report(case: dict, noisy: bool = False) -> str:
    findings = case["findings"]
    if not findings:
        body = "No acute cardiopulmonary abnormality is identified. Heart size is normal. Lungs are clear."
        impression = "No acute findings."
    else:
        body = " ".join(
            (PARAPHRASES if noisy else {}).get(f["label"], "") or _findings_to_sentence(f)
            for f in findings
        ) if not noisy else " ".join(PARAPHRASES[f["label"]] for f in findings)
        impression = "; ".join(f"{f['severity']} {LABEL_PHRASE[f['label']]}" for f in findings)
    return f"FINDINGS: {body}\nIMPRESSION: {impression}."



def write_llm_report(case: dict) -> str:
    import ollama

    findings_json = json.dumps(case["findings"], indent=2)

    prompt = f"""You are simulating a radiology report-generation model.

Given ONLY the structured findings below, write a standard two-section
chest X-ray report:

FINDINGS: ...
IMPRESSION: ...

Rules:
- Do NOT add any finding that is not present.
- Do NOT add severity that is not present.
- Do NOT add laterality that is not present.
- Do NOT omit any finding that IS present.
- Use professional radiology-report language.
- Output ONLY the report text.
- Do NOT include a preamble or explanation.

Structured findings:
{findings_json}
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()

def generate_reports(cases: list, mode: str = "template") -> list:

    out = []

    for case in cases:

        if mode == "llm":
            report = write_llm_report(case)
        else:
            report = write_template_report(case)

        out.append({
            **case,
            "report": report
        })

    return out



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["llm", "template"], default="template")
    ap.add_argument("--in", dest="inp", default="data/synthetic_findings.json")
    ap.add_argument("--out", default="data/clean_reports.json")
    args = ap.parse_args()
    cases = json.load(open(args.inp))
    reports = generate_reports(cases, mode=args.mode)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(reports, open(args.out, "w"), indent=2)
    print(f"Wrote {len(reports)} reports ({args.mode} mode) -> {args.out}")