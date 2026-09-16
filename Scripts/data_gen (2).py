import json
import random
from pathlib import Path

random.seed(42)

FINDING_LABELS = [
    "cardiomegaly", "opacity", "pleural_effusion", "pneumothorax",
    "atelectasis", "pulmonary_edema", "consolidation", "fracture", "support_devices",
]
SEVERITIES = ["mild", "moderate", "severe"]
LATERALITIES = ["left", "right", "bilateral", None]

def make_case(case_id:int)->dict:
    n_findings = random.choices([0, 1, 2, 3], weights=[0.25, 0.4, 0.25, 0.1])[0]
    chosen = random.sample(FINDING_LABELS, k=n_findings)

    findings = []
    for label in chosen:
        needs_laterality = label in {"pleural_effusion", "pneumothorax", "atelectasis"}
        findings.append({
            "label": label,
            "present": True,
            "severity": random.choice(SEVERITIES),
            "laterality": random.choice(LATERALITIES) if needs_laterality else None,
        })
    return {
        "case_id": f"case_{case_id:04d}",
        "findings": findings,
        "patient_context": {
            "age": random.randint(18, 90),
            "sex": random.choice(["M", "F"]),
            "indication": random.choice([
                "shortness of breath", "chest pain", "post-op check",
                "fever, rule out pneumonia", "routine pre-op",
            ]),
        },
    }
def generate(n_cases: int = 60, out_path: str = "data/synthetic_findings.json"):
    cases = [make_case(i) for i in range(n_cases)]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cases, open(out_path, "w"), indent=2)
    print(f"Wrote {len(cases)} cases -> {out_path}")
    return cases


if __name__ == "__main__":
    generate()
