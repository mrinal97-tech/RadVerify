"""
utils.py -- data loading and metric computation for the RadVerify dashboard.

Deliberately kept free of any Streamlit/UI code: app.py imports from here and
renders, this file only loads JSON and computes numbers. Every metric is derived
live from whatever results file is loaded -- nothing here is a hardcoded figure.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional

import pandas as pd

# Vocabulary used ONLY to recognize a label if it appears in free text (e.g. inside
# a "reason" string). This is a search aid, not an assumption that all of these
# exist in any given dataset -- extraction functions below only ever report labels
# they actually found.
KNOWN_LABELS = [
    "cardiomegaly", "opacity", "pleural_effusion", "pneumothorax",
    "atelectasis", "pulmonary_edema", "consolidation", "fracture", "support_devices",
]
_LABEL_SEARCH_TEXT = {lbl: lbl.replace("_", " ") for lbl in KNOWN_LABELS}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_results(path: Path):
    """Loads groundguard_results.json. Returns (records, error_message).
    Exactly one of the two is None."""
    if not path.exists():
        return None, f"Results file not found at `{path}`. Run the GroundGuard pipeline first, or point the sidebar at the correct file."
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        return None, f"`{path}` is not valid JSON ({e}). Check the file wasn't truncated by a partial pipeline run."
    if not isinstance(raw, list):
        return None, f"`{path}` was parsed but isn't a list of case records."
    if len(raw) == 0:
        return [], None  # valid, just empty -- caller must handle this
    return raw, None


def load_eval_set(path: Path):
    """Loads the optional richer data/real_eval_set.json (structured findings +
    generated report per case), keyed by case_id. Returns None if unavailable --
    this file is optional, the dashboard must work without it."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if not isinstance(raw, list):
        return None
    return {c.get("case_id"): c for c in raw if isinstance(c, dict) and c.get("case_id")}


# --------------------------------------------------------------------------- #
# Core classification (TP / TN / FP / FN)
# --------------------------------------------------------------------------- #

def classify(record: dict) -> str:
    """TP/TN/FP/FN from is_corrupted (ground truth) vs flagged (prediction).
    Missing fields default to False rather than crashing or being guessed."""
    actual = bool(record.get("is_corrupted", False))
    pred = bool(record.get("flagged", False))
    if actual and pred:
        return "TP"
    if actual and not pred:
        return "FN"
    if not actual and pred:
        return "FP"
    return "TN"


def dataframe(records: list) -> pd.DataFrame:
    """Builds the working dataframe, one row per case, with classification and
    true_error_type filled in defensively (missing fields never crash this)."""
    rows = []
    for r in records:
        rows.append({
            "case_id": r.get("case_id", "unknown"),
            "is_corrupted": bool(r.get("is_corrupted", False)),
            "flagged": bool(r.get("flagged", False)),
            "true_error_type": r.get("true_error_type") or ("CLEAN" if not r.get("is_corrupted") else "UNKNOWN"),
            "classification": classify(r),
            "n_unsupported": len(r.get("unsupported_claims") or []),
            "n_missing": len(r.get("missing_findings") or []),
            "unsupported_claims": r.get("unsupported_claims") or [],
            "missing_findings": r.get("missing_findings") or [],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(df: pd.DataFrame) -> dict:
    """All headline metrics, computed fresh from the dataframe. Returns zeros
    (not NaN, not a crash) on an empty dataframe."""
    if df.empty:
        keys = ["total", "corrupted", "clean", "tp", "tn", "fp", "fn",
                "accuracy", "precision", "recall", "f1", "specificity", "fpr"]
        return {k: 0 for k in keys}

    counts = df["classification"].value_counts()
    tp, tn, fp, fn = (int(counts.get(k, 0)) for k in ("TP", "TN", "FP", "FN"))
    total = len(df)
    corrupted = int(df["is_corrupted"].sum())
    clean = total - corrupted

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "total": total, "corrupted": corrupted, "clean": clean,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": f1, "specificity": specificity, "fpr": fpr,
    }


def error_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of true_error_type, whatever values are actually present -- never
    assumes CLEAN/ADDED_FINDING/OMITTED_FINDING/SEVERITY_FLIP are all there."""
    if df.empty:
        return pd.DataFrame(columns=["error_type", "count"])
    counts = df["true_error_type"].value_counts().reset_index()
    counts.columns = ["error_type", "count"]
    return counts


# --------------------------------------------------------------------------- #
# Label extraction for false-positive / false-negative breakdowns
# --------------------------------------------------------------------------- #

def _extract_label(claim: dict) -> str:
    """Best-effort label extraction from one unsupported_claims / missing_findings
    entry. Tries, in order: an explicit 'label' field; a single-quoted token in
    'reason'; any known label name appearing as a substring of 'reason' or
    'sentence'. Falls back to 'unspecified' rather than guessing."""
    if claim.get("label"):
        return str(claim["label"])

    reason = str(claim.get("reason", ""))
    m = re.search(r"'([a-zA-Z_ ]+)'", reason)
    if m:
        candidate = m.group(1).strip().lower().replace(" ", "_")
        if candidate in KNOWN_LABELS:
            return candidate

    haystack = f"{reason} {claim.get('sentence', '')}".lower()
    for label, search_text in _LABEL_SEARCH_TEXT.items():
        if search_text in haystack or label in haystack:
            return label

    return "unspecified"


def false_positive_label_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """For FP cases only: which labels appear in unsupported_claims/missing_findings.
    Only reports labels actually found in the data -- never pads with zero-count
    labels from KNOWN_LABELS."""
    fp_rows = df[df["classification"] == "FP"]
    counter = Counter()
    for _, row in fp_rows.iterrows():
        for claim in row["unsupported_claims"]:
            counter[_extract_label(claim)] += 1
        for claim in row["missing_findings"]:
            counter[_extract_label(claim)] += 1
    if not counter:
        return pd.DataFrame(columns=["label", "count"])
    out = pd.DataFrame(counter.items(), columns=["label", "count"])
    return out.sort_values("count", ascending=False).reset_index(drop=True)


def false_negative_cases(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["classification"] == "FN"].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Engineering insights (rule-based, only claims what the data supports)
# --------------------------------------------------------------------------- #

def generate_insights(metrics: dict, fp_labels: pd.DataFrame) -> list:
    """Returns a list of {'kind': 'good'|'warning'|'neutral', 'text': str}.
    Every claim here is a direct read of the computed metrics -- nothing is
    asserted that the numbers don't support."""
    insights = []
    if metrics["total"] == 0:
        return [{"kind": "neutral", "text": "No cases loaded -- nothing to report yet."}]

    if metrics["fn"] == 0 and metrics["corrupted"] > 0:
        insights.append({"kind": "good", "text": f"All {metrics['corrupted']} corrupted cases were detected -- zero false negatives."})
    elif metrics["fn"] > 0:
        insights.append({"kind": "warning", "text": f"{metrics['fn']} corrupted case(s) were missed (false negatives) -- the clinically riskier failure mode."})

    if metrics["recall"] >= 0.99 and metrics["precision"] < metrics["recall"] - 0.15:
        insights.append({"kind": "warning", "text": f"Recall ({metrics['recall']:.1%}) is substantially higher than precision ({metrics['precision']:.1%}) -- false positives, not missed errors, are the primary remaining issue."})

    if metrics["fp"] > 0:
        insights.append({"kind": "warning", "text": f"{metrics['fp']} false positive(s) out of {metrics['clean']} clean cases (FPR {metrics['fpr']:.1%})."})

    if not fp_labels.empty:
        top = fp_labels.iloc[0]
        share = top["count"] / fp_labels["count"].sum()
        insights.append({"kind": "neutral", "text": f"'{top['label']}' accounts for {int(top['count'])} of {int(fp_labels['count'].sum())} labeled false-positive mentions ({share:.0%}) -- the largest single contributor in this dataset."})

    return insights


def detection_by_error_type(df: pd.DataFrame) -> pd.DataFrame:
    """For each non-CLEAN true_error_type actually present: how many of those
    cases GroundGuard flagged. This is a per-failure-mode view -- a global
    accuracy number hides whether e.g. SEVERITY_FLIP is caught reliably while
    OMITTED_FINDING is not."""
    corrupted = df[df["true_error_type"] != "CLEAN"]
    if corrupted.empty:
        return pd.DataFrame(columns=["error_type", "detected", "total", "rate"])
    rows = []
    for et in sorted(corrupted["true_error_type"].unique()):
        subset = corrupted[corrupted["true_error_type"] == et]
        total = len(subset)
        detected = int(subset["flagged"].sum())
        rows.append({"error_type": et, "detected": detected, "total": total,
                      "rate": detected / total if total else 0.0})
    return pd.DataFrame(rows)


def parse_severity_flip(reason: str):
    """Extracts (found_severity, expected_severity) from a SEVERITY_FLIP reason
    string such as: severity 'severe' contradicts finding severity 'mild'.
    Returns (None, None) if the format doesn't match -- never guesses."""
    m = re.findall(r"'([a-zA-Z]+)'", reason or "")
    if len(m) >= 2:
        return m[0], m[1]
    return None, None


def recommended_investigation(fp_labels: pd.DataFrame) -> str:
    if fp_labels.empty:
        return "No labeled false positives to investigate in the current dataset."
    top = fp_labels.iloc[0]
    return (
        f"Start with cases where GroundGuard's false positives involve '{top['label']}' "
        f"({int(top['count'])} occurrences) -- inspect those cases in the Case Explorer to "
        f"see whether the checker is over-triggering on phrasing variants, negation, or "
        f"a genuine ambiguity in how that finding is described."
    )
