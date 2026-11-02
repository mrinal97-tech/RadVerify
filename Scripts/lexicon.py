"""
lexicon.py

ONE shared vocabulary for RadVerify.

Before this file existed, the labeller (real_data_adapter.py) and the checker
(groundguard.py) each had their own finding phrases. The labeller knew that
"catheter" or "pacemaker" means support devices and that "opacities" or a bare
"effusion" count; the checker did not. All 34 false positives in the first
evaluation came from that mismatch, not from real report errors.

Every module now imports its finding patterns, negation rules, severity cues
and laterality cues from here.
"""

import re

# ---------------------------------------------------------
# FINDINGS: label -> regex patterns (matched on lowercase text)
# ---------------------------------------------------------

FINDING_PATTERNS = {
    "cardiomegaly": [
        r"cardiomegaly",
        r"heart (?:is |size )?enlarged",
        r"enlarged (?:cardiac|heart)",
        r"cardiac silhouette is enlarged",
    ],
    "opacity": [
        r"opacit(?:y|ies)",
        r"increased density",
    ],
    "pleural_effusion": [
        r"pleural effusions?",
        r"\beffusions?\b",
        r"pleural fluid",
        r"blunting of the costophrenic",
    ],
    "pneumothorax": [
        r"pneumothora(?:x|ces)",
        r"visceral pleural line",
    ],
    "atelectasis": [
        r"atelectasis",
        r"atelectatic",
        r"volume loss",
    ],
    "pulmonary_edema": [
        r"pulmonary edema",
        r"\bedema\b",
        r"fluid overload",
    ],
    "consolidation": [
        r"consolidation",
        r"airspace disease",
    ],
    "fracture": [
        r"fractures?",
        r"cortical discontinuity",
    ],
    "support_devices": [
        r"support devices?",
        r"\bcatheter\b",
        r"\bpacemaker\b",
        r"\bsternotomy\b",
        r"\baicd\b",
        r"\bdrainage tube\b",
        r"line/tube",
    ],
}

# Phrase used when an injector has to WRITE a finding in IU-report style.
WRITE_PHRASE = {
    "cardiomegaly": "cardiomegaly",
    "opacity": "opacity",
    "pleural_effusion": "pleural effusion",
    "pneumothorax": "pneumothorax",
    "atelectasis": "atelectasis",
    "pulmonary_edema": "pulmonary edema",
    "consolidation": "airspace consolidation",
    "fracture": "rib fracture",
    "support_devices": "pacemaker",
}

# ---------------------------------------------------------
# NEGATION
# ---------------------------------------------------------

# Cues that negate a finding when they appear BEFORE it in the same clause.
NEGATION_BEFORE = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bfree of\b",
    r"\bclear of\b",
    r"\babsence of\b",
    r"\brule out\b",
    r"\bresolved\b",
    r"\bnot\b",
]

# Cues that negate a finding when they appear AFTER it in the same clause,
# e.g. "pleural effusion is not seen", "the pneumothorax has resolved".
NEGATION_AFTER = [
    r"\b(?:is|are) not (?:seen|identified|present|visualized)\b",
    r"\bnot seen\b",
    r"\b(?:has|have) resolved\b",
]

# Words that end a clause, so "no pneumothorax, but small left effusion"
# does not negate the effusion.
CLAUSE_BREAK = re.compile(r",\s*but\b|;|\bhowever\b|\balthough\b|\bwhereas\b")

# ---------------------------------------------------------
# SEVERITY AND LATERALITY
# ---------------------------------------------------------

SEVERITY_CUES = {
    "mild": ["mild", "minimal", "small", "trace", "tiny"],
    "moderate": ["moderate"],
    "severe": ["severe", "large", "extensive", "complete", "significant"],
}

LATERALITY_CUES = {
    "bilateral": [r"\bbilateral(?:ly)?\b", r"\bboth\b"],
    "left": [r"\bleft\b"],
    "right": [r"\bright\b"],
}

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------


def split_sentences(text: str) -> list:
    """Split a report into sentences, dropping FINDINGS:/IMPRESSION: headings."""
    body = re.sub(r"\b(?:FINDINGS|IMPRESSION):\s*", " ", text, flags=re.IGNORECASE)
    parts = re.split(r"(?<=[.!?])\s+|\n+", body)
    return [p.strip() for p in parts if p.strip()]


def clause_bounds(sentence: str, start: int, end: int) -> tuple:
    """Return (clause_start, clause_end) of the clause containing [start, end)."""
    low = sentence.lower()
    c_start = 0
    for m in CLAUSE_BREAK.finditer(low, 0, start):
        c_start = m.end()
    nxt = CLAUSE_BREAK.search(low, end)
    c_end = nxt.start() if nxt else len(low)
    return c_start, c_end


def is_negated(sentence: str, start: int, end: int) -> bool:
    """True if the mention at [start, end) is negated within its clause."""
    low = sentence.lower()
    c_start, c_end = clause_bounds(low, start, end)
    before = low[c_start:start]
    after = low[end:c_end]
    if any(re.search(p, before) for p in NEGATION_BEFORE):
        return True
    if any(re.search(p, after) for p in NEGATION_AFTER):
        return True
    return False


def mentions(sentence: str, label: str) -> list:
    """All (start, end) matches of a label in a sentence, negated or not."""
    low = sentence.lower()
    spans = set()
    for pat in FINDING_PATTERNS[label]:
        for m in re.finditer(pat, low):
            spans.add((m.start(), m.end()))
    return sorted(spans)


def positive_mentions(sentence: str, label: str) -> list:
    """Non-negated (start, end) matches of a label in a sentence."""
    return [(s, e) for s, e in mentions(sentence, label) if not is_negated(sentence, s, e)]


def is_positively_mentioned(report: str, label: str) -> bool:
    return any(positive_mentions(s, label) for s in split_sentences(report))


def severity_at(sentence: str, start: int, end: int) -> str:
    """Severity word in the same clause as the mention, else 'unspecified'."""
    low = sentence.lower()
    c_start, c_end = clause_bounds(low, start, end)
    clause = low[c_start:c_end]
    for sev, cues in SEVERITY_CUES.items():
        if any(re.search(rf"\b{re.escape(c)}\b", clause) for c in cues):
            return sev
    return "unspecified"


def laterality_at(sentence: str, start: int, end: int):
    """'left', 'right', 'bilateral' or None, from the mention's clause."""
    low = sentence.lower()
    c_start, c_end = clause_bounds(low, start, end)
    clause = low[c_start:c_end]
    found = {
        side for side, pats in LATERALITY_CUES.items()
        if any(re.search(p, clause) for p in pats)
    }
    if "bilateral" in found or {"left", "right"} <= found:
        return "bilateral"
    if found:
        return found.pop()
    return None
