# RadVerify — GroundGuard

**AI-powered radiology report consistency & groundedness verification.**

LLM-assisted radiology reporting is fast, but fluent reports can still say things the underlying structured findings don't support — or leave things out. GroundGuard is a verification layer that checks a generated report against the structured findings it was supposed to be based on, and flags it when the two disagree.

---

## The problem

A radiology report can be well-written and still be wrong in two distinct ways:

- **Unsupported claims** — the report states something (a finding, a severity, a laterality) that isn't backed by the structured findings it was generated from.
- **Omissions** — the structured findings include something the report never mentions.

Both are silent failure modes: nothing about a fluent, confident report signals that it drifted from its source data. GroundGuard checks for exactly this, automatically, before a report reaches a human reviewer.

---

## How it works

GroundGuard runs two independent checks on every report:

1. **Groundedness (precision check)** — every claim in the report must trace back to a structured finding. A mentioned finding with no corresponding structured entry is flagged as an unsupported claim.
2. **Completeness (recall check)** — every structured finding must appear in the report. A structured finding the report never mentions is flagged as an omission.

A case is flagged if either check fails.

Negation is handled explicitly — `"no pneumothorax"` is not treated as a positive mention of pneumothorax. Sentence boundaries are respected when checking for negation cues, so a negation earlier in a paragraph can't incorrectly suppress a real finding in the next sentence.

### Two backends

- **`heuristic`** (default) — deterministic, rule-based. Sentence-splits the report, checks each structured finding for a positively-mentioned phrase, and uses a negation-cue list scoped to the containing sentence. No external dependencies, fully reproducible.
- **`ollama`** — routes the same groundedness/completeness check through a local LLM (default model `llama3.2`) via [Ollama](https://ollama.com), for comparison against the heuristic backend.

### Pipeline

structured findings (synthetic or real)
│
▼
report_gen.py ──────────► generated report (template or paraphrased)
│
▼
error_injection.py / real_error_injection.py
│ (corrupts a subset of reports: ADDED_FINDING, OMITTED_FINDING, SEVERITY_FLIP)
▼
eval_set.json / real_eval_set.json (reports + ground-truth corruption labels)
│
▼
groundguard.py ────────► output/groundguard_results.json (flagged / not flagged, per case)
│
▼
evaluate.py ────────────► accuracy / precision / recall against ground truth


Two parallel data tracks exist:

- **Synthetic**: `data_gen.py` generates structured findings from scratch → `report_gen.py` writes clean reports from them → `error_injection.py` corrupts a subset.
- **Real**: `real_data_adapter.py` parses real chest X-ray reports/captions (regex-based extraction of finding, negation, severity, and laterality cues) into the same structured format → `real_error_injection.py` corrupts a subset the same way.

The real track is what the results below were run on.

---

## Results

Run via `evaluate.py` against `output/groundguard_results.json` (heuristic backend):

| Metric | Value |
|---|---|
| Total cases | 360 |
| Accuracy | 0.906 |
| Precision | 0.638 |
| Recall | 1.000 |
| True Positive | 60 |
| True Negative | 266 |
| False Positive | 34 |
| False Negative | 0 |

**Reading this**: GroundGuard catches every injected error in this eval set (recall = 1.0) — it never lets a corrupted report through unflagged. The cost is a real false-positive rate: 34 of 300 clean reports were also flagged (precision = 0.638). In a real workflow that means it's a safe first-pass filter (nothing corrupted slips past it) but not a final arbiter — flagged cases still need a human look, some of which will turn out to be false alarms.

**Error-type breakdown** of the 60 corrupted cases: 52 `ADDED_FINDING`, 5 `OMITTED_FINDING`, 3 `SEVERITY_FLIP`. The eval set skews heavily toward unsupported-claim errors; the completeness (omission) and severity-flip checks have far less evidence behind them and should be read as less validated than the headline numbers suggest.

---

## Repo structure

Scripts/
├── data_gen.py # generates synthetic structured findings
├── report_gen.py # renders findings → report text (template + paraphrase)
├── real_data_adapter.py # parses real report captions → structured findings
├── error_injection.py # corrupts synthetic reports for eval
├── real_error_injection.py # corrupts real reports for eval
├── groundguard.py # the verification engine (heuristic + ollama backends)
├── evaluate.py # computes accuracy/precision/recall from results
├── inspect_errors.py # inspect flagged cases manually
├── inspect_fn_cases.py # inspect false-negative cases manually
├── dashboard/
│ ├── app.py # Streamlit evaluation console (read-only viewer)
│ └── utils.py
├── data/ # generated + real structured findings and eval sets
├── output/
│ └── groundguard_results.json # latest run's verdicts
└── requirements.txt


---

## Setup

```bash
cd Scripts
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

`report_gen.py` and other scripts load a `.env` file (via `python-dotenv`) for any API keys used in report generation — this file is git-ignored and not included in the repo. Create your own `Scripts/.env` if you're regenerating reports through an API-backed path.

## Running the pipeline

**Synthetic track:**
```bash
python data_gen.py                          # → data/synthetic_findings.json
python report_gen.py                        # → data/clean_reports.json
python error_injection.py                   # → data/eval_set.json
python groundguard.py --in data/eval_set.json --out output/groundguard_results.json
python evaluate.py
```

**Real-data track** (requires a source CSV of real report captions):
```bash
python real_data_adapter.py --in path/to/your/reports.csv --out data/real_clean_reports.json
python real_error_injection.py               # → data/real_eval_set.json
python groundguard.py --in data/real_eval_set.json --out output/groundguard_results.json
python evaluate.py
```

**Dashboard** (read-only viewer over the latest results):
```bash
cd dashboard
streamlit run app.py
```

**Using the Ollama backend** instead of the default heuristic:
```bash
python groundguard.py --backend ollama --model llama3.2 --in data/real_eval_set.json
```
Requires [Ollama](https://ollama.com) installed locally with the chosen model pulled.

---

## Limitations

- **Precision, not just recall, matters in deployment.** A 64% precision rate means roughly one in three flags is a false alarm — fine as a triage filter, not fine as an autonomous gate.
- **Error-type coverage is uneven.** The eval set is dominated by added-finding errors (52/60); omission and severity-flip detection have much thinner evidence (5 and 3 cases respectively) and shouldn't be assumed to generalize as well as the headline numbers imply.
- **Heuristic backend is phrase-matching, not semantic understanding.** It relies on a fixed vocabulary of finding labels and negation cues (`report_gen.py` / `real_data_adapter.py`). Findings phrased in ways outside that vocabulary, or negation patterns outside the cue list, won't be caught correctly.
- **Real-data extraction is regex-based.** `real_data_adapter.py` extracts structured findings from free-text captions using pattern matching, which is itself an imperfect ground truth — any adapter error propagates into the eval set as if it were correct.
- **No clinical validation.** This is an independent proof-of-work project evaluated on synthetic corruption of report text, not a validated clinical tool, and has not been tested against real radiologist-identified errors or deployed in any clinical setting.

## What this demonstrates

An end-to-end pipeline for the AI-reliability problem in radiology reporting: generating structured ground truth, rendering it to natural-language reports, deliberately corrupting a subset for evaluation, verifying reports against their source data with two independently-swappable backends, and measuring the result with a real confusion matrix rather than a demo. The honest finding — high recall, moderate precision, uneven error-type coverage — is itself part of the point: a groundedness checker's value depends on knowing exactly where it's strong and where it isn't.
