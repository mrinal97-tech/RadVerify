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

Negation is handled explicitly — `"no pneumothorax"` is not treated as a positive mention of pneumothorax. Negation is scoped to the clause, so in `"no pneumothorax, but small left effusion"` the effusion still counts as present. Severity and laterality (left / right / bilateral) are checked against the structured findings too.

All finding synonyms, negation cues, severity words and laterality words live in one file, `lexicon.py`, shared by the labeller, the checker and the error injector.

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
│ (balanced corruption: ADDED_FINDING, OMITTED_FINDING, SEVERITY_FLIP, NEGATION_FLIP, LATERALITY_FLIP)
▼
eval_set.json / real_eval_set.json (reports + ground-truth corruption labels)
│
▼
groundguard.py ────────► output/groundguard_results.json (flagged / not flagged, per case)
│
▼
evaluate.py ────────────► precision / recall with 95% CIs, recall per error type, false-positive causes


Two parallel data tracks exist:

- **Synthetic**: `data_gen.py` generates structured findings from scratch → `report_gen.py` writes clean reports from them → `error_injection.py` corrupts a subset.
- **Real**: `real_data_adapter.py` parses real chest X-ray reports/captions (regex-based extraction of finding, negation, severity, and laterality cues) into the same structured format → `real_error_injection.py` corrupts a subset the same way.

The real track is what the results below were run on.

---

## Results

### v1 (first run): what the false positives revealed

The first run on 360 real IU reports (60 injected errors, 300 clean) gave recall 1.00 and precision 0.64. **All 34 false positives had one cause:** the labeller and the checker used different vocabularies. The labeller counted "catheter", "pacemaker" and "sternotomy" as support devices, and matched "opacities" and a bare "effusion"; the checker only knew phrases like "support devices" and "pulmonary opacity". Breakdown: support devices 20, opacity 13, pleural effusion 7, cardiomegaly 1.

Moving every rule into one shared `lexicon.py` removed all 34. But that exposed the real limitation: on the real-data track, the structured findings are **extracted by regex from the same report the checker reads**. Once both use the same vocabulary, the benchmark mostly measures the parser agreeing with itself. The v1 error set was also unbalanced (52 added findings, 5 omissions, 3 severity flips), and the added findings were template sentences pasted at the start of the report.

### v2: balanced error set (still circular — not a quality claim)

| Error type | Caught |
|---|---|
| ADDED_FINDING (IU-style sentence inserted mid-report) | 25/25 |
| OMITTED_FINDING | 23/23 |
| SEVERITY_FLIP | 22/22 |
| NEGATION_FLIP ("no pneumothorax" → "there is pneumothorax") | 25/25 |
| LATERALITY_FLIP (left ↔ right) | 25/25 |
| False positives on 300 clean reports | 0 |

These are near-perfect **because truth and checker share a parser**. They show the pipeline is internally consistent and that each error type is wired up end to end, nothing more.

### v3: independent ground truth (hand-labelled)

`make_gold_sheet.py` samples 50 real reports for manual labelling; `evaluate_gold.py` then measures (1) how accurate the regex labeller is against human labels and (2) the checker's false-positive rate and recall when the truth does not come from its own parser.

| Metric | Value |
|---|---|
| Labeller precision / recall vs independent labels | 0.77 / 0.87 |
| Clean reports wrongly flagged (of 50) | 16 |
| Recall on ADDED_FINDING / NEGATION_FLIP | 43/43, 24/24 |

Labels were produced with Claude and reviewed by me (judgement calls are in the `notes` column of `data/gold_sheet.csv`).

**What this shows:** recall holds up against independent truth, but 32% of clean reports are falsely flagged. The causes:
- **Hedged language (~10 cases):** "atelectasis or scarring", "may reflect effusions", "possibly lingular atelectasis" are read as definite findings.
- **Missing device vocabulary (4):** PICC, NG tube, feeding tube, central line, surgical clips.
- **Missing opacity vocabulary (2):** "opacification", "infiltrates".
- **Context (1):** "evaluation for pneumothorax is limited" read as pneumothorax present.

**Next fix:** model hedging as its own state (present / absent / uncertain) instead of forcing a yes/no.

---

## Repo structure

Scripts/
├── data_gen.py # generates synthetic structured findings
├── report_gen.py # renders findings → report text (template + paraphrase)
├── lexicon.py # shared vocabulary: finding synonyms, negation, severity, laterality
├── real_data_adapter.py # parses real report captions → structured findings
├── error_injection.py # corrupts synthetic reports for eval
├── real_error_injection.py # corrupts real reports for eval
├── groundguard.py # the verification engine (heuristic + ollama backends)
├── evaluate.py # precision/recall with CIs, per-type recall, FP causes → output/metrics.json
├── make_gold_sheet.py # samples 50 real reports for hand-labelling
├── evaluate_gold.py # scores labeller and checker against hand labels
├── inspect_errors.py # inspect flagged cases manually
├── inspect_fn_cases.py # print every missed error and false alarm from the latest run
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
python real_data_adapter.py --in path/to/your/reports.csv --limit 1000 --out data/real_clean_reports.json
#   (or re-extract from the existing JSON:  python real_data_adapter.py --from-json data/real_clean_reports.json)
python real_error_injection.py --per_type 25 # → data/real_eval_set.json (balanced)
python groundguard.py --in data/real_eval_set.json --out output/groundguard_results.json
python evaluate.py
python inspect_fn_cases.py
```

**Independent evaluation (hand labels):**
```bash
python make_gold_sheet.py      # → data/gold_sheet.csv; fill in 1 where a finding is present
python evaluate_gold.py
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

- **The real-data benchmark is partly circular.** Structured findings are regex-extracted from the same report text the checker reads. Treat v1/v2 numbers as pipeline checks; only the hand-labelled evaluation (v3) measures verification quality.
- **Errors are synthetic.** Injected corruptions are cleaner than the mistakes a real report-generation model makes. The next step is testing on reports actually drafted by a model from structured findings.
- **Text-to-findings only, not image-grounded.** RadVerify checks a report against structured findings; it does not look at the image. In a real system those findings would have to come from an image model or a radiologist.
- **Phrase matching, not semantic understanding.** The heuristic backend only knows the vocabulary in `lexicon.py`; hedged language ("cannot exclude", "possible") is not modelled.
- **Small sample.** A few hundred reports from one public dataset (IU Chest X-Ray).
- **No clinical validation.** An independent proof-of-work project, not a clinical tool; not tested against radiologist-identified errors or used in any clinical setting.

## What this demonstrates

An end-to-end pipeline for the AI-reliability problem in radiology reporting: generating structured ground truth, rendering it to natural-language reports, deliberately corrupting a subset for evaluation, verifying reports against their source data with two independently-swappable backends, and measuring the result with a real confusion matrix rather than a demo. The honest finding is itself part of the point: the first run's false positives traced back to a vocabulary mismatch, which in turn showed the benchmark was grading the parser against itself. A groundedness checker's value depends on knowing exactly how it is being measured.
