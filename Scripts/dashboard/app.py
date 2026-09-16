"""
app.py -- RadVerify evaluation dashboard.

Read-only visualization/observability layer over the existing GroundGuard
pipeline. Never modifies groundguard.py, evaluate.py, real_error_injection.py,
real_data_adapter.py, report_gen.py, or any JSON file it reads -- it only loads
output/groundguard_results.json (and optionally data/real_eval_set.json) and
computes everything else at render time.

Run with: streamlit run app.py
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

import utils

# --------------------------------------------------------------------------- #
# Page config
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="RadVerify | Evaluation Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Design tokens -- restrained, semantic-only color use (no rainbow charts)
# --------------------------------------------------------------------------- #

BG = "#0a0a0f"
SURFACE = "#111218"
SURFACE_ALT = "#15161d"
BORDER = "#1f212b"
TEXT = "#e4e6ea"
TEXT_MUTED = "#8a8d98"
ACCENT = "#6366f1"      # indigo -- primary system accent
GOOD = "#22c55e"
WARN = "#eab308"
BAD = "#ef4444"

st.markdown(f"""
<style>
    .stApp {{ background-color: {BG}; }}
    html, body, [class*="css"] {{
        font-family: -apple-system, BlinkMacSystemFont, "Inter", system-ui, sans-serif;
    }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    .rv-header {{ padding: 4px 0 20px 0; border-bottom: 1px solid {BORDER}; margin-bottom: 22px; }}
    .rv-eyebrow {{
        font-size: 11px; font-weight: 600; color: {TEXT_MUTED}; letter-spacing: 0.08em;
        text-transform: uppercase; margin-bottom: 4px;
    }}
    .rv-title {{ font-size: 24px; font-weight: 600; color: {TEXT}; letter-spacing: -0.01em; margin: 0; }}
    .rv-subtitle {{ font-size: 13.5px; color: {TEXT_MUTED}; margin-top: 3px; }}
    .rv-status {{
        display: inline-flex; align-items: center; gap: 6px; font-size: 11.5px;
        font-weight: 600; letter-spacing: 0.04em; color: {GOOD}; margin-top: 10px;
    }}
    .rv-status-dot {{ width: 6px; height: 6px; border-radius: 50%; background: {GOOD}; }}

    div[data-testid="stMetric"] {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 10px;
        padding: 14px 16px 10px 16px; transition: border-color 150ms ease;
    }}
    div[data-testid="stMetric"]:hover {{ border-color: #2a2d3a; }}
    div[data-testid="stMetricLabel"] {{
        color: {TEXT_MUTED} !important; font-size: 11.5px !important;
        text-transform: uppercase; letter-spacing: 0.05em;
    }}
    div[data-testid="stMetricValue"] {{ color: {TEXT} !important; font-size: 23px !important; font-weight: 600; }}

    .rv-badge {{
        display: inline-block; padding: 2px 9px; border-radius: 4px;
        font-size: 11.5px; font-weight: 600; letter-spacing: 0.01em;
    }}
    .rv-badge-good {{ background: rgba(34,197,94,0.10); color: {GOOD}; border: 1px solid rgba(34,197,94,0.25); }}
    .rv-badge-warn {{ background: rgba(234,179,8,0.10); color: {WARN}; border: 1px solid rgba(234,179,8,0.25); }}
    .rv-badge-bad {{ background: rgba(239,68,68,0.10); color: {BAD}; border: 1px solid rgba(239,68,68,0.25); }}
    .rv-badge-muted {{ background: rgba(138,141,152,0.10); color: {TEXT_MUTED}; border: 1px solid rgba(138,141,152,0.2); }}
    .rv-badge-accent {{ background: rgba(99,102,241,0.10); color: {ACCENT}; border: 1px solid rgba(99,102,241,0.25); }}

    .rv-callout {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-left: 2px solid {TEXT_MUTED};
        border-radius: 6px; padding: 11px 14px; margin: 5px 0; font-size: 13.5px; color: {TEXT};
    }}
    .rv-insight-good {{ border-left-color: {GOOD}; }}
    .rv-insight-warn {{ border-left-color: {WARN}; }}
    .rv-insight-neutral {{ border-left-color: {TEXT_MUTED}; }}

    .rv-report-box {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px;
        padding: 16px 18px; font-size: 13.5px; color: {TEXT}; line-height: 1.65;
        white-space: pre-wrap;
    }}
    .rv-report-box b {{ color: {ACCENT}; }}

    .rv-section-label {{
        font-size: 12px; font-weight: 600; color: {TEXT_MUTED}; text-transform: uppercase;
        letter-spacing: 0.05em; margin: 20px 0 8px 0;
    }}

    .rv-flow-row {{ display: flex; align-items: center; flex-wrap: wrap; gap: 0; margin: 6px 0; }}
    .rv-flow-step {{
        background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 6px;
        padding: 8px 14px; font-size: 12.5px; color: {TEXT}; white-space: nowrap;
    }}
    .rv-flow-arrow {{ color: {TEXT_MUTED}; padding: 0 8px; font-size: 13px; }}

    .rv-footer {{
        margin-top: 40px; padding-top: 16px; border-top: 1px solid {BORDER};
        color: {TEXT_MUTED}; font-size: 12px; text-align: center; line-height: 1.7;
    }}

    section[data-testid="stSidebar"] {{ background: #0d0e13; border-right: 1px solid {BORDER}; }}
    div[data-testid="stExpander"] {{ background: {SURFACE}; border: 1px solid {BORDER}; border-radius: 8px; }}
</style>
""", unsafe_allow_html=True)


def badge(text: str, kind: str = "muted") -> str:
    return f'<span class="rv-badge rv-badge-{kind}">{text}</span>'


def callout(text: str, kind: str = "neutral") -> str:
    icon = {"good": "\u2713", "warn": "\u26a0", "neutral": "\u2022"}.get(kind, "\u2022")
    return f'<div class="rv-callout rv-insight-{kind}">{icon}&nbsp;&nbsp;{text}</div>'


# --------------------------------------------------------------------------- #
# Cached data loading -- keyed on path + mtime, so editing/regenerating the
# source JSON invalidates the cache automatically, and repeated widget
# interactions never re-read disk.
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def _cached_results(path_str: str, _mtime: float):
    return utils.load_results(Path(path_str))


@st.cache_data(show_spinner=False)
def _cached_eval_set(path_str: str, _mtime: float):
    return utils.load_eval_set(Path(path_str))


def get_results(path: Path):
    """Not-found is a valid, expected state (wrong path, pipeline not yet run),
    so it's produced fresh every time rather than cached -- only a successful
    read is cached, keyed on the file's own mtime."""
    if not path.exists():
        return utils.load_results(path)
    return _cached_results(str(path), path.stat().st_mtime)


def get_eval_set(path: Path):
    if not path.exists():
        return None
    return _cached_eval_set(str(path), path.stat().st_mtime)


# BASE_DIR anchors default paths to this file's location rather than the
# directory the user happened to launch `streamlit run` from -- app.py lives in
# dashboard/, and output/ + data/ are its siblings one level up, so parent.parent
# from app.py's own resolved path lands on the project root regardless of cwd.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = BASE_DIR / "output" / "groundguard_results.json"
DEFAULT_EVAL_SET = BASE_DIR / "data" / "real_eval_set.json"


# --------------------------------------------------------------------------- #
# Render functions
# --------------------------------------------------------------------------- #

def render_header(n_shown: int, n_total: int):
    st.markdown(f'''
    <div class="rv-header">
        <p class="rv-eyebrow">RadVerify</p>
        <p class="rv-title">Evaluation Console</p>
        <p class="rv-subtitle">AI-generated radiology report verification &mdash; engineering evaluation &amp; error analysis
        &nbsp;&middot;&nbsp; {n_shown} of {n_total} cases shown</p>
        <div class="rv-status"><span class="rv-status-dot"></span>SYSTEM STATUS &middot; EVALUATION READY</div>
    </div>
    ''', unsafe_allow_html=True)


def render_kpi_cards(metrics: dict):
    if metrics["recall"] >= 0.999 and metrics["fn"] == 0:
        c1, _ = st.columns([1, 6])
        with c1:
            st.markdown(badge("100% Recall", "good"), unsafe_allow_html=True)
        st.write("")

    r1 = st.columns(4)
    r1[0].metric("Total Cases", metrics["total"])
    r1[1].metric("Corrupted Cases", metrics["corrupted"])
    r1[2].metric("Clean Cases", metrics["clean"])
    r1[3].metric("Accuracy", f"{metrics['accuracy']:.1%}")

    r2 = st.columns(4)
    r2[0].metric("Precision", f"{metrics['precision']:.1%}")
    r2[1].metric("Recall", f"{metrics['recall']:.1%}")
    r2[2].metric("False Positives", metrics["fp"])
    r2[3].metric("False Negatives", metrics["fn"])


def render_confusion_matrix(metrics: dict):
    st.markdown('<p class="rv-section-label">Confusion Matrix</p>', unsafe_allow_html=True)
    z = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=["Predicted Clean", "Predicted Corrupted"],
        y=["Actual Clean", "Actual Corrupted"],
        text=z, texttemplate="%{text}", textfont={"size": 19, "color": TEXT},
        colorscale=[[0, SURFACE_ALT], [1, ACCENT]],
        showscale=False, xgap=2, ygap=2,
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12), height=320,
        margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, width="stretch")


def render_prediction_breakdown(metrics: dict):
    st.markdown('<p class="rv-section-label">Prediction Breakdown</p>', unsafe_allow_html=True)
    pred_df = pd.DataFrame([
        {"Category": "TP", "Label": "Correctly flagged corrupt", "Count": metrics["tp"], "Kind": "TP"},
        {"Category": "TN", "Label": "Correctly accepted clean", "Count": metrics["tn"], "Kind": "TN"},
        {"Category": "FP", "Label": "Incorrectly flagged clean", "Count": metrics["fp"], "Kind": "FP"},
        {"Category": "FN", "Label": "Missed corrupt", "Count": metrics["fn"], "Kind": "FN"},
    ])
    color_map = {"TP": GOOD, "TN": ACCENT, "FP": WARN, "FN": BAD}
    fig = px.bar(pred_df, x="Count", y="Label", orientation="h", color="Kind",
                 color_discrete_map=color_map, text="Count")
    fig.update_traces(textposition="outside")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12), height=320, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")


def render_error_distribution(df: pd.DataFrame):
    st.markdown('<p class="rv-section-label">Error Type Distribution</p>', unsafe_allow_html=True)
    dist = utils.error_type_distribution(df)
    total = dist["count"].sum()
    dist["pct"] = dist["count"] / total if total else 0
    fig = px.bar(dist, x="error_type", y="count", custom_data=["pct"],
                 color_discrete_sequence=[ACCENT])
    fig.update_traces(hovertemplate="%{x}: %{y} cases (%{customdata[0]:.1%})<extra></extra>")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12), height=300, showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Cases",
    )
    st.plotly_chart(fig, width="stretch")


def render_pipeline_flow():
    st.markdown('<p class="rv-section-label">Evaluation Pipeline (observed, not executed by this dashboard)</p>', unsafe_allow_html=True)
    row1 = ["Real Reports", "Real Data Adapter", "Structured Findings", "Error Injection", "Evaluation Set"]
    row2 = ["GroundGuard", "Verdict", "Evaluation Metrics", "Dashboard"]

    def flow_html(steps):
        parts = []
        for i, step in enumerate(steps):
            parts.append(f'<span class="rv-flow-step">{step}</span>')
            if i < len(steps) - 1:
                parts.append('<span class="rv-flow-arrow">&rarr;</span>')
        return f'<div class="rv-flow-row">{"".join(parts)}</div>'

    st.markdown(flow_html(row1), unsafe_allow_html=True)
    st.markdown(flow_html(row2), unsafe_allow_html=True)
    st.caption("This dashboard reads the outputs of this pipeline. It does not rerun GroundGuard, regenerate reports, or inject new errors.")


def render_system_performance(df: pd.DataFrame):
    st.markdown('<p class="rv-section-label">System Performance by Error Type</p>', unsafe_allow_html=True)
    det = utils.detection_by_error_type(df)
    if det.empty:
        st.markdown(callout("No corrupted cases in the current filter selection.", "neutral"), unsafe_allow_html=True)
        return
    cols = st.columns(len(det))
    for col, (_, row) in zip(cols, det.iterrows()):
        with col:
            st.markdown(f"**{row['error_type']}**")
            st.markdown(f"Detected: **{row['detected']} / {row['total']}**")
            kind = "good" if row["rate"] >= 0.99 else ("warn" if row["rate"] >= 0.6 else "bad")
            st.markdown(badge(f"{row['rate']:.0%}", kind), unsafe_allow_html=True)


def render_error_analysis(df: pd.DataFrame, df_all: pd.DataFrame, records: list, metrics: dict):
    fp_labels = utils.false_positive_label_breakdown(df)

    st.markdown('<p class="rv-section-label">False Positive Rate</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("False Positives", metrics["fp"])
    c2.metric("False Positive Rate", f"{metrics['fpr']:.1%}")
    c3.metric("Clean Cases Evaluated", metrics["clean"])
    st.caption(f"{metrics['fp']} / {metrics['clean']} clean cases were incorrectly flagged.")

    st.write("")
    st.markdown('<p class="rv-section-label">False Positives by Finding Label</p>', unsafe_allow_html=True)
    if fp_labels.empty:
        st.markdown(callout("No labeled false positives in the current selection.", "neutral"), unsafe_allow_html=True)
    else:
        label_filter = st.multiselect("Filter labels", fp_labels["label"].tolist(), default=fp_labels["label"].tolist())
        shown = fp_labels[fp_labels["label"].isin(label_filter)] if label_filter else fp_labels
        fig = px.bar(shown, x="label", y="count", color_discrete_sequence=[WARN])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_MUTED, size=12), height=300,
            margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="Mentions",
        )
        st.plotly_chart(fig, width="stretch")

    st.write("")
    st.markdown('<p class="rv-section-label">False Negative Analysis</p>', unsafe_allow_html=True)
    if metrics["fn"] == 0:
        st.markdown(callout("<b>No false negatives.</b> All corrupted evaluation cases were detected in the currently loaded evaluation set.", "good"), unsafe_allow_html=True)
    else:
        fn_df = utils.false_negative_cases(df)
        sel = st.selectbox("Inspect a false negative case", fn_df["case_id"].tolist(), key="fn_select")
        if sel:
            record = next((r for r in records if r.get("case_id") == sel), {})
            row = df_all[df_all["case_id"] == sel].iloc[0]
            m1, m2 = st.columns(2)
            m1.markdown(f"**True error type**<br>{badge(row['true_error_type'], 'muted')}", unsafe_allow_html=True)
            m2.markdown(f"**GroundGuard verdict**<br>{badge('SAFE (missed)', 'bad')}", unsafe_allow_html=True)
            st.markdown("**Unsupported claims**")
            st.json(record.get("unsupported_claims") or [])
            st.markdown("**Missing findings**")
            st.json(record.get("missing_findings") or [])

    return fp_labels


def render_case_explorer(df: pd.DataFrame, df_all: pd.DataFrame, records: list, eval_set):
    search_col, dd_col = st.columns([1, 2])
    with search_col:
        search_term = st.text_input("Search by case ID", value="")
    explorer_df = df
    if search_term:
        explorer_df = explorer_df[explorer_df["case_id"].str.contains(search_term, case=False, na=False)]

    with dd_col:
        if explorer_df.empty:
            st.warning("No cases match the current filters/search.")
            selected_id = None
        else:
            selected_id = st.selectbox("Select a case", explorer_df["case_id"].tolist())

    if not selected_id:
        return

    row = df_all[df_all["case_id"] == selected_id].iloc[0]
    record = next((r for r in records if r.get("case_id") == selected_id), {})

    st.write("")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f"**Case ID**<br><code>{row['case_id']}</code>", unsafe_allow_html=True)
    m2.markdown(f"**Ground truth**<br>{badge('CORRUPTED', 'warn') if row['is_corrupted'] else badge('CLEAN', 'good')}", unsafe_allow_html=True)
    m3.markdown(f"**True error type**<br>{badge(row['true_error_type'], 'muted')}", unsafe_allow_html=True)
    cls_kind = {"TP": "good", "TN": "good", "FP": "bad", "FN": "bad"}[row["classification"]]
    m4.markdown(f"**Classification**<br>{badge(row['classification'], cls_kind)}", unsafe_allow_html=True)

    gg_kind = "warn" if row["flagged"] else "good"
    st.markdown(f"**GroundGuard verdict:** {badge('FLAGGED' if row['flagged'] else 'SAFE', gg_kind)}", unsafe_allow_html=True)

    st.write("")
    ev_case = eval_set.get(selected_id) if eval_set else None

    if ev_case:
        st.markdown('<p class="rv-section-label">Structured Findings</p>', unsafe_allow_html=True)
        findings = ev_case.get("findings", [])
        if findings:
            fdf = pd.DataFrame(findings)
            for col in ["label", "present", "severity", "laterality", "source_sentence"]:
                if col not in fdf.columns:
                    fdf[col] = None
            st.dataframe(fdf[["label", "present", "severity", "laterality", "source_sentence"]],
                         width="stretch", hide_index=True)
        else:
            st.markdown(callout("No positive findings for this case.", "neutral"), unsafe_allow_html=True)

        st.markdown('<p class="rv-section-label">Report</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="rv-report-box">{ev_case.get("report", "(no report text available)")}</div>', unsafe_allow_html=True)
    else:
        st.markdown(callout("Structured findings / report text unavailable for this case &mdash; load the evaluation set JSON in the sidebar for richer inspection.", "neutral"), unsafe_allow_html=True)

    st.write("")
    st.markdown('<p class="rv-section-label">GroundGuard Analysis</p>', unsafe_allow_html=True)
    uc = record.get("unsupported_claims") or []
    mf = record.get("missing_findings") or []
    if not uc and not mf:
        st.markdown(callout("No errors &mdash; grounded and complete.", "good"), unsafe_allow_html=True)
    else:
        for claim in uc:
            ctype = claim.get("type", "ADDED_FINDING")
            if ctype == "SEVERITY_FLIP":
                found, expected = utils.parse_severity_flip(claim.get("reason", ""))
                if found and expected:
                    st.markdown(callout(
                        f"<b>SEVERITY_FLIP</b> &mdash; Original: <code>{expected}</code> &rarr; Changed: <code>{found}</code><br>"
                        f"<span style='color:{TEXT_MUTED}'>{claim.get('sentence', '')}</span>", "warn"), unsafe_allow_html=True)
                    continue
            label = "Unsupported claim" if ctype != "SEVERITY_FLIP" else "Severity mismatch"
            st.markdown(callout(f"<b>{ctype}</b> &mdash; {label}: {claim.get('sentence', claim.get('reason', ''))}", "warn"), unsafe_allow_html=True)
        for claim in mf:
            st.markdown(callout(f"<b>{claim.get('type', 'OMITTED_FINDING')}</b> &mdash; Missing: {claim.get('label', claim.get('reason', ''))}", "warn"), unsafe_allow_html=True)


def render_evaluation_progress():
    st.markdown(callout("Development Experiment History &mdash; manually recorded across development iterations. Not calculated from the currently loaded results file.", "neutral"), unsafe_allow_html=True)

    # Edit this table to add future experiment runs.
    experiment_history = pd.DataFrame([
        {"Version": "Initial",     "Accuracy": 43.5, "Precision": 18.8, "Recall": 100.0, "FN": 0},
        {"Version": "Iteration 1", "Accuracy": 78.3, "Precision": 37.5, "Recall": 100.0, "FN": 0},
        {"Version": "Iteration 2", "Accuracy": 89.4, "Precision": 62.2, "Recall": 93.3,  "FN": 4},
        {"Version": "Current",     "Accuracy": 90.6, "Precision": 63.8, "Recall": 100.0, "FN": 0},
    ])

    fig = go.Figure()
    for metric_name, color in [("Accuracy", ACCENT), ("Precision", WARN), ("Recall", GOOD)]:
        fig.add_trace(go.Scatter(
            x=experiment_history["Version"], y=experiment_history[metric_name],
            mode="lines+markers", name=metric_name, line=dict(width=2.5, color=color), marker=dict(size=8),
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_MUTED, size=12), height=360,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=40, b=10), yaxis_title="%", xaxis_title="",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(experiment_history, width="stretch", hide_index=True)


def render_engineering_insights(metrics: dict, fp_labels: pd.DataFrame):
    insights = utils.generate_insights(metrics, fp_labels)
    for insight in insights:
        st.markdown(callout(insight["text"], insight["kind"]), unsafe_allow_html=True)

    st.write("")
    st.markdown('<p class="rv-section-label">Recommended Next Investigation</p>', unsafe_allow_html=True)
    st.markdown(callout(utils.recommended_investigation(fp_labels), "neutral"), unsafe_allow_html=True)


def render_export_section(df: pd.DataFrame, metrics: dict):
    st.markdown('<p class="rv-section-label">Export</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    export_df = df.drop(columns=["unsupported_claims", "missing_findings"], errors="ignore")
    c1.download_button(
        "Download filtered results (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="radverify_filtered_results.csv", mime="text/csv",
        width="stretch",
    )
    import json as _json
    c2.download_button(
        "Download metrics summary (JSON)",
        data=_json.dumps(metrics, indent=2).encode("utf-8"),
        file_name="radverify_metrics_summary.json", mime="application/json",
        width="stretch",
    )
    st.caption("Exports reflect the currently filtered view. Source files are never modified.")


def render_footer():
    st.markdown('''
    <div class="rv-footer">
        RadVerify<br>
        AI-generated Radiology Report Verification &mdash; Evaluation &amp; Error Analysis
    </div>
    ''', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Main -- sidebar widgets are created exactly ONCE here, nowhere else
# --------------------------------------------------------------------------- #

st.sidebar.markdown("### Data Source")
results_path = Path(st.sidebar.text_input("Results JSON", value=str(DEFAULT_RESULTS)))
eval_set_path = Path(st.sidebar.text_input("Evaluation Set JSON (optional)", value=str(DEFAULT_EVAL_SET)))

records, load_error = get_results(results_path)

if load_error:
    render_header(0, 0)
    st.error(load_error)
    st.info("Point the sidebar field at your `groundguard_results.json`. The default assumes `output/` sits next to `dashboard/` in your project root.")
    st.stop()

if len(records) == 0:
    render_header(0, 0)
    st.warning("The results file loaded successfully but contains zero cases. Nothing to display yet.")
    st.stop()

df_all = utils.dataframe(records)
eval_set = get_eval_set(eval_set_path)

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")
truth_filter = st.sidebar.multiselect("Ground truth", ["CLEAN", "CORRUPTED"], default=["CLEAN", "CORRUPTED"])
class_filter = st.sidebar.multiselect("Classification", ["TP", "TN", "FP", "FN"], default=["TP", "TN", "FP", "FN"])
error_types_present = sorted(df_all["true_error_type"].unique().tolist())
error_type_filter = st.sidebar.multiselect("True error type", error_types_present, default=error_types_present)

df = df_all[
    df_all["is_corrupted"].map(lambda x: "CORRUPTED" if x else "CLEAN").isin(truth_filter)
    & df_all["classification"].isin(class_filter)
    & df_all["true_error_type"].isin(error_type_filter)
].reset_index(drop=True)

metrics = utils.compute_metrics(df)

render_header(len(df), len(df_all))

tab_overview, tab_errors, tab_explorer, tab_progress, tab_insights = st.tabs(
    ["Overview", "Error Analysis", "Case Explorer", "Evaluation Progress", "Engineering Insights"]
)

with tab_overview:
    render_kpi_cards(metrics)
    st.write("")
    col_cm, col_pred = st.columns(2)
    with col_cm:
        render_confusion_matrix(metrics)
    with col_pred:
        render_prediction_breakdown(metrics)
    render_error_distribution(df)
    st.write("")
    render_system_performance(df)
    st.write("")
    render_pipeline_flow()

with tab_errors:
    fp_labels_for_insights = render_error_analysis(df, df_all, records, metrics)

with tab_explorer:
    render_case_explorer(df, df_all, records, eval_set)

with tab_progress:
    render_evaluation_progress()

with tab_insights:
    fp_labels_current = utils.false_positive_label_breakdown(df)
    render_engineering_insights(metrics, fp_labels_current)

st.write("")
render_export_section(df, metrics)
render_footer()