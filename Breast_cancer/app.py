import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import os

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Syne+Mono&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #f5f0e8;
    color: #1a1a1a;
}

[data-testid="stSidebar"] {
    background: #1a1a1a !important;
    border-right: none;
}
[data-testid="stSidebar"] * { color: #f5f0e8 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMarkdown p { color: #a0a0a0 !important; }

.logo-block {
    padding: 0 0 2rem 0;
    border-bottom: 1px solid #333;
    margin-bottom: 2rem;
}
.logo-main {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f5f0e8 !important;
    line-height: 1;
    letter-spacing: -0.03em;
}
.logo-tag {
    font-family: 'Syne Mono', monospace;
    font-size: 0.58rem;
    color: #e8a020 !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.page-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    margin-bottom: 0.3rem;
}
.page-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #1a1a1a;
    line-height: 1;
}
.page-num {
    font-family: 'Syne Mono', monospace;
    font-size: 0.75rem;
    color: #999;
    letter-spacing: 0.15em;
}
.page-desc {
    font-family: 'Syne Mono', monospace;
    font-size: 0.68rem;
    color: #888;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 2rem;
    border-left: 3px solid #e8a020;
    padding-left: 0.7rem;
}

.stButton > button {
    background: #1a1a1a !important;
    color: #f5f0e8 !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    padding: 0.75rem 2.5rem !important;
    text-transform: uppercase !important;
    transition: background 0.2s, transform 0.1s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #333 !important;
    transform: translateY(-1px) !important;
}

.res-card {
    border-radius: 8px;
    padding: 2rem 2.5rem;
    margin: 1.5rem 0;
}
.res-card.malignant {
    background: #1a1a1a;
    border-left: 5px solid #e85020;
}
.res-card.benign {
    background: #1a1a1a;
    border-left: 5px solid #20c850;
}
.res-verdict {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    line-height: 1;
}
.res-verdict.malignant { color: #e85020; }
.res-verdict.benign    { color: #20c850; }
.res-desc {
    font-family: 'Syne Mono', monospace;
    font-size: 0.72rem;
    color: #888;
    margin-top: 0.6rem;
    letter-spacing: 0.05em;
    line-height: 1.6;
}
.res-prob {
    font-family: 'Syne Mono', monospace;
    font-size: 1.1rem;
    color: #f5f0e8;
    margin-top: 1rem;
}

.stat-tile {
    background: #1a1a1a;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.stat-label {
    font-family: 'Syne Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.4rem;
}
.stat-val {
    font-size: 1.8rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    color: #e8a020;
}

[data-testid="stFileUploader"] {
    background: white;
    border: 2px dashed #ccc;
    border-radius: 8px;
    padding: 0.5rem;
}

.stNumberInput > div > div > input {
    background: white !important;
    border: 1px solid #ddd !important;
    border-radius: 4px !important;
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.85rem !important;
    color: #1a1a1a !important;
}
.stNumberInput label {
    font-family: 'Syne Mono', monospace !important;
    font-size: 0.68rem !important;
    color: #666 !important;
    letter-spacing: 0.05em !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    gap: 0.3rem;
    border-bottom: 2px solid #ddd;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border: none;
    font-family: 'Syne Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #999;
    padding: 0.5rem 1rem;
    border-bottom: 2px solid transparent;
    margin-bottom: -2px;
}
.stTabs [aria-selected="true"] {
    color: #1a1a1a !important;
    border-bottom: 2px solid #1a1a1a !important;
    background: transparent !important;
}

.model-status {
    background: #0d2e0d;
    border: 1px solid #1a5c1a;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-family: 'Syne Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: #44cc44 !important;
    margin-top: 1.5rem;
}
.model-error {
    background: #2e0d0d;
    border: 1px solid #5c1a1a;
    border-radius: 4px;
    padding: 0.6rem 0.8rem;
    font-family: 'Syne Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: #cc4444 !important;
    margin-top: 1.5rem;
}

#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.5rem; max-width: 1100px; }
</style>
""", unsafe_allow_html=True)


# ─── Load Saved Model ─────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        base_dir      = os.path.dirname(os.path.abspath(__file__))
        pipeline      = joblib.load(os.path.join(base_dir, "pipeline.pkl"))
        feature_names = joblib.load(os.path.join(base_dir, "feature_names.pkl"))
        return pipeline, feature_names, None
    except FileNotFoundError as e:
        return None, None, str(e)

# ✅ This line was missing — unpacking the return values
pipeline, feature_names, load_error = load_model()


# ─── Feature Metadata (label, min, max, default, step) ───────────────────────
FEATURE_META = {
    "radius_mean":             ("Radius Mean",             5.0,    30.0,   14.1,   0.01),
    "texture_mean":            ("Texture Mean",            9.0,    40.0,   19.3,   0.01),
    "perimeter_mean":          ("Perimeter Mean",          40.0,   190.0,  92.0,   0.1),
    "area_mean":               ("Area Mean",               140.0,  2500.0, 654.9,  1.0),
    "smoothness_mean":         ("Smoothness Mean",         0.05,   0.17,   0.096,  0.0001),
    "compactness_mean":        ("Compactness Mean",        0.02,   0.35,   0.104,  0.0001),
    "concavity_mean":          ("Concavity Mean",          0.0,    0.43,   0.089,  0.0001),
    "concave points_mean":     ("Concave Points Mean",     0.0,    0.20,   0.048,  0.0001),
    "symmetry_mean":           ("Symmetry Mean",           0.10,   0.30,   0.181,  0.0001),
    "fractal_dimension_mean":  ("Fractal Dimension Mean",  0.05,   0.10,   0.063,  0.0001),
    "radius_se":               ("Radius SE",               0.1,    3.0,    0.405,  0.001),
    "texture_se":              ("Texture SE",              0.3,    5.0,    1.217,  0.001),
    "perimeter_se":            ("Perimeter SE",            0.7,    22.0,   2.866,  0.01),
    "area_se":                 ("Area SE",                 6.0,    550.0,  40.3,   0.1),
    "smoothness_se":           ("Smoothness SE",           0.001,  0.031,  0.007,  0.0001),
    "compactness_se":          ("Compactness SE",          0.002,  0.135,  0.025,  0.0001),
    "concavity_se":            ("Concavity SE",            0.0,    0.40,   0.032,  0.0001),
    "concave points_se":       ("Concave Points SE",       0.0,    0.053,  0.012,  0.0001),
    "symmetry_se":             ("Symmetry SE",             0.007,  0.079,  0.020,  0.0001),
    "fractal_dimension_se":    ("Fractal Dimension SE",    0.001,  0.030,  0.004,  0.0001),
    "radius_worst":            ("Radius Worst",            7.0,    36.0,   16.3,   0.01),
    "texture_worst":           ("Texture Worst",           12.0,   50.0,   25.7,   0.01),
    "perimeter_worst":         ("Perimeter Worst",         50.0,   252.0,  107.3,  0.1),
    "area_worst":              ("Area Worst",              180.0,  4254.0, 880.6,  1.0),
    "smoothness_worst":        ("Smoothness Worst",        0.07,   0.22,   0.132,  0.0001),
    "compactness_worst":       ("Compactness Worst",       0.03,   1.06,   0.254,  0.0001),
    "concavity_worst":         ("Concavity Worst",         0.0,    1.25,   0.272,  0.0001),
    "concave points_worst":    ("Concave Points Worst",    0.0,    0.29,   0.115,  0.0001),
    "symmetry_worst":          ("Symmetry Worst",          0.15,   0.66,   0.290,  0.0001),
    "fractal_dimension_worst": ("Fractal Dimension Worst", 0.055,  0.208,  0.084,  0.0001),
}

MEAN_FEATS  = [k for k in FEATURE_META if k.endswith("_mean")]
SE_FEATS    = [k for k in FEATURE_META if k.endswith("_se")]
WORST_FEATS = [k for k in FEATURE_META if k.endswith("_worst")]


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="logo-block">
        <div class="logo-main">Breast Cancer<br>Predictor</div>
        <div class="logo-tag">Clinical ML Tool · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["Single Prediction", "Batch Prediction"],
        label_visibility="collapsed"
    )

    if load_error:
        st.markdown(f"""
        <div class="model-error">
            ✗ MODEL NOT FOUND<br><br>
            {load_error}<br><br>
            Ensure pipeline.pkl and feature_names.pkl are in the same directory as app.py.
        </div>
        """, unsafe_allow_html=True)
    else:
        model = pipeline.named_steps["model"]
        st.markdown(f"""
        <div class="model-status">
            ✓ MODEL LOADED<br><br>
            PIPELINE &nbsp;· StandardScaler + LR<br>
            FEATURES · {len(feature_names)}<br>
            SOLVER &nbsp;&nbsp;· {model.solver}<br>
            CLASS WT · {model.class_weight}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne Mono,monospace;font-size:0.58rem;color:#444;line-height:2'>
        Dataset · Wisconsin Breast Cancer<br>
        Algorithm · Logistic Regression<br>
        Classes · Benign (0) / Malignant (1)
    </div>
    """, unsafe_allow_html=True)


# ─── Guard ────────────────────────────────────────────────────────────────────
if pipeline is None:
    st.error("⚠️  Could not load `pipeline.pkl` or `feature_names.pkl`.\n\nPlace both files in the same folder as `app.py` and restart.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 · SINGLE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
if page == "Single Prediction":

    st.markdown("""
    <div class="page-header">
        <span class="page-title">Single Sample</span>
        <span class="page-num">01 / 02</span>
    </div>
    <div class="page-desc">Manually enter cell nucleus measurements · all 30 features</div>
    """, unsafe_allow_html=True)

    input_values = {}

    tab_mean, tab_se, tab_worst = st.tabs(["Mean", "Standard Error", "Worst"])

    for tab, feat_list in [
        (tab_mean,  MEAN_FEATS),
        (tab_se,    SE_FEATS),
        (tab_worst, WORST_FEATS),
    ]:
        with tab:
            cols = st.columns(2)
            for i, key in enumerate(feat_list):
                if key not in feature_names:
                    continue
                lbl, mn, mx, default, step = FEATURE_META[key]
                with cols[i % 2]:
                    input_values[key] = st.number_input(
                        lbl,
                        min_value=float(mn),
                        max_value=float(mx),
                        value=float(default),
                        step=float(step),
                        format="%.4f",
                        key=f"inp_{key}"
                    )

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("ANALYSE SAMPLE →")

    if run:
        row = [input_values.get(f, FEATURE_META.get(f, (f, 0, 1, 0, 0.01))[2])
               for f in feature_names]
        input_df = pd.DataFrame([row], columns=feature_names)

        pred      = pipeline.predict(input_df)[0]
        proba     = pipeline.predict_proba(input_df)[0]
        mal_prob  = proba[1]
        ben_prob  = proba[0]

        if pred == 1:
            st.markdown(f"""
            <div class="res-card malignant">
                <div class="res-verdict malignant">MALIGNANT</div>
                <div class="res-desc">
                    The model classifies this sample as <strong>malignant</strong>.<br>
                    This result is for research purposes only — consult a medical professional.
                </div>
                <div class="res-prob">
                    Malignancy probability &nbsp;→&nbsp;
                    <span style="color:#e85020;font-size:1.5rem;font-weight:800">
                        {mal_prob:.1%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="res-card benign">
                <div class="res-verdict benign">BENIGN</div>
                <div class="res-desc">
                    The model classifies this sample as <strong>benign</strong>.<br>
                    This result is for research purposes only — regular monitoring is advised.
                </div>
                <div class="res-prob">
                    Benign probability &nbsp;→&nbsp;
                    <span style="color:#20c850;font-size:1.5rem;font-weight:800">
                        {ben_prob:.1%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_g, col_b = st.columns(2)

        with col_g:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(mal_prob * 100, 2),
                title={"text": "Malignancy %",
                       "font": {"family": "Syne Mono", "size": 11, "color": "#888"}},
                number={"suffix": "%",
                        "font": {"family": "Syne", "size": 40,
                                 "color": "#e85020" if pred == 1 else "#20c850"}},
                gauge={
                    "axis": {"range": [0, 100],
                             "tickfont": {"family": "Syne Mono", "size": 9, "color": "#666"}},
                    "bar": {"color": "#e85020" if pred == 1 else "#20c850", "thickness": 0.25},
                    "bgcolor": "#f5f0e8",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0,  30],  "color": "#e8f5e8"},
                        {"range": [30, 60],  "color": "#f5f0e8"},
                        {"range": [60, 100], "color": "#f5e8e8"},
                    ],
                    "threshold": {"line": {"color": "#1a1a1a", "width": 2}, "value": 50}
                }
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=240, margin=dict(t=40, b=0, l=20, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=[ben_prob * 100], y=[""],
                orientation="h", marker_color="#20c850",
                text=f"Benign  {ben_prob:.1%}", textposition="inside",
                textfont=dict(family="Syne Mono", size=11, color="white"),
                insidetextanchor="middle",
            ))
            fig2.add_trace(go.Bar(
                x=[mal_prob * 100], y=[""],
                orientation="h", marker_color="#e85020",
                text=f"Malignant  {mal_prob:.1%}", textposition="inside",
                textfont=dict(family="Syne Mono", size=11, color="white"),
                insidetextanchor="middle",
            ))
            fig2.update_layout(
                barmode="stack",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                xaxis=dict(range=[0, 100], visible=False),
                yaxis=dict(visible=False),
                height=240, margin=dict(t=60, b=0, l=0, r=0),
                title=dict(text="Probability Breakdown",
                           font=dict(family="Syne Mono", size=11, color="#888"), x=0)
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("VIEW INPUT VALUES"):
            summary_df = pd.DataFrame({
                "Feature": [FEATURE_META.get(f, (f,))[0] for f in feature_names],
                "Value":   [f"{input_values.get(f, 0):.4f}" for f in feature_names],
            })
            st.dataframe(summary_df.set_index("Feature"), use_container_width=True)

        st.markdown("""
        <div style='font-family:Syne Mono,monospace;font-size:0.58rem;color:#bbb;
                    text-align:center;padding:1.5rem 0 0 0;border-top:1px solid #ddd;margin-top:1.5rem'>
            ⚠ FOR RESEARCH & EDUCATIONAL USE ONLY — NOT A MEDICAL DEVICE
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 · BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class="page-header">
        <span class="page-title">Batch Prediction</span>
        <span class="page-num">02 / 02</span>
    </div>
    <div class="page-desc">Upload a CSV file to classify multiple samples at once</div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style='background:white;border:1px solid #ddd;border-radius:6px;
                padding:1rem 1.5rem;margin-bottom:1.5rem;
                font-family:Syne Mono,monospace;font-size:0.68rem;color:#555;line-height:2'>
        <strong style='color:#1a1a1a'>Expected CSV format</strong><br>
        · {len(feature_names)} numeric feature columns matching training data<br>
        · Column names must match exactly (case-sensitive)<br>
        · <code>id</code>, <code>Unnamed: 32</code>, <code>diagnosis</code> columns are auto-removed<br>
        · First 5 required columns: <code>{", ".join(feature_names[:5])}</code> ...
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded:
        try:
            df_raw = pd.read_csv(uploaded)
            df_raw.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")

            has_gt = "diagnosis" in df_raw.columns
            if has_gt:
                true_labels = df_raw["diagnosis"].str.strip().map({"M": 1, "B": 0})
                df_raw.drop(columns=["diagnosis"], inplace=True, errors="ignore")

            missing = [f for f in feature_names if f not in df_raw.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
                st.stop()

            df_feat = df_raw[feature_names].apply(pd.to_numeric, errors="coerce")
            n_nulls = df_feat.isnull().sum().sum()
            if n_nulls > 0:
                st.warning(f"{n_nulls} NaN values found — affected rows dropped.")
                df_feat.dropna(inplace=True)

            preds  = pipeline.predict(df_feat)
            probas = pipeline.predict_proba(df_feat)[:, 1]

            results = pd.DataFrame({
                "Sample":         range(1, len(preds) + 1),
                "Prediction":     ["Malignant" if p == 1 else "Benign" for p in preds],
                "Malignant_%":    (probas * 100).round(2),
                "Benign_%":       ((1 - probas) * 100).round(2),
            })
            if has_gt and len(true_labels) == len(results):
                results["True_Label"] = ["Malignant" if t == 1 else "Benign"
                                         for t in true_labels.values]

            n_mal = (preds == 1).sum()
            n_ben = (preds == 0).sum()
            total = len(preds)

            c1, c2, c3, c4 = st.columns(4)
            for col, (lbl, val, clr) in zip(
                [c1, c2, c3, c4],
                [("TOTAL",     str(total),           "#e8a020"),
                 ("BENIGN",    str(n_ben),            "#20c850"),
                 ("MALIGNANT", str(n_mal),            "#e85020"),
                 ("MAL. RATE", f"{n_mal/total:.1%}",  "#888")]
            ):
                with col:
                    st.markdown(f"""
                    <div class="stat-tile">
                        <div class="stat-label">{lbl}</div>
                        <div class="stat-val" style="color:{clr}">{val}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_chart, col_hist = st.columns(2)
            with col_chart:
                fig_bar = go.Figure(go.Bar(
                    x=["Benign", "Malignant"], y=[n_ben, n_mal],
                    marker_color=["#20c850", "#e85020"],
                    text=[n_ben, n_mal], textposition="outside",
                    textfont=dict(family="Syne", size=18, color="#1a1a1a"),
                    width=0.4,
                ))
                fig_bar.update_layout(
                    title=dict(text="Prediction Distribution",
                               font=dict(family="Syne Mono", size=11, color="#888")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(tickfont=dict(family="Syne Mono", size=11, color="#555"), showgrid=False),
                    yaxis=dict(gridcolor="#eee", tickfont=dict(family="Syne Mono", size=9, color="#999")),
                    showlegend=False, height=260,
                    margin=dict(t=40, b=20, l=20, r=20),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_hist:
                fig_hist = go.Figure(go.Histogram(
                    x=probas * 100, nbinsx=20,
                    marker_color="#e8a020",
                    marker_line_color="white", marker_line_width=0.5,
                ))
                fig_hist.update_layout(
                    title=dict(text="Malignancy Probability Distribution",
                               font=dict(family="Syne Mono", size=11, color="#888")),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(title="Malignancy %",
                               tickfont=dict(family="Syne Mono", size=9, color="#999"),
                               gridcolor="#eee"),
                    yaxis=dict(title="Count",
                               tickfont=dict(family="Syne Mono", size=9, color="#999"),
                               gridcolor="#eee"),
                    showlegend=False, height=260,
                    margin=dict(t=40, b=40, l=40, r=20),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("""
            <div style='font-family:Syne Mono,monospace;font-size:0.6rem;letter-spacing:0.2em;
                        text-transform:uppercase;color:#e8a020;margin-bottom:0.5rem'>
                Prediction Results
            </div>""", unsafe_allow_html=True)

            display_cols = ["Sample", "Prediction", "Malignant_%", "Benign_%"]
            if has_gt and "True_Label" in results.columns:
                display_cols.append("True_Label")

            st.dataframe(results[display_cols].set_index("Sample"),
                         use_container_width=True, height=320)

            csv_out = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="↓  DOWNLOAD RESULTS (.csv)",
                data=csv_out,
                file_name="breast_cancer_predictions.csv",
                mime="text/csv",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

    else:
        st.markdown("""
        <div style='background:white;border:2px dashed #ccc;border-radius:8px;
                    padding:3rem;text-align:center;'>
            <div style='font-size:2.5rem;margin-bottom:1rem'>📂</div>
            <div style='font-family:Syne Mono,monospace;font-size:0.7rem;color:#999;
                        letter-spacing:0.1em'>
                DRAG & DROP or click above to upload your CSV
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:Syne Mono,monospace;font-size:0.58rem;color:#bbb;
                text-align:center;padding:1.5rem 0 0 0;border-top:1px solid #ddd;margin-top:2rem'>
        ⚠ FOR RESEARCH & EDUCATIONAL USE ONLY — NOT A MEDICAL DEVICE
    </div>
    """, unsafe_allow_html=True)