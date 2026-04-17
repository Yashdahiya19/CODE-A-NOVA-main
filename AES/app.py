import streamlit as st
from  aes_model import get_or_train_model, predict

st.set_page_config(page_title="Essay Scorer", page_icon="📝")
st.title(" Automated Essay Scorer")

CSV_PATH = "ASAP.csv"  # ← change if your CSV is in a different location

#  Load model ONCE using Streamlit cache — never retrains on refresh
@st.cache_resource
def load():
    return get_or_train_model(CSV_PATH)

with st.spinner("Loading model..."):
    data = load()

model      = data["model"]
vectorizer = data["vectorizer"]
score_min  = data["score_min"]
score_max  = data["score_max"]
metrics    = data["metrics"]

# ── Sidebar: model stats ──────────────────────────────────────────────────────
with st.sidebar:
    st.header(" Model Stats")
    st.metric("R² Score",     f"{metrics['r2']:.3f}")
    st.metric("RMSE",         f"{metrics['rmse']:.3f}")
    st.metric("Exact Match",  f"{metrics['exact_match']*100:.1f}%")
    st.metric("Within ±1",    f"{metrics['within_one']*100:.1f}%")
    st.caption(f"Score range: {score_min} – {score_max}")
    st.caption(f"Trained on {metrics['n_train']} essays")

    if st.button("🔄 Force Retrain"):
        import os
        from aes_model import MODEL_PATH
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
        st.cache_resource.clear()
        st.rerun()

# ── Main: essay input ─────────────────────────────────────────────────────────
essay = st.text_area("Paste your essay here:", height=300,
                     placeholder="Write or paste your essay here...")

if st.button("Score Essay", type="primary"):
    if not essay.strip():
        st.warning("Please enter an essay first.")
    else:
        result = predict(essay, model, vectorizer, score_min, score_max)

        st.subheader(" Score")
        col1, col2 = st.columns(2)
        col1.metric("Final Score",    f"{result['rounded']} / {score_max}")
        col2.metric("Raw Score",      f"{result['raw']:.2f}")

        st.subheader("Writing Features")
        feats = result["feats"]
        col1, col2, col3 = st.columns(3)
        col1.metric("Word Count",       feats["word_count"])
        col1.metric("Sentence Count",   feats["sentence_count"])
        col2.metric("Unique Word Ratio", f"{feats['unique_ratio']:.2f}")
        col2.metric("Avg Word Length",   f"{feats['avg_word_len']:.2f}")
        col3.metric("Flesch Score",      f"{feats['flesch_score']:.1f}")
        col3.metric("Difficult Words",   feats["difficult_words"])