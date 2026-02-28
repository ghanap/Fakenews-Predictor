import streamlit as st
import re
import nltk
import numpy as np
import joblib
import os
from pathlib import Path

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

/* ── Globals ── */
html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e0d0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

/* ── Header ── */
.hero {
    border: 1px solid #2a2a2a;
    background: linear-gradient(135deg, #111 0%, #1a1a1a 100%);
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "VERIFY";
    position: absolute;
    right: -10px;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif;
    font-size: 9rem;
    font-weight: 800;
    color: rgba(255,255,255,0.03);
    pointer-events: none;
    letter-spacing: -4px;
}
.hero-tag {
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #f0c040;
    margin-bottom: 0.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    line-height: 1.05;
    margin: 0 0 0.5rem 0;
    color: #f5f0e8;
}
.hero-sub {
    font-size: 0.75rem;
    color: #666;
    letter-spacing: 0.05em;
}

/* ── Model selector pills ── */
.model-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    flex-wrap: wrap;
}
.model-pill {
    border: 1px solid #333;
    padding: 0.4rem 1rem;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    cursor: pointer;
    background: transparent;
    color: #999;
    transition: all 0.15s;
}
.model-pill:hover { border-color: #f0c040; color: #f0c040; }

/* ── Verdict cards ── */
.verdict-wrap {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-top: 1.5rem;
}
.verdict-card {
    border: 1px solid #222;
    padding: 1.5rem;
    background: #111;
    position: relative;
}
.verdict-card .model-label {
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #555;
    margin-bottom: 0.75rem;
}
.verdict-card .verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}
.verdict-card.real { border-left: 3px solid #4caf82; }
.verdict-card.fake { border-left: 3px solid #e05a5a; }
.verdict-card.uncertain { border-left: 3px solid #f0c040; }
.verdict-card .verdict-text.real { color: #4caf82; }
.verdict-card .verdict-text.fake { color: #e05a5a; }
.verdict-card .conf-bar-wrap {
    margin-top: 0.75rem;
    height: 2px;
    background: #222;
}
.verdict-card .conf-bar {
    height: 100%;
    transition: width 0.4s ease;
}
.conf-bar.real { background: #4caf82; }
.conf-bar.fake { background: #e05a5a; }
.conf-label {
    font-size: 0.65rem;
    color: #555;
    margin-top: 0.3rem;
    letter-spacing: 0.1em;
}

/* ── Comparison table ── */
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.75rem;
    margin-top: 1rem;
}
.cmp-table th {
    text-align: left;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #222;
    padding: 0.5rem 0.75rem;
}
.cmp-table td {
    padding: 0.6rem 0.75rem;
    border-bottom: 1px solid #181818;
    color: #ccc;
}
.cmp-table tr:last-child td { border-bottom: none; }
.badge-real { color: #4caf82; font-weight: 700; }
.badge-fake { color: #e05a5a; font-weight: 700; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Textarea ── */
textarea {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 0 !important;
    color: #e8e0d0 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
textarea:focus { border-color: #f0c040 !important; box-shadow: none !important; }

/* ── Buttons ── */
.stButton > button {
    background: #f0c040 !important;
    color: #0d0d0d !important;
    border: none !important;
    border-radius: 0 !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Divider ── */
hr { border-color: #1e1e1e !important; }

/* ── Info/warning boxes ── */
.stAlert { border-radius: 0 !important; }

/* ── Metric ── */
[data-testid="stMetric"] {
    background: #111;
    border: 1px solid #1e1e1e;
    padding: 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.6rem !important; letter-spacing: 0.2em; color: #555 !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; color: #f5f0e8 !important; }

/* ── Section headers ── */
.section-label {
    font-size: 0.6rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #1e1e1e;
    padding-bottom: 0.4rem;
    margin-bottom: 1rem;
}

.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 0.4rem;
    background: #4caf82;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)


# ── NLTK setup ─────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ['punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()


# ── Text preprocessing ─────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)


# ── Load SentenceBERT ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading SentenceBERT model…")
def load_sbert(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


# ── Load classifiers ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classifiers…")
def load_classifiers(mlp_path: str, sgd_path: str):
    mlp = joblib.load(mlp_path) if Path(mlp_path).exists() else None
    sgd = joblib.load(sgd_path) if Path(sgd_path).exists() else None
    return mlp, sgd


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_with_proba(model, embedding):
    pred = int(model.predict(embedding)[0])
    label = "REAL" if pred == 1 else "FAKE"
    css_cls = "real" if pred == 1 else "fake"
    # Get confidence if available
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(embedding)[0]
        conf = float(max(proba))
    elif hasattr(model, 'decision_function'):
        df = model.decision_function(embedding)[0]
        conf = float(min(abs(df) / (abs(df) + 1) + 0.5, 0.99))
    else:
        conf = 0.0
    return pred, label, css_cls, conf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>⚙ Configuration</div>", unsafe_allow_html=True)

    sbert_model = st.selectbox(
        "SentenceBERT Model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0,
        help="MiniLM is ~5x faster. mpnet is more accurate but slower.",
    )

    st.markdown("---")
    st.markdown("<div class='section-label'>Model Files</div>", unsafe_allow_html=True)

    mlp_path = st.text_input("MLP model path", value="mlp_classifier_model.pkl")
    sgd_path = st.text_input("SGD model path", value="sgd_classifier_model.pkl")

    st.markdown("---")
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.68rem; color:#555; line-height:1.8;'>
<b style='color:#888'>Pipeline</b><br>
Text → Lemmatize → SentenceBERT → Classifier<br><br>
<b style='color:#888'>Models</b><br>
• MLP (neural network)<br>
• SGD (linear / SVM-like)<br><br>
<b style='color:#888'>Labels</b><br>
<span style='color:#4caf82'>■</span> 1 = Real news<br>
<span style='color:#e05a5a'>■</span> 0 = Fake news
</div>
""", unsafe_allow_html=True)


# ── Load resources ─────────────────────────────────────────────────────────────
sbert = load_sbert(sbert_model)
mlp_model, sgd_model = load_classifiers(mlp_path, sgd_path)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="status-dot"></span>Live · SentenceBERT</div>
    <div class="hero-title">Fake News<br>Detector</div>
    <div class="hero-sub">MLP &amp; SGD classifiers · Dual-model verdict</div>
</div>
""", unsafe_allow_html=True)


# ── Model status ───────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SBERT Model", sbert_model.split("/")[-1].replace("all-", ""))
with col2:
    st.metric("MLP", "✓ Loaded" if mlp_model else "✗ Not found")
with col3:
    st.metric("SGD", "✓ Loaded" if sgd_model else "✗ Not found")

st.markdown("---")

# ── Input ──────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-label'>Input Text</div>", unsafe_allow_html=True)

# Sample news options
SAMPLES = {
    "— select a sample —": "",
    "🟢 Real-style: Climate report": (
        "Scientists from the IPCC released their latest assessment report confirming that "
        "global temperatures have risen by 1.1°C above pre-industrial levels. The report, "
        "compiled by hundreds of researchers, urges immediate reduction in carbon emissions "
        "to limit warming to 1.5°C by the end of the century."
    ),
    "🔴 Fake-style: Satire / absurd": (
        "WASHINGTON — President signed an executive order renaming the White House refrigerator "
        "the U.S. Department of Ham, citing Thomas Jefferson's 1801 ice house as precedent. "
        "Officials say the move restores the nation's historic commitment to cold cuts."
    ),
}

sample_choice = st.selectbox("Load a sample", list(SAMPLES.keys()))
default_text = SAMPLES[sample_choice]

news_text = st.text_area(
    "Paste article text here",
    value=default_text,
    height=200,
    placeholder="Paste a news article, headline, or paragraph…",
    label_visibility="collapsed",
)

char_count = len(news_text)
word_count = len(news_text.split()) if news_text.strip() else 0
st.markdown(
    f"<div style='font-size:0.65rem;color:#444;text-align:right;margin-top:-0.5rem'>"
    f"{word_count} words · {char_count} chars</div>",
    unsafe_allow_html=True,
)

analyze_btn = st.button("ANALYZE", disabled=(not news_text.strip()))


# ── Analysis ───────────────────────────────────────────────────────────────────
if analyze_btn and news_text.strip():
    if not mlp_model and not sgd_model:
        st.error(
            "No models found. Train and save your models first using the notebook, "
            "then make sure the .pkl paths above are correct."
        )
    else:
        with st.spinner("Preprocessing and embedding…"):
            cleaned = preprocess_text(news_text)
            embedding = sbert.encode([cleaned])

        st.markdown("---")
        st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)

        # ── Preprocessed preview
        with st.expander("View preprocessed text"):
            st.code(cleaned, language=None)

        # ── Verdict cards
        results = []
        cards_html = '<div class="verdict-wrap">'

        for model_obj, model_name in [(mlp_model, "MLP"), (sgd_model, "SGD")]:
            if model_obj is None:
                cards_html += f"""
                <div class="verdict-card uncertain">
                    <div class="model-label">{model_name} Classifier</div>
                    <div class="verdict-text" style="color:#555">N/A</div>
                    <div class="conf-label">Model not loaded</div>
                </div>"""
                results.append({"Model": model_name, "Verdict": "N/A", "Confidence": None})
                continue

            pred, label, css_cls, conf = predict_with_proba(model_obj, embedding)
            results.append({"Model": model_name, "Verdict": label, "Pred": pred, "Confidence": conf})

            conf_pct = round(conf * 100, 1)
            cards_html += f"""
            <div class="verdict-card {css_cls}">
                <div class="model-label">{model_name} Classifier</div>
                <div class="verdict-text {css_cls}">{label}</div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar {css_cls}" style="width:{conf_pct}%"></div>
                </div>
                <div class="conf-label">Confidence: {conf_pct}%</div>
            </div>"""

        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)

        # ── Agreement / disagreement callout
        verdicts = [r["Verdict"] for r in results if r["Verdict"] != "N/A"]
        if len(verdicts) == 2:
            if verdicts[0] == verdicts[1]:
                emoji = "✅" if verdicts[0] == "REAL" else "🚨"
                st.success(f"{emoji} Both models agree: **{verdicts[0]}**")
            else:
                st.warning("⚠️ Models disagree — treat result with caution.")

        # ── Comparison table
        st.markdown("---")
        st.markdown("<div class='section-label'>Comparison</div>", unsafe_allow_html=True)

        table_rows = ""
        for r in results:
            if r["Verdict"] == "N/A":
                table_rows += f"<tr><td>{r['Model']}</td><td style='color:#555'>N/A</td><td style='color:#555'>—</td></tr>"
            else:
                badge_cls = "badge-real" if r["Verdict"] == "REAL" else "badge-fake"
                conf_str = f"{round(r['Confidence']*100,1)}%" if r["Confidence"] else "—"
                table_rows += f"""
                <tr>
                    <td>{r['Model']}</td>
                    <td class='{badge_cls}'>{r['Verdict']}</td>
                    <td>{conf_str}</td>
                </tr>"""

        st.markdown(f"""
        <table class="cmp-table">
            <thead><tr><th>Model</th><th>Verdict</th><th>Confidence</th></tr></thead>
            <tbody>{table_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.6rem;color:#333;text-align:center;letter-spacing:0.15em;'>"
    "FAKE NEWS DETECTOR · SBERT + MLP / SGD · FOR RESEARCH USE ONLY"
    "</div>",
    unsafe_allow_html=True,
)
