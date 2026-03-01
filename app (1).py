import streamlit as st
import re
import nltk
import numpy as np
import joblib
import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

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

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #0d0d0d;
    color: #e8e0d0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

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
.hero-tag { font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: #f0c040; margin-bottom: 0.5rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1.05; margin: 0 0 0.5rem 0; color: #f5f0e8; }
.hero-sub { font-size: 0.75rem; color: #666; letter-spacing: 0.05em; }

.verdict-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
.verdict-card { border: 1px solid #222; padding: 1.5rem; background: #111; position: relative; }
.verdict-card .model-label { font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase; color: #555; margin-bottom: 0.75rem; }
.verdict-card .verdict-text { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
.verdict-card.real { border-left: 3px solid #4caf82; }
.verdict-card.fake { border-left: 3px solid #e05a5a; }
.verdict-card.uncertain { border-left: 3px solid #f0c040; }
.verdict-card .verdict-text.real { color: #4caf82; }
.verdict-card .verdict-text.fake { color: #e05a5a; }
.verdict-card .conf-bar-wrap { margin-top: 0.75rem; height: 2px; background: #222; }
.verdict-card .conf-bar { height: 100%; transition: width 0.4s ease; }
.conf-bar.real { background: #4caf82; }
.conf-bar.fake { background: #e05a5a; }
.conf-label { font-size: 0.65rem; color: #555; margin-top: 0.3rem; letter-spacing: 0.1em; }

.cmp-table { width: 100%; border-collapse: collapse; font-size: 0.75rem; margin-top: 1rem; }
.cmp-table th { text-align: left; font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: #555; border-bottom: 1px solid #222; padding: 0.5rem 0.75rem; }
.cmp-table td { padding: 0.6rem 0.75rem; border-bottom: 1px solid #181818; color: #ccc; }
.cmp-table tr:last-child td { border-bottom: none; }
.badge-real { color: #4caf82; font-weight: 700; }
.badge-fake { color: #e05a5a; font-weight: 700; }

section[data-testid="stSidebar"] { background: #0a0a0a; border-right: 1px solid #1e1e1e; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

textarea {
    background: #111 !important; border: 1px solid #2a2a2a !important;
    border-radius: 0 !important; color: #e8e0d0 !important;
    font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important;
}
textarea:focus { border-color: #f0c040 !important; box-shadow: none !important; }

.stButton > button {
    background: #f0c040 !important; color: #0d0d0d !important; border: none !important;
    border-radius: 0 !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.85rem !important;
    letter-spacing: 0.1em !important; padding: 0.65rem 2rem !important;
    width: 100% !important; cursor: pointer !important; transition: opacity 0.15s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #1e1e1e !important; }
.stAlert { border-radius: 0 !important; }

[data-testid="stMetric"] { background: #111; border: 1px solid #1e1e1e; padding: 1rem; }
[data-testid="stMetricLabel"] { font-size: 0.6rem !important; letter-spacing: 0.2em; color: #555 !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; color: #f5f0e8 !important; }

.section-label { font-size: 0.6rem; letter-spacing: 0.3em; text-transform: uppercase; color: #555; border-bottom: 1px solid #1e1e1e; padding-bottom: 0.4rem; margin-bottom: 1rem; }

.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 0.4rem; background: #4caf82; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

/* ── How It Works ── */
.algo-block {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border: 1px solid #222;
    margin-bottom: 2rem;
    overflow: hidden;
}
.algo-code {
    background: #0a0a0a;
    border-right: 1px solid #222;
    padding: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    line-height: 1.8;
    color: #9db8a8;
    overflow-x: auto;
}
.algo-math {
    background: #111;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0.75rem;
}
.algo-math-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #f5f0e8;
    margin-bottom: 0.25rem;
}
.algo-math-desc {
    font-size: 0.68rem;
    color: #666;
    line-height: 1.8;
}
.algo-tag {
    display: inline-block;
    font-size: 0.55rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #f0c040;
    border: 1px solid #3a3000;
    padding: 0.15rem 0.5rem;
    margin-bottom: 0.5rem;
    width: fit-content;
}
.kw { color: #c792ea; }
.fn { color: #82aaff; }
.cm { color: #3d5a4a; font-style: italic; }
.st { color: #c3e88d; }
.nm { color: #f78c6c; }

/* ── Explainability ── */
.lime-word { display: inline-block; padding: 2px 4px; margin: 1px; border-radius: 2px; font-size: 0.8rem; font-family: 'Space Mono', monospace; }
.expl-card { border: 1px solid #1e1e1e; background: #111; padding: 1.5rem; margin-bottom: 1rem; }
.expl-title { font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase; color: #555; margin-bottom: 1rem; }
.expl-snippet { font-size: 0.72rem; color: #888; line-height: 1.7; margin-bottom: 1rem; border-left: 2px solid #222; padding-left: 0.75rem; font-style: italic; }
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

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource(show_spinner="Loading SentenceBERT model…")
def load_sbert(model_name: str):
    return SentenceTransformer(model_name)

@st.cache_resource(show_spinner="Loading classifiers…")
def load_classifiers(mlp_path: str, sgd_path: str):
    mlp = joblib.load(mlp_path) if Path(mlp_path).exists() else None
    sgd = joblib.load(sgd_path) if Path(sgd_path).exists() else None
    return mlp, sgd

def predict_with_proba(model, embedding):
    pred = int(model.predict(embedding)[0])
    label = "REAL" if pred == 1 else "FAKE"
    css_cls = "real" if pred == 1 else "fake"
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
SBERT_MODEL = "all-MiniLM-L6-v2"

with st.sidebar:
    st.markdown("<div class='section-label'>⚙ Configuration</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;color:#888;margin-bottom:0.5rem'>"
        "🤖 SentenceBERT: <span style='color:#f0c040'>all-MiniLM-L6-v2</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.markdown("<div class='section-label'>Model Files</div>", unsafe_allow_html=True)
    mlp_path = st.text_input("MLP model path", value="mlp_classifier_model.pkl")
    sgd_path = st.text_input("SGD model path", value="sgd_classifier_model.pkl")
    expl_path = st.text_input("Explanations JSON", value="explanations.json")
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

sbert = load_sbert(SBERT_MODEL)
mlp_model, sgd_model = load_classifiers(mlp_path, sgd_path)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="status-dot"></span>Live · SentenceBERT</div>
    <div class="hero-title">Fake News<br>Detector</div>
    <div class="hero-sub">MLP &amp; SGD classifiers · Dual-model verdict</div>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("SBERT Model", "MiniLM-L6-v2")
with col2:
    st.metric("MLP", "✓ Loaded" if mlp_model else "✗ Not found")
with col3:
    st.metric("SGD", "✓ Loaded" if sgd_model else "✗ Not found")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_explain, tab_how = st.tabs(["🔍 Analyze", "📊 Explainability", "📖 How It Works"])


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 1 — ANALYZE                                            ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_predict:
    st.markdown("<div class='section-label'>Input Text</div>", unsafe_allow_html=True)

    SAMPLES = {
        "— select a sample —": "",
        "🟢 Real: Senate passes bipartisan infrastructure bill": (
            "WASHINGTON — The Senate passed a sweeping bipartisan infrastructure bill on Tuesday in a "
            "69-30 vote, sending the $1.2 trillion measure to the House for consideration. The legislation "
            "represents one of the largest investments in American infrastructure in decades, allocating "
            "funds for roads, bridges, broadband internet expansion, public transit, water systems, and "
            "the electrical grid. Senate Majority Leader Chuck Schumer called it a once-in-a-generation "
            "investment in the backbone of the nation. Several Republican senators who crossed the aisle "
            "argued the bill addressed urgent, long-neglected needs in their home states. The White House "
            "praised the bipartisan vote and urged the House to act swiftly. Transportation Secretary Pete "
            "Buttigieg said the funding would create hundreds of thousands of jobs over the next decade. "
            "Analysts noted the bill passed with more Republican support than many had expected, reflecting "
            "widespread public demand for infrastructure improvements. Construction industry groups welcomed "
            "the news, saying projects had been stalled for years due to lack of federal funding."
        ),
        "🟢 Real: Federal Reserve raises interest rates again": (
            "The Federal Reserve raised its benchmark interest rate by a quarter percentage point on "
            "Wednesday, the tenth increase in just over a year as the central bank continues its sustained "
            "campaign against inflation. Fed Chair Jerome Powell said policymakers remain firmly committed "
            "to bringing inflation back down to the 2% target, though he acknowledged the broader economy "
            "faces growing uncertainty. Markets had widely anticipated the move, and stocks showed little "
            "reaction following the announcement. The federal funds rate now sits at its highest level in "
            "16 years. Some economists have warned the aggressive rate hikes risk tipping the economy into "
            "recession, while others argue the labor market remains strong enough to absorb the pressure. "
            "Powell said the Fed would continue to make decisions meeting by meeting based on incoming data. "
            "Consumer prices have cooled significantly from their peak but remain above the Fed's target. "
            "Housing markets have slowed sharply in response to higher mortgage rates, which have more than "
            "doubled over the past 18 months, pricing many first-time buyers out of the market."
        ),
        "🔴 Fake: Obama signs secret UN military takeover order": (
            "BREAKING: President Obama has quietly signed an executive order handing operational control of "
            "the United States military to the United Nations, multiple sources close to the White House "
            "have confirmed exclusively to this outlet. The order, buried inside a routine budget amendment "
            "released late Friday evening, grants UN peacekeeping forces the authority to deploy on American "
            "soil without congressional approval or public notice. Legal experts who reviewed the document "
            "say it effectively bypasses the Constitution and strips Congress of its war powers. Patriot "
            "groups across the country are calling it the single biggest act of treason ever committed by "
            "an American president. The mainstream media is completely ignoring the story. Gun rights "
            "organizations are urging citizens to contact their representatives immediately. Several "
            "retired generals have allegedly refused to comply with the new directive. The Pentagon has "
            "not responded to requests for comment. Share this article before it gets scrubbed from the "
            "internet. The globalist agenda to disarm and occupy America is no longer a theory — it is "
            "happening right now, and Washington insiders say the window to stop it is closing fast."
        ),
        "🔴 Fake: Hillary Clinton secretly arrested at Dulles airport": (
            "Hillary Clinton was detained at Dulles International Airport in the early hours of Friday "
            "morning by federal agents operating under a sealed indictment connected to her private email "
            "server and alleged involvement in a sprawling international trafficking network, according to "
            "multiple high-level sources who spoke on condition of anonymity. The arrest was carried out "
            "quietly to avoid media attention, and Clinton was reportedly transported to an undisclosed "
            "government facility for questioning. The mainstream media, which has long protected Clinton, "
            "is completely refusing to cover the story. Insiders say the indictment has been sitting with "
            "a federal judge for months, waiting for the right moment. The Deep State is now in full panic "
            "mode, desperately working to suppress the information before it reaches ordinary Americans. "
            "Social media platforms have already begun removing posts about the arrest. Former colleagues "
            "of Clinton reportedly received calls warning them to stay silent. Citizens are urged to "
            "screenshot and share this report immediately before it disappears. Justice, long denied, "
            "may finally be coming to one of Washington's most protected political figures."
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

    word_count = len(news_text.split()) if news_text.strip() else 0
    st.markdown(
        f"<div style='font-size:0.65rem;color:#444;text-align:right;margin-top:-0.5rem'>"
        f"{word_count} words · {len(news_text)} chars</div>",
        unsafe_allow_html=True,
    )

    analyze_btn = st.button("ANALYZE", disabled=(not news_text.strip()))

    if analyze_btn and news_text.strip():
        if not mlp_model and not sgd_model:
            st.error("No models found. Train and save your models first using the notebook.")
        else:
            with st.spinner("Preprocessing and embedding…"):
                cleaned = preprocess_text(news_text)
                embedding = sbert.encode([cleaned])

            st.markdown("---")
            st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)

            with st.expander("View preprocessed text"):
                st.code(cleaned, language=None)

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

            verdicts = [r["Verdict"] for r in results if r["Verdict"] != "N/A"]
            if len(verdicts) == 2:
                if verdicts[0] == verdicts[1]:
                    emoji = "✅" if verdicts[0] == "REAL" else "🚨"
                    st.success(f"{emoji} Both models agree: **{verdicts[0]}**")
                else:
                    st.warning("⚠️ Models disagree — treat result with caution.")

            st.markdown("---")
            st.markdown("<div class='section-label'>Comparison</div>", unsafe_allow_html=True)
            table_rows = ""
            for r in results:
                if r["Verdict"] == "N/A":
                    table_rows += f"<tr><td>{r['Model']}</td><td style='color:#555'>N/A</td><td style='color:#555'>—</td></tr>"
                else:
                    badge_cls = "badge-real" if r["Verdict"] == "REAL" else "badge-fake"
                    conf_str = f"{round(r['Confidence']*100,1)}%" if r["Confidence"] else "—"
                    table_rows += f"<tr><td>{r['Model']}</td><td class='{badge_cls}'>{r['Verdict']}</td><td>{conf_str}</td></tr>"

            st.markdown(f"""
            <table class="cmp-table">
                <thead><tr><th>Model</th><th>Verdict</th><th>Confidence</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 2 — EXPLAINABILITY                                     ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_explain:
    expl_file = Path(expl_path)

    if not expl_file.exists():
        st.warning("No `explanations.json` found. Run the Colab export notebook first, then upload the file alongside app.py.")
        st.markdown("""
<div style='font-size:0.75rem; color:#555; line-height:2; margin-top:1rem;'>
<b style='color:#888'>To generate explanations:</b><br>
1. Open your Colab notebook<br>
2. Add and run the <b>LIME + SHAP export cells</b> (see export notebook)<br>
3. Download <code>explanations.json</code><br>
4. Place it in the same folder as <code>app.py</code>
</div>""", unsafe_allow_html=True)
    else:
        with open(expl_file) as f:
            expl_data = json.load(f)

        st.markdown("<div class='section-label'>Confusion Matrix</div>", unsafe_allow_html=True)

        import plotly.figure_factory as ff
        import plotly.graph_objects as go

        for model_key, model_label in [("mlp", "MLP"), ("sgd", "SGD")]:
            if model_key not in expl_data.get("confusion", {}):
                continue
            cm = expl_data["confusion"][model_key]
            fig = ff.create_annotated_heatmap(
                z=[[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                x=["Pred: Fake", "Pred: Real"],
                y=["True: Fake", "True: Real"],
                colorscale=[[0, "#0d0d0d"], [1, "#f0c040"]],
                font_colors=["#e8e0d0"],
            )
            fig.update_layout(
                title=dict(text=f"{model_label} — Confusion Matrix", font=dict(color="#e8e0d0", size=12)),
                paper_bgcolor="#111", plot_bgcolor="#111",
                font=dict(family="Space Mono", color="#e8e0d0"),
                margin=dict(l=10, r=10, t=50, b=10),
                height=300,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-label'>Confidence Calibration</div>", unsafe_allow_html=True)
        st.caption("How well does the model's confidence match its actual accuracy? A perfectly calibrated model follows the diagonal.")

        if "calibration" in expl_data:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(color="#333", dash="dash", width=1))
            colors = {"mlp": "#f0c040", "sgd": "#4caf82"}
            for model_key, model_label in [("mlp", "MLP"), ("sgd", "SGD")]:
                if model_key not in expl_data["calibration"]:
                    continue
                cal = expl_data["calibration"][model_key]
                fig.add_trace(go.Scatter(
                    x=cal["mean_predicted"], y=cal["fraction_positive"],
                    mode="lines+markers", name=model_label,
                    line=dict(color=colors[model_key], width=2),
                    marker=dict(size=6),
                ))
            fig.update_layout(
                paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
                font=dict(family="Space Mono", color="#e8e0d0", size=11),
                xaxis=dict(title="Mean Predicted Confidence", gridcolor="#1e1e1e", range=[0,1]),
                yaxis=dict(title="Fraction Positive (Actual)", gridcolor="#1e1e1e", range=[0,1]),
                legend=dict(bgcolor="#111", bordercolor="#222"),
                margin=dict(l=10, r=10, t=20, b=10),
                height=350,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-label'>SHAP — Top Feature Dimensions</div>", unsafe_allow_html=True)
        st.caption("Mean absolute SHAP values across the 384 embedding dimensions. Higher = more influential to the prediction.")

        if "shap" in expl_data:
            for model_key, model_label in [("mlp", "MLP"), ("sgd", "SGD")]:
                if model_key not in expl_data["shap"]:
                    continue
                shap_vals = expl_data["shap"][model_key]
                top_dims = shap_vals[:20]
                fig = go.Figure(go.Bar(
                    x=[v["value"] for v in top_dims],
                    y=[f"dim {v['dim']}" for v in top_dims],
                    orientation="h",
                    marker=dict(color="#f0c040" if model_key == "mlp" else "#4caf82"),
                ))
                fig.update_layout(
                    title=dict(text=f"{model_label} — Top 20 SHAP Dimensions", font=dict(color="#e8e0d0", size=12)),
                    paper_bgcolor="#111", plot_bgcolor="#0d0d0d",
                    font=dict(family="Space Mono", color="#e8e0d0", size=10),
                    xaxis=dict(title="Mean |SHAP|", gridcolor="#1e1e1e"),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(l=10, r=10, t=40, b=10),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-label'>LIME — Word-level Explanations</div>", unsafe_allow_html=True)
        st.caption("Words highlighted in green pushed the prediction toward REAL. Red pushed toward FAKE. Intensity = strength of influence.")

        if "lime" in expl_data:
            samples = expl_data["lime"]
            for i, sample in enumerate(samples[:6]):
                model_label = sample.get("model", "").upper()
                true_label = "REAL" if sample.get("true_label") == 1 else "FAKE"
                pred_label = "REAL" if sample.get("pred_label") == 1 else "FAKE"
                match = true_label == pred_label
                border_color = "#4caf82" if match else "#e05a5a"

                word_weights = {w: v for w, v in sample.get("word_weights", {}).items()}
                max_abs = max(abs(v) for v in word_weights.values()) if word_weights else 1

                text_tokens = sample.get("text", "").split()
                highlighted = []
                for word in text_tokens:
                    w_clean = re.sub(r'[^a-z]', '', word.lower())
                    weight = word_weights.get(w_clean, 0)
                    intensity = int(min(abs(weight) / max_abs * 180, 180))
                    if weight > 0.01:
                        bg = f"rgba(76,175,130,{intensity/255:.2f})"
                        fg = "#fff" if intensity > 100 else "#e8e0d0"
                    elif weight < -0.01:
                        bg = f"rgba(224,90,90,{intensity/255:.2f})"
                        fg = "#fff" if intensity > 100 else "#e8e0d0"
                    else:
                        bg, fg = "transparent", "#777"
                    highlighted.append(f"<span class='lime-word' style='background:{bg};color:{fg}'>{word}</span>")

                highlighted_html = " ".join(highlighted)
                correct_str = "✓ Correct" if match else "✗ Incorrect"
                correct_color = "#4caf82" if match else "#e05a5a"

                st.markdown(f"""
                <div class="expl-card" style="border-left: 3px solid {border_color}">
                    <div class="expl-title">
                        Sample {i+1} · {model_label} ·
                        True: <b style='color:#e8e0d0'>{true_label}</b> ·
                        Pred: <b style='color:#e8e0d0'>{pred_label}</b> ·
                        <span style='color:{correct_color}'>{correct_str}</span>
                    </div>
                    <div style='line-height:2.2;'>{highlighted_html}</div>
                </div>""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TAB 3 — HOW IT WORKS                                       ║
# ╚══════════════════════════════════════════════════════════════╝
with tab_how:
    st.markdown("<div class='section-label'>Algorithm Explanations</div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;color:#555;margin-bottom:2rem;'>Code on the left · Math on the right</div>",
        unsafe_allow_html=True
    )

    # ── 1. SentenceBERT ──────────────────────────────────────────
    st.markdown("""
<div class="algo-block">
  <div class="algo-code"><pre><span class="cm"># 1. Load pretrained transformer</span>
<span class="kw">from</span> sentence_transformers <span class="kw">import</span> SentenceTransformer

model = SentenceTransformer(<span class="st">"all-MiniLM-L6-v2"</span>)

<span class="cm"># 2. Encode raw text → dense vector</span>
texts = [<span class="st">"Senate passes infrastructure bill"</span>,
         <span class="st">"Clinton arrested by feds"</span>]

embeddings = model.encode(texts)
<span class="cm"># shape: (2, 384)</span>

<span class="cm"># 3. Cosine similarity between docs</span>
sim = model.similarity(embeddings[<span class="nm">0</span>],
                       embeddings[<span class="nm">1</span>])
</pre></div>
  <div class="algo-math">
    <div class="algo-tag">SentenceBERT · Embedding Model</div>
    <div class="algo-math-title">Sentence Embeddings via Siamese BERT</div>
    <div class="algo-math-desc">
      A pretrained BERT encoder maps variable-length text to a fixed-size dense vector via mean pooling over token representations:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">u = MeanPool(BERT(tokens))</code><br><br>
      The model is fine-tuned on sentence pairs using a siamese network with cosine similarity loss, so semantically similar sentences cluster close together in the 384-dimensional space:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">sim(u, v) = (u · v) / (‖u‖ ‖v‖)</code><br><br>
      Why use it? Bag-of-words models miss meaning. "Bank robbery" and "financial institution heist" share no words but have near-identical embeddings.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 2. MLP ───────────────────────────────────────────────────
    st.markdown("""
<div class="algo-block">
  <div class="algo-code"><pre><span class="kw">from</span> sklearn.neural_network <span class="kw">import</span> MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=(<span class="nm">256</span>, <span class="nm">128</span>),
    activation=<span class="st">"relu"</span>,
    solver=<span class="st">"adam"</span>,
    max_iter=<span class="nm">200</span>,
    random_state=<span class="nm">42</span>,
)

<span class="cm"># Train on SBERT embeddings</span>
mlp.fit(train_embeddings, train_labels)

<span class="cm"># Predict with probability</span>
proba = mlp.predict_proba(test_embeddings)
<span class="cm"># [[0.03, 0.97], [0.91, 0.09], ...]</span>
</pre></div>
  <div class="algo-math">
    <div class="algo-tag">MLP · Multi-Layer Perceptron</div>
    <div class="algo-math-title">Feedforward Neural Network</div>
    <div class="algo-math-desc">
      Each layer applies a linear transformation followed by a non-linear activation:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">h⁽ˡ⁾ = ReLU(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)</code><br><br>
      The final layer uses softmax to output a probability distribution over classes:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">p(y=k | x) = exp(zₖ) / Σⱼ exp(zⱼ)</code><br><br>
      Weights are updated by backpropagating the cross-entropy loss with the Adam optimiser. The MLP can learn non-linear decision boundaries in the embedding space — useful when real vs. fake news clusters aren't linearly separable.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── 3. SGD ───────────────────────────────────────────────────
    st.markdown("""
<div class="algo-block">
  <div class="algo-code"><pre><span class="kw">from</span> sklearn.linear_model <span class="kw">import</span> SGDClassifier

sgd = SGDClassifier(
    loss=<span class="st">"modified_huber"</span>,  <span class="cm"># gives proba</span>
    class_weight=<span class="st">"balanced"</span>,
    random_state=<span class="nm">42</span>,
)

<span class="cm"># Single pass through data</span>
sgd.fit(train_embeddings, train_labels)

<span class="cm"># Decision boundary distance</span>
scores = sgd.decision_function(test_emb)
<span class="cm"># positive = leans real, negative = fake</span>
</pre></div>
  <div class="algo-math">
    <div class="algo-tag">SGD · Stochastic Gradient Descent</div>
    <div class="algo-math-title">Linear Classifier with SGD Optimisation</div>
    <div class="algo-math-desc">
      Finds a hyperplane <b>w</b> that separates real from fake news in embedding space, minimising the hinge loss on one sample at a time:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">L(w) = max(0, 1 − yᵢ (w · xᵢ + b))</code><br><br>
      Weight update per sample:<br><br>
      <code style="color:#c3e88d; font-size:0.75rem">w ← w − η ∇L(w)</code><br><br>
      Because embeddings are already rich and high-dimensional (384 dims), a linear boundary often works well. SGD trains in a single pass, making it far faster than MLP — but it cannot capture non-linear patterns in the embedding space.
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label'>Why This Pipeline?</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.72rem; color:#666; line-height:2; max-width:700px;'>
Traditional fake news detectors use TF-IDF bag-of-words features, which treat each word independently and ignore context.
SentenceBERT encodes the <em>meaning</em> of the full article into a single vector, letting the downstream classifiers
focus on semantic patterns rather than surface vocabulary.<br><br>
Running both MLP and SGD lets us see whether the relationship between embeddings and labels
is linear (SGD performs well) or requires a more complex boundary (MLP outperforms SGD).
Comparing the two is itself a diagnostic tool.
</div>
""", unsafe_allow_html=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='font-size:0.6rem;color:#333;text-align:center;letter-spacing:0.15em;'>"
    "FAKE NEWS DETECTOR · SBERT + MLP / SGD · FOR RESEARCH USE ONLY"
    "</div>",
    unsafe_allow_html=True,
)

@st.cache_resource
def setup_nltk():
    for pkg in ['punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return ' '.join(tokens)

@st.cache_resource(show_spinner="Loading SentenceBERT model…")
def load_sbert(name): return SentenceTransformer(name)

@st.cache_resource(show_spinner="Loading classifiers…")
def load_classifiers(mp, sp):
    mlp = joblib.load(mp) if Path(mp).exists() else None
    sgd = joblib.load(sp) if Path(sp).exists() else None
    return mlp, sgd

def predict_with_proba(model, emb):
    pred = int(model.predict(emb)[0])
    label = "REAL" if pred == 1 else "FAKE"
    css = "real" if pred == 1 else "fake"
    if hasattr(model, 'predict_proba'):
        conf = float(max(model.predict_proba(emb)[0]))
    elif hasattr(model, 'decision_function'):
        d = model.decision_function(emb)[0]
        conf = float(min(abs(d)/(abs(d)+1)+0.5, 0.99))
    else:
        conf = 0.0
    return pred, label, css, conf

def dark_fig(w=6, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#111111")
    for s in ax.spines.values(): s.set_edgecolor("#2a2a2a")
    ax.tick_params(colors="#555")
    ax.xaxis.label.set_color("#888"); ax.yaxis.label.set_color("#888"); ax.title.set_color("#888")
    ax.grid(True, color="#1e1e1e", linestyle="--", linewidth=0.5)
    return fig, ax


SBERT_MODEL = "all-MiniLM-L6-v2"
with st.sidebar:
    st.markdown("<div class='section-label'>⚙ Configuration</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#888;margin-bottom:0.5rem'>🤖 SentenceBERT: <span style='color:#f0c040'>all-MiniLM-L6-v2</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='section-label'>Model Files</div>", unsafe_allow_html=True)
    mlp_path = st.text_input("MLP model path", value="mlp_classifier_model.pkl")
    sgd_path = st.text_input("SGD model path", value="sgd_classifier_model.pkl")
    st.markdown("---")
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem; color:#555; line-height:1.8;'><b style='color:#888'>Pipeline</b><br>Text → Lemmatize → SentenceBERT → Classifier<br><br><b style='color:#888'>Labels</b><br><span style='color:#4caf82'>■</span> 1 = Real news<br><span style='color:#e05a5a'>■</span> 0 = Fake news</div>", unsafe_allow_html=True)

sbert = load_sbert(SBERT_MODEL)
mlp_model, sgd_model = load_classifiers(mlp_path, sgd_path)

st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="status-dot"></span>Live · SentenceBERT</div>
    <div class="hero-title">Fake News<br>Detector</div>
    <div class="hero-sub">MLP &amp; SGD classifiers · Dual-model verdict</div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("SBERT Model", "MiniLM-L6-v2")
with c2: st.metric("MLP", "✓ Loaded" if mlp_model else "✗ Not found")
with c3: st.metric("SGD", "✓ Loaded" if sgd_model else "✗ Not found")
st.markdown("---")


tab_predict, tab_explain, tab_howit = st.tabs(["Predict", "Analysis & Explainability", "How It Works"])

# ══ TAB 1 — PREDICT ══════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("<div class='section-label'>Input Text</div>", unsafe_allow_html=True)
    SAMPLES = {
        "— select a sample —": "",
        "🟢 Real: Senate passes bipartisan infrastructure bill": (
            "WASHINGTON — The Senate passed a sweeping bipartisan infrastructure bill on Tuesday in a "
            "69-30 vote, sending the $1.2 trillion measure to the House for consideration. The legislation "
            "represents one of the largest investments in American infrastructure in decades, allocating "
            "funds for roads, bridges, broadband internet expansion, public transit, water systems, and "
            "the electrical grid. Senate Majority Leader Chuck Schumer called it a once-in-a-generation "
            "investment in the backbone of the nation. Several Republican senators who crossed the aisle "
            "argued the bill addressed urgent, long-neglected needs in their home states. The White House "
            "praised the bipartisan vote and urged the House to act swiftly. Transportation Secretary Pete "
            "Buttigieg said the funding would create hundreds of thousands of jobs over the next decade. "
            "Analysts noted the bill passed with more Republican support than many had expected, reflecting "
            "widespread public demand for infrastructure improvements. Construction industry groups welcomed "
            "the news, saying projects had been stalled for years due to lack of federal funding."
        ),
        "🟢 Real: Federal Reserve raises interest rates again": (
            "The Federal Reserve raised its benchmark interest rate by a quarter percentage point on "
            "Wednesday, the tenth increase in just over a year as the central bank continues its sustained "
            "campaign against inflation. Fed Chair Jerome Powell said policymakers remain firmly committed "
            "to bringing inflation back down to the 2% target, though he acknowledged the broader economy "
            "faces growing uncertainty. Markets had widely anticipated the move, and stocks showed little "
            "reaction following the announcement. The federal funds rate now sits at its highest level in "
            "16 years. Some economists have warned the aggressive rate hikes risk tipping the economy into "
            "recession, while others argue the labor market remains strong enough to absorb the pressure. "
            "Powell said the Fed would continue to make decisions meeting by meeting based on incoming data. "
            "Consumer prices have cooled significantly from their peak but remain above the Fed's target. "
            "Housing markets have slowed sharply in response to higher mortgage rates, which have more than "
            "doubled over the past 18 months, pricing many first-time buyers out of the market."
        ),
        "🔴 Fake: Obama signs secret UN military takeover order": (
            "BREAKING: President Obama has quietly signed an executive order handing operational control of "
            "the United States military to the United Nations, multiple sources close to the White House "
            "have confirmed exclusively to this outlet. The order, buried inside a routine budget amendment "
            "released late Friday evening, grants UN peacekeeping forces the authority to deploy on American "
            "soil without congressional approval or public notice. Legal experts who reviewed the document "
            "say it effectively bypasses the Constitution and strips Congress of its war powers. Patriot "
            "groups across the country are calling it the single biggest act of treason ever committed by "
            "an American president. The mainstream media is completely ignoring the story. Gun rights "
            "organizations are urging citizens to contact their representatives immediately. Several "
            "retired generals have allegedly refused to comply with the new directive. The Pentagon has "
            "not responded to requests for comment. Share this article before it gets scrubbed from the "
            "internet. The globalist agenda to disarm and occupy America is no longer a theory — it is "
            "happening right now, and Washington insiders say the window to stop it is closing fast."
        ),
        "🔴 Fake: Hillary Clinton secretly arrested at Dulles airport": (
            "Hillary Clinton was detained at Dulles International Airport in the early hours of Friday "
            "morning by federal agents operating under a sealed indictment connected to her private email "
            "server and alleged involvement in a sprawling international trafficking network, according to "
            "multiple high-level sources who spoke on condition of anonymity. The arrest was carried out "
            "quietly to avoid media attention, and Clinton was reportedly transported to an undisclosed "
            "government facility for questioning. The mainstream media, which has long protected Clinton, "
            "is completely refusing to cover the story. Insiders say the indictment has been sitting with "
            "a federal judge for months, waiting for the right moment. The Deep State is now in full panic "
            "mode, desperately working to suppress the information before it reaches ordinary Americans. "
            "Social media platforms have already begun removing posts about the arrest. Former colleagues "
            "of Clinton reportedly received calls warning them to stay silent. Citizens are urged to "
            "screenshot and share this report immediately before it disappears. Justice, long denied, "
            "may finally be coming to one of Washington's most protected political figures."
        ),
    }
    sample_choice = st.selectbox("Load a sample", list(SAMPLES.keys()))
    news_text = st.text_area("text", value=SAMPLES[sample_choice], height=200,
                              placeholder="Paste a news article, headline, or paragraph…", label_visibility="collapsed")
    wc = len(news_text.split()) if news_text.strip() else 0
    st.markdown(f"<div style='font-size:0.65rem;color:#444;text-align:right;margin-top:-0.5rem'>{wc} words · {len(news_text)} chars</div>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE", disabled=(not news_text.strip()))

    if analyze_btn and news_text.strip():
        if not mlp_model and not sgd_model:
            st.error("No models found. Train and save models first.")
        else:
            with st.spinner("Preprocessing and embedding…"):
                cleaned = preprocess_text(news_text)
                embedding = sbert.encode([cleaned])
            st.session_state.update({"last_text": news_text, "last_cleaned": cleaned, "last_embedding": embedding})

            st.markdown("---")
            st.markdown("<div class='section-label'>Results</div>", unsafe_allow_html=True)
            with st.expander("View preprocessed text"):
                st.code(cleaned, language=None)

            results = []
            cards_html = '<div class="verdict-wrap">'
            for model_obj, model_name in [(mlp_model, "MLP"), (sgd_model, "SGD")]:
                if model_obj is None:
                    cards_html += f'<div class="verdict-card uncertain"><div class="model-label">{model_name} Classifier</div><div class="verdict-text" style="color:#555">N/A</div><div class="conf-label">Model not loaded</div></div>'
                    results.append({"Model": model_name, "Verdict": "N/A", "Confidence": None})
                    continue
                pred, label, css, conf = predict_with_proba(model_obj, embedding)
                results.append({"Model": model_name, "Verdict": label, "Pred": pred, "Confidence": conf})
                cp = round(conf*100, 1)
                cards_html += f'<div class="verdict-card {css}"><div class="model-label">{model_name} Classifier</div><div class="verdict-text {css}">{label}</div><div class="conf-bar-wrap"><div class="conf-bar {css}" style="width:{cp}%"></div></div><div class="conf-label">Confidence: {cp}%</div></div>'
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            verdicts = [r["Verdict"] for r in results if r["Verdict"] != "N/A"]
            if len(verdicts) == 2:
                if verdicts[0] == verdicts[1]:
                    emoji = "✅" if verdicts[0] == "REAL" else "🚨"
                    st.success(f"{emoji} Both models agree: **{verdicts[0]}**")
                else:
                    st.warning("⚠️ Models disagree — treat result with caution.")

            st.markdown("---")
            st.markdown("<div class='section-label'>Comparison</div>", unsafe_allow_html=True)
            rows = ""
            for r in results:
                if r["Verdict"] == "N/A":
                    rows += f"<tr><td>{r['Model']}</td><td style='color:#555'>N/A</td><td style='color:#555'>—</td></tr>"
                else:
                    bc = "badge-real" if r["Verdict"] == "REAL" else "badge-fake"
                    cs = f"{round(r['Confidence']*100,1)}%" if r["Confidence"] else "—"
                    rows += f"<tr><td>{r['Model']}</td><td class='{bc}'>{r['Verdict']}</td><td>{cs}</td></tr>"
            st.markdown(f'<table class="cmp-table"><thead><tr><th>Model</th><th>Verdict</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)
            st.info("💡 Head to **Analysis & Explainability** to see LIME, SHAP, calibration, and confusion matrix.")


# ══ TAB 2 — ANALYSIS ═════════════════════════════════════════════════════════
with tab_explain:
    st.markdown("<div class='section-label'>Analysis & Explainability</div>", unsafe_allow_html=True)
    if "last_embedding" not in st.session_state:
        st.markdown("<div style='color:#555;font-size:0.8rem;padding:3rem 0;text-align:center;'>Run a prediction in the <b>Predict</b> tab first.</div>", unsafe_allow_html=True)
    else:
        embedding = st.session_state["last_embedding"]
        cleaned   = st.session_state["last_cleaned"]
        raw_text  = st.session_state["last_text"]

        # ── LIME ──────────────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>LIME — Word-level Local Explanation</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:1.25rem;line-height:1.8;'>LIME removes words one by one and watches how the prediction changes. Words that flip the result most are most influential. <span style='color:#4caf82'>Green</span> = pushes toward REAL, <span style='color:#e05a5a'>red</span> = pushes toward FAKE.</div>", unsafe_allow_html=True)
        try:
            from lime.lime_text import LimeTextExplainer
            def pipeline_mlp(texts):
                embs = sbert.encode([preprocess_text(t) for t in texts])
                if hasattr(mlp_model, 'predict_proba'): return mlp_model.predict_proba(embs)
                p = mlp_model.predict(embs).astype(float); return np.column_stack([1-p, p])
            def pipeline_sgd(texts):
                embs = sbert.encode([preprocess_text(t) for t in texts])
                if hasattr(sgd_model, 'predict_proba'): return sgd_model.predict_proba(embs)
                d = sgd_model.decision_function(embs); p = 1/(1+np.exp(-d)); return np.column_stack([1-p, p])
            explainer = LimeTextExplainer(class_names=["FAKE", "REAL"])
            lime_cols = st.columns(2)
            for col, (model_obj, model_name, fn) in zip(lime_cols, [(mlp_model,"MLP",pipeline_mlp),(sgd_model,"SGD",pipeline_sgd)]):
                with col:
                    if model_obj is None:
                        st.markdown(f"<div style='color:#444;font-size:0.75rem;padding:1rem'>{model_name} not loaded.</div>", unsafe_allow_html=True); continue
                    with st.spinner(f"Running LIME for {model_name}…"):
                        exp = explainer.explain_instance(raw_text, fn, num_features=12, num_samples=400)
                    items = exp.as_list()
                    if not items: st.write("No explanation generated."); continue
                    words, weights = zip(*items)
                    colors = ["#4caf82" if w > 0 else "#e05a5a" for w in weights]
                    fig, ax = dark_fig(5, 4.2)
                    ax.barh(range(len(words)), weights, color=colors, height=0.55, zorder=2)
                    ax.set_yticks(range(len(words))); ax.set_yticklabels(words, fontsize=8, color="#ccc")
                    ax.axvline(0, color="#333", linewidth=1)
                    ax.set_title(f"LIME — {model_name}", fontsize=9, pad=8)
                    ax.set_xlabel("Weight (→ REAL)", fontsize=8)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except ImportError:
            st.warning("Install LIME: `pip install lime`")

        st.markdown("---")

        # ── SHAP ──────────────────────────────────────────────────────────────
        st.markdown("<div class='section-label'>SHAP — Embedding Dimension Importance</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:1.25rem;line-height:1.8;'>SHAP assigns each of the 384 embedding dimensions a contribution score. We show the top 15 by absolute impact. <span style='color:#4caf82'>Green</span> = toward REAL, <span style='color:#e05a5a'>red</span> = toward FAKE.</div>", unsafe_allow_html=True)
        try:
            import shap
            shap_cols = st.columns(2)
            for col, (model_obj, model_name) in zip(shap_cols, [(mlp_model,"MLP"),(sgd_model,"SGD")]):
                with col:
                    if model_obj is None:
                        st.markdown(f"<div style='color:#444;font-size:0.75rem;padding:1rem'>{model_name} not loaded.</div>", unsafe_allow_html=True); continue
                    with st.spinner(f"Running SHAP for {model_name}…"):
                        bg = np.zeros((1, embedding.shape[1]))
                        if hasattr(model_obj, 'predict_proba'):
                            fn = model_obj.predict_proba
                        else:
                            def fn(x):
                                d = model_obj.decision_function(x); p = 1/(1+np.exp(-d)); return np.column_stack([1-p, p])
                        ex = shap.KernelExplainer(fn, bg)
                        sv = ex.shap_values(embedding, nsamples=150, silent=True)
                        sv_real = sv[1][0] if isinstance(sv, list) else sv[0]
                    top_k = 15
                    top_idx = np.argsort(np.abs(sv_real))[-top_k:][::-1]
                    top_vals = sv_real[top_idx]
                    top_labels = [f"dim {i}" for i in top_idx]
                    colors = ["#4caf82" if v > 0 else "#e05a5a" for v in top_vals]
                    fig, ax = dark_fig(5, 4.2)
                    ax.barh(range(top_k), top_vals[::-1], color=colors[::-1], height=0.55, zorder=2)
                    ax.set_yticks(range(top_k)); ax.set_yticklabels(top_labels[::-1], fontsize=7, color="#ccc")
                    ax.axvline(0, color="#333", linewidth=1)
                    ax.set_title(f"SHAP — {model_name} (top {top_k} dims)", fontsize=9, pad=8)
                    ax.set_xlabel("SHAP value (→ REAL)", fontsize=8)
                    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        except ImportError:
            st.warning("Install SHAP: `pip install shap`")

        st.markdown("---")

        # ── Confidence calibration ─────────────────────────────────────────────
        st.markdown("<div class='section-label'>Confidence Calibration</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:1.25rem;line-height:1.8;'>How confident is each model in this prediction? The dashed line marks the 50% decision boundary. MLP uses softmax so scores are well-calibrated probabilities. SGD's score is derived from its decision function distance and should be interpreted more loosely.</div>", unsafe_allow_html=True)
        model_confs = []
        for model_obj, model_name in [(mlp_model,"MLP"),(sgd_model,"SGD")]:
            if model_obj is None: continue
            _, label, css, conf = predict_with_proba(model_obj, embedding)
            model_confs.append((model_name, conf, label, "#4caf82" if css=="real" else "#e05a5a"))
        if model_confs:
            fig, ax = dark_fig(7, 2.2)
            y_labels = [f"{m}  ({l})" for m,_,l,_ in model_confs]
            confs = [c for _,c,_,_ in model_confs]
            colors = [col for _,_,_,col in model_confs]
            bars = ax.barh(y_labels, confs, color=colors, height=0.35, zorder=2)
            ax.set_xlim(0, 1)
            ax.axvline(0.5, color="#f0c040", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.text(0.51, -0.55, "decision boundary", fontsize=7, color="#f0c040", alpha=0.7)
            for bar, conf in zip(bars, confs):
                ax.text(min(conf+0.02, 0.93), bar.get_y()+bar.get_height()/2, f"{conf*100:.1f}%", va="center", fontsize=9, color="#e8e0d0")
            ax.set_xlabel("Confidence score", fontsize=8)
            ax.set_title("Model Confidence — Current Prediction", fontsize=9, pad=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)

        st.markdown("---")

        # ── Confusion Matrix ───────────────────────────────────────────────────
        st.markdown("<div class='section-label'>Confusion Matrix</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:1.25rem;line-height:1.8;'>Upload test set predictions to see how each model performs across all examples. Generate the CSV from your notebook:</div>", unsafe_allow_html=True)
        st.code("pd.DataFrame({'true_label': test_df['label'], 'mlp_pred': mlp_tp, 'sgd_pred': sgd_tp}).to_csv('test_predictions.csv', index=False)", language="python")
        uploaded = st.file_uploader("Upload test_predictions.csv", type="csv", key="cm_upload")
        if uploaded:
            import pandas as pd
            from sklearn.metrics import confusion_matrix, classification_report
            df_cm = pd.read_csv(uploaded)
            if not {"true_label","mlp_pred","sgd_pred"}.issubset(df_cm.columns):
                st.error("CSV must have columns: true_label, mlp_pred, sgd_pred")
            else:
                cm_cols = st.columns(2)
                for col, (pc, mn) in zip(cm_cols, [("mlp_pred","MLP"),("sgd_pred","SGD")]):
                    with col:
                        cm = confusion_matrix(df_cm["true_label"], df_cm[pc])
                        fig, ax = dark_fig(4, 3.5)
                        ax.imshow(cm, cmap="YlOrRd", aspect="auto", vmin=0)
                        ax.set_xticks([0,1]); ax.set_yticks([0,1])
                        ax.set_xticklabels(["FAKE","REAL"], fontsize=9, color="#ccc")
                        ax.set_yticklabels(["FAKE","REAL"], fontsize=9, color="#ccc")
                        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
                        ax.set_title(f"Confusion Matrix — {mn}", fontsize=9, pad=8)
                        for i in range(2):
                            for j in range(2):
                                ax.text(j, i, str(cm[i,j]), ha="center", va="center", fontsize=16, color="white", fontweight="bold")
                        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
                        rpt = classification_report(df_cm["true_label"], df_cm[pc], target_names=["FAKE","REAL"], output_dict=True)
                        st.markdown(f"<div style='font-size:0.7rem;color:#666;margin-top:0.5rem;'>Accuracy: <span style='color:#f0c040'>{rpt['accuracy']*100:.1f}%</span> &nbsp;·&nbsp; FAKE F1: <span style='color:#e05a5a'>{rpt['FAKE']['f1-score']:.3f}</span> &nbsp;·&nbsp; REAL F1: <span style='color:#4caf82'>{rpt['REAL']['f1-score']:.3f}</span></div>", unsafe_allow_html=True)


# ══ TAB 3 — HOW IT WORKS ═════════════════════════════════════════════════════
with tab_howit:
    st.markdown("<div class='section-label'>Algorithm Explanations</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.72rem;color:#555;margin-bottom:2rem;line-height:1.8;'>Implementation on the left · Mathematics on the right.</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="algo-card">
  <div class="algo-header">
    <span class="algo-name">SentenceBERT</span>
    <span class="algo-tag">Embedding · Stage 1</span>
  </div>
  <div class="algo-body">
    <div class="algo-code">
<pre style="margin:0;color:#e8e0d0;font-size:0.7rem;line-height:1.8;background:transparent;"><span style="color:#f0c040">from</span> sentence_transformers <span style="color:#f0c040">import</span> SentenceTransformer

<span style="color:#555"># Distilled 6-layer BERT, 384-dim output</span>
model = SentenceTransformer(
    <span style="color:#4caf82">"all-MiniLM-L6-v2"</span>
)

<span style="color:#555"># Any length article → fixed 384-dim vector</span>
embedding = model.encode([cleaned_text])

<span style="color:#555"># Semantic similarity check</span>
sim = model.similarity(emb_a, emb_b)
<span style="color:#555"># 1.0 = same meaning, 0.0 = unrelated</span></pre>
    </div>
    <div class="algo-math">
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">TOKEN EMBEDDING (per layer)</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;"><b>h</b><sub>i</sub> = BERT<sub>1..6</sub>( token<sub>i</sub> )</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">MEAN POOLING</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;"><b>e</b> = (1/n) · Σ<sub>i=1..n</sub> <b>h</b><sub>i</sub></div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">COSINE SIMILARITY</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">sim(<b>u</b>,<b>v</b>) = <b>u</b>·<b>v</b> / (‖<b>u</b>‖·‖<b>v</b>‖)</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">TRIPLET TRAINING LOSS</div>
      <div style="font-size:0.82rem;color:#ccc;">L = max(‖<b>e</b><sub>a</sub>−<b>e</b><sub>p</sub>‖ − ‖<b>e</b><sub>a</sub>−<b>e</b><sub>n</sub>‖ + ε, 0)</div>
    </div>
  </div>
  <div class="algo-desc">
    BERT processes every token in context of all others via self-attention across 6 transformer layers. Token outputs are mean-pooled into a single 384-dimensional vector — so an article of any length becomes one fixed-size point in semantic space. Unlike bag-of-words, the same word gets a different embedding depending on context. MiniLM-L6-v2 is a distilled model that retains most of BERT-base's performance at a fraction of the cost.
    <br><br>
    <b style="color:#888">Why it matters here:</b> Fake news articles share vocabulary with real ones but differ in tone, framing, and epistemic hedging. SBERT captures these subtle semantic differences in a way that simple word counts cannot. The embedding is the input to both classifiers below.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="algo-card">
  <div class="algo-header">
    <span class="algo-name">MLP Classifier</span>
    <span class="algo-tag">Neural Network · Non-linear · Stage 2</span>
  </div>
  <div class="algo-body">
    <div class="algo-code">
<pre style="margin:0;color:#e8e0d0;font-size:0.7rem;line-height:1.8;background:transparent;"><span style="color:#f0c040">from</span> sklearn.neural_network <span style="color:#f0c040">import</span> MLPClassifier

mlp = MLPClassifier(
    hidden_layer_sizes=<span style="color:#4caf82">(256, 128)</span>,
    activation=<span style="color:#4caf82">'relu'</span>,     <span style="color:#555"># non-linearity</span>
    solver=<span style="color:#4caf82">'adam'</span>,         <span style="color:#555"># adaptive lr</span>
    max_iter=<span style="color:#4caf82">200</span>,
    early_stopping=<span style="color:#4caf82">True</span>,   <span style="color:#555"># prevent overfit</span>
    random_state=<span style="color:#4caf82">42</span>
)
mlp.fit(train_embeddings, train_labels)

<span style="color:#555"># Calibrated probability output</span>
proba = mlp.predict_proba(embedding)
<span style="color:#555"># → [[P(FAKE), P(REAL)]]</span></pre>
    </div>
    <div class="algo-math">
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">FORWARD PASS (per layer l)</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;"><b>h</b><sup>(l)</sup> = ReLU( <b>W</b><sup>(l)</sup><b>h</b><sup>(l−1)</sup> + <b>b</b><sup>(l)</sup> )</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">SOFTMAX OUTPUT</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">P(y=k|<b>x</b>) = exp(<b>z</b><sub>k</sub>) / Σ<sub>j</sub> exp(<b>z</b><sub>j</sub>)</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">CROSS-ENTROPY LOSS</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">L = −Σ<sub>i</sub> y<sub>i</sub> log(ŷ<sub>i</sub>)</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">ADAM WEIGHT UPDATE</div>
      <div style="font-size:0.82rem;color:#ccc;"><b>w</b> ← <b>w</b> − η · m̂ / (√v̂ + ε)</div>
    </div>
  </div>
  <div class="algo-desc">
    The MLP stacks two hidden layers (384→256→128→2) with ReLU activations. Each layer learns increasingly abstract non-linear combinations of embedding dimensions. Adam adjusts the learning rate per-parameter, making convergence faster than plain gradient descent. Early stopping halts training when validation loss stops improving.
    <br><br>
    <b style="color:#888">Why we use it:</b> Decision boundaries in SBERT embedding space are unlikely to be flat planes — the hidden layers let the MLP learn curved boundaries that better separate fake from real. Softmax output gives well-calibrated probabilities. Downside: slower to train, more hyperparameters to tune than SGD.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="algo-card">
  <div class="algo-header">
    <span class="algo-name">SGD Classifier</span>
    <span class="algo-tag">Linear SVM · Fast Baseline · Stage 2</span>
  </div>
  <div class="algo-body">
    <div class="algo-code">
<pre style="margin:0;color:#e8e0d0;font-size:0.7rem;line-height:1.8;background:transparent;"><span style="color:#f0c040">from</span> sklearn.linear_model <span style="color:#f0c040">import</span> SGDClassifier

sgd = SGDClassifier(
    loss=<span style="color:#4caf82">'hinge'</span>,           <span style="color:#555"># linear SVM</span>
    class_weight=<span style="color:#4caf82">'balanced'</span>,  <span style="color:#555"># handle imbalance</span>
    random_state=<span style="color:#4caf82">42</span>
)
sgd.fit(train_embeddings, train_labels)

<span style="color:#555"># Signed distance from decision hyperplane</span>
score = sgd.decision_function(embedding)
<span style="color:#555"># positive → REAL, negative → FAKE</span>
<span style="color:#555"># larger |score| = further from boundary</span></pre>
    </div>
    <div class="algo-math">
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">LINEAR DECISION FUNCTION</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">f(<b>x</b>) = <b>w</b> · <b>x</b> + b</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">HINGE LOSS (SVM objective)</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">L = max( 0, 1 − y · f(<b>x</b>) )</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">REGULARISED OBJECTIVE</div>
      <div style="font-size:0.82rem;color:#ccc;margin-bottom:1rem;">min ½‖<b>w</b>‖² + C·Σ max(0, 1−y<sub>i</sub>f(<b>x</b><sub>i</sub>))</div>
      <div style="font-size:0.67rem;color:#555;margin-bottom:0.4rem;letter-spacing:0.1em;">STOCHASTIC WEIGHT UPDATE</div>
      <div style="font-size:0.82rem;color:#ccc;"><b>w</b> ← <b>w</b> − η ∇<sub><b>w</b></sub> L(<b>w</b>, <b>x</b><sub>i</sub>, y<sub>i</sub>)</div>
    </div>
  </div>
  <div class="algo-desc">
    The SGD classifier learns a single hyperplane in 384-dimensional space separating REAL from FAKE. With hinge loss it behaves exactly like a linear SVM — maximising the margin between the two classes. Weights are updated one sample at a time (stochastic), making training extremely fast. <code>class_weight='balanced'</code> automatically compensates if one class dominates the training set.
    <br><br>
    <b style="color:#888">Why we use it:</b> SBERT does the heavy lifting — the resulting embeddings are rich enough that a linear model is surprisingly competitive. SGD trains orders of magnitude faster than the MLP and is easier to inspect: the weight vector <b>w</b> directly encodes which embedding directions matter. It serves as a strong interpretable baseline to check whether the MLP's extra complexity is actually earning its keep.
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-size:0.65rem;color:#2a2a2a;line-height:1.8;text-align:center;'>Pipeline: Raw Text → Stopword removal + Lemmatization → SentenceBERT (384-dim) → [MLP | SGD] → {0: FAKE, 1: REAL}</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='font-size:0.6rem;color:#333;text-align:center;letter-spacing:0.15em;'>FAKE NEWS DETECTOR · SBERT + MLP / SGD · FOR RESEARCH USE ONLY</div>", unsafe_allow_html=True)
