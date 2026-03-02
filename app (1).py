import streamlit as st
import re
import nltk
import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Fake News Detector", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Space+Mono:ital,wght@0,400;0,700;1,400&display=swap');

html, body, [class*="css"] { font-family: 'Space Mono', monospace; background-color: #0d0d0d; color: #e8e0d0; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }

.hero { border: 1px solid #2a2a2a; background: linear-gradient(135deg, #111 0%, #1a1a1a 100%); padding: 2.5rem 3rem; margin-bottom: 2rem; position: relative; overflow: hidden; }
.hero::before { content: "VERIFY"; position: absolute; right: -10px; top: 50%; transform: translateY(-50%); font-family: 'Syne', sans-serif; font-size: 9rem; font-weight: 800; color: rgba(255,255,255,0.03); pointer-events: none; letter-spacing: -4px; }
.hero-tag { font-size: 0.65rem; letter-spacing: 0.3em; text-transform: uppercase; color: #f0c040; margin-bottom: 0.5rem; }
.hero-title { font-family: 'Syne', sans-serif; font-size: 2.8rem; font-weight: 800; line-height: 1.05; margin: 0 0 0.5rem 0; color: #f5f0e8; }
.hero-sub { font-size: 0.75rem; color: #666; letter-spacing: 0.05em; }

.verdict-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1.5rem; }
.verdict-card { border: 1px solid #222; padding: 1.5rem; background: #111; }
.verdict-card .model-label { font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase; color: #555; margin-bottom: 0.75rem; }
.verdict-card .verdict-text { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; }
.verdict-card.real { border-left: 3px solid #4caf82; }
.verdict-card.fake { border-left: 3px solid #e05a5a; }
.verdict-card.uncertain { border-left: 3px solid #f0c040; }
.verdict-card .verdict-text.real { color: #4caf82; }
.verdict-card .verdict-text.fake { color: #e05a5a; }
.verdict-card .conf-bar-wrap { margin-top: 0.75rem; height: 2px; background: #222; }
.verdict-card .conf-bar { height: 100%; }
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

textarea { background: #111 !important; border: 1px solid #2a2a2a !important; border-radius: 0 !important; color: #e8e0d0 !important; font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }
textarea:focus { border-color: #f0c040 !important; box-shadow: none !important; }

.stButton > button { background: #f0c040 !important; color: #0d0d0d !important; border: none !important; border-radius: 0 !important; font-family: 'Syne', sans-serif !important; font-weight: 700 !important; font-size: 0.85rem !important; letter-spacing: 0.1em !important; padding: 0.65rem 2rem !important; width: 100% !important; transition: opacity 0.15s !important; }
.stButton > button:hover { opacity: 0.85 !important; }

hr { border-color: #1e1e1e !important; }
.stAlert { border-radius: 0 !important; }

[data-testid="stMetric"] { background: #111; border: 1px solid #1e1e1e; padding: 1rem; }
[data-testid="stMetricLabel"] { font-size: 0.6rem !important; letter-spacing: 0.2em; color: #555 !important; }
[data-testid="stMetricValue"] { font-family: 'Syne', sans-serif !important; font-size: 1.6rem !important; color: #f5f0e8 !important; }

.section-label { font-size: 0.6rem; letter-spacing: 0.3em; text-transform: uppercase; color: #555; border-bottom: 1px solid #1e1e1e; padding-bottom: 0.4rem; margin-bottom: 1rem; }

.status-dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 0.4rem; background: #4caf82; animation: pulse 2s infinite; }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }

.algo-block { display: grid; grid-template-columns: 1fr 1fr; gap: 0; border: 1px solid #222; margin-bottom: 2rem; overflow: hidden; }
.algo-code { background: #0a0a0a; border-right: 1px solid #222; padding: 1.5rem; font-family: 'Space Mono', monospace; font-size: 0.7rem; line-height: 1.8; color: #9db8a8; overflow-x: auto; }
.algo-math { background: #111; padding: 1.5rem; display: flex; flex-direction: column; justify-content: center; gap: 0.75rem; }
.algo-math-title { font-family: 'Syne', sans-serif; font-size: 1rem; font-weight: 700; color: #f5f0e8; margin-bottom: 0.25rem; }
.algo-math-desc { font-size: 0.68rem; color: #666; line-height: 1.8; }
.algo-tag { display: inline-block; font-size: 0.55rem; letter-spacing: 0.2em; text-transform: uppercase; color: #f0c040; border: 1px solid #3a3000; padding: 0.15rem 0.5rem; margin-bottom: 0.5rem; width: fit-content; }
.kw { color: #c792ea; } .fn { color: #82aaff; } .cm { color: #3d5a4a; font-style: italic; } .st { color: #c3e88d; } .nm { color: #f78c6c; }
</style>
""", unsafe_allow_html=True)


# ── NLTK ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def setup_nltk():
    for pkg in ['punkt_tab', 'stopwords', 'wordnet']:
        nltk.download(pkg, quiet=True)
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

def preprocess_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    return ' '.join(lemmatizer.lemmatize(w) for w in tokens if w not in stop_words)


# ── Model loading ─────────────────────────────────────────────────────────────
SBERT_MODEL = "all-MiniLM-L6-v2"

@st.cache_resource(show_spinner="Loading SentenceBERT model…")
def load_sbert(name): return SentenceTransformer(name)

@st.cache_resource(show_spinner="Loading classifiers…")
def load_classifiers(mp, sp):
    mlp = joblib.load(mp) if Path(mp).exists() else None
    sgd = joblib.load(sp) if Path(sp).exists() else None
    return mlp, sgd

def predict_with_proba(model, emb):
    pred = int(model.predict(emb)[0])
    label, css = ("REAL", "real") if pred == 1 else ("FAKE", "fake")
    if hasattr(model, 'predict_proba'):
        conf = float(max(model.predict_proba(emb)[0]))
    elif hasattr(model, 'decision_function'):
        d = model.decision_function(emb)[0]
        conf = float(min(abs(d) / (abs(d) + 1) + 0.5, 0.99))
    else:
        conf = 0.0
    return pred, label, css, conf


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<div class='section-label'>⚙ Configuration</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#888;margin-bottom:0.5rem'>🤖 SentenceBERT: <span style='color:#f0c040'>all-MiniLM-L6-v2</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div class='section-label'>Model Files</div>", unsafe_allow_html=True)
    mlp_path = st.text_input("MLP model path", value="mlp_classifier_model.pkl", key="mlp_path")
    sgd_path = st.text_input("SGD model path", value="sgd_classifier_model.pkl", key="sgd_path")
    st.markdown("---")
    st.markdown("<div class='section-label'>About</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.68rem; color:#555; line-height:1.8;'>
<b style='color:#888'>Pipeline</b><br>Text → Lemmatize → SentenceBERT → Classifier<br><br>
<b style='color:#888'>Models</b><br>• MLP (neural network)<br>• SGD (linear / SVM-like)<br><br>
<b style='color:#888'>Labels</b><br>
<span style='color:#4caf82'>■</span> 1 = Real news<br>
<span style='color:#e05a5a'>■</span> 0 = Fake news
</div>""", unsafe_allow_html=True)


# ── Load resources ────────────────────────────────────────────────────────────
sbert = load_sbert(SBERT_MODEL)
mlp_model, sgd_model = load_classifiers(mlp_path, sgd_path)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="status-dot"></span>Live · SentenceBERT</div>
    <div class="hero-title">Fake News<br>Detector</div>
    <div class="hero-sub">MLP &amp; SGD classifiers · Dual-model verdict</div>
</div>""", unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)
with c1: st.metric("SBERT Model", "MiniLM-L6-v2")
with c2: st.metric("MLP", "✓ Loaded" if mlp_model else "✗ Not found")
with c3: st.metric("SGD", "✓ Loaded" if sgd_model else "✗ Not found")
st.markdown("---")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_predict, tab_how = st.tabs(["🔍 Analyze", "📖 How It Works"])


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 1 — ANALYZE                                    ║
# ╚══════════════════════════════════════════════════════╝
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

    sample_choice = st.selectbox("Load a sample", list(SAMPLES.keys()), key="sample_select")
    news_text = st.text_area(
        "Paste article text here",
        value=SAMPLES[sample_choice],
        height=200,
        placeholder="Paste a news article, headline, or paragraph…",
        label_visibility="collapsed",
        key="news_input",
    )

    word_count = len(news_text.split()) if news_text.strip() else 0
    st.markdown(f"<div style='font-size:0.65rem;color:#444;text-align:right;margin-top:-0.5rem'>{word_count} words · {len(news_text)} chars</div>", unsafe_allow_html=True)
    analyze_btn = st.button("ANALYZE", disabled=(not news_text.strip()), key="analyze_btn")

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
                    cards_html += f'<div class="verdict-card uncertain"><div class="model-label">{model_name} Classifier</div><div class="verdict-text" style="color:#555">N/A</div><div class="conf-label">Model not loaded</div></div>'
                    results.append({"Model": model_name, "Verdict": "N/A", "Confidence": None})
                    continue
                pred, label, css_cls, conf = predict_with_proba(model_obj, embedding)
                results.append({"Model": model_name, "Verdict": label, "Confidence": conf})
                conf_pct = round(conf * 100, 1)
                cards_html += f"""
                <div class="verdict-card {css_cls}">
                    <div class="model-label">{model_name} Classifier</div>
                    <div class="verdict-text {css_cls}">{label}</div>
                    <div class="conf-bar-wrap"><div class="conf-bar {css_cls}" style="width:{conf_pct}%"></div></div>
                    <div class="conf-label">Confidence: {conf_pct}%</div>
                </div>"""
            cards_html += '</div>'
            st.markdown(cards_html, unsafe_allow_html=True)

            verdicts = [r["Verdict"] for r in results if r["Verdict"] != "N/A"]
            if len(verdicts) == 2:
                if verdicts[0] == verdicts[1]:
                    st.success(f"{'✅' if verdicts[0] == 'REAL' else '🚨'} Both models agree: **{verdicts[0]}**")
                else:
                    st.warning("⚠️ Models disagree — treat result with caution.")

            st.markdown("---")
            st.markdown("<div class='section-label'>Comparison</div>", unsafe_allow_html=True)
            rows = ""
            for r in results:
                if r["Verdict"] == "N/A":
                    rows += f"<tr><td>{r['Model']}</td><td style='color:#555'>N/A</td><td style='color:#555'>—</td></tr>"
                else:
                    cls = "badge-real" if r["Verdict"] == "REAL" else "badge-fake"
                    conf_str = f"{round(r['Confidence']*100,1)}%" if r["Confidence"] else "—"
                    rows += f"<tr><td>{r['Model']}</td><td class='{cls}'>{r['Verdict']}</td><td>{conf_str}</td></tr>"
            st.markdown(f'<table class="cmp-table"><thead><tr><th>Model</th><th>Verdict</th><th>Confidence</th></tr></thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════╗
# ║  TAB 2 — HOW IT WORKS                               ║
# ╚══════════════════════════════════════════════════════╝
with tab_how:
    st.markdown("<div class='section-label'>Algorithm Explanations</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.7rem;color:#555;margin-bottom:2rem;'>Code on the left · Math on the right</div>", unsafe_allow_html=True)

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
                       embeddings[<span class="nm">1</span>])</pre></div>
  <div class="algo-math">
    <div class="algo-tag">SentenceBERT · Embedding Model</div>
    <div class="algo-math-title">Sentence Embeddings via Siamese BERT</div>
    <div class="algo-math-desc">
      A pretrained BERT encoder maps variable-length text to a fixed-size dense vector via mean pooling over token representations:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">u = MeanPool(BERT(tokens))</code><br><br>
      Fine-tuned on sentence pairs so semantically similar sentences cluster close together in 384-dimensional space:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">sim(u, v) = (u · v) / (‖u‖ ‖v‖)</code><br><br>
      Why use it? Bag-of-words models miss meaning. "Bank robbery" and "financial institution heist" share no words but have near-identical embeddings.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

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

mlp.fit(train_embeddings, train_labels)

proba = mlp.predict_proba(test_embeddings)
<span class="cm"># [[0.03, 0.97], [0.91, 0.09], ...]</span></pre></div>
  <div class="algo-math">
    <div class="algo-tag">MLP · Multi-Layer Perceptron</div>
    <div class="algo-math-title">Feedforward Neural Network</div>
    <div class="algo-math-desc">
      Each layer applies a linear transformation followed by ReLU activation:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">h⁽ˡ⁾ = ReLU(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾)</code><br><br>
      Final layer uses softmax to output class probabilities:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">p(y=k | x) = exp(zₖ) / Σⱼ exp(zⱼ)</code><br><br>
      Weights updated via backprop + Adam optimiser on cross-entropy loss. Can learn non-linear decision boundaries — useful if real vs. fake clusters aren't linearly separable.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("""
<div class="algo-block">
  <div class="algo-code"><pre><span class="kw">from</span> sklearn.linear_model <span class="kw">import</span> SGDClassifier

sgd = SGDClassifier(
    loss=<span class="st">"modified_huber"</span>,
    class_weight=<span class="st">"balanced"</span>,
    random_state=<span class="nm">42</span>,
)

sgd.fit(train_embeddings, train_labels)

scores = sgd.decision_function(test_emb)
<span class="cm"># positive = leans real, negative = fake</span></pre></div>
  <div class="algo-math">
    <div class="algo-tag">SGD · Stochastic Gradient Descent</div>
    <div class="algo-math-title">Linear Classifier with SGD Optimisation</div>
    <div class="algo-math-desc">
      Finds a hyperplane <b>w</b> separating real from fake in embedding space, minimising hinge loss one sample at a time:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">L(w) = max(0, 1 − yᵢ (w · xᵢ + b))</code><br><br>
      Weight update per sample:<br><br>
      <code style="color:#c3e88d;font-size:0.75rem">w ← w − η ∇L(w)</code><br><br>
      With 384-dim embeddings a linear boundary often works surprisingly well. Trains in a single pass — far faster than MLP — but can't capture non-linear patterns.
    </div>
  </div>
</div>""", unsafe_allow_html=True)

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
</div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<div style='font-size:0.6rem;color:#333;text-align:center;letter-spacing:0.15em;'>FAKE NEWS DETECTOR · SBERT + MLP / SGD · FOR RESEARCH USE ONLY</div>", unsafe_allow_html=True)
