import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import lime
import lime.lime_text
import shap
import plotly.graph_objects as go
import joblib
import os

# ======================================================
# NLTK DOWNLOAD (CACHE SAFE)
# ======================================================
@st.cache_resource
def download_nltk_data():
    packages = [
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    for path, package in packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)

download_nltk_data()

# CREATE OBJECTS OUTSIDE CACHE (IMPORTANT)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ======================================================
# TEXT PREPROCESSING
# ======================================================
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

# ======================================================
# LOAD MODEL + VECTORIZER
# ======================================================
@st.cache_resource
def load_model_and_vectorizer(
    model_path="model.joblib",
    vectorizer_path="vectorizer.joblib",
    train_path="train.csv"
):
    if not os.path.exists(model_path):
        st.error(f"❌ Missing model: {model_path}")
        st.stop()

    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Missing vectorizer: {vectorizer_path}")
        st.stop()

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    train_df = None
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path, sep=';')
            train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
            train_df = train_df.dropna(subset=['label'])
            train_df['label'] = train_df['label'].astype(int)
            train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
            train_df = train_df[train_df['cleaned_text'].str.strip() != ""]
        except:
            train_df = None

    return vectorizer, model, train_df

# ======================================================
# PREDICTION
# ======================================================
def predict_fake_news(text, vectorizer, model):
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return None
    X = vectorizer.transform([cleaned])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    return pred, proba

def predict_proba_for_lime(texts, vectorizer, model):
    cleaned = [preprocess_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    return model.predict_proba(X)

# ======================================================
# LIME
# ======================================================
def generate_lime_explanation(text, vectorizer, model):
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['Fake', 'Real']
    )
    return explainer.explain_instance(
        text,
        lambda x: predict_proba_for_lime(x, vectorizer, model),
        num_features=15,
        num_samples=500
    )

def plot_lime_explanation(exp):
    words, weights = zip(*exp.as_list())
    colors = ['#ef4444' if w < 0 else '#10b981' for w in weights]

    fig = go.Figure(go.Bar(
        x=weights[::-1],
        y=words[::-1],
        orientation='h',
        marker_color=colors[::-1]
    ))

    fig.update_layout(
        title="LIME Feature Importance",
        height=500
    )
    return fig

# ======================================================
# SHAP
# ======================================================
def generate_shap_explanation(text, vectorizer, model, train_df):
    if train_df is None:
        return None

    background = vectorizer.transform(
        train_df['cleaned_text'].sample(
            min(50, len(train_df)),
            random_state=42
        )
    )

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background,
        link="logit"
    )

    cleaned = preprocess_text(text)
    X = vectorizer.transform([cleaned])
    shap_values = explainer.shap_values(X, nsamples=100)

    return shap_values, X, vectorizer.get_feature_names_out(), model.predict(X)[0]

def plot_shap_explanation(shap_values, X, feature_names, cls):
    idx = X.toarray()[0].nonzero()[0]
    if len(idx) == 0:
        return None

    vals = shap_values[cls][0][idx]
    top = np.argsort(np.abs(vals))[-15:]

    fig = go.Figure(go.Bar(
        x=vals[top],
        y=[feature_names[i] for i in idx[top]],
        orientation='h'
    ))

    fig.update_layout(
        title="SHAP Feature Impact",
        height=500
    )
    return fig

# ======================================================
# STREAMLIT UI
# ======================================================
st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    page_icon="📰"
)

st.title("📰 Fake News Detector")

with st.spinner("Loading model..."):
    vectorizer, model, train_df = load_model_and_vectorizer()

st.success("✅ Model loaded")

news_text = st.text_area(
    "Paste a news article:",
    height=200
)

if st.button("🔍 Analyze", type="primary"):
    if not news_text.strip():
        st.warning("Please enter text.")
    else:
        pred, proba = predict_fake_news(news_text, vectorizer, model)

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.success("REAL NEWS ✅" if pred == 1 else "FAKE NEWS ❌")
        with col2:
            st.metric("Fake Confidence", f"{proba[0]:.2%}")
        with col3:
            st.metric("Real Confidence", f"{proba[1]:.2%}")

        st.progress(proba[1])

        st.markdown("---")
        st.subheader("🔍 LIME Explanation")
        lime_exp = generate_lime_explanation(news_text, vectorizer, model)
        st.plotly_chart(plot_lime_explanation(lime_exp), use_container_width=True)

        st.subheader("📊 SHAP Explanation")
        shap_data = generate_shap_explanation(
            news_text, vectorizer, model, train_df
        )
        if shap_data:
            shap_vals, X, feats, cls = shap_data
            fig = plot_shap_explanation(shap_vals, X, feats, cls)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
