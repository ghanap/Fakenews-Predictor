import streamlit as st
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import lime
import lime.lime_text
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import joblib
import os

# --- NLTK Data Download and Setup ---
@st.cache_resource
def setup_nltk():
    """Downloads necessary NLTK data and returns lemmatizer and stopwords."""
    nltk_packages = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, package in nltk_packages:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)
    
    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and lemmatizes text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Model Loading ---
@st.cache_resource
def load_model_and_vectorizer(model_path="model.joblib", vectorizer_path="vectorizer.joblib", train_path="train.csv"):
    """Loads saved model and vectorizer."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Vectorizer file not found: {vectorizer_path}")
        st.stop()
    
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    train_df = None
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path, sep=';')
            train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
            train_df = train_df[pd.notna(train_df['label'])]
            train_df['label'] = train_df['label'].astype(int)
            train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
            train_df = train_df[train_df['cleaned_text'].str.strip() != '']
        except:
            pass
    
    return tfidf_vectorizer, model, train_df

# --- Prediction Functions ---
def predict_fake_news(news_text, vectorizer, model):
    """Predicts if a news article is fake or real."""
    cleaned_text = preprocess_text(news_text)
    if not cleaned_text.strip():
        return None
    tfidf_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_text)
    proba = model.predict_proba(tfidf_text)
    return prediction[0], proba[0]

def predict_proba_for_lime(texts, vectorizer, model):
    """Returns prediction probabilities for LIME."""
    cleaned_texts = [preprocess_text(text) for text in texts]
    tfidf_vectors = vectorizer.transform(cleaned_texts)
    return model.predict_proba(tfidf_vectors)

# --- Custom LIME Visualization ---
def plot_lime_explanation(lime_exp, prediction):
    """Create a custom Plotly visualization for LIME."""
    # Get feature weights
    exp_list = lime_exp.as_list()
    
    # Separate by class
    words = [item[0] for item in exp_list]
    weights = [item[1] for item in exp_list]
    
    # Create color scheme based on prediction direction
    colors = ['#ef4444' if w < 0 else '#10b981' for w in weights]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=words[::-1],  # Reverse to show most important at top
        x=weights[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{w:.3f}' for w in weights[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Weight: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Feature Importance',
            font=dict(size=20, color='#1f2937')
        ),
        xaxis_title='Impact on Prediction',
        yaxis_title='',
        height=500,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#374151'),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0,0,0,0.3)'
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig

# --- Custom SHAP Visualization ---
def plot_shap_explanation(shap_values, feature_names, tfidf_input, predicted_class):
    """Create a custom Plotly visualization for SHAP."""
    # Get non-zero features
    non_zero_idx = tfidf_input.toarray()[0].nonzero()[0]
    
    if len(non_zero_idx) == 0:
        return None
    
    # Get top features by absolute SHAP value
    shap_vals = shap_values[predicted_class][0]
    top_indices = np.argsort(np.abs(shap_vals[non_zero_idx]))[-15:][::-1]
    
    top_features = [feature_names[non_zero_idx[i]] for i in top_indices]
    top_shap_values = [shap_vals[non_zero_idx[i]] for i in top_indices]
    
    # Create colors
    colors = ['#ef4444' if v < 0 else '#10b981' for v in top_shap_values]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=top_features[::-1],
        x=top_shap_values[::-1],
        orientation='h',
        marker=dict(
            color=colors[::-1],
            line=dict(color='rgba(0,0,0,0.3)', width=1)
        ),
        text=[f'{v:.3f}' for v in top_shap_values[::-1]],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text='Feature Impact',
            font=dict(size=20, color='#1f2937')
        ),
        xaxis_title='SHAP Value',
        yaxis_title='',
        height=500,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12, color='#374151'),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(0,0,0,0.3)'
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig

# --- Word Highlighting Function ---
def highlight_text_by_importance(text, lime_exp, max_words=50):
    """Highlight words in text based on LIME importance."""
    exp_dict = dict(lime_exp.as_list())
    
    words = text.split()
    if len(words) > max_words:
        words = words[:max_words]
        truncated = True
    else:
        truncated = False
    
    highlighted_words = []
    for word in words:
        cleaned_word = preprocess_text(word)
        
        if cleaned_word in exp_dict:
            weight = exp_dict[cleaned_word]
            if weight > 0:
                # Green for real news indicators
                intensity = min(int(abs(weight) * 200), 200)
                highlighted_words.append(
                    f'<span style="background-color: rgba(16, 185, 129, {min(abs(weight) * 2, 0.8)}); '
                    f'padding: 2px 4px; border-radius: 3px; margin: 2px;">{word}</span>'
                )
            else:
                # Red for fake news indicators
                intensity = min(int(abs(weight) * 200), 200)
                highlighted_words.append(
                    f'<span style="background-color: rgba(239, 68, 68, {min(abs(weight) * 2, 0.8)}); '
                    f'padding: 2px 4px; border-radius: 3px; margin: 2px;">{word}</span>'
                )
        else:
            highlighted_words.append(word)
    
    result = ' '.join(highlighted_words)
    if truncated:
        result += ' <span style="color: #6b7280;">...</span>'
    
    return result

# --- LIME Explanation Function ---
def generate_lime_explanation(news_article, vectorizer, model):
    """Generate LIME explanation."""
    try:
        explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
        exp = explainer.explain_instance(
            news_article,
            lambda x: predict_proba_for_lime(x, vectorizer, model),
            num_features=15,
            num_samples=500
        )
        return exp
    except Exception as e:
        st.error(f"Error generating LIME explanation: {str(e)}")
        return None

# --- SHAP Explanation Function ---
def generate_shap_explanation(news_article, vectorizer, model, train_df):
    """Generate SHAP explanation."""
    try:
        if train_df is None:
            st.warning("Training data not available for SHAP background.")
            return None
        
        background_size = min(50, len(train_df))
        background_data = train_df['cleaned_text'].sample(background_size, random_state=42).tolist()
        background_tfidf = vectorizer.transform(background_data)
        
        shap_explainer = shap.KernelExplainer(
            model.predict_proba, 
            background_tfidf,
            link="logit"
        )
        
        cleaned_input = preprocess_text(news_article)
        tfidf_input = vectorizer.transform([cleaned_input])
        
        shap_values = shap_explainer.shap_values(tfidf_input, nsamples=100)
        
        feature_names = vectorizer.get_feature_names_out()
        predicted_class = model.predict(tfidf_input)[0]
        
        return shap_values, feature_names, tfidf_input, predicted_class
    except Exception as e:
        st.error(f"Error generating SHAP explanation: {str(e)}")
        return None

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #f9fafb;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
    }
    .highlighted-text {
        font-size: 16px;
        line-height: 1.8;
        padding: 20px;
        background-color: #f9fafb;
        border-radius: 10px;
        border: 1px solid #e5e7eb;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("📰 Fake News Detector")

# Load model and vectorizer
with st.spinner("Loading model..."):
    tfidf_vectorizer, model, train_df_processed = load_model_and_vectorizer()

st.success("✅ Model loaded successfully!")

# Sample articles
with st.expander("📋 Try a sample article"):
    sample_fake = """BREAKING: Scientists Discover That The Earth Is Actually Flat After All! A team of researchers from the Institute of Revolutionary Science has announced shocking findings that contradict centuries of scientific consensus. Using advanced technology, they claim to have proven the Earth is flat. The government has been covering this up for years!"""
    
    sample_real = """WASHINGTON (Reuters) - The United States economy added 200,000 jobs last month, according to data released by the Bureau of Labor Statistics on Friday. The unemployment rate remained steady at 4.2 percent. Economists had predicted job growth of around 180,000. The report suggests continued strength in the labor market despite concerns about rising interest rates."""
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Use Sample Fake News"):
            st.session_state['news_article'] = sample_fake
    with col_s2:
        if st.button("Use Sample Real News"):
            st.session_state['news_article'] = sample_real

news_article = st.text_area(
    "Enter news article:", 
    height=200,
    value=st.session_state.get('news_article', ''),
    placeholder="Paste a news article to analyze..."
)

# Prediction section
if st.button("🔍 Analyze Article", type="primary"):
    if news_article.strip():
        with st.spinner("Analyzing..."):
            result = predict_fake_news(news_article, tfidf_vectorizer, model)
            
            if result is None:
                st.warning("⚠️ Unable to process the article. Please try with different text.")
            else:
                prediction, proba = result
                
                # Display prediction
                st.markdown("---")
                
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if prediction == 1:
                        st.success("### ✅ REAL NEWS")
                    else:
                        st.error("### ❌ FAKE NEWS")
                
                with col2:
                    st.metric("Confidence (Fake)", f"{proba[0]:.2%}")
                
                with col3:
                    st.metric("Confidence (Real)", f"{proba[1]:.2%}")
                
                st.progress(proba[1])
                
                # Generate LIME explanation
                st.markdown("---")
                with st.spinner("Generating LIME explanation..."):
                    lime_exp = generate_lime_explanation(news_article, tfidf_vectorizer, model)
                
                if lime_exp:
                    # Show highlighted text
                    st.subheader("📝 Text Analysis")
                    highlighted_html = highlight_text_by_importance(news_article, lime_exp)
                    st.markdown(
                        f'<div class="highlighted-text">{highlighted_html}</div>',
                        unsafe_allow_html=True
                    )
                    
                    # Create two columns for visualizations
                    col_lime, col_shap = st.columns(2)
                    
                    with col_lime:
                        st.subheader("🔍 LIME Analysis")
                        lime_fig = plot_lime_explanation(lime_exp, prediction)
                        st.plotly_chart(lime_fig, use_container_width=True)
                    
                    with col_shap:
                        st.subheader("📊 SHAP Analysis")
                        with st.spinner("Generating SHAP explanation..."):
                            shap_result = generate_shap_explanation(
                                news_article, tfidf_vectorizer, model, train_df_processed
                            )
                        
                        if shap_result:
                            shap_values, feature_names, tfidf_input, predicted_class = shap_result
                            shap_fig = plot_shap_explanation(
                                shap_values, feature_names, tfidf_input, predicted_class
                            )
                            if shap_fig:
                                st.plotly_chart(shap_fig, use_container_width=True)
                            else:
                                st.info("No significant features found for SHAP analysis.")
    else:
        st.warning("⚠️ Please enter a news article to analyze.")
