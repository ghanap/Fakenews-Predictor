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
import gdown
import requests

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

# --- Download Training Data from Google Drive ---
@st.cache_resource
def download_train_data(file_id="1eKQJ2iltCxHna3OGTnqu6ESFi5znvLn_", output_path="train.csv"):
    """Download train.csv from Google Drive if not present."""
    if os.path.exists(output_path):
        return output_path
    
    try:
        # Google Drive direct download URL
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return output_path
    except Exception as e:
        st.warning(f"Could not download training data from Google Drive: {e}")
        return None

# --- Model Loading ---
@st.cache_resource
def load_model_and_vectorizer(model_path="model.joblib", vectorizer_path="vectorizer.joblib", 
                              gdrive_file_id="1eKQJ2iltCxHna3OGTnqu6ESFi5znvLn_"):
    """Loads saved model and vectorizer."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        st.stop()
    
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Vectorizer file not found: {vectorizer_path}")
        st.stop()
    
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    # Try to download and load training data from Google Drive
    train_df = None
    train_path = "train.csv"
    
    if not os.path.exists(train_path):
        with st.spinner("Downloading training data from Google Drive..."):
            downloaded_path = download_train_data(gdrive_file_id, train_path)
            if downloaded_path:
                st.success("✅ Training data downloaded successfully!")
    
    if os.path.exists(train_path):
        try:
            with st.spinner("Loading training data..."):
                train_df = pd.read_csv(train_path, sep=';')
                train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
                train_df = train_df[pd.notna(train_df['label'])]
                train_df['label'] = train_df['label'].astype(int)
                train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
                train_df = train_df[train_df['cleaned_text'].str.strip() != '']
        except Exception as e:
            st.warning(f"Could not load training data: {e}")
            st.info("SHAP explanations may be slower or unavailable without training data.")
    else:
        st.warning("Training data not available. SHAP explanations will be limited.")
    
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
            font=dict(size=20, color='#e5e7eb')
        ),
        xaxis_title='Impact on Prediction',
        yaxis_title='',
        height=500,
        margin=dict(l=20, r=20, t=60, b=40),
        plot_bgcolor='#1e1e1e',
        paper_bgcolor='#1e1e1e',
        font=dict(size=12, color='#e5e7eb'),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(255,255,255,0.3)',
            color='#e5e7eb'
        ),
        yaxis=dict(
            showgrid=False,
            color='#e5e7eb'
        ),
        template='plotly_dark'
    )
    
    return fig

# --- Custom SHAP Visualization ---
def plot_shap_explanation(shap_values_for_class, feature_names, tfidf_input, predicted_class):
    """Create a custom Plotly visualization for SHAP."""
    try:
        # 1. Get non-zero features from the input text
        # Converting sparse matrix to dense to find indices of present words
        tfidf_dense = tfidf_input.toarray()[0]
        non_zero_idx = tfidf_dense.nonzero()[0]
        
        if len(non_zero_idx) == 0:
            return None
        
        # 2. Robustly handle SHAP value shapes
        # We need a 1D array where each index corresponds to a feature
        shap_vals = np.array(shap_values_for_class)
        
        # If the array is (2, 2), (N, 2, 2), or similar, we flatten it
        # to get individual scalar values
        if shap_vals.ndim > 1:
            shap_vals = shap_vals.flatten()
            
        # Ensure we don't have a mismatch between features and SHAP array length
        # This can happen with interaction values
        if len(shap_vals) > len(feature_names):
            shap_vals = shap_vals[:len(feature_names)]
        
        # 3. Filter and Sort
        # Only look at SHAP values for words actually in the user's text
        shap_vals_filtered = shap_vals[non_zero_idx]
        abs_shap_non_zero = np.abs(shap_vals_filtered)
        
        # Get top N features (limit to 15)
        n_features = min(15, len(non_zero_idx))
        top_indices_in_filtered = np.argsort(abs_shap_non_zero)[-n_features:]
        
        # Map back to feature names and values
        top_features = [feature_names[non_zero_idx[i]] for i in top_indices_in_filtered]
        top_shap_vals_final = [shap_vals_filtered[i] for i in top_indices_in_filtered]
        
        # 4. Create Colors
        # Use simple comparison since v is now guaranteed to be a scalar
        colors = ['#ef4444' if v < 0 else '#10b981' for v in top_shap_vals_final]

        # 5. Build the Plotly Bar Chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features,
            x=top_shap_vals_final,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='rgba(255,255,255,0.3)', width=1)
            ),
            # Each v is now a float, so formatting works perfectly
            text=[f'{float(v):.3f}' for v in top_shap_vals_final],
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>SHAP Value: %{x:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(
                text='Key Feature Impacts (SHAP)',
                font=dict(size=20, color='#e5e7eb')
            ),
            xaxis_title='Impact on Prediction',
            yaxis_title='',
            height=500,
            margin=dict(l=20, r=20, t=60, b=40),
            plot_bgcolor='rgba(0,0,0,0)', # Transparent to match container
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12, color='#e5e7eb'),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.3)',
            ),
            template='plotly_dark'
        )
        
        return fig

    except Exception as e:
        st.error(f"Error creating SHAP plot: {str(e)}")
        return None

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
def generate_shap_explanation(shap_values_for_class, feature_names, tfidf_input, predicted_class):
    try:
        # Get the actual words present in the article
        tfidf_dense = tfidf_input.toarray()[0]
        non_zero_idx = tfidf_dense.nonzero()[0]
        
        if len(non_zero_idx) == 0:
            return None
        
        # Flatten the SHAP values to ensure they are scalars for the bar chart
        shap_vals = np.array(shap_values_for_class).flatten()
        
        # Get values for words in the current text
        current_text_shap = shap_vals[non_zero_idx]
        
        # Sort by absolute impact (magnitude)
        n_features = min(15, len(non_zero_idx))
        top_indices = np.argsort(np.abs(current_text_shap))[-n_features:]
        
        # Prepare data for Plotly
        top_features = [feature_names[non_zero_idx[i]] for i in top_indices]
        top_vals = [current_text_shap[i] for i in top_indices]
        
        # Create the figure...
        # (Use the Plotly code from the previous step)
        
    except Exception as e:
        st.error(f"SHAP Error: {e}")
        return None
# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

# Custom CSS
st.markdown("""
<style>
    .stMetric {
        background-color: #1e1e1e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3a3a3a;
    }
    .highlighted-text {
        font-size: 16px;
        line-height: 1.8;
        padding: 20px;
        background-color: #1e1e1e;
        border-radius: 10px;
        border: 1px solid #3a3a3a;
        margin: 20px 0;
        color: #e5e7eb;
    }
    /* Ensure Plotly charts match dark theme */
    .js-plotly-plot {
        background-color: #1e1e1e !important;
    }
    /* Fix for Streamlit columns */
    [data-testid="stHorizontalBlock"] > div {
        background-color: transparent !important;
    }
    /* Ensure plotly container has proper background */
    .plotly {
        background-color: #1e1e1e !important;
        border-radius: 10px;
        padding: 10px;
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
                    
                    st.markdown("---")
                    
                    # Create two columns for visualizations
                    col_lime, col_shap = st.columns(2)
                    
                    with col_lime:
                        st.subheader("🔍 LIME Analysis")
                        try:
                            lime_fig = plot_lime_explanation(lime_exp, prediction)
                            if lime_fig:
                                st.plotly_chart(lime_fig, use_container_width=True)
                            else:
                                st.info("Unable to generate LIME visualization.")
                        except Exception as e:
                            st.error(f"Error displaying LIME chart: {str(e)}")
                    
                    with col_shap:
                        st.subheader("📊 SHAP Analysis")
                        with st.spinner("Generating SHAP explanation..."):
                            try:
                                shap_result = generate_shap_explanation(
                                    news_article, tfidf_vectorizer, model, train_df_processed
                                )
                            
                                if shap_result:
                                    shap_values_for_class, feature_names, tfidf_input, predicted_class = shap_result
                                    st.write(f"Sum of SHAP values: {np.sum(shap_values_for_class)}")
                                    st.write(f"Max SHAP value: {np.max(np.abs(shap_values_for_class))}")
                                    shap_fig = plot_shap_explanation(
                                        shap_values_for_class, feature_names, tfidf_input, predicted_class
                                    )
                                    if shap_fig:
                                        st.plotly_chart(shap_fig, use_container_width=True)
                                    else:
                                        st.info("No significant features found for SHAP analysis.")
                                else:
                                    st.info("SHAP analysis unavailable. Training data may be missing.")
                            except Exception as e:
                                st.error(f"Error generating SHAP analysis: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())
                                st.info("SHAP analysis requires training data. Continuing with LIME only.")
    else:
        st.warning("⚠️ Please enter a news article to analyze.")
