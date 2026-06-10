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

def generate_shap_explanation(news_article, vectorizer, model, train_df):
    """Generates SHAP explanation and returns a SHAP Explanation object."""
    try:
        if train_df is None or len(train_df) == 0:
            return None

        cleaned_text = preprocess_text(news_article)
        tfidf_input = vectorizer.transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Get the predicted class
        probs = model.predict_proba(tfidf_input)[0]
        predicted_class = np.argmax(probs)

        # Use a small representative background sample
        # Using a small fixed sample for speed in Streamlit
        background_sample = vectorizer.transform(train_df['cleaned_text'].sample(min(50, len(train_df)))).toarray()
        
        # KernelExplainer for MultinomialNB
        explainer = shap.KernelExplainer(model.predict_proba, background_sample)
        
        # Generate SHAP values
        shap_values = explainer.shap_values(tfidf_input.toarray(), nsamples=100)

        # Robustly handle SHAP values (can be list or array depending on version/model)
        if isinstance(shap_values, list):
            # If it's a list, select the array for the predicted class
            if predicted_class < len(shap_values):
                vals = shap_values[predicted_class]
            else:
                vals = shap_values[0]
            # Handle list of arrays case where each array is (1, N)
            if isinstance(vals, np.ndarray) and vals.ndim > 1:
                vals = vals[0]
        elif isinstance(shap_values, np.ndarray):
            # If it's a single array, handle dimensions
            if shap_values.ndim == 3: # (samples, features, classes) or similar
                # This depends on version, usually (n_samples, n_features, n_classes)
                # or (n_classes, n_samples, n_features)
                if shap_values.shape[0] == 2: # classes first
                    vals = shap_values[predicted_class][0]
                else: # classes last
                    vals = shap_values[0, :, predicted_class]
            elif shap_values.ndim == 2: # (samples, features)
                # Likely only one class returned
                vals = shap_values[0]
            else:
                vals = shap_values.flatten()
        else:
            vals = np.array(shap_values).flatten()

        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            if len(expected_val) > predicted_class:
                expected_val = expected_val[predicted_class]
            else:
                expected_val = expected_val[0]
        
        explanation = shap.Explanation(
            values=vals,
            base_values=float(expected_val),
            data=tfidf_input.toarray()[0],
            feature_names=feature_names
        )

        return {
            'explanation': explanation,
            'predicted_class': predicted_class,
            'tfidf_input': tfidf_input
        }

    except Exception as e:
        st.error(f"SHAP Computation Error: {str(e)}")
        return None

def plot_shap_explanation(shap_exp_dict):
    """Creates a custom Plotly bar chart from a SHAP explanation."""
    try:
        explanation = shap_exp_dict['explanation']
        tfidf_input = shap_exp_dict['tfidf_input']
        
        # 1. Get indices of words actually present in the current article
        tfidf_dense = tfidf_input.toarray()[0]
        non_zero_idx = tfidf_dense.nonzero()[0]
        
        if len(non_zero_idx) == 0:
            return None

        # 2. Match SHAP values to the words in the text
        vals = explanation.values[non_zero_idx]
        feature_names = np.array(explanation.feature_names)[non_zero_idx]
        
        # 3. Sort by magnitude to find the top influential words
        n_features = min(15, len(vals))
        top_indices = np.argsort(np.abs(vals))[-n_features:]
        
        top_features = feature_names[top_indices]
        top_vals = vals[top_indices]

        # 4. Create the Bar Chart
        colors = ['#ef4444' if v < 0 else '#10b981' for v in top_vals]
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            y=top_features,
            x=top_vals,
            orientation='h',
            marker=dict(color=colors, line=dict(color='rgba(255,255,255,0.3)', width=1)),
            text=[f'{float(v):.4f}' for v in top_vals],
            textposition='outside'
        ))

        fig.update_layout(  
            template='plotly_dark',
            title="Feature Impact (SHAP)",
            xaxis_title="Impact on Prediction Probability",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=20, r=20, t=50, b=20),
            height=400
        )
        return fig
    except Exception as e:
        st.error(f"Plotting Error: {e}")
        return None

def display_native_shap_plots(shap_exp_dict):
    """Displays native SHAP waterfall and force plots."""
    try:
        explanation = shap_exp_dict['explanation']
        
        # Waterfall Plot
        st.write("#### 🌊 Waterfall Plot")
        st.info("Shows how each word 'pushes' the prediction from the average (base value) to the final result.")
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        shap.plots.waterfall(explanation, max_display=10, show=False)
        
        # Theme adjustments
        plt.gcf().axes[0].tick_params(colors='white')
        plt.gcf().axes[0].xaxis.label.set_color('white')
        plt.gcf().axes[0].title.set_color('white')
        st.pyplot(fig)
        plt.close(fig)

        # Force Plot (Javascript-based)
        st.write("#### ⚡ Force Plot")
        st.info("Interactive view of how features contribute to the score.")
        
        # Capture the interactive HTML using shap.save_html
        import io
        shap_buffer = io.StringIO()
        
        # Generate the force plot object
        p = shap.force_plot(
            explanation.base_values, 
            explanation.values, 
            features=explanation.data, 
            feature_names=explanation.feature_names,
            matplotlib=False,
            show=False
        )
        
        # Save to buffer
        shap.save_html(shap_buffer, p)
        shap_html = shap_buffer.getvalue()
        
        # Wrap in a white container for visibility of labels
        styled_html = f"""
        <div style="background-color: white; padding: 10px; border-radius: 5px; min-height: 150px;">
            {shap_html}
        </div>
        """
        st.components.v1.html(styled_html, height=200, scrolling=True)

    except Exception as e:
        st.error(f"Native Plotting Error: {e}")
        import traceback
        st.code(traceback.format_exc())

@st.cache_data
def get_global_shap_summary(_vectorizer, _model, _train_df):
    """Computes global SHAP summary for a sample of data."""
    try:
        if _train_df is None or len(_train_df) == 0:
            return None
            
        # Use a small sample for speed in the UI
        sample_size = min(30, len(_train_df))
        sample_df = _train_df.sample(sample_size, random_state=42)
        X_sample = _vectorizer.transform(sample_df['cleaned_text']).toarray()
        
        # Using a fixed background for global consistency
        background = _vectorizer.transform(_train_df['cleaned_text'].head(10)).toarray()
        explainer = shap.KernelExplainer(_model.predict_proba, background)
        
        # nsamples controls the accuracy vs speed
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        
        # Determine the target class index
        real_class_idx = 1 if 1 in _model.classes_ else 0
        
        # Robustly extract the 2D array (samples, features)
        if isinstance(shap_values, list):
            target_vals = shap_values[min(real_class_idx, len(shap_values)-1)]
        elif isinstance(shap_values, np.ndarray):
            if shap_values.ndim == 3: # (samples, features, classes)
                target_vals = shap_values[:, :, real_class_idx]
            else:
                target_vals = shap_values
        else:
            target_vals = np.array(shap_values)

        # Create figure explicitly
        fig = plt.figure(figsize=(10, 6))
        fig.patch.set_facecolor('#1e1e1e')
        
        # Force a clear of the current figure state
        plt.clf()
        
        shap.summary_plot(
            target_vals, 
            X_sample, 
            feature_names=_vectorizer.get_feature_names_out(),
            show=False,
            max_display=12,
            plot_type="dot"
        )
        
        # Theme adjustments for Dark Mode
        ax = plt.gca()
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='white', labelsize=10)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        
        # Find the colorbar and fix its labels
        for child in fig.get_children():
            if isinstance(child, plt.Axes) and child != ax:
                child.tick_params(colors='white')
                child.yaxis.label.set_color('white')

        plt.tight_layout()
        return fig
        
    except Exception as e:
        # We can't use st.error here easily because it's cached, but we can log
        print(f"Global SHAP analysis failed: {str(e)}")
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
                                    # Custom Plotly chart
                                    shap_fig = plot_shap_explanation(shap_result)
                                    if shap_fig:
                                        st.plotly_chart(shap_fig, use_container_width=True)
                                    
                                    # Native plots (Waterfall & Force)
                                    with st.expander("🔍 Detailed SHAP Plots", expanded=True):
                                        display_native_shap_plots(shap_result)
                                else:
                                    st.info("SHAP analysis unavailable. Training data may be missing.")
                            except Exception as e:
                                st.error(f"Error generating SHAP analysis: {str(e)}")
                                import traceback
                                st.code(traceback.format_exc())

                    # --- Global Insights ---
                    st.markdown("---")
                    st.header("🌐 Global Model Insights")
                    col_global1, col_global2 = st.columns([1, 1.5])
                    
                    with col_global1:
                        st.write("### How the Model Learns")
                        st.write("""
                        This section shows the general patterns the model has learned from the training data.
                        
                        - **SHAP Summary Plot**: Shows the most influential words across many articles.
                        - **Interpretation**: Red indicates higher TF-IDF values (word frequency), blue indicates lower. 
                          The position on the X-axis shows the impact on the 'Real' prediction.
                        """)
                    
                    with col_global2:
                        if train_df_processed is not None:
                            with st.spinner("Generating Global SHAP Summary..."):
                                global_fig = get_global_shap_summary(tfidf_vectorizer, model, train_df_processed)
                                if global_fig:
                                    st.pyplot(global_fig)
                                else:
                                    st.info("Global summary unavailable.")
                        else:
                            st.info("Global summary requires training data.")
    else:
        st.warning("⚠️ Please enter a news article to analyze.")
