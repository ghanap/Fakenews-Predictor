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
import joblib
import os

# --- NLTK Data Download and Setup ---
@st.cache_resource
def setup_nltk():
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
        st.info("Please ensure you have trained the model and saved it as 'model.joblib'")
        st.stop()
    
    if not os.path.exists(vectorizer_path):
        st.error(f"❌ Vectorizer file not found: {vectorizer_path}")
        st.info("Please ensure you have saved the TF-IDF vectorizer as 'vectorizer.joblib'")
        st.stop()
    
    # Load model and vectorizer using joblib
    model = joblib.load(model_path)
    tfidf_vectorizer = joblib.load(vectorizer_path)
    
    # Load training data for SHAP background (optional, but improves SHAP)
    train_df = None
    if os.path.exists(train_path):
        try:
            train_df = pd.read_csv(train_path, sep=';')
            train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
            train_df = train_df[pd.notna(train_df['label'])]
            train_df['label'] = train_df['label'].astype(int)
            train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
            train_df = train_df[train_df['cleaned_text'].str.strip() != '']
        except Exception as e:
            st.warning(f"Could not load training data for SHAP: {e}")
            st.info("SHAP explanations may be slower without background data.")
    
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

# --- Streamlit UI ---
st.set_page_config(page_title="Fake News Detector", layout="wide", page_icon="📰")

st.title("📰 Fake News Detector")
st.markdown("""
This app uses a Multinomial Naive Bayes classifier trained on news articles to detect fake news.
Enter an article below and get predictions with explanations using LIME and SHAP.
""")

# Load model and vectorizer
with st.spinner("Loading model..."):
    tfidf_vectorizer, model, train_df_processed = load_model_and_vectorizer()

st.success("✅ Model loaded successfully!")

# Input section
st.markdown("---")
st.subheader("Enter News Article")

# Sample articles for testing
with st.expander("📋 Try a sample article"):
    sample_fake = """
    BREAKING: Scientists Discover That The Earth Is Actually Flat After All! 
    A team of researchers from the Institute of Revolutionary Science has announced 
    shocking findings that contradict centuries of scientific consensus. Using advanced 
    technology, they claim to have proven the Earth is flat. The government has been 
    covering this up for years!
    """
    
    sample_real = """
    WASHINGTON (Reuters) - The United States economy added 200,000 jobs last month, 
    according to data released by the Bureau of Labor Statistics on Friday. The unemployment 
    rate remained steady at 4.2 percent. Economists had predicted job growth of around 
    180,000. The report suggests continued strength in the labor market despite concerns 
    about rising interest rates.
    """
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("Use Sample Fake News"):
            st.session_state['news_article'] = sample_fake
    with col_s2:
        if st.button("Use Sample Real News"):
            st.session_state['news_article'] = sample_real

news_article = st.text_area(
    "Paste your news article here:", 
    height=250,
    value=st.session_state.get('news_article', ''),
    placeholder="Enter a news article to analyze..."
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
                st.subheader("📊 Prediction Results")
                
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
                
                # Confidence bar
                st.progress(proba[1])
    else:
        st.warning("⚠️ Please enter a news article to analyze.")

# Explanation section
st.markdown("---")
st.subheader("🔬 Model Explanations")

st.markdown("""
**LIME** (Local Interpretable Model-agnostic Explanations) shows which words influenced the prediction by highlighting important features.

**SHAP** (SHapley Additive exPlanations) provides a game-theory based approach to explain individual predictions.
""")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate LIME Explanation", use_container_width=True):
        if news_article.strip():
            with st.spinner("Generating LIME explanation... This may take a moment."):
                try:
                    explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
                    exp = explainer.explain_instance(
                        news_article,
                        lambda x: predict_proba_for_lime(x, tfidf_vectorizer, model),
                        num_features=15,
                        num_samples=500
                    )
                    
                    st.markdown("#### LIME Explanation")
                    st.markdown("Words highlighted in **orange/red** indicate fake news, while **green/blue** indicate real news.")
                    st.components.v1.html(exp.as_html(), height=500, scrolling=True)
                    
                    # Show feature importance
                    st.markdown("#### Top Contributing Features")
                    fig = exp.as_pyplot_figure()
                    st.pyplot(fig)
                    plt.close()
                    
                except Exception as e:
                    st.error(f"Error generating LIME explanation: {str(e)}")
        else:
            st.warning("⚠️ Please enter a news article first.")

with col2:
    if st.button("📈 Generate SHAP Explanation", use_container_width=True):
        if news_article.strip():
            with st.spinner("Generating SHAP explanation... This may take a moment."):
                try:
                    if train_df_processed is None:
                        st.error("Training data not available for SHAP background.")
                    else:
                        # Prepare background data
                        background_size = min(50, len(train_df_processed))
                        background_data = train_df_processed['cleaned_text'].sample(
                            background_size, random_state=42
                        ).tolist()
                        background_tfidf = tfidf_vectorizer.transform(background_data)
                        
                        # Initialize SHAP explainer
                        shap_explainer = shap.KernelExplainer(
                            model.predict_proba, 
                            background_tfidf,
                            link="logit"
                        )
                        
                        # Prepare input
                        cleaned_input = preprocess_text(news_article)
                        tfidf_input = tfidf_vectorizer.transform([cleaned_input])
                        
                        # Get SHAP values
                        shap_values = shap_explainer.shap_values(tfidf_input, nsamples=100)
                        
                        # Get feature names
                        feature_names = tfidf_vectorizer.get_feature_names_out()
                        
                        # Get predicted class
                        predicted_class = model.predict(tfidf_input)[0]
                        
                        st.markdown("#### SHAP Explanation")
                        st.markdown(f"Showing explanation for class: **{'Real' if predicted_class == 1 else 'Fake'}**")
                        
                        # Create waterfall plot
                        fig, ax = plt.subplots(figsize=(10, 8))
                        shap.plots.waterfall(
                            shap.Explanation(
                                values=shap_values[predicted_class][0],
                                base_values=shap_explainer.expected_value[predicted_class],
                                data=tfidf_input.toarray()[0],
                                feature_names=feature_names
                            ),
                            max_display=15,
                            show=False
                        )
                        st.pyplot(fig)
                        plt.close()
                        
                except Exception as e:
                    st.error(f"Error generating SHAP explanation: {str(e)}")
                    st.info("SHAP computation can be intensive. Try with a shorter article or reduce background samples.")
        else:
            st.warning("⚠️ Please enter a news article first.")

# Footer
st.markdown("---")
st.markdown("""
### About the Model
- **Algorithm**: Multinomial Naive Bayes
- **Features**: TF-IDF vectors (top 5000 features)
- **Preprocessing**: Lowercase, remove punctuation, stopword removal, lemmatization

### How to Use
1. Enter or paste a news article in the text area above
2. Click "Analyze Article" to get the prediction
3. Click "Generate LIME Explanation" or "Generate SHAP Explanation" to understand why the model made its decision

**Note**: LIME and SHAP explanations may take some time to generate, especially for longer articles.
""")
