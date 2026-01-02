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
import os

# --- NLTK Data Download and Setup ---
@st.cache_resource
def setup_nltk():
    """Downloads necessary NLTK data and returns lemmatizer and stopwords."""
    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except nltk.downloader.DownloadError:
        nltk.download('wordnet', quiet=True)

    return set(stopwords.words('english')), WordNetLemmatizer()

stop_words, lemmatizer = setup_nltk()

# --- Text Preprocessing Function ---
def preprocess_text(text):
    """Cleans and lemmatizes text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # Remove non-alphabetic characters
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# --- Model Loading and Training (Cached) ---
@st.cache_resource
def load_model_and_vectorizer(train_path="train.csv"):
    """Loads training data, preprocesses, trains TF-IDF vectorizer and Multinomial Naive Bayes model."""
    try:
        train_df = pd.read_csv(train_path, sep=';')
    except FileNotFoundError:
        st.error(f"Error: train.csv not found at {train_path}. Please ensure it's in the same directory as app.py.")
        st.stop()

    # Convert label column to numeric, coercing errors to NaN
    train_df['label'] = pd.to_numeric(train_df['label'], errors='coerce')
    # Filter out rows where label is NaN or not a number/integer before astype(int)
    train_df = train_df[pd.notna(train_df['label']) & (train_df['label'].apply(lambda x: isinstance(x, (int, float))))]
    train_df['label'] = train_df['label'].astype(int)

    # Apply preprocessing
    train_df['cleaned_text'] = train_df['text'].apply(preprocess_text)
    # Filter out rows with empty cleaned_text after preprocessing to avoid issues with TF-IDF
    train_df = train_df[train_df['cleaned_text'].str.strip() != '']

    # Initialize and fit TF-IDF Vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    train_tfidf = tfidf_vectorizer.fit_transform(train_df['cleaned_text'])

    # Train Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(train_tfidf, train_df['label'])

    return tfidf_vectorizer, model, train_df

# --- Prediction Functions ---
def predict_fake_news(news_text, vectorizer, model):
    """Predicts if a news article is fake or real."""
    cleaned_text = preprocess_text(news_text)
    tfidf_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(tfidf_text)
    return prediction[0]

def predict_proba_for_explainer(texts, vectorizer, model):
    """Returns prediction probabilities for explainers (LIME/SHAP)."""
    cleaned_texts = [preprocess_text(text) for text in texts]
    tfidf_vectors = vectorizer.transform(cleaned_texts)
    return model.predict_proba(tfidf_vectors)

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("Fake News Detector")

# Load model and vectorizer (will be cached)
tfidf_vectorizer, model, train_df_processed = load_model_and_vectorizer()

news_article = st.text_area("Enter a news article here:", height=300)

if st.button("Predict"):
    if news_article:
        prediction = predict_fake_news(news_article, tfidf_vectorizer, model)
        if prediction == 1:
            st.success(f"Prediction: Real News ({prediction})")
        else:
            st.error(f"Prediction: Fake News ({prediction})")
    else:
        st.warning("Please enter a news article to predict.")

st.markdown("---")
st.subheader("Explanation")

col1, col2 = st.columns(2)

with col1:
    if st.button("Explain (LIME)"):
        if news_article:
            with st.spinner("Generating LIME explanation..."):
                explainer = lime.lime_text.LimeTextExplainer(class_names=['Fake', 'Real'])
                exp = explainer.explain_instance(
                    news_article,
                    lambda x: predict_proba_for_explainer(x, tfidf_vectorizer, model),
                    num_features=10,
                    num_samples=1000 # Number of perturbations for LIME
                )
                st.components.v1.html(exp.as_html(), height=400, scrolling=True)
        else:
            st.warning("Please enter a news article to generate LIME explanation.")

with col2:
    if st.button("Explain (SHAP)"):
        if news_article:
            with st.spinner("Generating SHAP explanation..."):
                # Prepare background data for SHAP KernelExplainer
                # Sample a small number of cleaned texts for background for faster computation
                background_data_for_shap = train_df_processed['cleaned_text'].sample(100, random_state=42).tolist()
                background_tfidf_for_shap = tfidf_vectorizer.transform(background_data_for_shap)

                # Initialize SHAP KernelExplainer
                # The explainer works on the TF-IDF vectorized data directly
                shap_explainer = shap.KernelExplainer(model.predict_proba, background_tfidf_for_shap)

                # Preprocess and vectorize the input article for SHAP
                cleaned_input_text = preprocess_text(news_article)
                tfidf_input_text = tfidf_vectorizer.transform([cleaned_input_text])

                # Compute SHAP values for the input text
                shap_values = shap_explainer.shap_values(tfidf_input_text)

                # Get feature names from the vectorizer
                feature_names = tfidf_vectorizer.get_feature_names_out()

                # Determine the index for the predicted class
                predicted_class_idx = model.predict(tfidf_input_text)[0]
                output_idx = list(model.classes_).index(predicted_class_idx) # Get the index of the predicted class label
                class_names = ['Fake', 'Real'] # Assuming 0=Fake, 1=Real

                # Create a SHAP Explanation object for the predicted class
                shap_explanation = shap.Explanation(
                    values=shap_values[output_idx][0],
                    base_values=shap_explainer.expected_value[output_idx],
                    data=tfidf_input_text.toarray()[0],
                    feature_names=feature_names
                )

                # Display the SHAP explanation using a waterfall plot
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.plots.waterfall(shap_explanation, max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig) # Close the plot to free memory
        else:
            st.warning("Please enter a news article to generate SHAP explanation.")

st.markdown("---")
st.write("Do you have further questions or want to try another article?")
