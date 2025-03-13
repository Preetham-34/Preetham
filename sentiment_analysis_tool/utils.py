import re
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import joblib
import streamlit as st

# Text cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text)      # Remove extra spaces
    return text.strip()

# Model loading with caching
@st.cache_resource
def load_models():
    models = {
        "VADER": SentimentIntensityAnalyzer(),
        "TextBlob": None,
        "DistilBERT": pipeline("sentiment-analysis", 
                             model="distilbert-base-uncased-finetuned-sst-2-english"),
        "CustomModel": None
    }
    
    try:
        models["CustomModel"] = joblib.load('models/sentiment_model.pkl')
    except Exception as e:
        st.error(f"Custom model not loaded: {str(e)}")
    
    return models

# Sentiment analysis functions
def analyze_text(text, models):
    cleaned = clean_text(text)
    results = {"text": text}

    # VADER Analysis
    vader_scores = models["VADER"].polarity_scores(cleaned)
    results["VADER"] = {
        "compound": vader_scores['compound'],
        "label": "Positive" if vader_scores['compound'] >= 0.05 else 
                 "Negative" if vader_scores['compound'] <= -0.05 else "Neutral"
    }

    # TextBlob Analysis
    blob = TextBlob(cleaned)
    results["TextBlob"] = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "label": "Positive" if blob.sentiment.polarity > 0 else 
                 "Negative" if blob.sentiment.polarity < 0 else "Neutral"
    }

    # DistilBERT Analysis
    try:
        bert_result = models["DistilBERT"](cleaned)[0]
        results["DistilBERT"] = {
            "label": bert_result['label'],
            "score": bert_result['score']
        }
    except Exception as e:
        results["DistilBERT"] = {"error": str(e)}

    # Custom Model Analysis
    if models["CustomModel"]:
        try:
    models["CustomModel"] = joblib.load('models/sentiment_model.pkl')
except FileNotFoundError:
    st.warning("Custom model not found - using default models only")
    models["CustomModel"] = None
except Exception as e:
    st.error(f"Error loading custom model: {str(e)}")
    models["CustomModel"] = None

    return results

def analyze_data(data, models):
    return [analyze_text(text, models) for text in data if text.strip()]

# Visualization functions
def generate_visualizations(results):
    df = pd.DataFrame([{
        'text': r['text'],
        'VADER': r['VADER']['label'],
        'TextBlob': r['TextBlob']['label'],
        'DistilBERT': r['DistilBERT']['label'],
        'CustomModel': r.get('CustomModel', {}).get('label', 'N/A')
    } for r in results])

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📊 Summary", "🔍 Comparison", "☁️ Word Cloud"])

    with tab1:
        st.subheader("Sentiment Distribution")
        cols = st.columns(4)
        models = ['VADER', 'TextBlob', 'DistilBERT', 'CustomModel']
        for col, model in zip(cols, models):
            with col:
                if model in df.columns:
                    st.write(f"**{model}**")
                    st.bar_chart(df[model].value_counts())

    with tab2:
        st.subheader("Model Comparison")
        comparison_df = df[['text'] + models]
        st.dataframe(
            comparison_df.style.highlight_max(axis=1, color='#90EE90'),
            use_container_width=True,
            height=400
        )

    with tab3:
        st.subheader("Word Cloud")
        if not df.empty:
            all_text = ' '.join(df['text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)
        else:
            st.warning("No text available for word cloud")
