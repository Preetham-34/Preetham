import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib
import re
import os

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def train_model():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    try:
        df = pd.read_csv('data/sentimentdataset.csv')
    except FileNotFoundError:
        print("Error: Training data not found at 'data/social_media_data.csv'")
        return

    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove neutral samples for binary classification
    df = df[df['label'].isin([0, 1])]
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'], df['label'], test_size=0.2, random_state=42
    )
    
    model = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])
    
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'models/sentiment_model.pkl')
    print("\nModel saved to 'models/sentiment_model.pkl'")

if __name__ == "__main__":
    train_model()
