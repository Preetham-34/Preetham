import re
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def check_nltk_data():
    try:
        stopwords.words('english')
        WordNetLemmatizer()
    except LookupError:
        nltk.download(['stopwords', 'wordnet', 'omw-1.4'])

check_nltk_data()  # Ensure NLTK data is downloaded

def preprocess_text(text):
    """Clean and normalize text data"""
    # Normalization
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenization
    tokens = text.split()
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)
