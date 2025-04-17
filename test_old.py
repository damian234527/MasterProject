import requests
from bs4 import BeautifulSoup
import nltk
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import numpy as np
import pickle

# Download necessary NLTK resources
nltk.download('punkt')


# Function to extract article text and headline
def extract_article(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract headline (assumed to be in <title> or <h1>)
    headline = soup.title.text if soup.title else ""
    h1_tag = soup.find('h1')
    if h1_tag:
        headline = h1_tag.text.strip()

    # Extract article text (assumed to be in <p> tags)
    paragraphs = soup.find_all('p')
    article_text = " ".join([p.text for p in paragraphs])

    return headline, article_text


# Function to compare headline with article content
def compare_headline_content(headline, article_text):
    summarizer = pipeline("summarization")
    summary = summarizer(article_text[:1024], max_length=50, min_length=20, do_sample=False)[0]['summary_text']

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([headline, summary])
    similarity = (vectors * vectors.T).A[0, 1]
    return similarity


# Load pre-trained clickbait detection model
def load_model():
    try:
        with open('clickbait_model.pkl', 'rb') as file:
            model, vectorizer = pickle.load(file)
        return model, vectorizer
    except FileNotFoundError:
        return None, None


# Predict clickbait likelihood
def predict_clickbait(headline):
    model, vectorizer = load_model()
    if model and vectorizer:
        features = vectorizer.transform([headline])
        prediction = model.predict(features)
        return "Clickbait" if prediction[0] == 1 else "Not Clickbait"
    return "Model not trained"


# Main function
def analyse_article(url):
    headline, article_text = extract_article(url)
    if not headline or not article_text:
        return "Could not extract necessary information."

    similarity_score = compare_headline_content(headline, article_text)
    clickbait_prediction = predict_clickbait(headline)

    return {
        "Headline": headline,
        "Similarity Score": similarity_score,
        "Clickbait Prediction": clickbait_prediction
    }


# Example Usage
if __name__ == "__main__":
    test_url = "https://example.com/some-article"
    result = analyse_article(test_url)
    print(result)
