import requests
from bs4 import BeautifulSoup
import re
import joblib
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from urllib.parse import urlparse


class ArticleScraper:
    """Class for scraping and extracting content from an online article."""

    def __init__(self, url: str):
        self.url = url
        self.soup = self._fetch_content()

    def _fetch_content(self):
        """Fetch and parse the HTML content of the article."""
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, 'html.parser')

    def get_headline(self):
        """Extract the article's headline."""
        return self.soup.find('h1').text.strip() if self.soup.find('h1') else ""

    def get_content(self):
        """Extract the main body content of the article."""
        paragraphs = self.soup.find_all('p')
        return ' '.join(p.text.strip() for p in paragraphs if p.text.strip())


class ClickbaitDetector:
    """Class to analyse article content and predict if the headline aligns with the content."""

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.nlp = spacy.load('en_core_web_sm')

    def preprocess(self, text: str) -> str:
        """Clean and preprocess text for analysis."""
        doc = self.nlp(text.lower())
        return ' '.join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

    def predict(self, headline: str, content: str) -> str:
        """Predict if the article headline aligns with the content."""
        preprocessed_headline = self.preprocess(headline)
        preprocessed_content = self.preprocess(content)
        features = self.vectorizer.transform([preprocessed_headline + ' ' + preprocessed_content])
        prediction = self.model.predict(features)
        print(prediction)
        return "Likely Clickbait" if prediction[0] == 1 else "Likely Legitimate"


def main():
    url = input("Enter the URL of the article: ")
    scraper = ArticleScraper(url)
    headline = scraper.get_headline()
    content = scraper.get_content()

    if not headline or not content:
        print("Failed to extract necessary content from the article.")
        return

    detector = ClickbaitDetector('clickbait_model.pkl', 'vectorizer.pkl')
    result = detector.predict(headline, content)

    print(f"Headline: {headline}\n")
    print(f"Analysis Result: {result}")


if __name__ == "__main__":
    main()
