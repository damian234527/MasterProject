import requests
from bs4 import BeautifulSoup
import joblib
import spacy
import sys
import nltk
import pandas as pd
from datetime import datetime
from nltk.corpus import stopwords
from headline_classifier import HeadlineClassifier
from headline_content_similarity import (
    CosineSimilarityTFIDF,
    TransformerEmbeddingSimilarity,
    ClickbaitModelScore,
    HeadlineContentSimilarity
)

# Ensure necessary NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# ================ Scraper ================

class ArticleScraper:
    def __init__(self, url: str):
        self.url = url
        self.soup = self._fetch_content()

    def _fetch_content(self):
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def get_headline(self) -> str:
        headline_tag = self.soup.find("h1")
        return headline_tag.text.strip() if headline_tag else ""

    def get_content(self) -> str:
        paragraphs = self.soup.find_all("p")
        return " ".join(p.text.strip() for p in paragraphs if p.text.strip())

# ================ Detector ================

class ClickbaitAndSimilarityDetector:
    def __init__(self, clickbait_model_path: str = "headline_model.joblib", model_type: str = "logistic"):
        self.headline_classifier = HeadlineClassifier(model_path=clickbait_model_path, model_type=model_type)
        self.headline_classifier.load_model()
        self.spacy_nlp = spacy.load("en_core_web_sm")

    def preprocess_text(self, text: str) -> str:
        doc = self.spacy_nlp(text.lower())
        return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

    def detect_clickbait(self, headline: str) -> bool:
        prediction = self.headline_classifier.predict([headline])[0]
        return prediction

    def compare_similarity(self, headline: str, content: str) -> dict:
        methods = {
            "TF-IDF Cosine": CosineSimilarityTFIDF(),
            "Transformer Embedding": TransformerEmbeddingSimilarity(),
            "Clickbait Transformer Model": ClickbaitModelScore(model_type="transformer", model_name_or_path="bert-base-uncased") # ./models/transformer_bert-base-uncased_bert-base-uncased_1745798398/best_model
        }

        scores = {}
        for method_name, method in methods.items():
            comparator = HeadlineContentSimilarity(method)
            score = comparator.compare(headline, content)
            scores[method_name] = score
            print(f"{method_name} Similarity Score: {score:.4f}")
        return scores

# ================ Main Program ================

def main(urls: list[str] | str, model_type: str = "logistic"):
    if isinstance(urls, str):
        urls = [urls]

    detector = ClickbaitAndSimilarityDetector(model_type=model_type)
    results = []

    for url in urls:
        print(f"\nProcessing URL: {url}\n{'-'*50}")
        scraper = ArticleScraper(url)

        headline = scraper.get_headline()
        content = scraper.get_content()

        if not headline or not content:
            print("Error: Missing headline or content. Skipping this URL.")
            continue

        print(f"Headline: {headline}\n")
        print(f"Content Snippet: {content[:300]}...\n")

        is_clickbait = detector.detect_clickbait(headline)
        print(f"Clickbait Detection Result: {'CLICKBAIT' if is_clickbait else 'NOT CLICKBAIT'}\n")

        similarity_scores = detector.compare_similarity(headline, content)

        result = {
            "URL": url,
            "Headline": headline,
            "Clickbait": "Yes" if is_clickbait else "No",
            **similarity_scores
        }
        results.append(result)

    # Save results to timestamped CSV
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"headline_analysis_results_{timestamp}.csv"
        df = pd.DataFrame(results)
        df.to_csv(filename, index=False)
        print(f"\nAll results saved to '{filename}'.")

if __name__ == "__main__":
    # Example usage
    debug_urls = [
        "https://www.buzzfeed.com/stephaniemcneal/a-couple-did-a-stunning-photo-shoot-with-their-baby-after-le",
        # Add more URLs if needed
    ]

    if len(sys.argv) > 2:
        urls = sys.argv[1:-1]
        model_type = sys.argv[-1]
        main(urls, model_type)
    else:
        main(debug_urls, model_type="logistic")
