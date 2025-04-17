import requests
from bs4 import BeautifulSoup
import joblib
import spacy
import sys
# import headline_classifier
import headline_content_similarity

# Retrieval of article via URL
# Extraction of article header and content
# Tokenization
# Deletion of special characters and stop words
# Lemmatization or stemming
# POS-tagging

class ArticleScrapper:
    """Class for scraping and extracting the article"""

    def __init__(self, url_article: str):
        self.url = url_article
        self.soup = self._fetchContent()

    def _fetchContent(self):
        """Fetch the article content"""
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(self.url, headers=headers)
        response.raise_for_status()
        return BeautifulSoup(response.text, "html.parser")

    def getHeadline(self):
        """Returns article headline (h1 tag from HTML)"""
        if self.soup.find("h1"):
            return self.soup.find("h1").text.strip()
        # TODO what if there is no h1 tag

    def getContent(self):
        """Returns all paragraphs merged (p tags from HTML)"""
        paragraphs = self.soup.find_all("p")
        return "".join(paragraph.text.strip() for paragraph in paragraphs if paragraph.text.strip())

class ClickbaitDetector:
    """Class analysing whether the article is clickbait or not based on the headline and content"""

    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.natural_language_processor = spacy.load("en_core_web_sm") # TODO change

    def preprocess(self, text: str) -> str:
        """Clean text for analysis"""
        doc = self.natural_language_processor(text.lower()) #sequence of spaCy Token objects
        return "".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

    def predict(self, headline: str, content: str, comparison_methods: str | list[str] = "cosine") -> str:
        """Predicts whether the article headline is consistent with its content"""
        headline_preprocessed = headline
        content_preprocessed = content
        if isinstance(comparison_methods, str):
            comparison_methods = [comparison_methods]
        # headline_preprocessed = self.preprocess(headline)
        # content_preprocessed = self.preprocess(content)
        self.classifyHeadline(headline_preprocessed)
        self.compare(headline_preprocessed, content_preprocessed, comparison_methods)

    def classifyHeadline(self, headline: str):
        headline_result = self.model.predict([headline])
        print(f"Result of headline classification: {'CLICKBAIT' if headline_result else 'NOT CLICKBAIT'}")
        return headline_result

    # TODO cosine similarity not working
    def compare(self, headline: str, content: str, methods: list[str]):
        comparator = headline_content_similarity.HeadlineContentSimilarity(method="cosine")
        for method in methods:
            if method == "cosine":
                method = headline_content_similarity.CosineSimilarity()
            else:
                method = headline_content_similarity.TransformerSimilarity(method)

            comparator.setMethod(method)
            score = comparator.compare(headline, content)
            print(f"Similarity score: {score:.4f}")

def main(urls: str | list[str] = None, comparison_methods: str | list[str] = "distilbert-base-uncased"):
    detector = ClickbaitDetector('headline_model.joblib', 'vectorizer.pkl')  # TODO change the models
    if isinstance(urls, str):
        urls = [urls]
    for url in urls:
        if not url:
            url = input("Enter the URL of the article: ")
        scrapper = ArticleScrapper(url)
        headline = scrapper.getHeadline()
        content = scrapper.getContent()

        detector.predict(headline, content, comparison_methods)

if __name__ == "__main__":
    debug = 1
    debug_links = ["https://www.buzzfeed.com/stephaniemcneal/a-couple-did-a-stunning-photo-shoot-with-their-baby-after-le"]
    if not debug:
        main(sys.argv[1] if len(sys.argv) > 1 else None)
    else:
        main(debug_links)

    # https://www.hindustantimes.com/india-news/cracks-in-trinamool-kalyan-banerjee-whatsapp-chat-leak-puts-spotlight-on-party-infighting-ahead-of-election-101744181659155.html
    # https://www.dailymail.co.uk/news/article-14599119/Our-neighbour-hell-groomed-left-brink-suicide-ruined-lives.html

    # Clickbaits
    # https://www.buzzfeed.com/stephaniemcneal/a-couple-did-a-stunning-photo-shoot-with-their-baby-after-le