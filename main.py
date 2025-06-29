import logging_config
import logging
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
import joblib
import spacy
import sys
import numpy as np
import nltk
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from tqdm import tqdm
from config import ARTICLE_SCRAPING_CONFIG, HEADLINE_CONFIG, HEADLINE_CONTENT_CONFIG
# Updated import to get the class, not the instance
from headline_classifier import HeadlineClassifier
from headline_content_similarity import (
    CosineSimilarityTFIDF,
    TransformerEmbeddingSimilarity,
    ClickbaitModelScore,
    HeadlineContentSimilarity
)
from headline_content_evaluation import evaluate_clickbait_predictions
from headline_content_feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)

# Ensure necessary NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("vader_lexicon")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("vader_lexicon")


# ================ Scraper (No Changes) ================

class ArticleScraper:
    def __init__(self, url: str):
        self.url = url
        self.article_obj = Article(self.url)
        self._fetch_and_parse_article()

    def _fetch_and_parse_article(self):
        try:
            self.article_obj.download()
            self.article_obj.parse()
            self.article_obj.nlp()  # For keywords, summary if needed later
        except ArticleException as e:
            logging.error(f"Error processing article with newspaper3k from {self.url}: {e}")
            self.headline = ""
            self.content = ""
            self._fetch_content_basic()

    def _fetch_content_basic(self):
        """Fallback basic scraping if newspaper3k fails."""
        headers = {"User-Agent": "Mozilla/5.0"}
        try:
            response = requests.get(self.url, headers=headers)
            response.raise_for_status()
            self.soup = BeautifulSoup(response.text, "html.parser")
            headline_tag = self.soup.find("h1")
            self.headline = headline_tag.text.strip() if headline_tag else ""
            paragraphs = self.soup.find_all("p")
            self.content = " ".join(p.text.strip() for p in paragraphs if p.text.strip())
        except requests.RequestException as e:
            logging.error(f"Fallback HTTP error for {self.url}: {e}")
            self.headline = ""
            self.content = ""
        except Exception as e:
            logging.error(f"Fallback BS4 parsing error for {self.url}: {e}")
            self.headline = ""
            self.content = ""

    def get_headline(self) -> str:
        if hasattr(self.article_obj, 'title') and self.article_obj.title:
            return self.article_obj.title
        return getattr(self, 'headline', "")

    def get_content(self) -> str:
        if hasattr(self.article_obj, 'text') and self.article_obj.text:
            return self.article_obj.text
        return getattr(self, 'content', "")


# ================ NEW: Feature Extractor ================
# This class is adapted from your clickbait17_dataset.py to be used for single articles.

class ArticleFeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        # Basic regex for clickbait-y phrases for feature counting
        clickbait_terms = [
            "you won't believe", "what happens next", "the reason why", "secret to",
            "shocked", "reveals", "hidden", "amazing"
        ]
        pattern = r"\b(?:%s)\b" % "|".join(clickbait_terms)
        self.clickbait_regex = re.compile(pattern, re.IGNORECASE)

    def extract(self, headline: str, content: str) -> dict:
        """Extracts a dictionary of features from the headline and content."""
        headline_words = headline.split()
        content_words = content.split()

        headline_length_words = len(headline_words)
        content_length_words = len(content_words)

        features = {
            "Headline Words": headline_length_words,
            "Content Words": content_length_words,
            "Headline-Content Word Ratio": headline_length_words / max(content_length_words, 1),
            "Exclamation Marks in Headline": headline.count("!"),
            "Question Marks in Headline": headline.count("?"),
            "Uppercase Ratio in Headline": sum(c.isupper() for c in headline) / max(len(headline), 1),
            "Stopword Ratio in Headline": sum(w.lower() in self.stop_words for w in headline_words) / max(
                headline_length_words, 1),
            "Clickbait Phrase Count": len(self.clickbait_regex.findall(headline)),
            "Sentiment Polarity (Headline)": self.sentiment_analyzer.polarity_scores(headline)["compound"],
            "Sentiment Polarity (Content)": self.sentiment_analyzer.polarity_scores(content)["compound"]
        }

        # Round float values for cleaner display
        for key, value in features.items():
            if isinstance(value, float):
                features[key] = round(value, 4)

        return features


# ================ Detector (Updated) ================

class ClickbaitAndSimilarityDetector:
    def __init__(self,
                 headline_model_path: str = HEADLINE_CONFIG["model_path"],
                 headline_model_type: str = HEADLINE_CONFIG["model_type"],
                 headline_content_model_path: str = HEADLINE_CONTENT_CONFIG["model_path_default"][1],
                 headline_content_model_type: str = HEADLINE_CONTENT_CONFIG["model_type"][1],
                 headline_content_transformer: str = HEADLINE_CONTENT_CONFIG["model_name"]):
        self.headline_classifier = HeadlineClassifier(model_path=headline_model_path, model_type=headline_model_type)
        self.headline_classifier.load_model()
        self.feature_extractor = FeatureExtractor()

        self.headline_content_type = headline_content_model_type
        self.headline_content_path = headline_content_model_path
        self.headline_content_transformer = headline_content_transformer
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.headline_content_comparator = HeadlineContentSimilarity(
            ClickbaitModelScore(
                model_type=headline_content_model_type,
                model_name_or_path=headline_content_model_path
            )
        )

    def preprocess_text(self, text: str) -> str:
        doc = self.spacy_nlp(text.lower())
        return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

    def detect_clickbait(self, headline: str) -> tuple[bool, float]:
        """
        Detects if a headline is clickbait.
        Returns a tuple: (is_clickbait_boolean, clickbait_probability_score)
        """
        # UPDATED: Use the new predict_proba method
        score = self.headline_classifier.predict_proba([headline])[0]
        prediction = score >= 0.5  # Assuming 0.5 threshold
        return prediction, score

    # MODIFIED: Added 'post' argument and implemented the method body
    def compare_similarity(self, headline: str, content: str, post: str = None, headline_score: float = None) -> dict:
        """
        Calculates similarity scores using multiple methods.
        The 'post' text is used by the transformer model.
        """
        methods = {
            "TF-IDF Cosine": CosineSimilarityTFIDF(),
            "Transformer Embedding": TransformerEmbeddingSimilarity(model_name=self.headline_content_transformer),
            "Clickbait Transformer Model": ClickbaitModelScore(model_type=self.headline_content_type,
                                                               model_name_or_path=self.headline_content_path)
        }

        scores = {}
        for method_name, method in methods.items():
            comparator = HeadlineContentSimilarity(method)
            # MODIFIED: Pass the post to the comparator
            score = comparator.compare(headline, content, post=post)
            scores[method_name] = round(score, 4)
            logging.info(f"{method_name} Similarity Score: {score:.4f}")
        return scores

    # MODIFIED: Added 'post' argument
    def get_headline_content_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Calculates the similarity score using the primary headline-content model."""
        # MODIFIED: Pass the post and headline_score to the comparator
        return self.headline_content_comparator.compare(headline, content, post=post, headline_score=headline_score)

    def extract_features(self, post: str, headline: str, content: str) -> dict:
        """Wrapper to call the centralized feature extractor and return a dictionary."""
        # Returns a dictionary of all 22 features for display.
        return self.feature_extractor.extract(post, headline, content, as_dict=True)



# ================ Main Program (Heavily Updated) ================

def main(articles: list[dict], model_type: str = "logistic"):
    """
    Processes a list of articles, analyzes them for clickbait, and displays the results.

    Args:
        articles (list[dict]): A list of dictionaries. Each dictionary must have a 'url' key,
                               and can optionally have a 'post' key with the social media text.
        model_type (str): The type of headline classifier to use.
    """
    # MODIFIED: Added robust input handling for different formats
    if isinstance(articles, dict):
        articles = [articles]
    elif isinstance(articles, str):
        articles = [{'url': articles}]
    elif isinstance(articles, list) and all(isinstance(i, str) for i in articles):
        articles = [{'url': url} for url in articles]

    detector = ClickbaitAndSimilarityDetector(headline_model_type=model_type)
    results = []
    article_features = {}

    for article_data in articles:
        url = article_data.get("url")
        post = article_data.get("post")  # MODIFIED: Get the post text

        if not url:
            logging.warning(f"Skipping article data with no URL: {article_data}")
            continue

        print(f"\n\n{'=' * 25}\nProcessing URL: {url}\n{'=' * 25}")
        # MODIFIED: Display the post if it exists
        if post:
            print(f"Provided Post: '{post}'\n")

        scraper_successful = False
        headline = ""
        content = ""
        error_message = "N/A"

        try:
            scraper = ArticleScraper(url)
            headline = scraper.get_headline()
            content = scraper.get_content()

            if not headline or not content:
                error_message = "Headline or content extraction failed (empty)."
                logging.warning(f"{error_message} for URL: {url}")
            else:
                scraper_successful = True

        except Exception as e:
            error_message = f"Critical error during scraping: {str(e)}"
            logging.error(f"{error_message} for URL: {url}")

        if scraper_successful:
            logging.info(f"Successfully scraped: {headline[:30]}...")
            if len(content.split()) < ARTICLE_SCRAPING_CONFIG["content_length_min"]:
                logging.warning(
                    f"Content very short for URL {url}; length: {len(content.split())} words). May affect analysis.")

            is_clickbait, clickbait_score = detector.detect_clickbait(headline)
            clickbait_status = "CLICKBAIT" if is_clickbait else "NOT CLICKBAIT"
            print(f"📰 Headline: '{headline}'")
            print(f"🎯 Clickbait Detection Result: {clickbait_status} (Score: {clickbait_score:.4f})\n")

            similarity_scores = detector.compare_similarity(headline, content, post=post, headline_score=clickbait_score)
            article_features = detector.extract_features(post, headline, content)

            result = {
                "URL": url,
                "Post": post if post else "N/A",  # MODIFIED: Add post to results
                "Headline": headline,
                "Status": "Processed",
                "Error": "None",
                "Clickbait": "Yes" if is_clickbait else "No",
                "Clickbait Score": round(clickbait_score, 4),
                **similarity_scores,
                **article_features
            }
        else:
            result = {
                "URL": url,
                "Post": post if post else "N/A",  # MODIFIED: Add post to results
                "Headline": "N/A",
                "Status": "Failed",
                "Error": error_message,
                "Clickbait": "N/A",
                "Clickbait Score": "N/A",
            }
        results.append(result)

    if results:
        df = pd.DataFrame(results)

        # MODIFIED: Added 'Post' to the list of base columns
        base_cols = ["URL", "Post", "Clickbait", "Clickbait Score"]
        similarity_cols = [col for col in ["TF-IDF Cosine", "Transformer Embedding", "Clickbait Transformer Model"] if
                           col in df.columns]
        feature_cols = [col for col in article_features.keys() if col in df.columns]

        display_cols = base_cols + similarity_cols
        print("\n\n" + "=" * 50)
        print("📊 HEADLINE ANALYSIS SUMMARY")
        print("=" * 50)
        print(df[display_cols].to_string())

        print("\n\n" + "=" * 50)
        print("📝 EXTRACTED ARTICLE FEATURES")
        print("=" * 50)
        feature_display_cols = ["URL"] + feature_cols
        print(df[feature_display_cols].to_string())

        if len(df[df['Status'] == 'Processed']) > 1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                stats_df = df[numeric_cols].agg(['mean', 'median']).round(4)
                print("\n\n" + "=" * 50)
                print("📈 AGGREGATE STATISTICS (for processed articles)")
                print("=" * 50)
                print(stats_df.to_string())

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"headline_analysis_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n\n✅ All results saved to '{filename}'.")


# ================ NEW: Evaluation Function ================

def evaluate_on_test_set(csv_path: str, model_type: str = "logistic"): #
    """
    Runs an evaluation on a local test CSV file, calculates a weighted score,
    and prints the final performance metrics.
    """
    print(f"Loading test data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # MODIFIED: Ensure 'post' column exists, fill NaNs with empty string
        if 'post' not in df.columns:
            df['post'] = ''
        df['post'] = df['post'].fillna('')
        df = df.dropna(subset=['headline', 'content', 'clickbait_score']).reset_index(drop=True)
    except FileNotFoundError:
        logger.error(f"Test file not found at '{csv_path}'. Please check the path.")
        return

    detector = ClickbaitAndSimilarityDetector(headline_model_type=model_type) #

    y_true = []
    y_pred_final = []

    # Weights are now only used for the standard model
    HEADLINE_MODEL_WEIGHT = 1
    CONTENT_MODEL_WEIGHT = 2
    TOTAL_WEIGHT = HEADLINE_MODEL_WEIGHT + CONTENT_MODEL_WEIGHT

    print("Processing test set to generate predictions...")
    if detector.headline_content_type == 'hybrid':
        print("Evaluation Mode: Hybrid Model (direct output).")
    else:
        print("Evaluation Mode: Standard Model (weighted average).")

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating"):
        headline = row['headline']
        content = row['content']
        post = row['post']  # MODIFIED: Get the post from the CSV
        ground_truth_score = row['clickbait_score']

        _, headline_score = detector.detect_clickbait(headline) #
        # MODIFIED: Pass the 'post' to the content score function
        content_score = detector.get_headline_content_score(headline, content, post=post, headline_score=headline_score) #

        if pd.isna(headline_score) or pd.isna(content_score):
            logger.warning(f"Skipping row due to NaN score. Headline: '{headline[:30]}...'")
            continue

        # MODIFIED: Conditional scoring based on model type
        final_score = 0.0
        if detector.headline_content_type == 'hybrid':
            # For the hybrid model, its output, which has already used the headline_score
            # as a feature, is the final score.
            final_score = content_score
        else:
            # For standard models, perform the weighted average.
            weighted_score = (
                                     (headline_score * HEADLINE_MODEL_WEIGHT) +
                                     (content_score * CONTENT_MODEL_WEIGHT)
                             ) / TOTAL_WEIGHT
            final_score = weighted_score

        y_true.append(ground_truth_score)
        y_pred_final.append(final_score)

    print("\n" + "=" * 50)
    if detector.headline_content_type == 'hybrid':
        print("      FINAL HYBRID MODEL PERFORMANCE METRICS")
    else:
        print("      FINAL WEIGHTED MODEL PERFORMANCE METRICS")
    print("=" * 50)

    if not y_true or not y_pred_final:
        print("No valid scores were generated. Cannot compute metrics.")
        return

    evaluate_clickbait_predictions(
        y_true=y_true,
        y_pred=y_pred_final,
        verbose=True
    )
    print("=" * 50)


if __name__ == "__main__":
    # MODIFIED: Example usage now uses a list of dictionaries with 'url' and 'post'
    debug_articles = [
        {
            "url": "https://www.buzzfeed.com/stephaniemcneal/a-couple-did-a-stunning-photo-shoot-with-their-baby-after-le",
            "post": "This Couple's Photoshoot With Their Baby Is Going Viral For The Most Amazing Reason"
        },
        {
            "url": "https://apnews.com/live/israel-iran-attack",
            # This article has no associated "post"
        }
    ]

    if len(sys.argv) > 1:
        # Command-line processing now assumes all args are URLs and converts them to the new format
        if sys.argv[-1] in ["logistic", "naive_bayes", "random_forest", "svm"]:
            urls_from_args = sys.argv[1:-1]
            model_type_from_args = sys.argv[-1]
        else:
            urls_from_args = sys.argv[1:]
            model_type_from_args = "logistic"  # default

        # MODIFIED: Convert list of URL strings to the expected list of dictionaries
        articles_from_args = [{'url': url} for url in urls_from_args]
        main(articles_from_args, model_type_from_args)
    else:
        # Run the main analysis function with the debug examples
        print("\n--- Running Main Analysis on Debug Articles ---")
        # main(debug_articles)

        # Run the evaluation on the Clickbait17 test set
        print("\n--- Running Evaluation on Clickbait17 Test Set ---")
        test_csv_path = "data/clickbait17/models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_test_features.csv"
        evaluate_on_test_set(csv_path=test_csv_path)