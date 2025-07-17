# main.py
"""Main script for running the clickbait detection and analysis tool.

This script provides the primary interface for analyzing articles. It can be run
from the command line to process a list of URLs or used to evaluate the model
performance on a pre-labeled test set. The main functionalities include:
- Scraping article headlines and content from URLs.
- Detecting clickbait in headlines using a trained classifier.
- Comparing the similarity between headlines and content using multiple methods.
- Extracting a wide range of linguistic and structural features.
- Displaying and saving a comprehensive analysis report.
"""
import logging_config
import logging
import requests
from bs4 import BeautifulSoup
from newspaper import Article, ArticleException
import joblib
import spacy
import sys
import time
import numpy as np
import nltk
import pandas as pd
from datetime import datetime
from transformers import AutoTokenizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from tqdm import tqdm
from config import ARTICLE_SCRAPING_CONFIG, HEADLINE_CONFIG, HEADLINE_CONTENT_CONFIG, GENERAL_CONFIG
from headline_classifier import HeadlineClassifier
from headline_content_similarity import (
    CosineSimilarityTFIDF,
    TransformerEmbeddingSimilarity,
    ClickbaitModelScore,
    HeadlineContentSimilarity
)
from headline_content_evaluation import evaluate_clickbait_predictions
from headline_content_feature_extractor import FeatureExtractor
from utils import set_seed

logger = logging.getLogger(__name__)

# Pre-flight check for NLTK data dependencies. This block ensures that all
# required NLTK packages are downloaded before the program proceeds.
try:
    nltk.data.find("corpora/stopwords")
    nltk.data.find("tokenizers/punkt_tab")
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("stopwords")
    nltk.download("punkt_tab")
    nltk.download("vader_lexicon")

# Set seed for reproducibility
set_seed(GENERAL_CONFIG["seed"])

class ArticleScraper:
    """Scrapes the headline and content of an article from a given URL.

    This class uses the 'newspaper3k' library as its primary scraping tool.
    If 'newspaper3k' fails, it falls back to a basic scraping method using
    'requests' and 'BeautifulSoup' to find the <h1> and <p> tags.

    Attributes:
        url (str): The URL of the article to scrape.
        article_obj (newspaper.Article): An instance of the newspaper Article.
        headline (str): The scraped headline of the article.
        content (str): The scraped main content of the article.
        soup (BeautifulSoup): A BeautifulSoup object, used in fallback scraping.
    """

    def __init__(self, url: str):
        """Initializes the scraper and starts the scraping process.

        Args:
            url (str): The URL of the article to be scraped.
        """
        self.url = url
        self.article_obj = Article(self.url)
        self._fetch_and_parse_article()

    def _fetch_and_parse_article(self):
        """Fetches and parses the article using the newspaper3k library."""
        try:
            self.article_obj.download()
            self.article_obj.parse()
            self.article_obj.nlp()
        except ArticleException as e:
            # Log the error and attempt fallback scraping if newspaper3k fails.
            logging.error(f"Error processing article with newspaper3k from {self.url}: {e}")
            self.headline = ""
            self.content = ""
            self._fetch_content_basic()

    def _fetch_content_basic(self):
        """A fallback scraping method using requests and BeautifulSoup.

        This method is triggered if the primary newspaper3k scraper fails. It
        sends a direct GET request and parses the HTML to find the first <h1>
        as the headline and all <p> tags as the content.
        """
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
        """Returns the scraped headline.

        Prioritizes the headline from newspaper3k if available.

        Returns:
            The article headline as a string.
        """
        if hasattr(self.article_obj, 'title') and self.article_obj.title:
            return self.article_obj.title
        return getattr(self, 'headline', "")

    def get_content(self) -> str:
        """Returns the scraped content.

        Prioritizes the content from newspaper3k if available.

        Returns:
            The article content as a single string.
        """
        if hasattr(self.article_obj, 'text') and self.article_obj.text:
            return self.article_obj.text
        return getattr(self, 'content', "")


class ClickbaitAndSimilarityDetector:
    """Performs clickbait detection, similarity analysis, and feature extraction.

    This class orchestrates the entire analysis pipeline for a given article.
    It loads the necessary models for headline-only clickbait detection and
    for headline-content similarity analysis.

    Attributes:
        headline_classifier (HeadlineClassifier): A model for headline-only
            clickbait classification.
        feature_extractor (FeatureExtractor): A tool for extracting linguistic
            and structural features.
        headline_content_type (str): The type of the headline-content model
            (e.g., 'hybrid', 'standard').
        headline_content_path (str): The file path to the headline-content model.
        headline_content_transformer (str): The name of the transformer used by
            the similarity models.
        spacy_nlp (spacy.Language): An instance of a spaCy language model for
            text preprocessing.
        headline_content_comparator (HeadlineContentSimilarity): The primary
            model for comparing headline and content.
    """

    def __init__(self,
                 headline_model_path: str = HEADLINE_CONFIG["model_path"],
                 headline_model_type: str = HEADLINE_CONFIG["model_type"],
                 headline_content_model_path: str = HEADLINE_CONTENT_CONFIG["model_path_default"][1],
                 headline_content_model_type: str = HEADLINE_CONTENT_CONFIG["model_type"][1],
                 headline_content_transformer: str = HEADLINE_CONTENT_CONFIG["model_name"]):
        """Initializes the detector and loads all required models.

        Args:
            headline_model_path (str): Path to the trained headline classifier model.
            headline_model_type (str): The type of headline classifier to use.
            headline_content_model_path (str): Path to the trained headline-content model.
            headline_content_model_type (str): The type of the headline-content model.
            headline_content_transformer (str): The name or path of the transformer model.
        """
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
        """Preprocesses text by lemmatizing and removing stop words.

        Args:
            text (str): The input text to preprocess.

        Returns:
            The processed text as a string.
        """
        doc = self.spacy_nlp(text.lower())
        return " ".join(token.lemma_ for token in doc if token.is_alpha and not token.is_stop)

    def detect_clickbait(self, headline: str) -> tuple[bool, float]:
        """Detects if a headline is clickbait using the headline-only classifier.

        Args:
            headline (str): The article headline.

        Returns:
            A tuple containing a boolean (True if clickbait) and the raw
            clickbait probability score.
        """
        score = self.headline_classifier.predict_proba([headline])[0]
        prediction = score >= GENERAL_CONFIG["clickbait_threshold"]
        return prediction, score

    def compare_similarity(self, headline: str, content: str, post: str = None, headline_score: float = None) -> dict:
        """Calculates headline-content similarity scores using multiple methods.

        This method computes similarity using three different techniques: TF-IDF
        Cosine Similarity, Transformer Embedding Similarity, and a fine-tuned
        Clickbait Transformer Model.

        Args:
            headline (str): The article headline.
            content (str): The article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): The clickbait score from the
                headline-only model, required for the 'hybrid' model.

        Returns:
            A dictionary mapping each similarity method name to its calculated score.
        """
        methods = {
            "TF-IDF Cosine": CosineSimilarityTFIDF(),
            "Transformer Embedding": TransformerEmbeddingSimilarity(model_name=self.headline_content_transformer),
            "Clickbait Transformer Model": ClickbaitModelScore(model_type=self.headline_content_type,
                                                               model_name_or_path=self.headline_content_path)
        }

        kwargs = {
            'headline': headline,
            'content': content,
            'post': post
        }
        # The hybrid model requires the headline_score as an additional feature.
        if self.headline_content_type == "hybrid":
            kwargs['headline_score'] = headline_score
        scores = {}

        # Iterate through each method and compute the score.
        for method_name, method in methods.items():
            comparator = HeadlineContentSimilarity(method)
            score = comparator.compare(**kwargs)
            scores[method_name] = round(score, 4)
            logging.info(f"{method_name} Similarity Score: {score:.4f}")
        return scores

    def get_headline_content_score(self, headline: str, content: str, post: str = None,
                                   headline_score: float = None) -> float:
        """Calculates the score using the primary headline-content model.

        This is a convenience wrapper to get the score from the main configured
        similarity model (e.g., the 'hybrid' model).

        Args:
            headline (str): The article headline.
            content (str): The article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): The headline-only clickbait score.

        Returns:
            The similarity score as a float.
        """
        return self.headline_content_comparator.compare(headline, content, post=post, headline_score=headline_score)

    def extract_features(self, post: str, headline: str, content: str) -> dict:
        """Extracts linguistic and structural features from the article texts.

        This method acts as a wrapper around the `FeatureExtractor` class.

        Args:
            post (str): The social media post text.
            headline (str): The article headline.
            content (str): The article content.

        Returns:
            A dictionary where keys are feature names and values are the
            calculated feature values.
        """
        return self.feature_extractor.extract(post, headline, content, as_dict=True)


def main(articles: list[dict], model_type: str = "logistic"):
    """Processes a list of articles for clickbait and similarity analysis.

    This function iterates through a list of articles, scrapes their content,
    runs the full analysis pipeline, and then prints and saves a detailed

    report including individual results and aggregate statistics.

    Args:
        articles (list[dict]): A list where each element is a dictionary
            containing a 'url' key and an optional 'post' key. This function
            also handles a single URL string or a list of URL strings.
        model_type (str): The type of headline classifier to use.
    """
    # Standardize various input formats into a list of dictionaries.
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
        post = article_data.get("post")

        if not url:
            logging.warning(f"Skipping article data with no URL: {article_data}")
            continue

        print(f"\n\n{'=' * 25}\nProcessing URL: {url}\n{'=' * 25}")
        if post:
            print(f"Provided Post: '{post}'\n")

        # Initialize status variables for the current article.
        scraper_successful = False
        headline = ""
        content = ""
        error_message = "N/A"

        try:
            # Step 1: Scrape the article.
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
            # Step 2: Perform analysis if scraping was successful.
            logging.info(f"Successfully scraped: {headline[:30]}...")
            if len(content.split()) < ARTICLE_SCRAPING_CONFIG["content_length_min"]:
                logging.warning(
                    f"Content very short for URL {url}; length: {len(content.split())} words).")

            # Perform detection, comparison, and feature extraction.
            is_clickbait, clickbait_score = detector.detect_clickbait(headline)
            clickbait_status = "CLICKBAIT" if is_clickbait else "NOT CLICKBAIT"
            print(f"Headline: '{headline}'")
            print(f"Clickbait Detection Result: {clickbait_status} (Score: {clickbait_score:.4f})\n")

            similarity_scores = detector.compare_similarity(headline, content, post=post,
                                                            headline_score=clickbait_score)
            article_features = detector.extract_features(post, headline, content)

            # Compile results into a dictionary.
            result = {
                "URL": url,
                "Post": post if post else "N/A",
                "Headline": headline,
                "Status": "Processed",
                "Error": "None",
                "Clickbait": "Yes" if is_clickbait else "No",
                "Clickbait Score": round(clickbait_score, 4),
                **similarity_scores,
                **article_features
            }
        else:
            # Compile a failure report.
            result = {
                "URL": url,
                "Post": post if post else "N/A",
                "Headline": "N/A",
                "Status": "Failed",
                "Error": error_message,
                "Clickbait": "N/A",
                "Clickbait Score": "N/A",
            }
        results.append(result)

    # Step 3: Format and display the results.
    if results:
        df = pd.DataFrame(results)

        # Define column groups for structured display.
        base_cols = ["URL", "Post", "Clickbait", "Clickbait Score"]
        similarity_cols = [col for col in ["TF-IDF Cosine", "Transformer Embedding", "Clickbait Transformer Model"] if
                           col in df.columns]
        feature_cols = [col for col in article_features.keys() if col in df.columns]

        # Display the main analysis summary.
        display_cols = base_cols + similarity_cols
        print("\n\n" + "=" * 50)
        print("HEADLINE ANALYSIS SUMMARY")
        print("=" * 50)
        print(df[display_cols].to_string())

        # Display the extracted features.
        print("\n\n" + "=" * 50)
        print("EXTRACTED ARTICLE FEATURES")
        print("=" * 50)
        feature_display_cols = ["URL"] + feature_cols
        print(df[feature_display_cols].to_string())

        # Display aggregate statistics if multiple articles were processed.
        if len(df[df['Status'] == 'Processed']) > 1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                stats_df = df[numeric_cols].agg(['mean', 'median']).round(4)
                print("\n\n" + "=" * 50)
                print("AGGREGATE STATISTICS (for processed articles)")
                print("=" * 50)
                print(stats_df.to_string())

        # Save all results to a timestamped CSV file.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"headline_analysis_results_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\n\nAll results saved to '{filename}'.")


def evaluate_on_test_set(csv_path: str, model_type: str = "naive_bayes"):
    """Evaluates the combined model performance on a labeled test set.

    This function reads a CSV file containing headlines, content, posts, and
    ground-truth scores. It calculates a final, combined clickbait score and
    evaluates its performance using various metrics (MSE, F1, AUC, etc.).

    Args:
        csv_path (str): The file path to the test CSV.
        model_type (str): The type of headline classifier to use.
    """
    print(f"Loading test data from: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
        # Preprocess the dataframe to handle missing values.
        if 'post' not in df.columns:
            df['post'] = ''
        df['post'] = df['post'].fillna('')
        df = df.dropna(subset=['headline', 'content', 'clickbait_score']).reset_index(drop=True)
    except FileNotFoundError:
        logger.error(f"Test file not found at '{csv_path}'. Please check the path.")
        return
    time_start = time.perf_counter()
    detector = ClickbaitAndSimilarityDetector(headline_model_type=model_type)

    y_true = []
    y_pred_final = []

    # The weights are only used for standard (non-hybrid) model
    HEADLINE_MODEL_WEIGHT = 1
    CONTENT_MODEL_WEIGHT = 9
    TOTAL_WEIGHT = HEADLINE_MODEL_WEIGHT + CONTENT_MODEL_WEIGHT

    print("Processing test set to generate predictions...")
    if detector.headline_content_type == 'hybrid':
        print("Evaluation Mode: Hybrid Model (direct output).")
    else:
        print("Evaluation Mode: Standard Model (weighted average).")

    # Iterate through each row of the test set to generate a final prediction.
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating"):
        headline = row['headline']
        content = row['content']
        post = row['post']
        ground_truth_score = row['clickbait_score']

        # Get scores from both the headline-only and headline-content models.
        _, headline_score = detector.detect_clickbait(headline)
        content_score = detector.get_headline_content_score(headline, content, post=post,
                                                            headline_score=headline_score)

        if pd.isna(headline_score) or pd.isna(content_score):
            logger.warning(f"Skipping row due to NaN score. Headline: '{headline[:30]}...'")
            continue

        # Calculate the final score based on the model type.
        final_score = 0.0
        if detector.headline_content_type == 'hybrid':
            # For the hybrid model, its output is the final score.
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
        verbose=True,
        time_start=time_start
    )
    print("=" * 50)


if __name__ == "__main__":
    # Define example articles for debugging and demonstration.
    debug_articles = [
        {
            "url": "https://www.buzzfeed.com/stephaniemcneal/a-couple-did-a-stunning-photo-shoot-with-their-baby-after-le",
            "post": "This Couple's Photoshoot With Their Baby Is Going Viral For The Most Amazing Reason"
        },
        {
            "url": "https://apnews.com/live/israel-iran-attack",
        }
    ]

    # Process command-line arguments if provided.
    if len(sys.argv) > 1:
        if sys.argv[-1] in ["logistic", "naive_bayes", "random_forest", "svm"]:
            urls_from_args = sys.argv[1:-1]
            model_type_from_args = sys.argv[-1]
        else:
            urls_from_args = sys.argv[1:]
            model_type_from_args = "logistic"

        # Convert the list of URL strings to the expected list of dictionaries.
        articles_from_args = [{'url': url} for url in urls_from_args]
        main(articles_from_args, model_type_from_args)
    else:
        # Run the main analysis function with the debug examples.
        print("\n--- Running Main Analysis on Debug Articles ---")
        # main(debug_articles)

        # Run the evaluation on the Clickbait17 test set.
        print("\n--- Running Evaluation on Clickbait17 Test Set ---")
        test_csv_path = "data/clickbait17/models/default/clickbait17_test_features.csv"
        evaluate_on_test_set(csv_path=test_csv_path)