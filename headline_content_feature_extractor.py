"""A feature extractor for linguistic and structural properties of articles.

This module provides the `FeatureExtractor` class, which is designed to compute
a wide range of features from the social media post, headline, and content of
an article. These features are intended for use in machine learning models,
particularly the hybrid clickbait detection model.
"""
import os
import re
from typing import List, Dict, Union

import textstat
import spacy
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


def build_clickbait_regex(path):
    """Compiles a regular expression from a list of clickbait terms in a file.

    Each line in the file is treated as a term. The resulting regex will match
    any of these terms as whole words, ignoring case.

    Args:
        path (str): The path to the lexicon file containing one term per line.

    Returns:
        A compiled regular expression object.

    Raises:
        FileNotFoundError: If the specified path does not exist.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        # Read terms, escape special characters, and filter out empty lines/comments.
        terms = [re.escape(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]
    # Create a regex pattern that matches any of the terms as whole words.
    pattern = r"\b(?:%s)\b" % "|".join(terms)
    return re.compile(pattern, re.IGNORECASE)


class FeatureExtractor:
    """A class to extract features from post, headline, and content.

    This class initializes all necessary NLP tools (e.g., spaCy, NLTK) once
    to provide an efficient `extract` method for computing a fixed set of 22
    linguistic and structural features.

    Attributes:
        stop_words (set): A set of English stopwords from NLTK.
        sentiment_analyzer (SentimentIntensityAnalyzer): An NLTK sentiment
            analyzer.
        nlp (spacy.Language): A spaCy language model for NLP tasks like named
            entity recognition.
        clickbait_regex (re.Pattern): A compiled regex for detecting common
            clickbait phrases.
        feature_names (list[str]): An ordered list of the names of the 22
            features that are extracted.
    """

    def __init__(self):
        """Initializes NLP tools and defines the list of feature names."""
        # Initialize all required NLP libraries and models.
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Download the spaCy model if it's not already installed.
            print("Downloading 'en_core_web_sm' model for spaCy")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # Build the clickbait phrase regex from the lexicon file.
        clickbait_phrases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/clickbait_phrases.txt")
        if not os.path.exists(clickbait_phrases_path):
            clickbait_phrases_path = "clickbait_phrases.txt"
        self.clickbait_regex = build_clickbait_regex(clickbait_phrases_path)

        # Define the canonical list of feature names in the correct order.
        self.feature_names = [
            "Post Length (Words)", "Post Length (Chars)", "Headline Length (Words)", "Headline Length (Chars)",
            "Content Length (Words)", "Content Length (Chars)", "Post/Content Length Ratio (Words)",
            "Headline/Content Length Ratio (Words)", "Exclamations in Headline", "Questions in Headline",
            "Exclamations in Post", "Questions in Post", "Uppercase Ratio in Post",
            "Stopword Ratio in Post", "Clickbait Word Count", "Sentiment Difference (Post-Content)",
            "Readability (Flesch)",
            "Pronoun Count in Headline", "Question Word Count in Headline", "Jaccard Similarity (Post-Content)",
            "Named Entity Count in Post", "Named Entity Count in Content"
        ]

    def extract(self, post: str, headline: str, content: str, as_dict: bool = False) -> Union[
        List[float], Dict[str, float]]:
        """Extracts a comprehensive set of 22 features from the input texts.

        Args:
            post (str): The social media post text.
            headline (str): The article headline.
            content (str): The main article content.
            as_dict (bool, optional): If True, returns features as a dictionary
                mapping feature names to values. If False, returns a list of
                feature values in a fixed order. Defaults to False.

        Returns:
            A list or dictionary of the 22 extracted feature values.
        """
        # Ensure 'post' is a string to prevent errors with NoneType.
        post = post if post is not None else ""

        # Pre-tokenize texts for efficiency.
        post_words = word_tokenize(post.lower()) if post else []
        headline_words = word_tokenize(headline.lower()) if headline else []
        content_words = word_tokenize(content.lower()) if content else []

        # --- Feature Calculations ---
        post_length_words = float(len(post_words))
        post_length_chars = float(len(post))
        headline_length_words = float(len(headline_words))
        headline_length_chars = float(len(headline))
        content_length_words = float(len(content_words))
        content_length_chars = float(len(content))
        post_to_content_length_ratio = post_length_words / max(content_length_words, 1.0)
        headline_to_content_length_ratio = headline_length_words / max(content_length_words, 1.0)
        exclamation_count_headline = float(headline.count("!"))
        question_mark_count_headline = float(headline.count("?"))
        exclamation_count_post = float(post.count("!"))
        question_mark_count_post = float(post.count("?"))
        uppercase_ratio_post = sum(c.isupper() for c in post) / max(len(post), 1.0)
        stopword_ratio_post = sum(w.lower() in self.stop_words for w in post_words) / max(post_length_words, 1.0)
        clickbait_word_count = float(len(self.clickbait_regex.findall(post.lower())))
        sentiment_diff = abs(
            self.sentiment_analyzer.polarity_scores(post)["compound"] -
            self.sentiment_analyzer.polarity_scores(content)["compound"]
        )
        readability_score = textstat.flesch_reading_ease(content)
        headline_pos_tags = [tag for _, tag in pos_tag(headline_words)]
        pronoun_count = float(headline_pos_tags.count("PRP") + headline_pos_tags.count("PRP$"))
        question_word_count = float(sum(
            1 for word in headline_words if word in {"what", "who", "when", "where", "why", "how"}))
        post_word_set = set(post_words)
        content_word_set = set(content_words)
        jaccard_similarity = len(post_word_set.intersection(content_word_set)) / max(
            len(post_word_set.union(content_word_set)), 1.0)
        post_doc = self.nlp(post)
        content_doc = self.nlp(content)
        post_entity_count = float(len(post_doc.ents))
        content_entity_count = float(len(content_doc.ents))

        # Consolidate all features into a list in the correct order.
        features = [
            post_length_words, post_length_chars, headline_length_words, headline_length_chars,
            content_length_words, content_length_chars, post_to_content_length_ratio,
            headline_to_content_length_ratio, exclamation_count_headline, question_mark_count_headline,
            exclamation_count_post, question_mark_count_post, uppercase_ratio_post,
            stopword_ratio_post, clickbait_word_count, sentiment_diff, readability_score,
            pronoun_count, question_word_count, jaccard_similarity, post_entity_count, content_entity_count
        ]

        # Return features as a dictionary or a list based on the 'as_dict' flag.
        if as_dict:
            return dict(zip(self.feature_names, features))

        return features