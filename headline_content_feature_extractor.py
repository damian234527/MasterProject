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
    """Return compiled regex based on external lexicon file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        terms = [re.escape(line.strip()) for line in f if line.strip() and not line.strip().startswith("#")]
    pattern = r"\b(?:%s)\b" % "|".join(terms)
    return re.compile(pattern, re.IGNORECASE)


class FeatureExtractor:
    """
    A centralized class for extracting features from post, headline, and content.
    Initializes all NLP tools once to be used across the application.
    """

    def __init__(self):
        """Initializes NLP tools and defines feature names."""
        # --- Initialize NLP Tools ---
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model for spaCy")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # --- Build Clickbait Regex ---
        # Assumes clickbait_phrases.txt is in the parent directory of this file's location.
        # Adjust the path if your project structure is different.
        clickbait_phrases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/clickbait_phrases.txt")
        if not os.path.exists(clickbait_phrases_path):
             # Fallback for different structures, like when running from the project root.
             clickbait_phrases_path = "clickbait_phrases.txt"
        self.clickbait_regex = build_clickbait_regex(clickbait_phrases_path)

        # --- Define Feature Names ---
        self.feature_names = [
            "Post Length (Words)", "Post Length (Chars)", "Headline Length (Words)", "Headline Length (Chars)",
            "Content Length (Words)", "Content Length (Chars)", "Post/Content Length Ratio (Words)",
            "Headline/Content Length Ratio (Words)", "Exclamations in Headline", "Questions in Headline",
            "Exclamations in Post", "Questions in Post", "Uppercase Ratio in Post",
            "Stopword Ratio in Post", "Clickbait Word Count", "Sentiment Difference (Post-Content)", "Readability (Flesch)",
            "Pronoun Count in Headline", "Question Word Count in Headline", "Jaccard Similarity (Post-Content)",
            "Named Entity Count in Post", "Named Entity Count in Content"
        ]

    def extract(self, post: str, headline: str, content: str, as_dict: bool = False) -> Union[List[float], Dict[str, float]]:
        """
        Extracts a comprehensive set of 22 features.

        Args:
            post (str): The social media post text.
            headline (str): The article headline.
            content (str): The article content.
            as_dict (bool): If True, returns features as a dictionary with names.
                            If False, returns a list of feature values.

        Returns:
            A list or dictionary of the extracted features.
        """
        post = post if post is not None else ""

        post_words = word_tokenize(post.lower()) if post else []
        headline_words = word_tokenize(headline.lower()) if headline else []
        content_words = word_tokenize(content.lower()) if content else []

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

        features = [
            post_length_words, post_length_chars, headline_length_words, headline_length_chars,
            content_length_words, content_length_chars, post_to_content_length_ratio,
            headline_to_content_length_ratio, exclamation_count_headline, question_mark_count_headline,
            exclamation_count_post, question_mark_count_post, uppercase_ratio_post,
            stopword_ratio_post, clickbait_word_count, sentiment_diff, readability_score,
            pronoun_count, question_word_count, jaccard_similarity, post_entity_count, content_entity_count
        ]

        if as_dict:
            return dict(zip(self.feature_names, features))

        return features