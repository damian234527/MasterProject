import pandas as pd
import torch
from config import HEADLINE_CONTENT_CONFIG
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List
import string
import re
import os
import json
import numpy as np
from pathlib import Path
import textstat
import spacy
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize


# nltk.download("stopwords")
# nltk.download("vader_lexicon")
# nltk.download("averaged_perceptron_tagger")
# nltk.download("averaged_perceptron_tagger_eng")

def build_clickbait_regex(path):
    """Return compiled regex based on external lexicon file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        terms = [re.escape(line.strip()) for line in f if line.strip()]
    pattern = r"\b(?:%s)\b" % "|".join(terms)
    return re.compile(pattern, re.IGNORECASE)


clickbait_phrases_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clickbait_phrases.txt")
clickbait_regex = build_clickbait_regex(clickbait_phrases_path)


class ClickbaitDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        self.tokenizer = tokenizer
        self.length_max = length_max
        self.data = dataframe.dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.data)


class Clickbait17Dataset(ClickbaitDataset):
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Retrieve original values
        post_original, headline_original, content, label = row["post"], row["headline"], row["content"], row[
            "clickbait_score"]

        # Apply the logic for combining post and headline for the current row
        combined_text = ""
        if pd.isna(post_original):
            post_original = ""
        if pd.isna(headline_original):
            headline_original = ""

        if post_original and headline_original:
            combined_text = f"{post_original}: {headline_original}"
        elif post_original:
            combined_text = post_original
        elif headline_original:
            combined_text = headline_original
        # If both are empty, combined_text remains empty string. The tokenizer will handle this.

        encoding = self.tokenizer(
            text=combined_text,  # Use the newly combined text here
            text_pair=content,
            truncation=True,
            padding="max_length",
            max_length=self.length_max,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.float)
        }


class Clickbait17FeatureAugmentedDataset(ClickbaitDataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer,
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"]):
        super().__init__(dataframe, tokenizer, length_max)
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading 'en_core_web_sm' model for spaCy")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        # MODIFIED: The hybrid model now uses 23 features.
        features_number = 23
        self.feature_means = torch.zeros(features_number)
        self.feature_stds = torch.ones(features_number)

    def _extract_features(self, post: str, headline: str, content: str, headline_score: float,
                          normalise: bool = False) -> List[float]:
        post_words = word_tokenize(post.lower()) if post else []
        headline_words = word_tokenize(headline.lower()) if headline else []
        content_words = word_tokenize(content.lower()) if content else []

        post_length_words = len(post_words)
        post_length_chars = len(post)
        headline_length_words = len(headline_words)
        headline_length_chars = len(headline)
        content_length_words = len(content_words)
        content_length_chars = len(content)
        post_to_content_length_ratio = post_length_words / max(content_length_words, 1)
        headline_to_content_length_ratio = headline_length_words / max(content_length_words, 1)
        exclamation_count_headline = headline.count("!")
        question_mark_count_headline = headline.count("?")
        exclamation_count_post = post.count("!")
        question_mark_count_post = post.count("?")
        uppercase_ratio_post = sum(c.isupper() for c in post) / max(len(post), 1)
        stopword_ratio_post = sum(w.lower() in self.stop_words for w in post_words) / max(post_length_words, 1)
        clickbait_word_count = len(clickbait_regex.findall(post.lower()))
        sentiment_diff = abs(
            self.sentiment_analyzer.polarity_scores(post)["compound"] -
            self.sentiment_analyzer.polarity_scores(content)["compound"]
        )

        readability_score = textstat.flesch_reading_ease(content)

        # Syntactic features (Part-of-Speech)
        headline_pos_tags = [tag for _, tag in pos_tag(headline_words)]
        pronoun_count = headline_pos_tags.count("PRP") + headline_pos_tags.count("PRP$")  # I, you, he, she, etc.
        question_word_count = sum(
            1 for word in headline_words if word in {"what", "who", "when", "where", "why", "how"})

        # Jaccard similarity to measures topical similarity between headline and content.
        post_word_set = set(post_words)
        content_word_set = set(content_words)
        jaccard_similarity = len(post_word_set.intersection(content_word_set)) / max(
            len(post_word_set.union(content_word_set)), 1)

        # Named entities (people, places, organizations)
        post_doc = self.nlp(post)
        content_doc = self.nlp(content)
        post_entity_count = len(post_doc.ents)
        content_entity_count = len(content_doc.ents)

        features = [
            post_length_words,
            post_length_chars,
            headline_length_words,
            headline_length_chars,
            content_length_words,
            content_length_chars,
            post_to_content_length_ratio,
            headline_to_content_length_ratio,
            exclamation_count_headline,
            question_mark_count_headline,
            exclamation_count_post,
            question_mark_count_post,
            uppercase_ratio_post,
            stopword_ratio_post,
            clickbait_word_count,
            sentiment_diff,
            readability_score,
            pronoun_count,
            question_word_count,
            jaccard_similarity,
            post_entity_count,
            content_entity_count
        ]

        # MODIFIED: Add the headline score as the 23rd feature.
        features.append(headline_score)

        if normalise:
            features = torch.tensor(features, dtype=torch.float)
            features_normalised = (features - self.feature_means) / self.feature_stds
            return features_normalised
        return features

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # MODIFIED: headline_score is now a required column in the dataframe for this class.
        post, headline, content, label, headline_score = row["post"], row["headline"], row["content"], row[
            "clickbait_score"], row["headline_score"]

        # MODIFIED: Pass the headline_score to the feature extractor
        features = self._extract_features(post, headline, content, headline_score, normalise=True)

        combined_text = ""
        if pd.isna(post):
            post = ""
        if pd.isna(headline):
            headline = ""

        if post and headline:
            combined_text = f"{post}: {headline}"
        elif post:
            combined_text = post
        elif headline:
            combined_text = headline

        encoding = self.tokenizer(
            text=combined_text,
            text_pair=content,
            truncation=True,
            padding="max_length",
            max_length=self.length_max,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "features": torch.tensor(features, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.float)
        }

    def save_with_features(self, path: str):
        """
        Extracts all 23 features for each row in the dataset and saves the
        result to a new CSV file.
        """
        records = []
        # Use tqdm for progress bar
        from tqdm import tqdm

        for i in tqdm(range(len(self.data)), desc="Extracting features"):
            row = self.data.iloc[i]
            post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

            # The dataframe passed to this class MUST have the headline_score column.
            headline_score_val = row["headline_score"]

            # Extract all 23 features, without normalization for saving.
            features = self._extract_features(post, headline, content, headline_score_val, normalise=False)

            # Create a record for the new dataframe
            record_data = {
                "post": post,
                "headline": headline,
                "content": content,
                "clickbait_score": label,
                "headline_score": headline_score_val,  # Keep the raw score as a column
            }
            # Add the extracted features as f1, f2, f3...
            record_data.update({f"f{i + 1}": feat for i, feat in enumerate(features)})
            records.append(record_data)

        df_with_features = pd.DataFrame(records)
        df_with_features.to_csv(path, index=False)
        return df_with_features

    @classmethod
    def from_feature_csv(cls, csv_path: str, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        df = pd.read_csv(csv_path)
        with open(csv_path.replace(".csv", "_metadata.json")) as metadata_file:
            meta = json.load(metadata_file)
        obj = cls(df, tokenizer, length_max)
        obj.feature_means = torch.tensor(meta["features_mean"], dtype=torch.float)
        obj.feature_stds = torch.tensor(meta["features_std"], dtype=torch.float)
        return obj