import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List
import nltk
import string
import re
import os
import json
import numpy as np
from pathlib import Path
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

# nltk.download("stopwords")
# nltk.download("vader_lexicon")

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
        post_original, headline_original, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

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
            text=combined_text, # Use the newly combined text here
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
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        super().__init__(dataframe, tokenizer, length_max)
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        features_number = 14
        self.feature_means = torch.zeros(features_number)
        self.feature_stds = torch.ones(features_number)

    def _extract_features(self, post: str, headline: str, content: str, normalise: bool = False) -> List[float]:
        post_words = post.split()
        headline_words = headline.split()
        content_words = content.split()

        post_length_words = len(post_words)
        post_length_chars = len(post)
        headline_length_words = len(headline_words)
        headline_length_chars = len(headline)
        content_length_words = len(content_words)
        content_length_chars = len(content)
        post_to_content_length_ratio = post_length_words / max(content_length_words, 1)
        headline_to_content_length_ratio = headline_length_words / max(content_length_words, 1)
        exclamation_count = post.count("!")
        question_mark_count = post.count("?")
        uppercase_ratio_post = sum(c.isupper() for c in post) / max(len(post), 1)
        stopword_ratio_post = sum(w.lower() in self.stop_words for w in post_words) / max(post_length_words, 1)
        clickbait_word_count = len(clickbait_regex.findall(post.lower())) # TODO add more to regex phrases
        sentiment_diff = abs(
            self.sentiment_analyzer.polarity_scores(post)["compound"] -
            self.sentiment_analyzer.polarity_scores(content)["compound"]
        )

        features = [
            post_length_words,
            post_length_chars,
            headline_length_words,
            headline_length_chars,
            content_length_words,
            content_length_chars,
            post_to_content_length_ratio,
            headline_to_content_length_ratio,
            exclamation_count,
            question_mark_count,
            uppercase_ratio_post,
            stopword_ratio_post,
            clickbait_word_count,
            sentiment_diff]

        if normalise:
            features = torch.tensor(features, dtype=torch.float)
            features_normalised = (features - self.feature_means) / self.feature_stds
            return features_normalised
        return features

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

        features = self._extract_features(post, headline, content, normalise=True)

        encoding = self.tokenizer(
            text=post,
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
        records = []
        for i in range(len(self.data)):
            row = self.data.iloc[i]
            post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]
            features = self._extract_features(post, headline, content, normalise=False)
            records.append({
                "post": post,
                "headline": headline,
                "content": content,
                "clickbait_score": label,
                **{f"f{i+1}": feat for i, feat in enumerate(features)}
            })

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
        obj.feature_stds  = torch.tensor(meta["features_std"],  dtype=torch.float)
        return obj
        # return cls(df, tokenizer, length_max)
