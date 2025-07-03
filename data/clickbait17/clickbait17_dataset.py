"""PyTorch Dataset classes for the Clickbait17 dataset.

This module provides two Dataset classes:
- 'Clickbait17Dataset': A basic dataset that tokenizes the text fields
  ('post', 'headline', 'content') for use in a standard transformer model.
- 'Clickbait17FeatureAugmentedDataset': An advanced dataset that extracts 23
  linguistic and statistical features from the text, in addition to tokenizing
  it. This is designed for use in a hybrid model that combines a transformer
  with a feature-based classifier.
"""
import pandas as pd
import torch
from headline_content_feature_extractor import FeatureExtractor
from data.clickbait17.clickbait17_utils import combined_headline
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

def build_clickbait_regex(path):
    """Builds a compiled regex from a file of clickbait phrases.

    Args:
        path: The path to the text file containing clickbait phrases.

    Returns:
        A compiled regular expression pattern.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path, encoding="utf-8") as f:
        terms = [re.escape(line.strip()) for line in f if line.strip()]
    pattern = r"\b(?:%s)\b" % "|".join(terms)
    return re.compile(pattern, re.IGNORECASE)


clickbait_phrases_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clickbait_phrases.txt")
clickbait_regex = build_clickbait_regex(clickbait_phrases_path)


class ClickbaitDataset(Dataset):
    """Base class for Clickbait datasets."""

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        self.tokenizer = tokenizer
        self.length_max = length_max
        self.data = dataframe.dropna().reset_index(drop=True)

    def __len__(self):
        return len(self.data)


class Clickbait17Dataset(ClickbaitDataset):
    """A Dataset for the standard transformer"""

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # Retrieve values
        post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

        # Combine post and headline into a single text input
        combined_text = combined_headline(headline=headline, post=post)

        # Tokenize the combined text and content for the model
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
            "label": torch.tensor(label, dtype=torch.float)
        }


class Clickbait17FeatureAugmentedDataset(ClickbaitDataset):
    """A feature-augmented Dataset for hybrid transformer"""
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer,
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"]):
        super().__init__(dataframe, tokenizer, length_max)

        self.feature_extractor = FeatureExtractor()
        self.feature_columns = sorted([col for col in dataframe.columns if col.startswith('f')], key=lambda x: int(x[1:]))
        features_number = len(self.feature_columns)

        # For feature normalisation
        # self.feature_means = torch.zeros(features_number)
        # self.feature_stds = torch.ones(features_number)
        self.feature_median = torch.zeros(features_number)
        self.feature_iqr = torch.ones(features_number)

    def _extract_features(self, post: str, headline: str, content: str, headline_score: float,
                          normalise: bool = False) -> List[float]:
        """Extracts linguistic features by delegating to FeatureExtractor and adds the headline_score.

        Args:
            post: The post text.
            headline: The headline text.
            content: The content text.
            headline_score: The pre-computed headline score.

        Returns:
            A list of 23 feature values.
        """
        features = self.feature_extractor.extract(post, headline, content)

        # headline_score from headline classification
        features.append(headline_score)

        # normalization
        if normalise:
            features = torch.tensor(features, dtype=torch.float)
            # Add a small epsilon to IQR to avoid division by zero for constant features
            iqr_safe = self.feature_iqr.clone()
            iqr_safe[self.feature_iqr < 1e-5] = 1.0
            features_normalised = (features - self.feature_median) / iqr_safe
            return features_normalised.tolist()

        return features

    def __getitem__(self, idx):
        """Retrieves, tokenizes, and extracts features for a single sample.

        Args:
            idx: The index of the sample to retrieve.

        Returns:
            A dictionary containing tokenized inputs, normalised features, and the label.
        """
        row = self.data.iloc[idx]
        post, headline, content, label, headline_score = row["post"], row["headline"], row["content"], row[
            "clickbait_score"], row["headline_score"]

        # Load features
        features = torch.from_numpy(np.asarray(row[self.feature_columns].values, dtype=np.float32))

        # Normalize
        iqr_safe = self.feature_iqr.clone()
        iqr_safe[self.feature_iqr < 1e-5] = 1.0
        features = (features - self.feature_median) / iqr_safe

        combined_text = combined_headline(headline=headline, post=post)

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
            "features": features.clone().detach(),
            "label": torch.tensor(label, dtype=torch.float)
        }

    def save_with_features(self, path: str):
        """Extracts and saves all features to a new CSV file.

        Args:
            path: The path to save the new CSV file.

        Returns:
            A DataFrame with original data plus extracted feature columns.
        """
        from tqdm import tqdm
        records = []

        for i in tqdm(range(len(self.data)), desc="Extracting features"):
            row = self.data.iloc[i]
            post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

            # The dataframe passed to this class must have the headline_score column, trained classifier is needed
            headline_score_val = row["headline_score"]

            # Extract features without normalisation for saving.
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
        """Creates a dataset instance from a feature CSV and its metadata.

        Args:
            csv_path: The path to the feature CSV file.
            tokenizer: The Hugging Face tokenizer to use.
            length_max: The maximum sequence length for tokenization.

        Returns:
            A dataset instance with loaded normalisation statistics.
        """
        df = pd.read_csv(csv_path)
        with open(csv_path.replace(".csv", "_metadata.json")) as metadata_file:
            meta = json.load(metadata_file)
        obj = cls(df, tokenizer, length_max)
        # obj.feature_means = torch.tensor(meta["features_mean"], dtype=torch.float)
        # obj.feature_stds = torch.tensor(meta["features_std"], dtype=torch.float)
        obj.feature_median = torch.tensor(meta["features_median"], dtype=torch.float)
        obj.feature_iqr = torch.tensor(meta["features_iqr"], dtype=torch.float)
        return obj