"""PyTorch Dataset classes for the Clickbait17 dataset.

This module provides two Dataset classes:
- `Clickbait17Dataset`: A basic dataset that tokenizes the text fields
  ('post', 'headline', 'content') for use in a standard transformer model.
- `Clickbait17FeatureAugmentedDataset`: An advanced dataset that extracts 23
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
import csv
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
        path (str): The path to the text file containing clickbait phrases,
            with one phrase per line.

    Returns:
        A compiled regular expression pattern that matches any of the phrases
        as whole words, ignoring case.

    Raises:
        FileNotFoundError: If the file at the specified path does not exist.
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
    """A base PyTorch Dataset class for clickbait data.

    This class handles common initialization tasks such as loading a DataFrame,
    setting the tokenizer and max length, and dropping rows with missing
    essential data. It is intended to be subclassed.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        """Initializes the base dataset.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame containing the dataset.
            tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
            length_max (int, optional): The maximum sequence length for
                tokenization. Defaults to 512.
        """
        self.tokenizer = tokenizer
        self.length_max = length_max
        essential_columns = ["headline", "content", "clickbait_score"]
        self.data = dataframe.dropna(subset=essential_columns).reset_index(drop=True)

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)


class Clickbait17Dataset(ClickbaitDataset):
    """A Dataset for the standard transformer model.

    This class prepares data for a standard transformer by tokenizing the
    textual inputs (post, headline, content) and providing the clickbait score
    as a label.
    """

    def __getitem__(self, idx):
        """Retrieves and tokenizes a single sample from the dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing the tokenized inputs ('input_ids',
            'attention_mask') and the corresponding 'label', all as
            torch tensors.
        """
        row = self.data.iloc[idx]
        post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]

        # Combine post and headline into a single text input for the model.
        combined_text = combined_headline(headline=headline, post=post)

        # Tokenize the combined text and the main content as a pair.
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
    """A feature-augmented Dataset for the hybrid transformer model.

    This class extends the base dataset to handle pre-extracted linguistic
    features. It loads feature columns from the DataFrame and normalizes them
    using statistics (median and IQR) that are loaded from metadata. This is
    designed for a hybrid model that uses both text and structured features.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer,
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"]):
        """Initializes the feature-augmented dataset.

        Args:
            dataframe (pd.DataFrame): The DataFrame, which must contain feature
                columns (e.g., 'f1', 'f2', ...).
            tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer instance.
            length_max (int, optional): The maximum sequence length.
        """
        super().__init__(dataframe, tokenizer, length_max)
        self.feature_extractor = FeatureExtractor()
        self.feature_columns = sorted([col for col in dataframe.columns if col.startswith('f')],
                                      key=lambda x: int(x[1:]))
        features_number = len(self.feature_columns)
        # Initialize tensors for feature normalization statistics (Robust Scaler).
        self.feature_median = torch.zeros(features_number)
        self.feature_iqr = torch.ones(features_number)

    def _extract_features(self, post: str, headline: str, content: str, headline_score: float,
                          normalise: bool = False) -> List[float]:
        """Extracts and optionally normalizes features for a single sample.

        This method delegates to the `FeatureExtractor` to get base linguistic
        features, appends the pre-computed `headline_score`, and can apply
        normalization using the dataset's stored statistics.

        Args:
            post (str): The post text.
            headline (str): The headline text.
            content (str): The content text.
            headline_score (float): The pre-computed headline score.
            normalise (bool, optional): If True, applies normalization.
                Defaults to False.

        Returns:
            A list of 23 feature values.
        """
        features = self.feature_extractor.extract(post, headline, content)
        # Append the headline score from the pre-trained classifier.
        features.append(headline_score)

        # Apply Robust Scaler normalization if requested.
        if normalise:
            features = torch.tensor(features, dtype=torch.float)
            # Add a small epsilon to IQR to avoid division by zero.
            iqr_safe = self.feature_iqr.clone()
            iqr_safe[self.feature_iqr < 1e-5] = 1.0
            features_normalised = (features - self.feature_median) / iqr_safe
            return features_normalised.tolist()

        return features

    def __getitem__(self, idx):
        """Retrieves, tokenizes, and processes features for a single sample.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            A dictionary containing tokenized inputs, normalized features,
            and the label.
        """
        row = self.data.iloc[idx]
        post, headline, content, label, headline_score = row["post"], row["headline"], row["content"], row[
            "clickbait_score"], row["headline_score"]

        # Load the pre-calculated features directly from the DataFrame.
        features = torch.from_numpy(np.asarray(row[self.feature_columns].values, dtype=np.float32))

        # Normalize the features using the stored median and IQR.
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
        """Extracts features for all samples and saves them to a new CSV file.

        This is a utility method used during the data preparation phase to create
        the feature-augmented dataset file.

        Args:
            path (str): The path to save the new CSV file.

        Returns:
            A pandas DataFrame containing the original data plus the new
            extracted feature columns (f1, f2, etc.).
        """
        from tqdm import tqdm
        records = []
        for i in tqdm(range(len(self.data)), desc="Extracting features"):
            row = self.data.iloc[i]
            post, headline, content, label = row["post"], row["headline"], row["content"], row["clickbait_score"]
            headline_score_val = row["headline_score"]

            # Extract features without normalization for saving.
            features = self._extract_features(post, headline, content, headline_score_val, normalise=False)

            # Create a dictionary record for the new DataFrame.
            record_data = {
                "post": post, "headline": headline, "content": content,
                "clickbait_score": label, "headline_score": headline_score_val,
            }
            # Add the extracted features as f1, f2, f3...
            record_data.update({f"f{i + 1}": feat for i, feat in enumerate(features)})
            records.append(record_data)

        df_with_features = pd.DataFrame(records)
        df_with_features.to_csv(path, index=False, quoting=csv.QUOTE_ALL)
        return df_with_features

    @classmethod
    def from_feature_csv(cls, csv_path: str, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        """Creates a dataset instance from a feature CSV and its metadata.

        This factory method is the standard way to load a feature-augmented
        dataset for training or evaluation. It ensures that the crucial
        normalization statistics are loaded from the corresponding metadata
        file, which is essential for consistent data processing.

        Args:
            csv_path (str): The path to the feature-augmented CSV file.
            tokenizer (PreTrainedTokenizer): The Hugging Face tokenizer to use.
            length_max (int, optional): The maximum sequence length.

        Returns:
            An instance of `Clickbait17FeatureAugmentedDataset` with the
            normalization statistics correctly loaded.
        """
        df = pd.read_csv(csv_path, keep_default_na=False)
        # Load the metadata JSON file.
        with open(csv_path.replace(".csv", "_metadata.json")) as metadata_file:
            meta = json.load(metadata_file)
        # Create a new dataset instance.
        obj = cls(df, tokenizer, length_max)
        # Load and set the normalization statistics from the metadata.
        obj.feature_median = torch.tensor(meta["features_median"], dtype=torch.float)
        obj.feature_iqr = torch.tensor(meta["features_iqr"], dtype=torch.float)
        return obj