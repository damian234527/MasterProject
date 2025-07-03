"""Utility functions for handling the Clickbait17 dataset.

This module provides helper functions for constructing file paths and combining
text fields from the dataset.
"""

import os
import pandas as pd
from config import DATASETS_CONFIG

def get_dataset_folder(tokenizer_name: str) -> str:
    """Returns the folder path where datasets for the given tokenizer are stored.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.

    Returns:
        The path to the dataset folder.
    """
    # Create a filesystem-safe version of the tokenizer name
    safe_name = get_safe_name(tokenizer_name)
    # Construct the path to the 'models' sub-directory for the specific tokenizer
    return os.path.join(os.path.dirname(__file__), "models", safe_name)

def get_safe_name(tokenizer_name: str) -> str:
    """Creates a filesystem-safe name from a tokenizer name.

    Args:
        tokenizer_name: The name of the tokenizer.

    Returns:
        A safe version of the tokenizer name.
    """
    return tokenizer_name.replace("/", "_")

def get_basic_csv_paths(tokenizer_name: str) -> tuple:
    """Returns train and validation CSV paths for basic (no features) datasets.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.

    Returns:
        A tuple containing the paths to the training and validation CSV files.
    """
    folder = get_dataset_folder(tokenizer_name)
    train_csv = os.path.join(folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["train_suffix"]}.csv")
    val_csv = os.path.join(folder,f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["validation_suffix"]}.csv")
    return train_csv, val_csv


def get_feature_csv_paths(tokenizer_name: str) -> tuple:
    """Returns train and validation CSV paths for feature-augmented datasets.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.

    Returns:
        A tuple containing the paths to the training and validation feature
        CSV files.
    """
    folder = get_dataset_folder(tokenizer_name)
    train_csv = os.path.join(folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["train_suffix"]}_{DATASETS_CONFIG["features_suffix"]}.csv")
    val_csv = os.path.join(folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["validation_suffix"]}_{DATASETS_CONFIG["features_suffix"]}.csv")
    return train_csv, val_csv

def combined_headline(headline: str = None, post: str = None) -> str:
    # Combine post and headline for tokenization
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
    # If both are empty, combined_text remains empty string

    return combined_text


def combined_headline_series(df):
    """Combines the 'post' and 'headline' columns into a single series.

    The combination logic is as follows:
    - If both 'post' and 'headline' are present: "post: headline"
    - If only 'post' is present: "post"
    - If only 'headline' is present: "headline"
    - If neither is present, the result is an empty string.

    Args:
        df: A pandas DataFrame containing 'post' and 'headline' columns.

    Returns:
        A pandas Series with the combined headlines.
    """
    # Ensure "post" and "headline" columns exist
    if "post" not in df.columns or "headline" not in df.columns:
        raise ValueError("DataFrame must contain 'post' and 'headline' columns.")

    post_series = df["post"].fillna("").astype(str)
    headline_series = df["headline"].fillna("").astype(str)

    # Both post and headline are non-empty
    both_present = (post_series != "") & (headline_series != "")
    combined_both = post_series + ": " + headline_series

    # Only post is non-empty
    only_post = (post_series != "") & (headline_series == "")
    combined_post = post_series

    # Only headline is non-empty
    only_headline = (post_series == "") & (headline_series != "")
    combined_headline = headline_series

    new_headline_series = pd.Series("", index=df.index, dtype=str)

    # Assign new values based on conditions
    new_headline_series.loc[both_present] = combined_both
    new_headline_series.loc[only_post] = combined_post
    new_headline_series.loc[only_headline] = combined_headline

    return new_headline_series