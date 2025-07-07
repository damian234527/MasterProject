"""Utility functions for handling the Clickbait17 dataset.

This module provides helper functions for constructing file paths and combining
text fields from the dataset.
"""
import os
import pandas as pd
from tqdm import tqdm
from headline_content_feature_extractor import FeatureExtractor
from config import DATASETS_CONFIG
import csv

def get_dataset_folder(tokenizer_name: str, use_specific_tokenizer: bool = False) -> str: # MODIFIED
    """Returns the folder path where datasets for the given tokenizer are stored.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.
        use_specific_tokenizer: If False, uses a generic 'default' folder. # NEW
                                If True, uses a tokenizer-specific folder. # NEW

    Returns:
        The path to the dataset folder.
    """
    # Conditionally set the folder name
    if use_specific_tokenizer:
        safe_name = get_safe_name(tokenizer_name)
    else:
        safe_name = "default"
    # Construct the path to the 'models' sub-directory for the specific tokenizer (or to default)
    return os.path.join(os.path.dirname(__file__), "models", safe_name)


def get_safe_name(tokenizer_name: str) -> str:
    """Creates a filesystem-safe name from a tokenizer name.

    Args:
        tokenizer_name: The name of the tokenizer.

    Returns:
        A safe version of the tokenizer name.
    """
    return tokenizer_name.replace("/", "_")

def get_basic_csv_paths(tokenizer_name: str, use_specific_tokenizer: bool = False) -> tuple:
    """Returns train and validation CSV paths for basic (no features) datasets.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.
        use_specific_tokenizer: Passed to get_dataset_folder.

    Returns:
        A tuple containing the paths to the training and validation CSV files.
    """
    folder = get_dataset_folder(tokenizer_name, use_specific_tokenizer)
    train_csv = os.path.join(folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["train_suffix"]}.csv")
    val_csv = os.path.join(folder,f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["validation_suffix"]}.csv")
    return train_csv, val_csv


def get_feature_csv_paths(tokenizer_name: str, use_specific_tokenizer: bool = False) -> tuple:
    """Returns train and validation CSV paths for feature-augmented datasets.

    Args:
        tokenizer_name: The name of the Hugging Face tokenizer.
        use_specific_tokenizer: Passed to get_dataset_folder

    Returns:
        A tuple containing the paths to the training and validation feature
        CSV files.
    """
    folder = get_dataset_folder(tokenizer_name, use_specific_tokenizer)
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

def create_blank_post_csv(original_csv_path: str = "models/default/clickbait17_test.csv"):
    path, file_extension = os.path.splitext(original_csv_path)
    new_csv_path = f"{path}_no_post{file_extension}"
    df = pd.read_csv(original_csv_path)
    if "_features" not in original_csv_path:
        df["post"] = ""
        df.to_csv(new_csv_path, index=False, quoting=csv.QUOTE_ALL)
        return df
    else:
        feature_extractor = FeatureExtractor()
        feature_columns = sorted([col for col in df.columns if col.startswith('f')], key=lambda x: int(x[1:]))
        updated_rows = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
            new_row = row.copy()
            post_text = ""
            base_features = feature_extractor.extract(post_text, new_row["headline"], new_row["content"])
            all_features = base_features + [new_row["headline_score"]]
            new_row["post"] = post_text
            for i, col_name in enumerate(feature_columns):
                new_row[col_name] = all_features[i]
            updated_rows.append(new_row)
        new_df = pd.DataFrame(updated_rows)
        new_df.to_csv(new_csv_path, index=False, quoting=csv.QUOTE_ALL)

if __name__ == "__main__":
    create_blank_post_csv("models/default/clickbait17_test.csv")