"""Utility functions for handling the Clickbait17 dataset files and fields.

This module provides helper functions for common tasks such as constructing
standardized file paths, creating filesystem-safe names from model identifiers,
and combining text fields from the dataset in a consistent manner.
"""
import os
import pandas as pd
from tqdm import tqdm
from headline_content_feature_extractor import FeatureExtractor
from config import DATASETS_CONFIG
import csv


def get_dataset_folder(tokenizer_name: str, use_specific_tokenizer: bool = False) -> str:
    """Returns the folder path where datasets for a given tokenizer are stored.

    Args:
        tokenizer_name (str): The name of the Hugging Face tokenizer.
        use_specific_tokenizer (bool): If False, returns a generic 'default'
            folder path. If True, returns a path specific to the tokenizer name.

    Returns:
        The path to the dataset folder as a string.
    """
    # Conditionally set the folder name based on the flag.
    if use_specific_tokenizer:
        safe_name = get_safe_name(tokenizer_name)
    else:
        safe_name = "default"
    # Construct the full path to the appropriate 'models' sub-directory.
    return os.path.join(os.path.dirname(__file__), "models", safe_name)


def get_safe_name(tokenizer_name: str) -> str:
    """Creates a filesystem-safe name from a Hugging Face tokenizer name.

    Replaces characters like '/' with '_' to prevent issues with file paths.

    Args:
        tokenizer_name (str): The name of the tokenizer (e.g., 'bert-base-uncased').

    Returns:
        A safe version of the tokenizer name suitable for use in filenames.
    """
    return tokenizer_name.replace("/", "_")


def get_basic_csv_paths(tokenizer_name: str, use_specific_tokenizer: bool = False) -> tuple:
    """Returns standardized train and validation CSV paths for basic datasets.

    Args:
        tokenizer_name (str): The name of the Hugging Face tokenizer.
        use_specific_tokenizer (bool): Passed to `get_dataset_folder`.

    Returns:
        A tuple containing the string paths to the training and validation CSV files.
    """
    folder = get_dataset_folder(tokenizer_name, use_specific_tokenizer)
    train_csv = os.path.join(folder, f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}.csv")
    val_csv = os.path.join(folder,
                           f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['validation_suffix']}.csv")
    return train_csv, val_csv


def get_feature_csv_paths(tokenizer_name: str, use_specific_tokenizer: bool = False) -> tuple:
    """Returns standardized paths for feature-augmented datasets.

    Args:
        tokenizer_name (str): The name of the Hugging Face tokenizer.
        use_specific_tokenizer (bool): Passed to `get_dataset_folder`.

    Returns:
        A tuple containing the string paths to the training and validation
        feature-augmented CSV files.
    """
    folder = get_dataset_folder(tokenizer_name, use_specific_tokenizer)
    train_csv = os.path.join(folder,
                             f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}_{DATASETS_CONFIG['features_suffix']}.csv")
    val_csv = os.path.join(folder,
                           f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['validation_suffix']}_{DATASETS_CONFIG['features_suffix']}.csv")
    return train_csv, val_csv


def combined_headline(headline: str = None, post: str = None) -> str:
    """Combines a headline and a post into a single string.

    Args:
        headline (str, optional): The article headline. Defaults to None.
        post (str, optional): The social media post text. Defaults to None.

    Returns:
        The combined text string.
    """
    # Combine post and headline for a richer input to the models.
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

    return combined_text


def combined_headline_series(df):
    """Combines the 'post' and 'headline' columns of a DataFrame.

    This function applies a consistent logic for merging the two text fields
    into a single new pandas Series.

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
    if "post" not in df.columns or "headline" not in df.columns:
        raise ValueError("DataFrame must contain 'post' and 'headline' columns.")

    post_series = df["post"].fillna("").astype(str)
    headline_series = df["headline"].fillna("").astype(str)

    # Define conditions for combination.
    both_present = (post_series != "") & (headline_series != "")
    only_post = (post_series != "") & (headline_series == "")
    only_headline = (post_series == "") & (headline_series != "")

    # Create the combined strings.
    combined_both = post_series + ": " + headline_series
    combined_post = post_series
    combined_headline = headline_series

    # Apply the combinations based on the conditions.
    new_headline_series = pd.Series("", index=df.index, dtype=str)
    new_headline_series.loc[both_present] = combined_both
    new_headline_series.loc[only_post] = combined_post
    new_headline_series.loc[only_headline] = combined_headline

    return new_headline_series


def create_blank_post_csv(original_csv_path: str = "models/default/clickbait17_test.csv"):
    """Creates a version of a dataset CSV with the 'post' column blanked out.

    This is used for testing model performance in scenarios where no social
    media post is available. For feature-augmented datasets, it also re-calculates
    all features based on the now-empty post text.

    Args:
        original_csv_path (str): The path to the original dataset CSV.
    """
    path, file_extension = os.path.splitext(original_csv_path)
    new_csv_path = f"{path}_no_post{file_extension}"
    df = pd.read_csv(original_csv_path)

    # For basic CSVs, simply clear the 'post' column.
    if "_features" not in original_csv_path:
        df["post"] = ""
        df.to_csv(new_csv_path, index=False, quoting=csv.QUOTE_ALL)
        return df
    # For feature CSVs, re-extract all features with an empty post.
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
            # Update the feature columns with the newly calculated values.
            for i, col_name in enumerate(feature_columns):
                new_row[col_name] = all_features[i]
            updated_rows.append(new_row)
        new_df = pd.DataFrame(updated_rows)
        new_df.to_csv(new_csv_path, index=False, quoting=csv.QUOTE_ALL)


if __name__ == "__main__":
    # Example usage to generate a test set version with no post text.
    create_blank_post_csv("models/default/clickbait17_test.csv")