import os
import pandas as pd

def get_dataset_folder(tokenizer_name: str) -> str:
    """Returns the folder path where datasets for the given tokenizer are stored."""
    safe_name = get_safe_name(tokenizer_name)
    return os.path.join(os.path.dirname(__file__), "models", safe_name)

def get_safe_name(tokenizer_name: str) -> str:
    return tokenizer_name.replace('/', '_')

def get_basic_csv_paths(tokenizer_name: str) -> tuple:
    """Returns train and validation CSV paths for basic (no features) datasets."""
    folder = get_dataset_folder(tokenizer_name)
    train_csv = os.path.join(folder, "clickbait17_train.csv")
    val_csv = os.path.join(folder, "clickbait17_validation.csv")
    return train_csv, val_csv


def get_feature_csv_paths(tokenizer_name: str) -> tuple:
    """Returns train and validation CSV paths for feature-augmented datasets."""
    folder = get_dataset_folder(tokenizer_name)
    train_csv = os.path.join(folder, "clickbait17_train_features.csv")
    val_csv = os.path.join(folder, "clickbait17_validation_features.csv")
    return train_csv, val_csv

def create_combined_headline(df):
    # Ensure 'post' and 'headline' columns exist
    if 'post' not in df.columns or 'headline' not in df.columns:
        raise ValueError("DataFrame must contain 'post' and 'headline' columns.")

    post_series = df['post'].fillna("").astype(str)
    headline_series = df['headline'].fillna("").astype(str)

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

if __name__ == "__main__":
    # Example usage
    tokenizer = "bert-base-uncased"
    print(get_basic_csv_paths(tokenizer))
    print(get_feature_csv_paths(tokenizer))
