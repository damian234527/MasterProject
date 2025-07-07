"""Prepares the Clickbait17 dataset for training and evaluation.

The script performs the following steps:
1.  Creates basic CSV files from the raw JSONL data.
2.  Optionally, creates feature-augmented CSV files for hybrid models. This
    involves:
    - Loading a pre-trained headline classifier to generate a 'headline_score'
      for each sample.
    - Extracting a set of 23 linguistic and statistical features from the
      textual data.
    - Saving the data with these new features to separate CSV files.
3.  Saves metadata files alongside the generated CSVs, including the
    tokenizer used and normalization statistics for the features.
"""

import os
import json
from datetime import datetime
import pandas as pd
from transformers import AutoTokenizer
from data.clickbait17.clickbait17_preprocess import dataset17_create_csv
from data.clickbait17.clickbait17_dataset import Clickbait17FeatureAugmentedDataset
from data.clickbait17.clickbait17_utils import get_dataset_folder, get_safe_name
from config import DATASETS_CONFIG, HEADLINE_CONTENT_CONFIG, HEADLINE_CONFIG
from typing import Dict
import logging
from headline_classifier import HeadlineClassifier
from tqdm import tqdm
import numpy as np
from scipy.stats import boxcox
import csv

# Set pandas to use tqdm for progress bars on .apply()
tqdm.pandas()

logger = logging.getLogger(__name__)


def save_metadata(csv_path: str, tokenizer_name: str, extra: Dict = None):
    """Saves metadata to a JSON file alongside a CSV.

    Args:
        csv_path: The path to the CSV file.
        tokenizer_name: The name of the tokenizer used to process the data.
        extra: An optional dictionary of extra metadata to include.
    """
    metadata = {
        "tokenizer_name": tokenizer_name,
        "created": datetime.now().isoformat()
    }
    if extra:
        metadata.update(extra)
    json_path = csv_path.replace(".csv", "_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)


def metadata_matches(csv_path: str, tokenizer_name: str, use_specific_tokenizer: bool = False) -> bool:
    """Checks if a CSV file's metadata matches the given tokenizer.

    Args:
        csv_path: The path to the CSV file.
        tokenizer_name: The tokenizer name to check against.
        use_specific_tokenizer: If False, ignores tokenizer name check.

    Returns:
        True if the metadata file exists and the tokenizer name matches,
        False otherwise.
    """
    json_path = csv_path.replace(".csv", "_metadata.json")
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return False
    if use_specific_tokenizer is False:
        return True
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata.get("tokenizer_name") == tokenizer_name


def prepare_clickbait17_datasets(base_path: str = None, tokenizer_name: str = None, dataset_type: str = "both", use_specific_tokenizer: bool = False):
    """Prepares the Clickbait17 datasets.

    This function orchestrates the creation of basic and feature-augmented
    datasets for the Clickbait17 corpus.

    Args:
        base_path: The path to the directory containing the raw 'train',
            'validation', and 'test' subdirectories.
        tokenizer_name: The name of the Hugging Face tokenizer to use.
        dataset_type: The type of dataset to prepare. Can be 'basic',
            'hybrid', or 'both'. 'basic' creates datasets with only the
            textual content, while 'hybrid' creates feature-augmented
            datasets. 'both' creates both types.
        use_specific_tokenizer: If False, saves to a generic 'default' folder.
    """
    logger.info("\n--- Preparing Clickbait17 datasets ---")

    subsets = {
        "train": DATASETS_CONFIG["train_suffix"],
        "validation": DATASETS_CONFIG["validation_suffix"],
        "test": DATASETS_CONFIG["test_suffix"]
    }

    # Set default paths
    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "raw")

    # Set tokenizer
    if tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]

    # Initialize tokenizer and create the main directory for datasets processed
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset_folder = get_dataset_folder(tokenizer_name, use_specific_tokenizer)
    os.makedirs(dataset_folder, exist_ok=True)

    headline_classifier = None
    if dataset_type == "hybrid" or dataset_type == "both":
        logger.info("Loading headline classifier to generate scores...")
        headline_classifier = HeadlineClassifier(model_type=HEADLINE_CONFIG["model_type"])
        try:
            headline_classifier.load_model()
        except FileNotFoundError:
            logger.error(f"FATAL: Headline classifier model not found at '{headline_classifier.model_path}'.")
            logger.error(
                "Please run headline_classifier.py to train and save the model before preparing the feature-augmented dataset.")
            raise

    # For storing training set statistics
    training_set_stats = {}
    # For lambdas from training set
    boxcox_lambdas = {}

    # Process each subset one after another
    for subset_key, subset_name in subsets.items():
        subset_path = os.path.join(base_path, subset_name)
        df = None  # Reset dataframe

        # Basic CSV creation for standard model
        basic_csv_filename = os.path.join(dataset_folder,
                                          f"{DATASETS_CONFIG['dataset_headline_content_name']}_{subset_name}.csv")
        if not metadata_matches(basic_csv_filename, tokenizer_name, use_specific_tokenizer):
            logger.info(f"Creating basic CSV for {subset_name}...")
            df = dataset17_create_csv(subset_path)
            df.to_csv(basic_csv_filename, index=False, quoting=csv.QUOTE_ALL)
            save_metadata(basic_csv_filename, tokenizer_name)
        else:
            logger.info(f"Basic CSV for {subset_name} already exists. Skipping.")

        # Feature enhanced CSV for hybrid model
        if dataset_type == "hybrid" or dataset_type == "both":
            feature_csv_filename = os.path.join(dataset_folder,
                                                f"{DATASETS_CONFIG['dataset_headline_content_name']}_{subset_name}_{DATASETS_CONFIG['features_suffix']}.csv")

            # Load stats from training when processing validation/test sets
            if subset_key != "train" and not training_set_stats:
                train_meta_path = os.path.join(dataset_folder,
                                               f"{DATASETS_CONFIG['dataset_headline_content_name']}_train_{DATASETS_CONFIG['features_suffix']}_metadata.json")
                with open(train_meta_path, "r") as f:
                    meta = json.load(f)
                    training_set_stats = {
                        "features_median": meta.get("features_median"),
                        "features_iqr": meta.get("features_iqr")
                    }
                    boxcox_lambdas = meta.get("boxcox_lambdas", {})

            # Create the feature augmented file if it's missing or metadata is incorrect
            if not metadata_matches(feature_csv_filename, tokenizer_name, use_specific_tokenizer):
                logger.info(f"Creating feature-augmented CSV for {subset_name}...")
                if df is None:
                    df = pd.read_csv(basic_csv_filename, keep_default_na=False)

                # Generate headline scores using the pre-trained classifier
                logger.info(f"Generating headline scores for {subset_name}...")
                if "headline" in df.columns and not df["headline"].isnull().all():
                    headlines = df["headline"].fillna("").tolist()
                    scores = headline_classifier.predict_proba(headlines)
                    df["headline_score"] = scores
                else:
                    logger.warning("No 'headline' column found. Adding 'headline_score' column with zeros.")
                    df["headline_score"] = 0.0

                # Use the dataset class to handle feature extraction
                dataset = Clickbait17FeatureAugmentedDataset(df, tokenizer)
                df_features = dataset.save_with_features(feature_csv_filename)

                # Save a copy of the features before any transformation
                unscaled_csv_filename = feature_csv_filename.replace(".csv", "_original.csv")
                logger.info(f"Saving unscaled features to '{os.path.basename(unscaled_csv_filename)}'...")
                df_features.to_csv(unscaled_csv_filename, index=False, quoting=csv.QUOTE_ALL)

                feature_columns = [col for col in df_features.columns if col.startswith("f")]

                # For the training set, calculate and store feature means and standard deviations
                if subset_key == "train":
                    logger.info("Applying Box-Cox transformation to eligible training features...")
                    current_lambdas = {}
                    for feature in feature_columns:
                        # Box-Cox requires all data to be positive
                        if (df_features[feature] > 0).all():
                            transformed_feature, lmbda = boxcox(df_features[feature])
                            df_features[feature] = transformed_feature
                            current_lambdas[feature] = lmbda
                            logger.info(f"Transformed '{feature}' with lambda={lmbda:.4f}")
                    boxcox_lambdas = current_lambdas  # Save the learned lambdas
                    training_set_stats["boxcox_lambdas"] = boxcox_lambdas
                else:  # Apply learned transformations to validation/test sets
                    logger.info(f"Applying learned Box-Cox transformations to {subset_name}...")
                    if boxcox_lambdas:
                        for feature, lmbda in boxcox_lambdas.items():
                            if feature in df_features.columns:

                                # Identify rows with non-positive values that cannot be transformed.
                                non_positive_mask = df_features[feature] <= 0
                                num_non_positive = non_positive_mask.sum()

                                # Isolate the positive values that can be transformed
                                positive_mask = ~non_positive_mask

                                # Apply the Box-Cox transformation only to the positive values
                                if positive_mask.any():
                                    df_features.loc[positive_mask, feature] = boxcox(
                                        df_features.loc[positive_mask, feature], lmbda=lmbda)

                                # Assign the non-positive the minimum value from the newly transformed positive data
                                if num_non_positive > 0 and positive_mask.any():
                                    min_transformed_val = df_features.loc[positive_mask, feature].min()
                                    df_features.loc[non_positive_mask, feature] = min_transformed_val

                                logger.info(
                                    f"Transformed '{feature}' using lambda={lmbda:.4f}, safely handling {num_non_positive} non-positive value(s).")
                            else:
                                logger.warning(
                                    f"Feature '{feature}' from Box-Cox lambdas not found in {subset_name} columns.")

                # For the training set, calculate and store feature medians and IQRs
                if subset_key == "train":
                    mean = df_features[feature_columns].mean().tolist()
                    median = df_features[feature_columns].median().tolist()
                    std = df_features[feature_columns].std(ddof=0).tolist()
                    q1 = df_features[feature_columns].quantile(0.25)
                    q3 = df_features[feature_columns].quantile(0.75)
                    iqr = (q3 - q1).tolist()
                    training_set_stats.update({"features_mean": mean, "features_median": median, "features_std": std, "features_iqr": iqr})

                # Save the CSV with transformed features
                df_features.to_csv(feature_csv_filename, index=False, quoting=csv.QUOTE_ALL)
                save_metadata(feature_csv_filename, tokenizer_name, training_set_stats)
            else:
                logger.info(f"Feature-augmented CSV for {subset_name} already exists. Skipping.")

    logger.info(f"\nDataset preparation complete. All files stored in '{dataset_folder}'.")


def dataset_check(tokenizer_name: str) -> str:
    """Checks for and creates datasets if they are missing.

    Args:
        tokenizer_name: The name of the tokenizer to check for.

    Returns:
        The path to the directory containing the datasets.
    """
    dataset_main = os.path.join("data", DATASETS_CONFIG["dataset_headline_content_name"])
    dataset_directory = os.path.join(dataset_main, "models", get_safe_name(tokenizer_name))
    filename_train = f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}.csv"
    csv_path = os.path.join(dataset_directory, filename_train)
    # Commence the full preparation if the main training CSV is not found.
    if not os.path.exists(csv_path):
        prepare_clickbait17_datasets(base_path=os.path.join(dataset_main, "raw"), tokenizer_name=tokenizer_name)
    return dataset_directory
