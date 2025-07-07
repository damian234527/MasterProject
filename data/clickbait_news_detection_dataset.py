"""Prepares a custom dataset to be compatible with the project's models.

This script adapts an external dataset (like the 'Clickbait News Detection'
dataset from another source) for use with the existing model pipeline. Crucially,
it applies the *same* feature transformations (e.g., Box-Cox) and normalization
statistics (e.g., median, IQR) that were learned from the original Clickbait17
training set. This ensures that the custom test data is processed in a way that
is consistent with the model's training, which is essential for obtaining
meaningful evaluation results.
"""
import os
import json
import pandas as pd
import numpy as np
import logging
import csv
from transformers import AutoTokenizer
from scipy.stats import boxcox
import torch
from tqdm import tqdm

from headline_classifier import HeadlineClassifier
from data.clickbait17.clickbait17_dataset import Clickbait17FeatureAugmentedDataset
from data.clickbait17.clickbait17_prepare import save_metadata
import logging_config
from config import HEADLINE_CONTENT_CONFIG, HEADLINE_CONFIG, DATASETS_CONFIG

logger = logging.getLogger(__name__)
tqdm.pandas()


def prepare_cnd_dataset(
        input_csv_path: str,
        output_dir: str,
        clickbait17_train_meta_path: str,
        tokenizer_name: str = None
):
    """Prepares and transforms a custom dataset for testing.

    Args:
        input_csv_path (str): The path to the custom dataset's raw CSV file.
            It must contain "title", "text", and "label" columns.
        output_dir (str): The directory where the processed files will be saved.
        clickbait17_train_meta_path (str): Path to the metadata JSON from the
            *original* Clickbait17 training set. This is required for applying
            the correct data transformations.
        tokenizer_name (str, optional): The name of the Hugging Face tokenizer.
            If None, the default from the config is used.

    Returns:
        The path to the output directory.

    Raises:
        FileNotFoundError: If the input CSV or the Clickbait17 metadata file
            cannot be found.
    """
    logger.info("\n--- Preparing Custom Test Dataset ---")

    # Load the custom dataset and map its columns to the project's format.
    logger.info(f"Loading and transforming custom dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")
    df = pd.read_csv(input_csv_path)

    df_processed = pd.DataFrame()
    df_processed["post"] = ""  # Add an empty 'post' column as expected.
    df_processed["headline"] = df["title"]
    df_processed["content"] = df["text"]
    df_processed["clickbait_score"] = df["label"].apply(lambda x: 1.0 if str(x).lower() == "clickbait" else 0.0)

    # Clean and prepare the DataFrame.
    df_processed.dropna(subset=["headline", "content", "clickbait_score"], inplace=True)
    df_processed = df_processed.reset_index(drop=True)
    df_processed["post"] = df_processed["post"].fillna("").astype(str)
    df_processed["headline"] = df_processed["headline"].fillna("").astype(str)
    df_processed["content"] = df_processed["content"].fillna("").astype(str)
    logger.info(f"Transformed dataset contains {len(df_processed)} records.")

    # Set up the tokenizer and output directory.
    if tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save a basic version of the CSV for the standard transformer model.
    basic_csv_path = os.path.join(output_dir, "clickbait_news_detection.csv")
    logger.info(f"Saving basic processed CSV to: {basic_csv_path}")
    df_processed.to_csv(basic_csv_path, index=False, quoting=csv.QUOTE_ALL)
    save_metadata(basic_csv_path, tokenizer_name)

    # Create the feature-augmented version for the hybrid model.
    logger.info("--- Starting Feature Augmentation ---")
    logger.info("Loading headline classifier...")
    headline_classifier = HeadlineClassifier(model_type=HEADLINE_CONFIG["model_type"])
    try:
        headline_classifier.load_model()
    except FileNotFoundError as e:
        logger.error(f"FATAL: Headline classifier model not found. {e}")
        raise

    # Generate headline scores as a required feature for the hybrid model.
    logger.info("Generating headline scores for the custom dataset...")
    headlines = df_processed["headline"].fillna("").tolist()
    scores = headline_classifier.predict_proba(headlines)
    df_processed["headline_score"] = scores

    # Use the dataset class to extract the other 22 features.
    dataset = Clickbait17FeatureAugmentedDataset(df_processed, tokenizer)
    feature_csv_path = os.path.join(output_dir, "clickbait_news_detection_features.csv")
    logger.info(f"Extracting 23 linguistic features and saving to: {feature_csv_path}")
    df_features = dataset.save_with_features(feature_csv_path)
    feature_columns = [col for col in df_features.columns if col.startswith("f")]

    # Load the crucial transformation and normalization stats from the original set.
    logger.info(f"Loading normalization stats from: {clickbait17_train_meta_path}")
    if not os.path.exists(clickbait17_train_meta_path):
        raise FileNotFoundError(f"Clickbait17 training metadata not found: {clickbait17_train_meta_path}")

    with open(clickbait17_train_meta_path, "r") as f:
        meta = json.load(f)
        boxcox_lambdas = meta.get("boxcox_lambdas", {})
        # Use the median and IQR for robust scaling, as learned from the training set.
        dataset.feature_median = torch.tensor(meta.get("features_median"))
        dataset.feature_iqr = torch.tensor(meta.get("features_iqr"))

    # Apply the loaded transformations to the new dataset's features.
    logger.info("Applying learned Box-Cox transformations to new dataset...")
    if boxcox_lambdas:
        for feature, lmbda in boxcox_lambdas.items():
            if feature in df_features.columns:
                # Box-Cox can only be applied to positive values.
                positive_mask = df_features[feature] > 0
                if positive_mask.any():
                    df_features.loc[positive_mask, feature] = boxcox(
                        df_features.loc[positive_mask, feature], lmbda=lmbda
                    )

    logger.info(f"Saving final transformed features to: {feature_csv_path}")
    df_features.to_csv(feature_csv_path, index=False, quoting=csv.QUOTE_ALL)

    # Save metadata for the new feature-augmented file.
    feature_metadata = {
        "normalization_source": clickbait17_train_meta_path,
        "features_median": dataset.feature_median.tolist(),
        "features_iqr": dataset.feature_iqr.tolist(),
        "boxcox_lambdas_applied": boxcox_lambdas
    }
    save_metadata(feature_csv_path, tokenizer_name, extra=feature_metadata)

    logger.info(f"\nCustom dataset preparation complete. Files are in '{output_dir}'.")
    return output_dir