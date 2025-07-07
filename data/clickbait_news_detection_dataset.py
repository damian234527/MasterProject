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
    """
    Prepares a custom dataset for testing with the clickbait detection model.

    Args:
        input_csv_path: Path to the new dataset"s CSV file (columns: "id", "title", "text", "label").
        output_dir: Directory to save the processed files.
        clickbait17_train_meta_path: Path to the metadata JSON of the original Clickbait17
                                     feature-augmented training set. This is essential for
                                     applying the correct normalization and transformations.
        tokenizer_name: The name of the Hugging Face tokenizer to use.
    """
    logger.info("\n--- Preparing Custom Test Dataset ---")

    # 1. Load and Transform the Custom Dataset
    logger.info(f"Loading and transforming custom dataset from: {input_csv_path}")
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input file not found: {input_csv_path}")

    df = pd.read_csv(input_csv_path)

    # Map columns to the required format
    df_processed = pd.DataFrame()
    df_processed["post"] = ""  # Add empty "post" column as expected by the dataset class
    df_processed["headline"] = df["title"]
    df_processed["content"] = df["text"]
    df_processed["clickbait_score"] = df["label"].apply(lambda x: 1.0 if str(x).lower() == "clickbait" else 0.0)

    # Drop rows with missing essential data
    df_processed.dropna(subset=["headline", "content", "clickbait_score"], inplace=True)
    df_processed = df_processed.reset_index(drop=True)
    df_processed["post"] = df_processed["post"].fillna("").astype(str)
    df_processed["headline"] = df_processed["headline"].fillna("").astype(str)
    df_processed["content"] = df_processed["content"].fillna("").astype(str)
    
    logger.info(f"Transformed dataset contains {len(df_processed)} records.")

    # 2. Setup Tokenizer and Output Directory
    if tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Save Basic CSV (for standard transformer model)
    basic_csv_path = os.path.join(output_dir, "clickbait_news_detection.csv")
    logger.info(f"Saving basic processed CSV to: {basic_csv_path}")
    df_processed.to_csv(basic_csv_path, index=False, quoting=csv.QUOTE_ALL)
    save_metadata(basic_csv_path, tokenizer_name)

    # 4. Create Feature-Augmented CSV (for hybrid model)
    logger.info("--- Starting Feature Augmentation ---")

    # Load pre-trained headline classifier to generate scores
    logger.info("Loading headline classifier...")
    headline_classifier = HeadlineClassifier(model_type=HEADLINE_CONFIG["model_type"])
    try:
        headline_classifier.load_model()
    except FileNotFoundError as e:
        logger.error(f"FATAL: Headline classifier model not found. {e}")
        logger.error("Please ensure the headline classifier is trained and saved.")
        raise

    # Generate headline scores
    logger.info("Generating headline scores for the custom dataset...")
    headlines = df_processed["headline"].fillna("").tolist()
    scores = headline_classifier.predict_proba(headlines)
    df_processed["headline_score"] = scores

    # Use the dataset class to extract features
    dataset = Clickbait17FeatureAugmentedDataset(df_processed, tokenizer)
    feature_csv_path = os.path.join(output_dir, "clickbait_news_detection_features.csv")

    logger.info(f"Extracting 23 linguistic features and saving to: {feature_csv_path}")
    df_features = dataset.save_with_features(feature_csv_path)
    feature_columns = [col for col in df_features.columns if col.startswith("f")]

    # 5. Load Transformation and Normalization Stats from Clickbait17 Training Set
    logger.info(f"Loading normalization stats from: {clickbait17_train_meta_path}")
    if not os.path.exists(clickbait17_train_meta_path):
        raise FileNotFoundError(f"Clickbait17 training metadata not found: {clickbait17_train_meta_path}")

    with open(clickbait17_train_meta_path, "r") as f:
        meta = json.load(f)
        boxcox_lambdas = meta.get("boxcox_lambdas", {})
        # The dataset loader uses median and iqr for robust scaling
        # We must apply the *exact same* scaling to the test data
        dataset.feature_median = torch.tensor(meta.get("features_median"))
        dataset.feature_iqr = torch.tensor(meta.get("features_iqr"))

    # 6. Apply Loaded Transformations to the New Dataset"s Features
    logger.info("Applying learned Box-Cox transformations to new dataset...")
    if boxcox_lambdas:
        for feature, lmbda in boxcox_lambdas.items():
            if feature in df_features.columns:
                # Handle non-positive values that cannot be transformed with Box-Cox
                positive_mask = df_features[feature] > 0
                if positive_mask.any():
                    df_features.loc[positive_mask, feature] = boxcox(
                        df_features.loc[positive_mask, feature], lmbda=lmbda
                    )

    # Save the final, transformed feature set
    logger.info(f"Saving final transformed features to: {feature_csv_path}")
    df_features.to_csv(feature_csv_path, index=False, quoting=csv.QUOTE_ALL)

    # Save metadata for the feature-augmented file
    feature_metadata = {
        "normalization_source": clickbait17_train_meta_path,
        "features_median": dataset.feature_median.tolist(),
        "features_iqr": dataset.feature_iqr.tolist(),
        "boxcox_lambdas_applied": boxcox_lambdas
    }
    save_metadata(feature_csv_path, tokenizer_name, extra=feature_metadata)

    logger.info(f"\nCustom dataset preparation complete. Files are in '{output_dir}'.")
    return output_dir
