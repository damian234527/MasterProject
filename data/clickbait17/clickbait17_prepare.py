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
import logging_config
import logging
from headline_classifier import HeadlineClassifier  # MODIFIED: Import classifier
from tqdm import tqdm

# Set pandas to use tqdm for progress bars on .apply()
tqdm.pandas()

logger = logging.getLogger(__name__)


def save_metadata(csv_path: str, tokenizer_name: str, extra: Dict = None):
    metadata = {
        "tokenizer_name": tokenizer_name,
        "created": datetime.now().isoformat()
    }
    if extra:
        metadata.update(extra)
    json_path = csv_path.replace(".csv", "_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=4)


def metadata_matches(csv_path: str, tokenizer_name: str) -> bool:
    json_path = csv_path.replace(".csv", "_metadata.json")
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return False
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata.get("tokenizer_name") == tokenizer_name


def prepare_clickbait17_datasets(base_path: str = None, tokenizer_name: str = None, dataset_type: str = "both"):
    logging.info("\n--- Preparing Clickbait17 datasets ---")

    subsets = {
        "train": DATASETS_CONFIG["train_suffix"],
        "validation": DATASETS_CONFIG["validation_suffix"],
        "test": DATASETS_CONFIG["test_suffix"]
    }

    if base_path is None:
        base_path = os.path.join(os.path.dirname(__file__), "raw")

    if tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    dataset_folder = get_dataset_folder(tokenizer_name)
    os.makedirs(dataset_folder, exist_ok=True)

    # MODIFIED: Load the trained headline classifier once
    headline_classifier = None
    if dataset_type == "hybrid" or dataset_type == "both":
        logging.info("Loading headline classifier to generate scores...")
        headline_classifier = HeadlineClassifier(model_type=HEADLINE_CONFIG["model_type"])
        try:
            headline_classifier.load_model()
        except FileNotFoundError:
            logging.error(f"FATAL: Headline classifier model not found at '{headline_classifier.model_path}'.")
            logging.error(
                "Please run headline_classifier.py to train and save the model before preparing the feature-augmented dataset.")
            raise

    # Store training set statistics to prevent data leakage
    training_set_stats = {}

    # --- Process All Subsets ---
    for subset_key, subset_name in subsets.items():
        subset_path = os.path.join(base_path, subset_name)
        df = None  # Reset dataframe

        # --- Step 1: Basic CSV creation ---
        basic_csv_filename = os.path.join(dataset_folder,
                                          f"{DATASETS_CONFIG['dataset_headline_content_name']}_{subset_name}.csv")
        if not metadata_matches(basic_csv_filename, tokenizer_name):
            logging.info(f"Creating basic CSV for {subset_name}...")
            df = dataset17_create_csv(subset_path)
            df.to_csv(basic_csv_filename, index=False)
            save_metadata(basic_csv_filename, tokenizer_name)
        else:
            logging.info(f"Basic CSV for {subset_name} already exists. Skipping.")

        # --- Step 2: Feature-augmented CSV creation ---
        if dataset_type == "hybrid" or dataset_type == "both":
            feature_csv_filename = os.path.join(dataset_folder,
                                                f"{DATASETS_CONFIG['dataset_headline_content_name']}_{subset_name}_{DATASETS_CONFIG['features_suffix']}.csv")

            # Load stats from training if we are processing validation/test sets
            if subset_key != "train" and not training_set_stats:
                train_meta_path = os.path.join(dataset_folder,
                                               f"{DATASETS_CONFIG['dataset_headline_content_name']}_train_{DATASETS_CONFIG['features_suffix']}_metadata.json")
                with open(train_meta_path, "r") as f:
                    meta = json.load(f)
                    training_set_stats = {"features_mean": meta["features_mean"], "features_std": meta["features_std"]}

            if not metadata_matches(feature_csv_filename, tokenizer_name):
                logging.info(f"Creating feature-augmented CSV for {subset_name}...")
                if df is None:
                    df = pd.read_csv(basic_csv_filename)

                # MODIFICATION: Generate and add headline_score feature
                logging.info(f"Generating headline scores for {subset_name}...")
                if 'headline' in df.columns and not df['headline'].isnull().all():
                    headlines = df['headline'].fillna('').tolist()
                    scores = headline_classifier.predict_proba(headlines)
                    df['headline_score'] = scores
                else:
                    logging.warning("No 'headline' column found. Adding 'headline_score' column with zeros.")
                    df['headline_score'] = 0.0

                # The dataframe `df` now contains the required `headline_score` column.
                # Instantiate the dataset class which now expects this column.
                dataset = Clickbait17FeatureAugmentedDataset(df, tokenizer)

                # Save the dataframe with all 23 features.
                df_features = dataset.save_with_features(feature_csv_filename)

                # If this is the training set, calculate and store normalization stats.
                if subset_key == "train":
                    feature_columns = [col for col in df_features.columns if col.startswith("f")]
                    mean = df_features[feature_columns].mean().tolist()
                    std = df_features[feature_columns].std(ddof=0).tolist()  # Use ddof=0 for population std
                    training_set_stats = {"features_mean": mean, "features_std": std}

                # Save metadata for the new feature-augmented file.
                # Validation/test sets will use stats from the training set.
                save_metadata(feature_csv_filename, tokenizer_name, training_set_stats)
            else:
                logging.info(f"Feature-augmented CSV for {subset_name} already exists. Skipping.")

    logging.info(f"\nDataset preparation complete. All files stored in '{dataset_folder}'.")


def dataset_check(tokenizer_name: str) -> str:
    """Checks and ensures datasets for a given tokenizer exist.
    Returns the path to the directory with datasets.
    """
    dataset_main = os.path.join("data", DATASETS_CONFIG["dataset_headline_content_name"])
    dataset_directory = os.path.join(dataset_main, "models", get_safe_name(tokenizer_name))
    filename_train = f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}.csv"
    csv_path = os.path.join(dataset_directory, filename_train)
    if not os.path.exists(csv_path):
        prepare_clickbait17_datasets(base_path=os.path.join(dataset_main, "raw"), tokenizer_name=tokenizer_name)
    return dataset_directory


if __name__ == "__main__":
    # Example: prepare datasets only for the hybrid model
    prepare_clickbait17_datasets(
        tokenizer_name=HEADLINE_CONTENT_CONFIG["tokenizer_name"],
        dataset_type="hybrid"
    )