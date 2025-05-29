import os
import json
from datetime import datetime
from transformers import AutoTokenizer
from data.clickbait17.clickbait17_preprocess import dataset17_create_csv
from data.clickbait17.clickbait17_dataset import Clickbait17FeatureAugmentedDataset
from data.clickbait17.clickbait17_utils import get_dataset_folder
from config import DATASETS_CONFIG, HEADLINE_CONTENT_CONFIG
from typing import Dict

def save_metadata(csv_path: str, tokenizer_name: str, extra: Dict = None):
    metadata = {
        "tokenizer_name": tokenizer_name,
        "created": datetime.now().isoformat()
    }
    if extra:
        metadata.update(extra)
    json_path = csv_path.replace(".csv", "_metadata.json")
    with open(json_path, "w") as f:
        json.dump(metadata, f)


def metadata_matches(csv_path: str, tokenizer_name: str) -> bool:
    json_path = csv_path.replace(".csv", "_metadata.json")
    if not os.path.exists(csv_path) or not os.path.exists(json_path):
        return False
    with open(json_path, "r") as f:
        metadata = json.load(f)
    return metadata.get("tokenizer_name") == tokenizer_name


def prepare_clickbait17_datasets(base_path: str = None, tokenizer_name: str = None):
    print("\n--- Preparing Clickbait17 datasets ---")

    subsets = [DATASETS_CONFIG["train_suffix"], DATASETS_CONFIG["validation_suffix"], DATASETS_CONFIG["test_suffix"]]

    if base_path is None:
        base_path = os.path.dirname(__file__)

    if tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]
        #tokenizer_name = os.getenv("TOKENIZER_NAME", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset_folder = get_dataset_folder(tokenizer_name)
    os.makedirs(dataset_folder, exist_ok=True)

    for subset in subsets:
        subset_path = os.path.join(base_path, "raw", subset)

        # Step 1: Basic CSV
        basic_csv_filename = os.path.join(dataset_folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{subset}.csv")
        if not metadata_matches(basic_csv_filename, tokenizer_name):
            print(f"Creating basic CSV for {subset}...")
            df = dataset17_create_csv(subset_path)
            df.to_csv(basic_csv_filename, index=False)
            save_metadata(basic_csv_filename, tokenizer_name)
        else:
            print(f"Basic CSV for {subset} already exists and matches tokenizer. Skipping.")

        # Step 2: Feature-augmented CSV
        feature_csv_filename = os.path.join(dataset_folder, f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{subset}_{DATASETS_CONFIG["features_suffix"]}.csv")
        if not metadata_matches(feature_csv_filename, tokenizer_name):
            print(f"Creating feature-augmented CSV for {subset}...")
            df = dataset17_create_csv(subset_path)
            dataset = Clickbait17FeatureAugmentedDataset(df, tokenizer)
            df_features = dataset.save_with_features(feature_csv_filename)
            feature_column = [column for column in df_features.columns if column.startswith("f")]
            mean = df_features[feature_column].mean().tolist()
            std = df_features[feature_column].std(ddof=0).tolist()
            save_metadata(feature_csv_filename, tokenizer_name, {"features_mean": mean, "features_std": std})
        else:
            print(f"Feature-augmented CSV for {subset} already exists and matches tokenizer. Skipping.")

    print(f"\nDataset preparation complete. Stored in '{dataset_folder}'.")

def dataset_check(tokenizer_name: str) -> str:
    """Checks and ensures datasets for given tokenizer exists.
    Returns path to the directory with datasets
    """
    dataset_main = os.path.join("data", DATASETS_CONFIG["dataset_headline_content_name"])
    dataset_directory = os.path.join(dataset_main, "models", dataset_check(tokenizer_name))
    filename_train = f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}.csv"
    csv_path = os.path.join(dataset_directory, filename_train)
    if not os.path.exists(csv_path):
        prepare_clickbait17_datasets(base_path=dataset_main, tokenizer_name=tokenizer_name)
    return dataset_directory

if __name__ == "__main__":
    prepare_clickbait17_datasets()