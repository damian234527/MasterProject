import os
import pandas as pd
from datetime import datetime
from headline_classifier import HeadlineClassifier
from headline_content_models import (
    ClickbaitTransformer,
    ClickbaitFeatureEnhancedTransformer
)
from data.clickbait17.clickbait17_prepare import prepare_clickbait17_datasets
from data.clickbait17.clickbait17_utils import get_basic_csv_paths, get_feature_csv_paths
from config import HEADLINE_CONTENT_CONFIG
import logging

logger = logging.getLogger(__name__)

# ================ Headline Classifier Training ================

def train_headline_classifiers(data_path: str = "data/headline_clickbait.csv"):
    logging.info("\n--- Training Headline Classifiers ---")

    model_types = ["logistic", "naive_bayes", "random_forest", "svm"]

    df = pd.read_csv(data_path)

    for model_type in model_types:
        logging.info(f"\nTraining HeadlineClassifier ({model_type})...")
        classifier = HeadlineClassifier(model_path=f"models/headline_{model_type}.joblib", model_type=model_type)
        classifier.train(df)
        classifier.save_model()

# ================ Headline-Content Models Training ================

def train_headline_content_models(tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"], model_type: str = "standard", sampling_strategy: str = None, **kwargs):
    logging.info("\n--- Training Headline-Content Models ---")

    # Ensure datasets are prepared
    prepare_clickbait17_datasets(tokenizer_name=tokenizer_name, dataset_type=model_type)

    if model_type == "standard" or "both":
        train_csv_basic, val_csv_basic = get_basic_csv_paths(tokenizer_name)
        logging.info("\nTraining ClickbaitTransformer (Transformer Model)...")
        transformer = ClickbaitTransformer(
            model_name_or_path=tokenizer_name,
            output_directory=f"models/transformer_{tokenizer_name.replace('/', '_')}",
            **kwargs
        )
        transformer.train(train_csv_basic, sampling_strategy = sampling_strategy)

    elif model_type == "hybrid" or  "both":
        train_csv_features, val_csv_features = get_feature_csv_paths(tokenizer_name)
        logging.info("\nTraining ClickbaitFeatureEnhancedTransformer (Hybrid Model)...")
        hybrid = ClickbaitFeatureEnhancedTransformer(
            model_name=tokenizer_name,
            output_dir=f"models/hybrid_{tokenizer_name.replace('/', '_')}",
            **kwargs
        )
        hybrid.train(train_csv_features, sampling_strategy=sampling_strategy)

# ================ Train All ================

def train_all(tokenizer_name: str = HEADLINE_CONTENT_CONFIG["model_name"]):
    os.makedirs("models", exist_ok=True)
    train_headline_classifiers()
    train_headline_content_models(tokenizer_name=tokenizer_name)

# ================ Main ================

if __name__ == "__main__":
    # train_all()
    # train_headline_classifiers()
    # sampling_strategy="oversample",
    # use_weighted_loss=True
    train_headline_content_models(model_type="hybrid")
