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

# ================ Headline Classifier Training ================

def train_headline_classifiers(data_path: str = "data/headline_clickbait.csv"):
    print("\n--- Training Headline Classifiers ---")

    model_types = ["logistic", "naive_bayes", "random_forest", "svm"]

    df = pd.read_csv(data_path)

    for model_type in model_types:
        print(f"\nTraining HeadlineClassifier ({model_type})...")
        classifier = HeadlineClassifier(model_path=f"models/headline_{model_type}.joblib", model_type=model_type)
        classifier.train(df)
        classifier.save_model()

# ================ Headline-Content Models Training ================

def train_headline_content_models(tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"]):
    print("\n--- Training Headline-Content Models ---")

    # Ensure datasets are prepared
    prepare_clickbait17_datasets()
    train_csv_basic, val_csv_basic = get_basic_csv_paths(tokenizer_name)
    train_csv_features, val_csv_features = get_feature_csv_paths(tokenizer_name)

    # Transformer Model
    print("\nTraining ClickbaitTransformer (Transformer Model)...")
    transformer = ClickbaitTransformer(
        model_name_or_path=tokenizer_name,
        output_directory=f"models/transformer_{tokenizer_name.replace('/', '_')}"
    )
    transformer.train(train_csv_basic, val_csv_basic)

    # Hybrid Feature-Enhanced Model
    print("\nTraining ClickbaitFeatureEnhancedTransformer (Hybrid Model)...")
    hybrid = ClickbaitFeatureEnhancedTransformer(
        model_name=tokenizer_name,
        output_dir=f"models/hybrid_{tokenizer_name.replace('/', '_')}"
    )
    hybrid.train(train_csv_features, val_csv_features)

# ================ Train All ================

def train_all(tokenizer_name: str = "bert-base-uncased"):
    os.makedirs("models", exist_ok=True)
    train_headline_classifiers()
    train_headline_content_models(tokenizer_name)

# ================ Main ================

if __name__ == "__main__":
    #train_all()
    train_headline_content_models()
