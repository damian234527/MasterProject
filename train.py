"""Main script for training all models in the clickbait detection tool.

This script provides functions to train the various models used in the project,
including the scikit-learn based headline classifiers and the deep learning
based headline-content models (both standard and hybrid transformers).
It can be executed to train all models with default configurations.
"""
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


def train_headline_classifiers(data_path: str = "data/headline_clickbait.csv"):
    """Trains and saves multiple scikit-learn headline classifiers.

    This function iterates through a predefined list of model types (e.g.,
    'logistic', 'svm'), trains a `HeadlineClassifier` for each, and saves the
    trained model to disk.

    Args:
        data_path (str, optional): The path to the CSV file containing
            'headline' and 'clickbait' columns. Defaults to
            "data/headline_clickbait.csv".
    """
    logging.info("\n--- Training Headline Classifiers ---")

    # Define the scikit-learn model types to be trained.
    model_types = ["logistic", "naive_bayes", "random_forest", "svm"]

    df = pd.read_csv(data_path)

    # Loop through each model type, train it, and save it.
    for model_type in model_types:
        logging.info(f"\nTraining HeadlineClassifier ({model_type})...")
        classifier = HeadlineClassifier(model_path=f"models/headline_{model_type}.joblib", model_type=model_type)
        classifier.train(df)
        classifier.save_model()


def train_headline_content_models(tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"],
                                  model_type: str = "standard", sampling_strategy: str = None, **kwargs):
    """Trains the headline-content transformer models.

    This function first ensures that the necessary datasets are prepared.
    It can train the standard transformer model, the hybrid (feature-enhanced)
    model, or both, based on the `model_type` parameter.

    Args:
        tokenizer_name (str, optional): The name or path of the Hugging Face
            tokenizer to use. Defaults to the value in `HEADLINE_CONTENT_CONFIG`.
        model_type (str, optional): The type of model to train. Can be
            'standard', 'hybrid', or 'both'. Defaults to "standard".
        sampling_strategy (str, optional): The resampling strategy to use,
            e.g., 'oversample' or 'undersample'. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to the model's
            constructor and train method (e.g., `use_weighted_loss`).
    """
    logging.info("\n--- Training Headline-Content Models ---")

    # Prepare the Clickbait17 dataset files required for training.
    prepare_clickbait17_datasets(tokenizer_name=tokenizer_name, dataset_type=model_type)

    # Train the standard transformer model if requested.
    if model_type == "standard" or model_type == "both":
        train_csv_basic, val_csv_basic, test_csv_basic = get_basic_csv_paths(tokenizer_name)
        logging.info("\nTraining ClickbaitTransformer (Transformer Model)...")
        transformer = ClickbaitTransformer(
            model_name_or_path=tokenizer_name,
            output_directory=f"models/transformer_{tokenizer_name.replace('/', '_')}",
            **kwargs
        )
        transformer.train(train_csv_basic, sampling_strategy=sampling_strategy)

    # Train the feature-enhanced hybrid model if requested.
    if model_type == "hybrid" or model_type == "both":
        train_csv_features, val_csv_features, test_csv_features = get_feature_csv_paths(tokenizer_name)
        logging.info("\nTraining ClickbaitFeatureEnhancedTransformer (Hybrid Model)...")
        hybrid = ClickbaitFeatureEnhancedTransformer(
            model_name=tokenizer_name,
            output_dir=f"models/hybrid_{tokenizer_name.replace('/', '_')}",
            **kwargs
        )
        hybrid.train(train_csv_features, sampling_strategy=sampling_strategy)


def train_all(tokenizer_name: str = HEADLINE_CONTENT_CONFIG["model_name"]):
    """A convenience function to train all models in the project.

    This runs the training for both the simple headline classifiers and the
    more complex headline-content models.

    Args:
        tokenizer_name (str, optional): The tokenizer to use for the
            headline-content models. Defaults to the value in
            `HEADLINE_CONTENT_CONFIG`.
    """
    # Ensure the target directory for saving models exists.
    os.makedirs("models", exist_ok=True)
    train_headline_classifiers()
    train_headline_content_models(tokenizer_name=tokenizer_name)


if __name__ == "__main__":
    # Example of training all models.
    # train_all()

    # Example of training only the headline classifiers.
    # train_headline_classifiers()

    # Example of training headline-content models with oversampling and weighted loss.
    # train_headline_content_models(
    #     sampling_strategy="oversample",
    #     use_weighted_loss=True
    # )

    # Default execution: train both the standard and hybrid transformer models.
    train_headline_content_models(model_type="both")