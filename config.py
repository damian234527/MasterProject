"""Centralizes all configuration parameters for the project.

This file contains dictionaries that define settings for general use, model
hyperparameters, dataset paths, and other constants to ensure consistency and
ease of modification across the entire application.
"""

# General application settings.
GENERAL_CONFIG = {
    "seed": 42,  # Random seed for reproducibility of all experiments.
    "clickbait_threshold": 0.5,  # Score threshold for binary classification.
    "separator": "-" * 40  # A string separator for console output.
}

# Configuration for headline-content deep learning models.
HEADLINE_CONTENT_CONFIG = {
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",  # Default shared transformer model.
    "tokenizer_name": "sentence-transformers/all-MiniLM-L6-v2",  # Default tokenizer.
    "model_type": ["standard", "hybrid"],  # Supported model architectures.
    "model_path_default": ["models/standard/best_model", "models/hybrid/best_model"],  # Default paths to trained models.
    "length_max": 512,  # Maximum sequence length for tokenization.
    "batch_size": 32,  # Batch size for training and evaluation.
    "epochs": 3,  # Number of training epochs.
    "learning_rate": 2e-5,  # Optimizer learning rate.
    "weight_decay": 0.01,  # L2 regularization strength.
    "dropout_rate": 0.3,  # Dropout rate for the hybrid model's classifier head.
    "fp16": True,  # Whether to use 16-bit mixed-precision training.
    "output_directory": "models"  # Base directory for saving model outputs.
}

# Configuration for the simple headline-only classifier.
HEADLINE_CONFIG = {
    "model_type": "naive_bayes",  # Default scikit-learn model type.
    "model_path": "models/headline_models/naive_bayes.joblib",  # Path to the trained model file.
}

# Configuration for dataset names and file suffixes.
DATASETS_CONFIG = {
    "dataset_headline_content_name": "clickbait17",
    "dataset_headline_name": "headlines",
    "dataset_headline2_name": "clickbait_notclickbait_dataset",
    "train_suffix": "train",  # Suffix for training set files.
    "validation_suffix": "validation",  # Suffix for validation set files.
    "test_suffix": "test",  # Suffix for test set files.
    "features_suffix": "features"  # Suffix for feature-augmented files.
}

# A list of pre-trained transformer models from Hugging Face for experimentation.
HEADLINE_CONTENT_MODELS_PRETRAINED = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "bert-base-uncased",
    "google/bert_uncased_L-2_H-128_A-2",
    "roberta-base",
    "microsoft/mdeberta-v3-base",
    "christinacdl/mDeBERTa-Multilingual-Opus-mt-Clickbait-Detection",
    "microsoft/deberta-v3-small",
    "khalidalt/DeBERTa-v3-large-mnli",

    "distilroberta-base",
    "valurank/distilroberta-clickbait",
    "allenai/longformer-base-4096",
    "albert/albert-base-v2",
    "google/electra-base-discriminator",
    "xlnet/xlnet-base-cased",
    "facebook/bart-large-mnli",
    "google-t5/t5-base",
    "cross-encoder/ms-marco-MiniLM-L6-v2"
]

# Configuration for the article scraping process.
ARTICLE_SCRAPING_CONFIG = {
    "content_length_min": 50,  # Minimum number of words for content to be considered valid.
}