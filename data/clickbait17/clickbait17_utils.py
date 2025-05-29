import os

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


if __name__ == "__main__":
    # Example usage
    tokenizer = "bert-base-uncased"
    print(get_basic_csv_paths(tokenizer))
    print(get_feature_csv_paths(tokenizer))
