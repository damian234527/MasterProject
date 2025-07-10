"""Hyperparameter tuning for headline-content models using Optuna.

This script uses the Optuna framework to perform Bayesian optimization on the
hyperparameters of the `ClickbaitTransformer` and
`ClickbaitFeatureEnhancedTransformer` models. It supports k-fold cross-validation
to find the best set of parameters (learning rate, dropout rate, etc.) that
minimize a chosen objective, such as the Normalized Mean Squared Error (NMSE) or
1 minus the PR-AUC score.
"""
import os
import optuna
import gc
import torch
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import random
import numpy as np
import json
from headline_content_models import ClickbaitTransformer, ClickbaitFeatureEnhancedTransformer
from data.clickbait17.clickbait17_prepare import dataset_check
from config import GENERAL_CONFIG, HEADLINE_CONTENT_CONFIG, DATASETS_CONFIG
from typing import Optional
from utils import set_seed
import logging


logger = logging.getLogger(__name__)

# Optuna pruner to stop unpromising trials early.
pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
# Seed for reproducibility.
seed = GENERAL_CONFIG["seed"]
# Directory to save optimization results.
RESULTS_DIR = "optimization_results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def train_and_evaluate(
        model_class: str,
        trial,
        train_csv: str,
        val_csv: str = None,
        test_run: bool = False,
        model_name: str = None,
        tokenizer_name: str = None,
        folds_number: int = 5):
    """The main objective function for an Optuna trial.

    This function is called by Optuna for each trial. It defines the
    hyperparameter search space, instantiates a model, trains it (using
    k-fold cross-validation if no validation set is provided), and evaluates
    its performance, returning a score for Optuna to minimize.

    Args:
        model_class (str): The type of model to train ('standard' or 'hybrid').
        trial (optuna.Trial): The Optuna trial object, used to suggest
            hyperparameters.
        train_csv (str): Path to the training dataset CSV.
        val_csv (str, optional): Path to a dedicated validation dataset CSV. If
            None, k-fold cross-validation is performed on `train_csv`.
            Defaults to None.
        test_run (bool, optional): If True, runs in a test mode with reduced
            data/epochs for quick checks. Defaults to False.
        model_name (str, optional): The name of the transformer model to use.
            Defaults to None.
        tokenizer_name (str, optional): The name of the tokenizer to use.
            Defaults to None.
        folds_number (int, optional): The number of folds for cross-validation.
            Defaults to 5.

    Returns:
        The objective value to be minimized (1 - PR-AUC score).

    Raises:
        ValueError: If an invalid `model_class` is provided.
        FileNotFoundError: If metadata files for the hybrid model are missing.
    """
    # Use a unique seed for each trial to ensure variety in model initialization.
    seed_trial = seed + trial.number
    set_seed(seed_trial)

    # Define the hyperparameter search space for the trial.
    trial_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "weight_decay": trial.suggest_float("weight_decay", 0, 0.3)
    }

    # Update the base configuration with the suggested hyperparameters.
    config = HEADLINE_CONTENT_CONFIG.copy()
    config.update(trial_params)
    if test_run:
        config["output_directory"] = "test"
    if model_name:
        config["model_name"] = model_name
    if tokenizer_name:
        config["tokenizer_name"] = tokenizer_name

    # Select the correct model class based on the input string.
    model_class_ref = {
        "standard": ClickbaitTransformer,
        "hybrid": ClickbaitFeatureEnhancedTransformer
    }.get(model_class)
    if not model_class_ref:
        raise ValueError("Invalid model type")

    # Prepare keyword arguments for model instantiation.
    kwargs = dict(
        model_name_or_path=config["model_name"],
        tokenizer_name=config["tokenizer_name"],
        length_max=config["length_max"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        fp16=config["fp16"],
        output_directory=config["output_directory"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        test_run=test_run
    )

    model_subfolder = f"{model_class}_{config['model_name']}"
    os.makedirs(os.path.join(RESULTS_DIR, model_subfolder), exist_ok=True)

    # If no validation set is provided, perform k-fold cross-validation.
    if val_csv is None:
        df = pd.read_csv(train_csv).dropna()

        # Load metadata for the hybrid model to ensure correct normalization.
        original_metadata = None
        if model_class == "hybrid":
            metadata_path = train_csv.replace(".csv", "_metadata.json")
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(
                    f"Metadata file not found for the main training CSV: {metadata_path}. "
                    "Please ensure the data preparation script has been run."
                )
            with open(metadata_path, 'r') as f:
                original_metadata = json.load(f)

        # Use StratifiedKFold if possible to maintain class distribution.
        bins = pd.qcut(df['clickbait_score'], q=2, labels=False, duplicates='drop')
        if bins.nunique() > 1 and all(bins.value_counts() >= folds_number):
            kf = StratifiedKFold(n_splits=folds_number, shuffle=True, random_state=seed_trial)
            split_generator = kf.split(df, bins)
        else:
            logging.warning("Insufficient samples for stratification, using KFold")
            kf = KFold(n_splits=folds_number, shuffle=True, random_state=seed_trial)
            split_generator = kf.split(df)

        fold_losses: list[float] = []
        fold_metrics: list[dict[str, float]] = []

        # Iterate over each fold.
        for fold, (train_index, val_index) in enumerate(split_generator, start=1):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]

            # Create a temporary directory for fold data if it doesn't exist
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)

            train_path = os.path.join(temp_dir, "train.csv")
            val_path = os.path.join(temp_dir, "validation.csv")
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

            # For hybrid models, write metadata files for the temporary fold CSVs.
            if model_class == "hybrid" and original_metadata:
                feature_cols = [col for col in train_df.columns if col.startswith("f")]
                if not feature_cols:
                    raise ValueError("No feature columns found in the dataframe for hybrid model.")

                # Calculate median and IQR from the current training fold.
                # This prevents data leakage from the validation fold and matches the model's expectation.
                features_median = train_df[feature_cols].median().tolist()
                q1 = train_df[feature_cols].quantile(0.25)
                q3 = train_df[feature_cols].quantile(0.75)
                features_iqr = (q3 - q1).tolist()

                # Include the original Box-Cox lambdas in the fold's metadata.
                fold_metadata = {
                    "features_median": features_median,
                    "features_iqr": features_iqr,
                    "boxcox_lambdas": original_metadata.get("boxcox_lambdas", {})
                }

                train_meta_path = train_path.replace(".csv", "_metadata.json")
                val_meta_path = val_path.replace(".csv", "_metadata.json")

                # Write metadata for both temp train and validation files.
                # The model loader will use the training fold's stats for both, which is correct.

                with open(train_meta_path, 'w') as f:
                    json.dump(fold_metadata, f, indent=4)
                with open(val_meta_path, 'w') as f:
                    json.dump(fold_metadata, f, indent=4)

            # Train and evaluate the model for the current fold.
            model = model_class_ref(**kwargs)
            model.train(train_path, val_path)
            metrics, _ = model.test(val_path)
            nmse = metrics.get("NMSE", float("inf"))

            fold_losses.append(nmse)
            fold_metrics.append({"fold": fold, **{k: float(v) for k, v in metrics.items()}})

            # Clean up memory to prevent CUDA out-of-memory errors.
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate the mean metrics across all folds.
        df_folds = pd.DataFrame(fold_metrics)
        df_mean = df_folds.mean(numeric_only=True).to_dict()
        objective = 1 - df_mean["PR-AUC"]
        metrics_final = df_mean

        # Save the fold-level results to a CSV.
        df_mean_row = {"fold": "mean", **metrics_final}
        df_out = pd.concat([df_folds, pd.DataFrame([df_mean_row])], ignore_index=True)
        folds_file = os.path.join(
            RESULTS_DIR, f"{model_subfolder}_fold_metrics.csv"
        )
        header_needed = not os.path.exists(folds_file)
        df_out.to_csv(folds_file, mode="a", header=header_needed, index=False)
    else:
        # If a validation set is provided, train and evaluate once.
        model = model_class_ref(**kwargs)
        model.train(train_csv, val_csv)
        metrics, _ = model.test(val_csv)
        metrics_final = {k: float(v) for k, v in metrics.items()}
        objective = 1 - metrics_final["PR-AUC"]

        # Clean up memory.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save the results of the current trial.
    suffix = "_test" if test_run else ""
    result_file = os.path.join(RESULTS_DIR, f"{model_subfolder}{suffix}.csv")
    df_result = pd.DataFrame([{
        "trial_number": trial.number,
        **trial_params,
        **metrics_final
    }])
    if not os.path.exists(result_file):
        df_result.to_csv(result_file, index=False)
    else:
        df_result.to_csv(result_file, mode="a", header=False, index=False)

    return objective


def create_objective_standard(train_csv: str, validation_csv: Optional[str] = None, test: bool = False, model_name: str = None, tokenizer_name: str = None):
    """Creates the objective function for the standard transformer model.

    This function acts as a factory, returning a callable that Optuna can use.
    It closes over the necessary arguments like file paths and model names.

    Args:
        train_csv (str): Path to the training dataset CSV.
        validation_csv (Optional[str]): Path to the validation dataset CSV.
        test (bool): Whether to run in test mode.
        model_name (str): The name of the transformer model.
        tokenizer_name (str): The name of the tokenizer.

    Returns:
        A callable objective function for an Optuna trial.
    """
    def objective_standard(trial):
        return train_and_evaluate(
            "standard",
            trial,
            train_csv,
            validation_csv,
            test_run=test,
            model_name=model_name,
            tokenizer_name=tokenizer_name
        )
    return objective_standard


def create_objective_hybrid(train_csv: str, validation_csv: Optional[str] = None, test: bool = False, model_name: str = None, tokenizer_name: str = None):
    """Creates the objective function for the hybrid transformer model.

    This function acts as a factory, returning a callable that Optuna can use.
    It closes over the necessary arguments like file paths and model names.

    Args:
        train_csv (str): Path to the training dataset CSV with features.
        validation_csv (Optional[str]): Path to the validation dataset CSV.
        test (bool): Whether to run in test mode.
        model_name (str): The name of the transformer model.
        tokenizer_name (str): The name of the tokenizer.

    Returns:
        A callable objective function for an Optuna trial.
    """
    def objective_hybrid(trial):
        return train_and_evaluate(
            "hybrid",
            trial,
            train_csv,
            validation_csv,
            test_run=test,
            model_name=model_name,
            tokenizer_name=tokenizer_name
        )
    return objective_hybrid


if __name__ == "__main__":
    # Configuration for the optimization run.
    test = False
    hybrid = True

    # Define the model and tokenizer to be used.
    model_name = None
    tokenizer_name = None
    if model_name == "" or model_name is None:
        model_name = HEADLINE_CONTENT_CONFIG["model_name"]
    if tokenizer_name == "" or tokenizer_name is None:
        tokenizer_name = HEADLINE_CONTENT_CONFIG["tokenizer_name"]

    # Number of trials for the optimization.
    trials_standard = 20
    #trials_initial = 1

    # Prepare dataset paths.
    # path_basic = dataset_check(tokenizer_name)
    path_basic = "data/clickbait17/models/default/"
    filename_train = f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['train_suffix']}"
    filename_validation = f"{DATASETS_CONFIG['dataset_headline_content_name']}_{DATASETS_CONFIG['validation_suffix']}"
    filename_used = filename_train

    # Run the optimization study.
    if not hybrid:
        study_standard = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner)
        study_standard.optimize(create_objective_standard(train_csv=os.path.join(path_basic, f"{filename_train}.csv"), validation_csv=os.path.join(path_basic, f"{filename_validation}.csv"), test=test, model_name=model_name, tokenizer_name=tokenizer_name), n_trials=trials_standard)
        logging.info("Best standard transformer params:", study_standard.best_trial.params)
    else:
        study_hybrid = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner)
        study_hybrid.optimize(create_objective_hybrid(train_csv=os.path.join(path_basic, f"{filename_train}_{DATASETS_CONFIG['features_suffix']}.csv"), validation_csv=os.path.join(path_basic, f"{filename_validation}_{DATASETS_CONFIG['features_suffix']}.csv"), test=test, model_name=model_name, tokenizer_name=tokenizer_name), n_trials=trials_standard)
        logging.info("Best hybrid transformer params:", study_hybrid.best_trial.params)