import os
import optuna
import gc
import torch
from sklearn.model_selection import KFold
import pandas as pd
from headline_content_models import ClickbaitTransformer, ClickbaitFeatureEnhancedTransformer
from config import GENERAL_CONFIG, HEADLINE_CONTENT_CONFIG, DATASETS_CONFIG
import json

pruner = optuna.pruners.MedianPruner(n_warmup_steps=1)
seed = GENERAL_CONFIG["seed"]
RESULTS_DIR = "optimization_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def train_and_evaluate(model_class, trial, train_csv, val_csv=None, test_run=False):
    trial_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
    }

    config = HEADLINE_CONTENT_CONFIG.copy()
    config.update(trial_params)
    if test_run:
        config["output_directory"] = "test"

    model_class_ref = {
        "standard": ClickbaitTransformer,
        "hybrid": ClickbaitFeatureEnhancedTransformer
    }.get(model_class)
    if not model_class_ref:
        raise ValueError("Invalid model type")

    kwargs = dict(
        model_name_or_path=config["model_name"],
        length_max=config["length_max"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        fp16=config["fp16"],
        output_directory=config["output_directory"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"],
        test_run=test_run
    )

    if val_csv is None:
        df = pd.read_csv(train_csv).dropna()
        # if test_run:
            # df = df.sample(frac=0.3, random_state=seed)
        kf = KFold(n_splits=5, shuffle=True, random_state=seed)
        losses = []
        for train_index, val_index in kf.split(df):
            train_df = df.iloc[train_index]
            val_df = df.iloc[val_index]
            train_path = "temp_train.csv"
            val_path = "temp_val.csv"
            train_df.to_csv(train_path, index=False)
            val_df.to_csv(val_path, index=False)

            model = model_class_ref(**kwargs)
            model.train(train_path, val_path)
            try:
                metrics, _ = model.test(val_path)
                nmse = metrics.get("NMSE", float("inf"))
            except Exception:
                nmse = float("inf")
            losses.append(nmse)

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        final_nmse = sum(losses) / len(losses)
        final_metrics = {"NMSE": final_nmse}
    else:
        model = model_class_ref(**kwargs)
        model.train(train_csv, val_csv)
        try:
            metrics, _ = model.test(val_csv)
            final_metrics = {k: float(v) for k, v in metrics.items()}
        except Exception:
            final_metrics = {"NMSE": float("inf")}

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "model_class": model_class,
        "trial_number": trial.number,
        "params": trial_params,
        "metrics": {k: float(v) for k, v in final_metrics.items()}
    }

    suffix = "_test" if test_run else ""
    result_file = os.path.join("optimization_results", f"{model_class}_{config["model_name"]}{suffix}.json")
    with open(result_file, "a+") as f:
        json.dump(result, f, indent=2)

    return final_metrics.get("NMSE", float("inf"))


def create_objective_standard(train_csv, validation_csv=None, test=False):
    def objective_standard(trial):
        return train_and_evaluate(
            "standard",
            trial,
            train_csv,
            validation_csv,
            test_run=test
        )
    return objective_standard

def create_objective_hybrid(train_csv, validation_csv=None, test=False):
    def objective_hybrid(trial):
        return train_and_evaluate(
            "hybrid",
            trial,
            train_csv,
            validation_csv,
            test_run=test
        )
    return objective_hybrid

if __name__ == "__main__":
    test = True
    hybrid = False

    path_basic = os.path.join("data", DATASETS_CONFIG["dataset_headline_content_name"], "models", HEADLINE_CONTENT_CONFIG["model_name"])
    filename_train = f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["train_suffix"]}"
    filename_validation = f"{DATASETS_CONFIG["dataset_headline_content_name"]}_{DATASETS_CONFIG["validation_suffix"]}"

    if not hybrid:
        study_standard = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner)
        study_standard.optimize(create_objective_standard(train_csv=os.path.join(path_basic, f"{filename_validation}.csv"), test=test), n_trials=2)
        print("Best standard transformer params:", study_standard.best_trial.params)
    else:
        study_hybrid = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=pruner)
        study_hybrid.optimize(create_objective_hybrid(train_csv=os.path.join(path_basic, f"{filename_validation}_{DATASETS_CONFIG["features_suffix"]}.csv"), test=test), n_trials=2)
        print("Best hybrid transformer params:", study_hybrid.best_trial.params)
