import optuna
import gc
import torch
from sklearn.model_selection import KFold
import pandas as pd
from headline_content_models import ClickbaitTransformer, ClickbaitFeatureEnhancedTransformer
from config import GENERAL_CONFIG, HEADLINE_CONTENT_CONFIG

def train_and_evaluate(model_class, trial, train_csv, val_csv=None):
    trial_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
    }

    config = HEADLINE_CONTENT_CONFIG.copy()
    config.update(trial_params)
    config["output_dir"] = f"trial_outputs/{model_class}_{trial.number}"

    kwargs = dict(
        model_name_or_path=config["model_name"],
        length_max=config["length_max"],
        batch_size=config["batch_size"],
        epochs=config["epochs"],
        fp16=config["fp16"],
        output_directory=config["output_dir"],
        learning_rate=config["learning_rate"],
        dropout_rate=config["dropout_rate"]
    )

    if model_class == "standard":
        model_class_ref = ClickbaitTransformer
    elif model_class == "hybrid":
        model_class_ref = ClickbaitFeatureEnhancedTransformer
    else:
        raise ValueError("Invalid model_class")

    if val_csv is None:
        # Use K-fold cross-validation on train_csv
        df = pd.read_csv(train_csv).dropna()
        kf = KFold(n_splits=5, shuffle=True, random_state=GENERAL_CONFIG["seed"])
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

        return sum(losses) / len(losses)

    else:
        model = model_class_ref(**kwargs)
        model.train(train_csv, val_csv)

        try:
            metrics, _ = model.test(val_csv)
            nmse = metrics.get("NMSE", float("inf"))
        except Exception:
            nmse = float("inf")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return nmse

def objective_standard(trial):
    return train_and_evaluate(
        "standard",
        trial,
        "./data/clickbait17/models/bert-base-uncased/clickbait17_train.csv",
        # "./data/clickbait17/models/bert-base-uncased/clickbait17_validation.csv"
    )

def objective_hybrid(trial):
    return train_and_evaluate(
        "hybrid",
        trial,
        "./data/clickbait17/models/bert-base-uncased/clickbait17_train_features.csv",
        # "./data/clickbait17/models/bert-base-uncased/clickbait17_validation_features.csv"
    )

if __name__ == "__main__":
    study_standard = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    study_standard.optimize(objective_standard, n_trials=20)

    # study_hybrid = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner(n_warmup_steps=1))
    # study_hybrid.optimize(objective_hybrid, n_trials=20)

    print("Best standard transformer params:", study_standard.best_trial.params)
    # print("Best hybrid transformer params:", study_hybrid.best_trial.params)
