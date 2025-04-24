import numpy as np
import pandas as pd
import time
from sklearn.metrics import (
    mean_squared_error, f1_score, precision_score,
    recall_score, accuracy_score, roc_auc_score
)

def evaluate_clickbait_predictions(y_true, y_pred, save_path: str = None, verbose: bool = True):
    """
    Evaluates predictions for a clickbait detection task using various metrics.

    Args:
        y_true (list or np.ndarray): Ground truth clickbait scores (float).
        y_pred (list or np.ndarray): Predicted clickbait scores (float).
        save_path (str, optional): If provided, saves metrics as CSV to this path.
        verbose (bool): Whether to print metrics to stdout.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    start_time = time.time()

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_binary = (y_pred >= 0.5).astype(int)

    mse = mean_squared_error(y_true, y_pred)
    nmse = mse / np.var(y_true) if np.var(y_true) > 0 else float("inf")
    f1 = f1_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    accuracy = accuracy_score(y_true, y_pred_binary)

    try:
        roc_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        roc_auc = float("nan")

    runtime = time.time() - start_time

    metrics = {
        "MSE": mse,
        "NMSE": nmse,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "Runtime": runtime
    }

    if verbose:
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    if save_path:
        pd.DataFrame([metrics]).to_csv(save_path, index=False)
        print(f"Metrics saved to {save_path}")

    return metrics
