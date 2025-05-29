"""Clickbait evaluation utilities.

Provides separate regression and classification metrics while avoiding
information loss during AUC computation. Designed for batch use; emits no
stdout unless `verbose=True`.
"""

from __future__ import annotations

import os
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)

__all__ = ["evaluate_clickbait_predictions"]


def _binarise(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Convert continuous clickbait scores to hard labels."""
    return (scores >= threshold).astype(int)


def evaluate_clickbait_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    threshold: float = 0.5,
    save_path: str | None = None,
    verbose: bool = True,
) -> Mapping[str, float]:
    """Compute regression and classification metrics for clickbait scoring.

    Parameters
    ----------
    y_true
        Ground-truth clickbait scores in [0, 1].
    y_pred
        Model-predicted scores or probabilities in [0, 1].
    threshold
        Score threshold for deriving hard labels.
    save_path
        CSV path to append metrics (directories auto-created).
    verbose
        Emit metrics via logging when `True`.

    Returns
    -------
    dict
        Metric names mapped to values.
    """
    start = time.perf_counter()

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")

    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    nmse = mse / np.var(y_true) if np.var(y_true) else float("inf")

    # Classification metrics
    y_true_bin = _binarise(y_true, threshold)
    y_pred_bin = _binarise(y_pred, threshold)

    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    roc_auc = roc_auc_score(y_true_bin, y_pred)
    pr_auc = average_precision_score(y_true_bin, y_pred)

    runtime = time.perf_counter() - start

    metrics = {
        "MSE": mse,
        "NMSE": nmse,
        "F1": f1,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc,
        "Runtime, s": runtime,
    }

    if save_path:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        df = pd.DataFrame([metrics])
        header = not os.path.exists(save_path)
        df.to_csv(save_path, mode="a", index=False, header=header)

    if verbose:
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print(confusion_matrix(y_true_bin, y_pred_bin))

    return metrics
