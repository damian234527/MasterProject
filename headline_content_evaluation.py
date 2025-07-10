"""Clickbait prediction evaluation utilities.

This module provides functions to compute a suite of regression and
classification metrics for evaluating clickbait scoring models. It is designed
to handle continuous score predictions and ground truth values, converting them
to binary labels for classification metrics based on a specified threshold.
"""

from __future__ import annotations

import os
import time
from typing import Mapping, Sequence
from config import GENERAL_CONFIG
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
import logging

logger = logging.getLogger(__name__)

__all__ = ["evaluate_clickbait_predictions"]


def _binarise(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Converts continuous clickbait scores to binary labels.

    Args:
        scores (np.ndarray): An array of continuous scores.
        threshold (float): The cutoff value for classifying as positive (1).

    Returns:
        An array of binary labels (0 or 1).
    """
    return (scores >= threshold).astype(int)


def evaluate_clickbait_predictions(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    threshold: float = GENERAL_CONFIG["clickbait_threshold"],
    save_path: str | None = None,
    verbose: bool = True,
    time_start = time.perf_counter()
) -> Mapping[str, float]:
    """Computes regression and classification metrics for clickbait scores.

    This function takes ground-truth and predicted scores, calculates
    regression metrics (MSE, NMSE) on the raw scores, and then binarises the
    scores to compute classification metrics (F1, Precision, Recall, AUC, etc.).

    Args:
        y_true (Sequence[float]): A sequence of ground-truth clickbait scores,
            typically in the range [0, 1].
        y_pred (Sequence[float]): A sequence of model-predicted scores or
            probabilities, typically in the range [0, 1].
        threshold (float, optional): The score threshold used to convert
            continuous scores into hard labels for classification metrics.
            Defaults to the value in `GENERAL_CONFIG`.
        save_path (str | None, optional): If provided, the computed metrics
            will be appended to a CSV file at this path. Directories are
            created automatically. Defaults to None.
        verbose (bool, optional): If True, the computed metrics and a confusion
            matrix are logged to the console. Defaults to True.

    Returns:
        A dictionary mapping metric names to their computed float values.

    Raises:
        ValueError: If the shapes of `y_true` and `y_pred` do not match.
    """
    start = time_start

    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred shapes differ")

    # Calculate regression metrics on the continuous scores.
    mse = mean_squared_error(y_true, y_pred)
    # Normalized MSE is MSE divided by the variance of the true values.
    nmse = mse / np.var(y_true) if np.var(y_true) else float("inf")

    # Binarise scores to calculate classification metrics.
    y_true_bin = _binarise(y_true, threshold)
    y_pred_bin = _binarise(y_pred, threshold)

    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    precision = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    recall = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    accuracy = accuracy_score(y_true_bin, y_pred_bin)
    # ROC-AUC is calculated on the original probabilities for better resolution.
    roc_auc = roc_auc_score(y_true_bin, y_pred)
    pr_auc = average_precision_score(y_true_bin, y_pred)

    runtime = time.perf_counter() - start

    # Consolidate all metrics into a dictionary.
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

    # Save metrics to a CSV file if a path is specified.
    if save_path:
        dir_ = os.path.dirname(save_path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        df = pd.DataFrame([metrics])
        # Add a header only if the file does not already exist.
        header = not os.path.exists(save_path)
        df.to_csv(save_path, mode="a", index=False, header=header)

    # Log metrics to the console if verbose is enabled.
    if verbose:
        for k, v in metrics.items():
            logging.info(f"{k}: {v:.4f}")
        logging.info(f"Confusion matrix:\n{confusion_matrix(y_true_bin, y_pred_bin)}")

    return metrics