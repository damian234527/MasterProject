"""Custom loss functions and weighting for clickbait model training.

This module provides utilities for handling the regression task of predicting a
continuous clickbait score, particularly when the data distribution is imbalanced.
It includes a function to calculate class weights by binning the continuous
scores and a custom Hugging Face Trainer subclass that applies these weights
during loss computation.
"""

import torch
from torch import nn
from transformers import Trainer
import pandas as pd
from collections import Counter
import logging
import numpy as np
from config import GENERAL_CONFIG

logger = logging.getLogger(__name__)


def calculate_class_weights(train_csv_path: str, threshold: float = GENERAL_CONFIG["clickbait_threshold"], strength: float = 0.5) -> dict:
    """Calculates class weights for binned scores with a tunable strength.

    This function reads a training CSV, bins the continuous 'clickbait_score'
    into two classes (0 and 1) based on a threshold, and computes weights to
    counteract class imbalance. The strength parameter allows for tuning the
    intensity of this re-weighting.

    Args:
        train_csv_path (str): The path to the original training data CSV file.
        threshold (float, optional): The value used to split scores into two
            bins (low and high). Defaults to value from config, typically 0.5.
        strength (float, optional): A factor between 0.0 and 1.0 that controls
            the intensity of the weighting. A value of 0.0 results in no
            weighting (all weights are 1.0), while 1.0 applies the full
            inverse frequency weighting. Defaults to 0.5.

    Returns:
        A dictionary mapping the binned labels {0, 1} to their calculated
        weights, or None if the file is not found or has insufficient diversity.
    """
    logger.info(f"Calculating class weights with threshold {threshold} and strength {strength}...")
    # Ensure the strength parameter is within the valid [0.0, 1.0] range.
    strength = max(0.0, min(1.0, strength))

    try:
        df = pd.read_csv(train_csv_path)
        # Bin the continuous scores into two discrete classes (0 and 1).
        binned_labels = (df['clickbait_score'] >= threshold).astype(int)
        score_counts = Counter(binned_labels)

        # If there's only one class, weights cannot be computed.
        if not score_counts or len(score_counts) < 2:
            logger.warning("Could not calculate weights; not enough class diversity.")
            return None

        total_samples = sum(score_counts.values())
        num_classes = len(score_counts)

        # Calculate raw weights based on inverse frequency.
        raw_weights = {label: total_samples / count for label, count in score_counts.items()}
        # Normalize the raw weights.
        sum_of_weights = sum(raw_weights.values())
        normalized_weights = {label: (w * num_classes) / sum_of_weights for label, w in raw_weights.items()}
        # Interpolate between uniform weights (1.0) and normalized weights using the strength factor.
        final_weights = {label: 1.0 * (1 - strength) + w * strength for label, w in normalized_weights.items()}

        logger.info(f"Calculated final class weights for bins: {final_weights}")
        return final_weights

    except FileNotFoundError:
        logger.error(f"Training CSV not found at {train_csv_path}. Cannot calculate weights.")
        return None


class WeightedLossTrainer(Trainer):
    """A custom Trainer that applies weights to binned regression targets.

    This Trainer subclass overrides the default loss computation to apply custom
    weights to the Mean Squared Error (MSE) loss. It bins the continuous labels
    on-the-fly during training to select the appropriate weight for each sample,
    effectively giving more importance to under-represented score ranges.
    """

    def __init__(self, *args, class_weights=None, weight_threshold: float = 0.5, **kwargs):
        """Initializes the WeightedLossTrainer.

        Args:
            *args: Positional arguments passed to the parent Trainer class.
            class_weights (dict, optional): A dictionary mapping binned labels
                {0, 1} to their weights. Defaults to None.
            weight_threshold (float, optional): The threshold used to bin the
                continuous labels during loss computation. Defaults to 0.5.
            **kwargs: Keyword arguments passed to the parent Trainer class.
        """
        super().__init__(*args, **kwargs)
        self.class_weights_map = class_weights
        self.weight_threshold = weight_threshold
        if class_weights:
            logger.info(f"WeightedLossTrainer initialized with weights and threshold {self.weight_threshold}.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Computes the loss with optional class weighting.

        If class weights are provided, this method calculates a weighted MSE
        loss. Otherwise, it falls back to the default loss computation of the
        parent Trainer class.

        Args:
            model: The model for which to compute the loss.
            inputs (dict): The inputs from the dataloader.
            return_outputs (bool): Whether to return the model's outputs along
                with the loss.

        Returns:
            The computed loss as a float, or a tuple of (loss, outputs) if
            `return_outputs` is True.
        """
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Apply weighted loss only if labels and weights are available.
        if labels is not None and self.class_weights_map is not None:
            # For each continuous label in the batch, determine its bin (0 or 1)
            # and select the corresponding weight.
            weights = torch.tensor(
                [self.class_weights_map[1] if label.item() >= self.weight_threshold else self.class_weights_map[0] for
                 label in labels],
                device=labels.device
            )
            # Calculate MSE loss without reduction to get per-sample losses.
            loss_fct = nn.MSELoss(reduction='none')
            loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))
            # Apply the weights to the per-sample losses and then take the mean.
            weighted_loss = (loss * weights).mean()
            return (weighted_loss, outputs) if return_outputs else weighted_loss

        # Fallback to the standard Hugging Face Trainer loss computation.
        return super().compute_loss(model, inputs, return_outputs)