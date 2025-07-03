# loss_functions.py (Revised for Continuous Scores)
import torch
from torch import nn
from transformers import Trainer
import pandas as pd
from collections import Counter
import logging
import numpy as np

logger = logging.getLogger(__name__)


def calculate_class_weights(train_csv_path: str, threshold: float = 0.5, strength: float = 0.5) -> dict:
    """
    Calculates class weights for binned scores with a tunable strength.

    Args:
        train_csv_path (str): Path to the original training CSV.
        threshold (float): The value to split scores into two bins (0 and 1).
        strength (float): Controls the intensity of weighting (0.0 to 1.0).

    Returns:
        dict: A dictionary mapping bin labels {0, 1} to their weights.
    """
    logger.info(f"Calculating class weights with threshold {threshold} and strength {strength}...")
    strength = max(0.0, min(1.0, strength))

    try:
        df = pd.read_csv(train_csv_path)
        # Bin scores into two classes based on the threshold
        binned_labels = (df['clickbait_score'] >= threshold).astype(int)
        score_counts = Counter(binned_labels)
        if not score_counts or len(score_counts) < 2:
            logger.warning("Could not calculate weights; not enough class diversity.")
            return None

        total_samples = sum(score_counts.values())
        num_classes = len(score_counts)

        raw_weights = {label: total_samples / count for label, count in score_counts.items()}
        sum_of_weights = sum(raw_weights.values())
        normalized_weights = {label: (w * num_classes) / sum_of_weights for label, w in raw_weights.items()}
        final_weights = {label: 1.0 * (1 - strength) + w * strength for label, w in normalized_weights.items()}

        logger.info(f"Calculated final class weights for bins: {final_weights}")
        return final_weights

    except FileNotFoundError:
        logger.error(f"Training CSV not found at {train_csv_path}. Cannot calculate weights.")
        return None


class WeightedLossTrainer(Trainer):
    """
    A custom Trainer that applies weights to binned regression targets.
    """
    def __init__(self, *args, class_weights=None, weight_threshold: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights_map = class_weights
        self.weight_threshold = weight_threshold # Store the threshold for use in loss computation
        if class_weights:
            logger.info(f"WeightedLossTrainer initialized with weights and threshold {self.weight_threshold}.")

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        if labels is not None and self.class_weights_map is not None:
            # Determine the weight for each label by binning it on the fly
            weights = torch.tensor(
                [self.class_weights_map[1] if label.item() >= self.weight_threshold else self.class_weights_map[0] for label in labels],
                device=labels.device
            )
            loss_fct = nn.MSELoss(reduction='none')
            loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))
            weighted_loss = (loss * weights).mean()
            return (weighted_loss, outputs) if return_outputs else weighted_loss

        return super().compute_loss(model, inputs, return_outputs)