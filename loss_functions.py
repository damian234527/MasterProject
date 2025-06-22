# loss_functions.py
import torch
from torch import nn
from transformers import Trainer
import pandas as pd
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def calculate_class_weights(train_csv_path: str) -> dict:
    """
    Calculates inverse frequency class weights from the original training data.
    The weights are normalized to prevent exploding gradients.

    Args:
        train_csv_path (str): Path to the original (non-resampled) training CSV.

    Returns:
        dict: A dictionary mapping class labels (scores) to their calculated weights.
    """
    logger.info("Calculating class weights for weighted loss...")

    try:
        original_train_df = pd.read_csv(train_csv_path)
        # Round scores to handle potential float inaccuracies and group them
        score_counts = Counter(round(score, 2) for score in original_train_df['clickbait_score'])

        if not score_counts:
            logger.warning("Could not calculate class weights, training data might be empty.")
            return None

        total_samples = sum(score_counts.values())

        # Calculate inverse frequency weights
        class_weights = {
            score: total_samples / count
            for score, count in score_counts.items()
        }

        # Normalize weights to a [0, 1] range to improve stability
        max_weight = float(max(class_weights.values()))
        class_weights = {score: weight / max_weight for score, weight in class_weights.items()}

        logger.info(f"Calculated class weights: {class_weights}")
        return class_weights

    except FileNotFoundError:
        logger.error(f"Training CSV not found at {train_csv_path}. Cannot calculate weights.")
        return None


class WeightedLossTrainer(Trainer):
    """
    A custom Hugging Face Trainer that applies pre-calculated class weights
    to the regression loss function (MSE).
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights_map = class_weights
            logger.info("WeightedLossTrainer initialized with custom class weights.")
        else:
            self.class_weights_map = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """
        Overrides the default loss computation to apply weights.
        """
        # The base Trainer calls the model's forward pass.
        # We get the outputs (logits) to calculate a custom loss.
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")

        # Proceed with weighted loss only if weights and labels are available
        if labels is not None and self.class_weights_map is not None:
            # Map labels in the current batch to their corresponding weights
            # Rounding the label is important to match the keys in the weight map
            weights = torch.tensor(
                [self.class_weights_map.get(round(label.item(), 2), 1.0) for label in labels],
                device=labels.device
            )

            # Use MSELoss with no reduction to get per-element losses
            loss_fct = nn.MSELoss(reduction='none')
            loss = loss_fct(logits.squeeze(-1), labels.squeeze(-1))

            # Apply weights element-wise and then compute the mean
            weighted_loss = (loss * weights).mean()

            return (weighted_loss, outputs) if return_outputs else weighted_loss

        # If no weights, fall back to the model's own loss calculation or the default behavior
        if outputs.loss is not None:
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        return super().compute_loss(model, inputs, return_outputs)