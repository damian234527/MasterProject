"""Data resampling utilities for handling class imbalance with continuous targets.

This module provides a function to apply over- or under-sampling techniques
to a pandas DataFrame. It is designed for regression tasks where the target
variable is continuous by first binning the target into discrete classes.
"""
import pandas as pd
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)


def apply_sampling(df: pd.DataFrame, strategy: str, seed: int, threshold: float = 0.5) -> pd.DataFrame:
    """Applies a sampling strategy to a DataFrame by binning a continuous target.

    This function takes a DataFrame and a sampling strategy ('undersample' or
    'oversample'). It creates temporary binary labels by splitting the
    'clickbait_score' column based on the provided threshold. It then uses
    these binary labels to perform resampling with an imblearn sampler, returning
    a new DataFrame with a more balanced distribution of scores around the
    threshold, while preserving the original continuous score values.

    Args:
        df (pd.DataFrame): The input DataFrame, which must contain a
            'clickbait_score' column.
        strategy (str): The sampling strategy to apply. Must be either
            'undersample' or 'oversample'.
        seed (int): The random seed for the sampler to ensure reproducibility.
        threshold (float, optional): The value used to split the continuous
            'clickbait_score' into two bins for balancing. Defaults to 0.5.

    Returns:
        A new, resampled pandas DataFrame. If the strategy is invalid or not
        provided, the original DataFrame is returned.
    """
    if not strategy or strategy not in ['undersample', 'oversample']:
        logger.warning(f"Invalid or no sampling strategy specified ('{strategy}'). Returning original data.")
        return df

    logger.info(f"Applying '{strategy}' with threshold {threshold}...")

    # Create temporary, discrete binary labels (0 or 1) based on the threshold.
    # This is used only for the purpose of balancing and does not modify the
    # original 'clickbait_score' column in the returned DataFrame.
    y_labels = (df['clickbait_score'] >= threshold).astype(int)

    logger.info(f"Class distribution before sampling (0: <{threshold}, 1: >={threshold}):\n"
                f"{y_labels.value_counts().sort_index().to_string()}")

    # Select the appropriate sampler from the imblearn library.
    if strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
    else:
        sampler = RandomOverSampler(random_state=seed)

    # Perform resampling. The sampler uses the binary `y_labels` to determine
    # which rows of the original `df` to duplicate or remove.
    resampled_df, _ = sampler.fit_resample(df, y_labels)

    logger.info(f"Data size after sampling: {len(resampled_df)}")
    logger.info(f"Distribution of scores after resampling:\n"
                f"{resampled_df['clickbait_score'].value_counts().sort_index().to_string()}")

    return resampled_df