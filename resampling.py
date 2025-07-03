# resampling.py (Revised for Continuous Scores)
import pandas as pd
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)


def apply_sampling(df: pd.DataFrame, strategy: str, seed: int, threshold: float = 0.5) -> pd.DataFrame:
    """
    Applies sampling to a DataFrame with a continuous target by binning it.

    Args:
        df (pd.DataFrame): The input DataFrame.
        strategy (str): 'undersample' or 'oversample'.
        seed (int): The random seed for reproducibility.
        threshold (float): The value to split the continuous score into two bins.

    Returns:
        pd.DataFrame: A resampled DataFrame with original continuous scores.
    """
    if not strategy or strategy not in ['undersample', 'oversample']:
        logger.warning(f"Invalid or no sampling strategy specified ('{strategy}'). Returning original data.")
        return df

    logger.info(f"Applying '{strategy}' with threshold {threshold}...")

    # Create discrete, binary labels (0 or 1) based on the threshold for balancing purposes.
    # This does NOT change the original 'clickbait_score' column.
    y_labels = (df['clickbait_score'] >= threshold).astype(int)

    logger.info(f"Class distribution before sampling (0: <{threshold}, 1: >={threshold}):\n"
                f"{y_labels.value_counts().sort_index().to_string()}")

    if strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
    else:  # strategy == 'oversample'
        sampler = RandomOverSampler(random_state=seed)

    # We resample the entire dataframe (as X) using the binary labels (as y).
    # imblearn cleverly returns the original dataframe rows that have been sampled.
    resampled_df, _ = sampler.fit_resample(df, y_labels)

    logger.info(f"Data size after sampling: {len(resampled_df)}")
    logger.info(f"Distribution of scores after resampling:\n"
                f"{resampled_df['clickbait_score'].value_counts().sort_index().to_string()}")

    return resampled_df