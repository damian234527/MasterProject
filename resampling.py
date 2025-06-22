# resampling.py
import pandas as pd
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)


def apply_sampling(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    """
    Applies a random sampling strategy to a DataFrame to address class imbalance.

    Args:
        df (pd.DataFrame): The input DataFrame to be resampled.
        strategy (str): The sampling strategy to apply.
                        Can be 'undersample' or 'oversample'.
        seed (int): The random seed for reproducibility.

    Returns:
        pd.DataFrame: A new DataFrame with the sampling strategy applied.
    """
    if not strategy:
        return df

    logger.info(f"Applying '{strategy}' to the training data...")
    logger.info(f"Original training data size: {len(df)}")

    X = df
    y_continuous = df['clickbait_score']

    # --- REVISED FIX ---
    # Convert the float scores into discrete string categories for the sampler.
    # This guarantees that the sampler treats them as class labels (e.g., '0.0', '0.33').
    y_discrete = y_continuous.round(2).astype(str)
    # --- REVISED FIX END ---

    if strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
    elif strategy == 'oversample':
        sampler = RandomOverSampler(random_state=seed)
    else:
        logger.warning(f"Invalid sampling strategy '{strategy}'. Returning original data.")
        return df

    # Use the discrete STRING labels to guide the resampling process.
    X_resampled, _ = sampler.fit_resample(X, y_discrete)

    resampled_df = pd.DataFrame(X_resampled, columns=df.columns)

    # Ensure numeric columns retain their original types after being resampled
    for col in df.columns:
        if df[col].dtype.kind in 'if':  # if integer or float
            resampled_df[col] = pd.to_numeric(resampled_df[col])

    logger.info(f"Training data size after sampling: {len(resampled_df)}")

    return resampled_df