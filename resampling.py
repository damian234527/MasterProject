# resampling.py
import pandas as pd
import logging
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

logger = logging.getLogger(__name__)


def apply_sampling(df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    """
    Applies a random sampling strategy to a DataFrame to address class imbalance
    for a discrete regression target (0, 0.33, 0.67, 1.0).

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
    y_scores = df['clickbait_score']

    # --- REVISED FIX (v2): Using discrete scores directly ---
    # Based on the information that scores are discrete (0, 0.33, 0.67, 1.0),
    # we can treat each score as a distinct class for resampling by converting to a string.
    # This is simpler and more accurate than binning for this specific dataset.
    y_labels = y_scores.astype(str)

    logger.info(f"Class distribution for sampling:\n{y_labels.value_counts().to_string()}")

    if strategy == 'undersample':
        sampler = RandomUnderSampler(random_state=seed)
    elif strategy == 'oversample':
        sampler = RandomOverSampler(random_state=seed)
    else:
        logger.warning(f"Invalid sampling strategy '{strategy}'. Returning original data.")
        return df

    # Use the string labels ('0.0', '0.33', etc.) to guide the resampling process.
    X_resampled, _ = sampler.fit_resample(X, y_labels)

    # Convert the resampled NumPy array back to a DataFrame.
    resampled_df = pd.DataFrame(X_resampled, columns=df.columns)

    # Ensure numeric columns retain their original dtypes.
    for col in df.columns:
        if df[col].dtype.kind in 'if':  # if integer or float
            resampled_df[col] = pd.to_numeric(resampled_df[col])

    logger.info(f"Training data size after sampling: {len(resampled_df)}")
    logger.info(f"Distribution of scores after resampling:\n{resampled_df['clickbait_score'].value_counts().sort_index().to_string()}")

    return resampled_df