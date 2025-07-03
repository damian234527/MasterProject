"""Analyzes and stores statistics about the features extracted from the Clickbait17 dataset.

This script reads a feature-augmented CSV file, calculates various statistical
measures (mean, std, p25, p75) and histogram data for each feature, and stores
these statistics in a JSON file. The statistics are computed separately for
clickbait and non-clickbait articles to facilitate comparison and enable
precise, non-parametric percentile calculations.
"""

import pandas as pd
import numpy as np
import json
import os
from config import GENERAL_CONFIG
import logging

logger = logging.getLogger(__name__)

# The order of feature names must match the order in FeatureExtractor
FEATURE_NAMES = [
    "Post Length (Words)", "Post Length (Chars)", "Headline Length (Words)", "Headline Length (Chars)",
    "Content Length (Words)", "Content Length (Chars)", "Post/Content Length Ratio (Words)",
    "Headline/Content Length Ratio (Words)", "Exclamations in Headline", "Questions in Headline",
    "Exclamations in Post", "Questions in Post", "Uppercase Ratio in Post",
    "Stopword Ratio in Post", "Clickbait Word Count", "Sentiment Difference (Post-Content)", "Readability (Flesch)",
    "Pronoun Count in Headline", "Question Word Count in Headline", "Jaccard Similarity (Post-Content)",
    "Named Entity Count in Post", "Named Entity Count in Content"
]


def analyze_and_store_features(feature_csv_path: str, output_path: str, clickbait_threshold: float = 0.5, num_bins: int = 100):
    """Analyzes features from a CSV and stores the statistics in a JSON file.

    Args:
        feature_csv_path: The path to the feature-augmented CSV file.
        output_path: The path to save the output JSON file.
        clickbait_threshold: The threshold for classifying articles as clickbait.
        num_bins: The number of bins to use for histogram generation.
    """
    logger.info(f"Loading feature data from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        logger.error(f"Error: File not found at {feature_csv_path}. Please run the data preparation script first.")
        return

    df = pd.read_csv(feature_csv_path)
    feature_cols = [f"f{i + 1}" for i in range(len(FEATURE_NAMES))]
    if not all(col in df.columns for col in feature_cols):
        logger.error("Error: Feature columns (f1, f2, ...) not found.")
        return

    # Calculate global min/max
    global_ranges = {}
    for i, f_col in enumerate(feature_cols):
        feature_name = FEATURE_NAMES[i]
        global_ranges[feature_name] = {
            "min": float(df[f_col].min()),
            "max": float(df[f_col].max())
        }

    # Split the DataFrame into clickbait and non-clickbait groups
    df_clickbait = df[df["clickbait_score"] > clickbait_threshold].copy()
    df_non_clickbait = df[df["clickbait_score"] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        logger.warning(f"Dataset must contain both clickbait and non-clickbait examples. Cannot generate stats.")
        return

    logger.info(f"Analyzing {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles.")

    def get_stats_for_group(dataframe, feature_names_list, feature_cols_list):
        """Calculates statistics, including histogram data, for a given group."""
        group_stats = {}
        for i, f_col in enumerate(feature_cols_list):
            feature_name = feature_names_list[i]
            feature_data = dataframe[f_col].dropna()

            # Basic stats
            mean = float(feature_data.mean())
            std = float(feature_data.std())
            p25 = float(feature_data.quantile(0.25))
            p75 = float(feature_data.quantile(0.75))

            # Histogram and cumulative distribution data
            hist, bin_edges = np.histogram(feature_data, bins=num_bins)
            cumulative_counts = np.cumsum(hist)

            group_stats[feature_name] = {
                "mean": mean,
                "std": std,
                "p25": p25,
                "p75": p75,
                "histogram": {
                    "cumulative_counts": cumulative_counts.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            }
        return group_stats

    output_stats = {
        "clickbait_profile": get_stats_for_group(df_clickbait, FEATURE_NAMES, feature_cols),
        "non_clickbait_profile": get_stats_for_group(df_non_clickbait, FEATURE_NAMES, feature_cols),
        "global_ranges": global_ranges
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_stats, f, indent=4)

    logger.info(f"Successfully saved feature statistics to: {output_path}")


if __name__ == "__main__":
    import logging_config
    feature_csv_path = "models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features.csv"
    output_json_path = "feature_statistics.json"
    clickbait_threshold = GENERAL_CONFIG["clickbait_threshold"]

    # Create the output directory if it doesn't exist
    #os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    analyze_and_store_features(feature_csv_path, output_json_path, clickbait_threshold)