# clickbait17_feature_analysis.py
"""Analyzes and stores statistics about features from the Clickbait17 dataset.

This script reads a feature-augmented CSV file (produced by the data preparation
scripts), calculates various statistical measures (mean, std, quartiles) and
histogram data for each linguistic feature, and stores these statistics in a
JSON file. The statistics are computed separately for clickbait and non-clickbait
articles, which is crucial for generating comparative explanations in the main
analysis tool.
"""
import pandas as pd
import numpy as np
import json
import os
from config import GENERAL_CONFIG
import logging

logger = logging.getLogger(__name__)

# The order of feature names must exactly match the order in FeatureExtractor.
FEATURE_NAMES = [
    "Post Length (Words)", "Post Length (Chars)", "Headline Length (Words)", "Headline Length (Chars)",
    "Content Length (Words)", "Content Length (Chars)", "Post/Content Length Ratio (Words)",
    "Headline/Content Length Ratio (Words)", "Exclamations in Headline", "Questions in Headline",
    "Exclamations in Post", "Questions in Post", "Uppercase Ratio in Post",
    "Stopword Ratio in Post", "Clickbait Word Count", "Sentiment Difference (Post-Content)", "Readability (Flesch)",
    "Pronoun Count in Headline", "Question Word Count in Headline", "Jaccard Similarity (Post-Content)",
    "Named Entity Count in Post", "Named Entity Count in Content"
]


def analyze_and_store_features(feature_csv_path: str, output_path: str, clickbait_threshold: float = GENERAL_CONFIG["clickbait_threshold"],
                               num_bins: int = 100):
    """Analyzes features from a CSV and stores the statistics in a JSON file.

    Args:
        feature_csv_path (str): The path to the feature-augmented CSV file.
        output_path (str): The path where the output JSON statistics file will be saved.
        clickbait_threshold (float, optional): The threshold for classifying
            articles as clickbait. Defaults to config value that is typically 0.5.
        num_bins (int, optional): The number of bins to use for generating
            histograms. Defaults to 100.
    """
    logger.info(f"Loading feature data from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        logger.error(f"Error: File not found at {feature_csv_path}. Please run data preparation first.")
        return

    df = pd.read_csv(feature_csv_path)
    feature_cols = [f"f{i + 1}" for i in range(len(FEATURE_NAMES))]
    if not all(col in df.columns for col in feature_cols):
        logger.error("Error: Feature columns (f1, f2, ...) not found in the CSV.")
        return

    # Calculate the global min and max for each feature for visualization scaling.
    global_ranges = {}
    for i, f_col in enumerate(feature_cols):
        feature_name = FEATURE_NAMES[i]
        global_ranges[feature_name] = {
            "min": float(df[f_col].min()),
            "max": float(df[f_col].max())
        }

    # Split the DataFrame into clickbait and non-clickbait groups.
    df_clickbait = df[df["clickbait_score"] > clickbait_threshold].copy()
    df_non_clickbait = df[df["clickbait_score"] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        logger.warning("Dataset must contain both clickbait and non-clickbait examples to generate stats.")
        return

    logger.info(f"Analyzing {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles.")

    def get_stats_for_group(dataframe, feature_names_list, feature_cols_list):
        """Calculates statistics, including histogram data, for a given group."""
        group_stats = {}
        for i, f_col in enumerate(feature_cols_list):
            feature_name = feature_names_list[i]
            feature_data = dataframe[f_col].dropna()

            # Basic descriptive statistics.
            mean = float(feature_data.mean())
            std = float(feature_data.std())
            p25 = float(feature_data.quantile(0.25))
            p75 = float(feature_data.quantile(0.75))

            # Histogram and cumulative distribution data for precise percentiles.
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

    # Compile the final statistics object.
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
    # Define paths for the script.
    feature_csv_path = "models/default/clickbait17_train_features_original.csv"
    output_json_path = "feature_statistics.json"
    clickbait_threshold = GENERAL_CONFIG["clickbait_threshold"]

    # Run the analysis.
    analyze_and_store_features(feature_csv_path, output_json_path, clickbait_threshold)