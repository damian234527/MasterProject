"""Analyzes and stores statistics about similarity scores from the Clickbait17 dataset.

This script reads a dataset CSV file, computes two types of similarity scores
(TF-IDF Cosine and Transformer Embedding), and then calculates statistical measures
(mean, std, p25, p75) and histogram data for these scores.

The statistics are computed separately for clickbait and non-clickbait articles
to facilitate comparison and are stored in a new JSON file.
"""

import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from config import GENERAL_CONFIG, HEADLINE_CONTENT_CONFIG
from headline_content_similarity import (
    CosineSimilarityTFIDF,
    TransformerEmbeddingSimilarity,
)
import logging
import logging_config

logger = logging.getLogger(__name__)

# Define the names for the similarity metrics we will be analyzing
SIMILARITY_NAMES = [
    "TF-IDF Cosine Similarity",
    "Transformer Embedding Similarity"
]


def analyze_and_store_similarity(
        dataset_csv_path: str,
        output_path: str,
        clickbait_threshold: float = 0.5,
        num_bins: int = 100
):
    """
    Analyzes similarity scores from a CSV and stores the statistics in a JSON file.

    Args:
        dataset_csv_path: Path to the dataset CSV file (e.g., test or train set).
        output_path: Path to save the output JSON file.
        clickbait_threshold: Threshold for classifying articles as clickbait.
        num_bins: The number of bins to use for histogram generation.
    """
    logger.info(f"Loading dataset from: {dataset_csv_path}")
    if not os.path.exists(dataset_csv_path):
        logger.error(f"Error: File not found at {dataset_csv_path}.")
        return

    df = pd.read_csv(dataset_csv_path)
    df = df.dropna(subset=["headline", "content", "clickbait_score", "post"]).copy()

    if df.empty:
        logger.warning("DataFrame is empty after dropping NaNs. Aborting analysis.")
        return

    # Initialize similarity calculators
    tfidf_calculator = CosineSimilarityTFIDF()
    transformer_calculator = TransformerEmbeddingSimilarity(model_name=HEADLINE_CONTENT_CONFIG["model_name"])

    # --- Calculate Similarity Scores ---
    logger.info("Calculating similarity scores for the dataset...")
    tqdm.pandas(desc="Calculating Similarities")

    df["tfidf_similarity"] = df.progress_apply(
        lambda row: tfidf_calculator.compute_score(row["headline"], row["content"], row["post"]),
        axis=1
    )
    df["transformer_similarity"] = df.progress_apply(
        lambda row: transformer_calculator.compute_score(row["headline"], row["content"], row["post"]),
        axis=1
    )
    logger.info("Finished calculating similarity scores.")

    similarity_cols = ["tfidf_similarity", "transformer_similarity"]

    # --- Analyze Scores ---
    global_ranges = {name: {"min": float(df[col].min()), "max": float(df[col].max())} for name, col in zip(SIMILARITY_NAMES, similarity_cols)}

    df_clickbait = df[df["clickbait_score"] > clickbait_threshold].copy()
    df_non_clickbait = df[df["clickbait_score"] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        logger.warning("Dataset must contain both clickbait and non-clickbait examples to generate stats.")
        return

    logger.info(f"Analyzing {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles.")

    def get_stats_for_group(dataframe, sim_names_list, sim_cols_list):
        """Calculates statistics, including histogram data, for a given group."""
        group_stats = {}
        for i, sim_col in enumerate(sim_cols_list):
            sim_name = sim_names_list[i]
            sim_data = dataframe[sim_col].dropna()

            # Basic stats
            mean = float(sim_data.mean())
            std = float(sim_data.std())
            p25 = float(sim_data.quantile(0.25))
            p75 = float(sim_data.quantile(0.75))

            # Histogram and cumulative distribution data
            hist, bin_edges = np.histogram(sim_data, bins=num_bins)
            cumulative_counts = np.cumsum(hist)

            group_stats[sim_name] = {
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
        "clickbait_profile": get_stats_for_group(df_clickbait, SIMILARITY_NAMES, similarity_cols),
        "non_clickbait_profile": get_stats_for_group(df_non_clickbait, SIMILARITY_NAMES, similarity_cols),
        "global_ranges": global_ranges
    }

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output_stats, f, indent=4)

    logger.info(f"Successfully saved similarity statistics to: {output_path}")


if __name__ == "__main__":
    dataset_csv_path = "models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features.csv"
    output_json_path = "similarity_statistics.json"
    clickbait_threshold = GENERAL_CONFIG.get("clickbait_threshold", 0.5)

    # os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    analyze_and_store_similarity(dataset_csv_path, output_json_path, clickbait_threshold)