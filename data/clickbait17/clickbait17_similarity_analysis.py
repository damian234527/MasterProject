"""Analyzes and stores statistics for similarity scores on the Clickbait17 dataset.

This script reads a dataset CSV file, computes two types of similarity scores
(TF-IDF Cosine and Transformer Embedding) for each entry, and then calculates
detailed statistical measures for these scores. The statistics, including
histograms for percentile calculations, are computed separately for clickbait
and non-clickbait articles and stored in a JSON file for later use by the
explanation generation module.
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

# Define the canonical names for the similarity metrics to be analyzed.
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
    """Analyzes similarity scores from a CSV and stores statistics in JSON.

    Args:
        dataset_csv_path (str): The path to the dataset CSV file.
        output_path (str): The path where the output JSON statistics file will be saved.
        clickbait_threshold (float, optional): The threshold for classifying
            articles as clickbait. Defaults to 0.5.
        num_bins (int, optional): The number of bins to use for generating
            histograms. Defaults to 100.
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

    # Initialize the similarity calculation methods.
    tfidf_calculator = CosineSimilarityTFIDF()
    transformer_calculator = TransformerEmbeddingSimilarity(model_name=HEADLINE_CONTENT_CONFIG["model_name"])

    # Calculate both types of similarity scores for every row in the dataset.
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

    # Calculate global min/max for scaling visualizations.
    global_ranges = {name: {"min": float(df[col].min()), "max": float(df[col].max())} for name, col in
                     zip(SIMILARITY_NAMES, similarity_cols)}

    # Split the DataFrame into clickbait and non-clickbait groups.
    df_clickbait = df[df["clickbait_score"] > clickbait_threshold].copy()
    df_non_clickbait = df[df["clickbait_score"] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        logger.warning("Dataset must contain both clickbait and non-clickbait examples to generate stats.")
        return

    logger.info(
        f"Analyzing {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles.")

    def get_stats_for_group(dataframe, sim_names_list, sim_cols_list):
        """Helper function to calculate statistics for a given DataFrame group."""
        group_stats = {}
        for i, sim_col in enumerate(sim_cols_list):
            sim_name = sim_names_list[i]
            sim_data = dataframe[sim_col].dropna()

            # Basic descriptive statistics.
            mean = float(sim_data.mean())
            std = float(sim_data.std())
            p25 = float(sim_data.quantile(0.25))
            p75 = float(sim_data.quantile(0.75))

            # Histogram data for non-parametric percentile calculations.
            hist, bin_edges = np.histogram(sim_data, bins=num_bins)
            cumulative_counts = np.cumsum(hist)

            group_stats[sim_name] = {
                "mean": mean, "std": std, "p25": p25, "p75": p75,
                "histogram": {
                    "cumulative_counts": cumulative_counts.tolist(),
                    "bin_edges": bin_edges.tolist()
                }
            }
        return group_stats

    # Compile the final statistics object.
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
    # Define paths for the script.
    dataset_csv_path = "models/default/clickbait17_train_features_original.csv"
    output_json_path = "similarity_statistics.json"
    clickbait_threshold = GENERAL_CONFIG.get("clickbait_threshold", 0.5)

    # Run the analysis.
    analyze_and_store_similarity(dataset_csv_path, output_json_path, clickbait_threshold)