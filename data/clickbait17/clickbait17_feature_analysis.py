import pandas as pd
import json
import argparse
import os
from config import GENERAL_CONFIG

# The order of feature names must match the order in _extract_features
# in Clickbait17FeatureAugmentedDataset
FEATURE_NAMES = [
    "post_length_words", "post_length_chars", "headline_length_words",
    "headline_length_chars", "content_length_words", "content_length_chars",
    "post_to_content_length_ratio", "headline_to_content_length_ratio",
    "exclamation_count_headline", "question_mark_count_headline",
    "exclamation_count_post", "question_mark_count_post", "uppercase_ratio_post",
    "stopword_ratio_post", "clickbait_word_count", "sentiment_diff",
    "readability_score", "pronoun_count", "question_word_count",
    "jaccard_similarity", "post_entity_count", "content_entity_count"
]


def analyze_and_store_features(feature_csv_path: str, output_path: str, clickbait_threshold: float = 0.5):
    """
    Analyzes the features of clickbait and non-clickbait articles in a dataset
    and stores the statistics for both.
    """
    print(f"Loading feature data from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        print(f"Error: File not found at {feature_csv_path}. Please run the main data preparation script first.")
        return

    df = pd.read_csv(feature_csv_path)

    # --- MODIFICATION START ---

    # Define the two groups
    df_clickbait = df[df['clickbait_score'] > clickbait_threshold].copy()
    df_non_clickbait = df[df['clickbait_score'] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        print(
            f"Dataset must contain both clickbait and non-clickbait examples based on threshold {clickbait_threshold}. Cannot generate stats.")
        return

    print(
        f"Found {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles to analyze.")

    # Get the feature columns (f1, f2, ...)
    feature_cols = [f'f{i + 1}' for i in range(len(FEATURE_NAMES))]

    if not all(col in df.columns for col in feature_cols):
        print("Error: Feature columns (f1, f2, ...) not found. Ensure the CSV was created with save_with_features.")
        return

    # Helper function to calculate and structure stats
    def get_stats_for_group(dataframe, feature_names_list, feature_cols_list):
        stats = dataframe[feature_cols_list].agg(['mean', 'median'])
        group_stats = {}
        for i, f_col in enumerate(feature_cols_list):
            feature_name = feature_names_list[i]
            group_stats[feature_name] = {
                'mean': stats.loc['mean', f_col],
                'median': stats.loc['median', f_col]
            }
        return group_stats

    # Calculate stats for both groups
    output_stats = {
        "clickbait": get_stats_for_group(df_clickbait, FEATURE_NAMES, feature_cols),
        "non_clickbait": get_stats_for_group(df_non_clickbait, FEATURE_NAMES, feature_cols)
    }

    # --- MODIFICATION END ---

    with open(output_path, 'w') as f:
        json.dump(output_stats, f, indent=4)

    print(f"Successfully saved typical clickbait and non-clickbait feature stats to: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyze and store typical feature values for clickbait articles.")
    parser.add_argument("feature_csv",
                        help="Path to the input CSV file containing features (e.g., data/clickbait17/models/bert-base-uncased/clickbait17_train_features.csv).")
    parser.add_argument("output_json",
                        help="Path to save the output JSON statistics file (e.g., typical_clickbait_features.json).")
    parser.add_argument("--threshold", type=float, default=GENERAL_CONFIG["clickbait_threshold"], help="Clickbait score threshold.")

    #args = parser.parse_args()
    # analyze_and_store_features(args.feature_csv, args.output_json, args.threshold)
    analyze_and_store_features("models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features.csv", "models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features_analysis_metadata.json")