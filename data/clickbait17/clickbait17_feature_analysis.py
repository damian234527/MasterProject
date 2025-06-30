import pandas as pd
import json
import os
from config import GENERAL_CONFIG

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


def analyze_and_store_features(feature_csv_path: str, output_path: str, clickbait_threshold: float = 0.5):
    print(f"Loading feature data from: {feature_csv_path}")
    if not os.path.exists(feature_csv_path):
        print(f"Error: File not found at {feature_csv_path}. Please run the data preparation script first.")
        return

    df = pd.read_csv(feature_csv_path)
    feature_cols = [f'f{i + 1}' for i in range(len(FEATURE_NAMES))]
    if not all(col in df.columns for col in feature_cols):
        print("Error: Feature columns (f1, f2, ...) not found.")
        return

    # Calculate global min/max for visualization scaling
    global_ranges = {}
    for i, f_col in enumerate(feature_cols):
        feature_name = FEATURE_NAMES[i]
        # --- FIX: Convert numpy types to native Python types ---
        global_ranges[feature_name] = {
            'min': float(df[f_col].min()),
            'max': float(df[f_col].max())
        }

    df_clickbait = df[df['clickbait_score'] > clickbait_threshold].copy()
    df_non_clickbait = df[df['clickbait_score'] <= clickbait_threshold].copy()

    if df_clickbait.empty or df_non_clickbait.empty:
        print(f"Dataset must contain both clickbait and non-clickbait examples. Cannot generate stats.")
        return

    print(f"Analyzing {len(df_clickbait)} clickbait articles and {len(df_non_clickbait)} non-clickbait articles.")

    def get_stats_for_group(dataframe, feature_names_list, feature_cols_list):
        agg_funcs = ['mean', 'std', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
        stats = dataframe[feature_cols_list].agg(agg_funcs)
        stats.index = ['mean', 'std', 'p25', 'p75']

        group_stats = {}
        for i, f_col in enumerate(feature_cols_list):
            feature_name = feature_names_list[i]
            # --- FIX: Convert all numpy stats to native Python floats ---
            group_stats[feature_name] = {
                'mean': float(stats.loc['mean', f_col]),
                'std': float(stats.loc['std', f_col]),
                'p25': float(stats.loc['p25', f_col]),
                'p75': float(stats.loc['p75', f_col])
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

    with open(output_path, 'w') as f:
        json.dump(output_stats, f, indent=4)

    print(f"Successfully saved detailed feature statistics to: {output_path}")


if __name__ == '__main__':
    feature_csv_path = "models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features.csv"
    output_json_path = "feature_statistics.json"
    clickbait_threshold = 0.5
    analyze_and_store_features(feature_csv_path, output_json_path, clickbait_threshold)