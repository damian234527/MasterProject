# clickbait17_visualization.py
"""Generates visualizations for feature distributions in the Clickbait17 dataset.

This script is intended for exploratory data analysis. It loads a
feature-augmented dataset and creates a series of plots (histograms and box plots)
for each feature, saved as image files. These visualizations help in
understanding the differences in feature distributions between clickbait and
non-clickbait articles.
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from config import GENERAL_CONFIG

# Configuration for the script.
INPUT_CSV_PATH = "models/default/clickbait17_train_features_original.csv"
OUTPUT_DIR = "visualization"
CLICKBAIT_THRESHOLD = GENERAL_CONFIG["clickbait_threshold"]


def visualize_features(csv_path: str, output_dir: str, threshold: float = GENERAL_CONFIG["clickbait_threshold"]):
    """Loads a feature dataset and generates plots for each feature.

    For each feature column in the CSV, this function generates and saves four plots:
    1. An overall histogram of the feature's distribution.
    2. An overall box plot to show quartiles and outliers.
    3. A comparative histogram showing distributions for clickbait vs. non-clickbait.
    4. A comparative box plot for clickbait vs. non-clickbait.

    Args:
        csv_path (str): The path to the input feature-augmented CSV file.
        output_dir (str): The directory where the plot images will be saved.
        threshold (float): The score threshold used to classify an entry as clickbait.
    """
    # --- 1. Setup and Data Loading ---
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found at '{csv_path}'.")
        return

    # Create the output directory if it doesn't exist.
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to '{output_dir}'")

    df = pd.read_csv(csv_path)

    # Create a binary 'is_clickbait' column for easier plotting with seaborn.
    df['is_clickbait'] = df['clickbait_score'] >= threshold

    # Find all feature columns, which are assumed to be named f1, f2, etc.
    feature_columns = sorted([col for col in df.columns if re.match(r'^f\d+$', col)])
    if not feature_columns:
        print("Error: No feature columns (e.g., 'f1', 'f2') found in the CSV.")
        return

    print(f"Found {len(feature_columns)} features. Generating plots...")

    # --- 2. Plot Generation Loop ---
    for feature in feature_columns:
        print(f"  - Processing {feature}...")

        # Define file paths for the four plots for the current feature.
        hist_overall_path = os.path.join(output_dir, f"{feature}_histogram_overall.png")
        box_overall_path = os.path.join(output_dir, f"{feature}_boxplot_overall.png")
        hist_comp_path = os.path.join(output_dir, f"{feature}_histogram_comparison.png")
        box_comp_path = os.path.join(output_dir, f"{feature}_boxplot_comparison.png")

        # Generate and save the overall histogram.
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Overall Histogram for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(hist_overall_path)
        plt.close()

        # Generate and save the overall box plot.
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[feature])
        plt.title(f'Overall Box Plot for {feature}')
        plt.ylabel(feature)
        plt.savefig(box_overall_path)
        plt.close()

        # Generate and save the comparative histogram.
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df, x=feature, hue='is_clickbait', kde=True, palette='viridis')
        plt.title(f'{feature}: Clickbait vs. Non-Clickbait Distribution')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(hist_comp_path)
        plt.close()

        # Generate and save the comparative box plot.
        plt.figure(figsize=(10, 8))
        sns.boxplot(data=df, x='is_clickbait', y=feature, palette='viridis')
        plt.title(f'{feature}: Clickbait vs. Non-Clickbait Comparison')
        plt.xlabel('Is Clickbait?')
        plt.ylabel(feature)
        plt.xticks([0, 1], ['Non-Clickbait', 'Clickbait'])
        plt.savefig(box_comp_path)
        plt.close()

    print("\nAll plots have been generated successfully!")


if __name__ == '__main__':
    # Run the visualization function with the configured paths.
    visualize_features(
        csv_path=INPUT_CSV_PATH,
        output_dir=OUTPUT_DIR,
        threshold=CLICKBAIT_THRESHOLD
    )