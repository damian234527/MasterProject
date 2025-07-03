import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import re
from config import GENERAL_CONFIG

# TODO
INPUT_CSV_PATH = "models/sentence-transformers_all-MiniLM-L6-v2/clickbait17_train_features.csv"
OUTPUT_DIR = "visualization"
CLICKBAIT_THRESHOLD = GENERAL_CONFIG.get("clickbait_threshold", 0.5)


def visualize_features(csv_path: str, output_dir: str, threshold: float):
    """
    Loads a feature dataset and generates histograms and box plots for each feature.

    For each feature, it creates:
    1. An overall histogram.
    2. An overall box plot.
    3. A comparative histogram (clickbait vs. non-clickbait).
    4. A comparative box plot (clickbait vs. non-clickbait).

    Args:
        csv_path (str): The path to the input CSV file with features.
        output_dir (str): The directory where plot images will be saved.
        threshold (float): The score threshold to classify an entry as clickbait.
    """
    # --- 1. Setup and Data Loading ---
    print(f"Loading data from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: Input file not found at '{csv_path}'. Please update the INPUT_CSV_PATH variable.")
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    print(f"Plots will be saved to '{output_dir}'")

    df = pd.read_csv(csv_path)

    # Add a binary 'is_clickbait' column for easier plotting
    df['is_clickbait'] = df['clickbait_score'] >= threshold

    # Find all feature columns (assuming they are named f1, f2, etc.)
    feature_columns = sorted([col for col in df.columns if re.match(r'^f\d+$', col)])
    if not feature_columns:
        print("Error: No feature columns (e.g., 'f1', 'f2') found in the CSV.")
        return

    print(f"Found {len(feature_columns)} features. Generating plots...")

    # --- 2. Plot Generation Loop ---
    for feature in feature_columns:
        print(f"  - Processing {feature}...")

        # Define file paths for the plots
        hist_overall_path = os.path.join(output_dir, f"{feature}_histogram_overall.png")
        box_overall_path = os.path.join(output_dir, f"{feature}_boxplot_overall.png")
        hist_comp_path = os.path.join(output_dir, f"{feature}_histogram_comparison.png")
        box_comp_path = os.path.join(output_dir, f"{feature}_boxplot_comparison.png")

        # Overall Histogram
        plt.figure(figsize=(10, 6))
        sns.histplot(df[feature], kde=True)
        plt.title(f'Overall Histogram for {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(hist_overall_path)
        plt.close()

        # Overall Box Plot
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=df[feature])
        plt.title(f'Overall Box Plot for {feature}')
        plt.ylabel(feature)
        plt.savefig(box_overall_path)
        plt.close()

        # Comparative Histogram
        plt.figure(figsize=(12, 7))
        sns.histplot(data=df, x=feature, hue='is_clickbait', kde=True, palette='viridis')
        plt.title(f'{feature}: Clickbait vs. Non-Clickbait Distribution')
        plt.xlabel(feature)
        plt.ylabel('Frequency')
        plt.savefig(hist_comp_path)
        plt.close()

        # Comparative Box Plot
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
    visualize_features(
        csv_path=INPUT_CSV_PATH,
        output_dir=OUTPUT_DIR,
        threshold=CLICKBAIT_THRESHOLD
    )