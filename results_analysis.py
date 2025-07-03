import json
import math
import os
import numpy as np


class ExplanationGenerator:
    def __init__(self, feature_stats_path: str, similarity_stats_path: str, top_n_features: int = 5):
        """
        Initializes the generator by loading and merging statistics from two separate files.

        Args:
            feature_stats_path (str): Path to the JSON file with linguistic feature stats.
            similarity_stats_path (str): Path to the JSON file with similarity score stats.
            top_n_features (int): The number of top features to explain.
        """
        self.stats = {}
        self.top_n_features = top_n_features
        try:
            with open(feature_stats_path, 'r') as f:
                feature_stats = json.load(f)
            with open(similarity_stats_path, 'r') as f:
                similarity_stats = json.load(f)

            # Merge the statistics from both files
            self.clickbait_profile = {**feature_stats.get("clickbait_profile", {}),
                                      **similarity_stats.get("clickbait_profile", {})}
            self.non_clickbait_profile = {**feature_stats.get("non_clickbait_profile", {}),
                                          **similarity_stats.get("non_clickbait_profile", {})}
            self.global_ranges = {**feature_stats.get("global_ranges", {}),
                                  **similarity_stats.get("global_ranges", {})}
            self.stats = {
                "clickbait_profile": self.clickbait_profile,
                "non_clickbait_profile": self.non_clickbait_profile,
                "global_ranges": self.global_ranges
            }

        except FileNotFoundError as e:
            print(f"Warning: Statistics file not found: {e}. Explanations may be incomplete.")
            self.stats = {}
            self.clickbait_profile = {}
            self.non_clickbait_profile = {}
            self.global_ranges = {}

    def _prepare_visualization_data(self, feature_name: str, value: float) -> dict:
        """Prepares the data needed for the CSS-based visualization gauge."""
        if feature_name not in self.global_ranges or not self.stats:
            return None

        g_range = self.global_ranges[feature_name]
        ncb_stats = self.non_clickbait_profile.get(feature_name)
        cb_stats = self.clickbait_profile.get(feature_name)

        if not ncb_stats or not cb_stats:
            return None

        total_range = g_range['max'] - g_range['min']
        if total_range == 0:
            return {'value_pos': 50, 'ncb_pos': 0, 'ncb_width': 100, 'cb_pos': 0, 'cb_width': 100}

        def get_pos(val):
            clamped_val = max(g_range['min'], min(val, g_range['max']))
            return ((clamped_val - g_range['min']) / total_range) * 100

        return {
            'value_pos': get_pos(value),
            'ncb_pos': get_pos(ncb_stats['p25']),
            'ncb_width': get_pos(ncb_stats['p75']) - get_pos(ncb_stats['p25']),
            'cb_pos': get_pos(cb_stats['p25']),
            'cb_width': get_pos(cb_stats['p75']) - get_pos(cb_stats['p25']),
        }

    def _get_precise_percentile_text(self, value: float, stats: dict, profile_name: str) -> str:
        """
        Generates a human-readable string with the precise percentile using histogram data.
        """
        histogram_data = stats.get('histogram')
        if not histogram_data:
            return "Percentile could not be calculated (no histogram data)."

        bin_edges = np.array(histogram_data['bin_edges'])  #
        cumulative_counts = np.array(histogram_data['cumulative_counts'])  #
        total_count = cumulative_counts[-1]

        if total_count == 0:
            return "Percentile could not be calculated (no data in profile)."

        # Find the index of the bin where the value falls
        # np.searchsorted finds the insertion point, which corresponds to the bin index
        bin_index = np.searchsorted(bin_edges, value, side='right')

        # Handle edge cases
        if bin_index == 0:
            percentile = 0
        else:
            # The count of items less than or equal to the current value's bin
            count_at_value = cumulative_counts[bin_index - 1]
            percentile = (count_at_value / total_count) * 100

        return f"This value is at the <strong>{percentile:.0f}th percentile</strong> for {profile_name} articles."

    def generate_percentile_analysis(self, analysis_data: dict) -> dict:
        """
        Generates percentile analysis for all features and similarity scores.
        """
        if not self.stats:
            return {}

        all_metrics = {**analysis_data.get("features", {}), **analysis_data.get("similarity_scores", {})}
        final_score = analysis_data.get("clickbait_score")
        is_clickbait = analysis_data.get("is_clickbait")
        percentiles = {}

        key_mapping = {
            "TF-IDF Cosine": "TF-IDF Cosine Similarity",
            "Transformer Embedding": "Transformer Embedding Similarity"
        }

        profiles_to_check = []
        if 0.4 <= final_score <= 0.6:
            profiles_to_check.append(("ambiguous (clickbait)", self.clickbait_profile))
            profiles_to_check.append(("ambiguous (non-clickbait)", self.non_clickbait_profile))
        elif is_clickbait:
            profiles_to_check.append(("clickbait", self.clickbait_profile))
        else:
            profiles_to_check.append(("non-clickbait", self.non_clickbait_profile))

        for metric_name, value in all_metrics.items():
            percentiles[metric_name] = []
            stats_key = key_mapping.get(metric_name, metric_name)

            for profile_name, profile_data in profiles_to_check:
                if stats_key in profile_data:
                    # UPDATED: Call the new function for precise percentile
                    text = self._get_precise_percentile_text(value, profile_data[stats_key],
                                                             profile_name.split(" (")[-1].replace(")", ""))
                    percentiles[metric_name].append(text)
        return percentiles

    def generate_explanations(self, article_features: dict, final_score: float) -> dict:
        """
        Dynamically finds the top N features that best explain the final score.
        """
        if not self.stats:
            return {"error": "Feature statistics not loaded."}

        is_clickbait_prediction = final_score > 0.5
        feature_support_scores = {}

        for feature_name, value in article_features.items():
            if feature_name not in self.clickbait_profile or feature_name not in self.non_clickbait_profile:
                continue

            cb_stats = self.clickbait_profile[feature_name]
            ncb_stats = self.non_clickbait_profile[feature_name]

            z_score_cb = abs((value - cb_stats['mean']) / cb_stats['std']) if cb_stats['std'] > 0 else float('inf')
            z_score_ncb = abs((value - ncb_stats['mean']) / ncb_stats['std']) if ncb_stats['std'] > 0 else float('inf')

            if is_clickbait_prediction:
                support_score = z_score_ncb - z_score_cb
            else:
                support_score = z_score_cb - z_score_ncb

            feature_support_scores[feature_name] = support_score

        top_features = sorted(feature_support_scores, key=feature_support_scores.get, reverse=True)[
                       :self.top_n_features]

        results = {}
        for feature_name in top_features:
            value = article_features[feature_name]
            z_score_cb = abs(
                (value - self.clickbait_profile[feature_name]['mean']) / self.clickbait_profile[feature_name]['std']) if \
                self.clickbait_profile[feature_name]['std'] > 0 else float('inf')
            z_score_ncb = abs(
                (value - self.non_clickbait_profile[feature_name]['mean']) /
                self.non_clickbait_profile[feature_name]['std']) if self.non_clickbait_profile[feature_name][
                                                                        'std'] > 0 else float('inf')

            explanation = ""
            tag = "neutral"

            if z_score_cb < z_score_ncb:
                tag = "pro-clickbait-mild"
                if z_score_cb < 0.5:
                    explanation = "Value is a very strong match for a typical clickbait article."
                    tag = "pro-clickbait"
                else:
                    explanation = "Value more closely resembles the profile of a clickbait article."
            elif z_score_ncb < z_score_cb:
                tag = "anti-clickbait"
                if z_score_ncb < 0.5:
                    explanation = "Value is a very strong match for a typical non-clickbait article."
                else:
                    explanation = "Value more closely resembles the profile of a non-clickbait article."
            else:
                explanation = "Value is equidistant from both profiles; this feature is neutral."

            results[feature_name] = {
                "text": explanation,
                "tag": tag,
                "visualization": self._prepare_visualization_data(feature_name, value)
            }
        return results