"""Generates human-readable analysis from model outputs and feature statistics.

This module provides the `ExplanationGenerator` class, which takes the raw
numerical outputs from the clickbait detection models (scores and features)
and translates them into qualitative explanations. It does this by comparing
an article's features against pre-computed statistical profiles of typical
clickbait and non-clickbait content.
"""

import json
import math
import os
import numpy as np


class ExplanationGenerator:
    """Generates explanations and percentile analysis for model predictions.

    This class loads statistical profiles (mean, std, quartiles, etc.) for
    various linguistic and similarity features, pre-calculated for both
    clickbait and non-clickbait articles. It uses these profiles to determine
    which features for a given article are most indicative of its final
    classification and to place its metrics within a percentile context.

    Attributes:
        stats (dict): A dictionary containing all loaded statistical data.
        top_n_features (int): The number of top contributing features to explain.
        clickbait_profile (dict): The statistical profile for clickbait articles.
        non_clickbait_profile (dict): The profile for non-clickbait articles.
        global_ranges (dict): The global min/max range for each feature.
    """

    def __init__(self, feature_stats_path: str, similarity_stats_path: str, top_n_features: int = 5):
        """Initializes the generator by loading statistical profiles.

        Args:
            feature_stats_path (str): Path to the JSON file with statistics
                for linguistic features.
            similarity_stats_path (str): Path to the JSON file with statistics
                for similarity scores.
            top_n_features (int, optional): The number of top features to
                select for generating detailed explanations. Defaults to 5.
        """
        self.stats = {}
        self.top_n_features = top_n_features
        try:
            # Load statistics from separate files for features and similarities.
            with open(feature_stats_path, 'r') as f:
                feature_stats = json.load(f)
            with open(similarity_stats_path, 'r') as f:
                similarity_stats = json.load(f)

            # Merge the loaded statistics into unified profiles.
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
        """Prepares data required for rendering a visual gauge in the front end.

        This method calculates the relative positions and widths for a value
        and the interquartile ranges of clickbait/non-clickbait profiles,
        scaled to a 0-100 range for CSS rendering.

        Args:
            feature_name (str): The name of the feature to visualize.
            value (float): The article's value for that feature.

        Returns:
            A dictionary with positional data for visualization, or None if
            stats for the feature are unavailable.
        """
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
            """Helper to scale a value to a 0-100 position."""
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
        """Generates a percentile description using pre-computed histogram data.

        This provides a more accurate percentile than assuming a normal
        distribution, especially for skewed data.

        Args:
            value (float): The value for which to find the percentile.
            stats (dict): The statistical profile for a feature, which must
                contain histogram data ('bin_edges', 'cumulative_counts').
            profile_name (str): The name of the profile (e.g., 'clickbait').

        Returns:
            A human-readable string describing the value's percentile.
        """
        histogram_data = stats.get('histogram')
        if not histogram_data:
            return "Percentile could not be calculated (no histogram data)."

        bin_edges = np.array(histogram_data['bin_edges'])
        cumulative_counts = np.array(histogram_data['cumulative_counts'])
        total_count = cumulative_counts[-1]

        if total_count == 0:
            return "Percentile could not be calculated (no data in profile)."

        # Find which histogram bin the value falls into.
        bin_index = np.searchsorted(bin_edges, value, side='right')

        if bin_index > len(cumulative_counts):
            bin_index = len(cumulative_counts)

        # Calculate percentile based on the cumulative count of the bin.
        if bin_index == 0:
            percentile = 0
        else:
            count_at_value = cumulative_counts[bin_index - 1]
            percentile = (count_at_value / total_count) * 100

        return f"This value is at the <strong>{percentile:.0f}th percentile</strong> for {profile_name} articles."

    def generate_percentile_analysis(self, analysis_data: dict) -> dict:
        """Generates percentile analysis for all features and similarity scores.

        It compares each metric from the analyzed article against the relevant
        statistical profiles (clickbait, non-clickbait, or both).

        Args:
            analysis_data (dict): A dictionary containing the analysis results,
                including 'features', 'similarity_scores', 'clickbait_score',
                and 'is_clickbait'.

        Returns:
            A dictionary where keys are metric names and values are a list of
            percentile description strings.
        """
        if not self.stats:
            return {}

        all_metrics = {**analysis_data.get("features", {}), **analysis_data.get("similarity_scores", {})}
        final_score = analysis_data.get("clickbait_score")
        is_clickbait = analysis_data.get("is_clickbait")
        percentiles = {}

        # Map display names to the keys used in the stats files.
        key_mapping = {
            "TF-IDF Cosine": "TF-IDF Cosine Similarity",
            "Transformer Embedding": "Transformer Embedding Similarity"
        }

        # Determine which profiles to compare against based on the final score.
        profiles_to_check = []
        if 0.4 <= final_score <= 0.6:
            # For ambiguous scores, compare against both profiles.
            profiles_to_check.append(("ambiguous (clickbait)", self.clickbait_profile))
            profiles_to_check.append(("ambiguous (non-clickbait)", self.non_clickbait_profile))
        elif is_clickbait:
            profiles_to_check.append(("clickbait", self.clickbait_profile))
        else:
            profiles_to_check.append(("non-clickbait", self.non_clickbait_profile))

        # Generate percentile text for each metric.
        for metric_name, value in all_metrics.items():
            percentiles[metric_name] = []
            stats_key = key_mapping.get(metric_name, metric_name)

            for profile_name, profile_data in profiles_to_check:
                if stats_key in profile_data:
                    text = self._get_precise_percentile_text(value, profile_data[stats_key],
                                                             profile_name.split(" (")[-1].replace(")", ""))
                    percentiles[metric_name].append(text)
        return percentiles

    def generate_explanations(self, article_features: dict, final_score: float) -> dict:
        """Identifies and explains the top features supporting the final score.

        This method calculates a "support score" for each feature based on how
        much closer the feature's value is to the mean of the predicted class's
        profile than to the mean of the other class's profile (measured in
        standard deviations, i.e., Z-scores).

        Args:
            article_features (dict): A dictionary of extracted feature values.
            final_score (float): The final clickbait score for the article.

        Returns:
            A dictionary where keys are the top feature names and values are
            explanation objects containing text, a tag, and visualization data.
        """
        if not self.stats:
            return {"error": "Feature statistics not loaded."}

        is_clickbait_prediction = final_score > 0.5
        feature_support_scores = {}

        # Calculate a support score for each feature.
        for feature_name, value in article_features.items():
            if feature_name not in self.clickbait_profile or feature_name not in self.non_clickbait_profile:
                continue

            cb_stats = self.clickbait_profile[feature_name]
            ncb_stats = self.non_clickbait_profile[feature_name]

            # Calculate Z-scores relative to both profiles.
            z_score_cb = abs((value - cb_stats['mean']) / cb_stats['std']) if cb_stats['std'] > 0 else float('inf')
            z_score_ncb = abs((value - ncb_stats['mean']) / ncb_stats['std']) if ncb_stats['std'] > 0 else float('inf')

            # The support score is the difference in Z-scores. A higher score
            # means the feature provides more support for the predicted class.
            if is_clickbait_prediction:
                support_score = z_score_ncb - z_score_cb
            else:
                support_score = z_score_cb - z_score_ncb

            feature_support_scores[feature_name] = support_score

        # Select the top N features with the highest support scores.
        top_features = sorted(feature_support_scores, key=feature_support_scores.get, reverse=True)[
                       :self.top_n_features]

        # Generate the final explanation objects for the top features.
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

            # Generate explanation text based on which profile the value is closer to.
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