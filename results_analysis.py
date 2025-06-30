import json
import math


class ExplanationGenerator:
    # No changes to __init__, _calculate_discriminative_features, or _prepare_visualization_data
    def __init__(self, stats_path: str, top_n_features: int = 5):
        try:
            with open(stats_path, 'r') as f:
                self.stats = json.load(f)
            self.clickbait_profile = self.stats.get("clickbait_profile", {})
            self.non_clickbait_profile = self.stats.get("non_clickbait_profile", {})
            self.global_ranges = self.stats.get("global_ranges", {})
            # We'll use top_n_features in the generate_explanations method now
            self.top_n_features = top_n_features
        except FileNotFoundError:
            print(f"Warning: Statistics file not found at {stats_path}. Explanations will not be available.")
            self.stats = {}

    def _prepare_visualization_data(self, feature_name: str, value: float) -> dict:
        """Prepares the data needed for the CSS-based visualization gauge."""
        if feature_name not in self.global_ranges or not self.stats:
            return None

        g_range = self.global_ranges[feature_name]
        ncb_stats = self.non_clickbait_profile[feature_name]
        cb_stats = self.clickbait_profile[feature_name]

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

    # --- THIS IS THE UPDATED METHOD ---
    def generate_explanations(self, article_features: dict, final_score: float) -> dict:
        """
        Dynamically finds the top N features that best explain the final score.
        """
        if not self.stats:
            return {"error": "Feature statistics not loaded."}

        is_clickbait_prediction = final_score > 0.5
        feature_support_scores = {}

        # First, calculate a "support score" for every feature
        for feature_name, value in article_features.items():
            if feature_name not in self.clickbait_profile:
                continue

            cb_stats = self.clickbait_profile[feature_name]
            ncb_stats = self.non_clickbait_profile[feature_name]

            z_score_cb = abs((value - cb_stats['mean']) / cb_stats['std']) if cb_stats['std'] > 0 else float('inf')
            z_score_ncb = abs((value - ncb_stats['mean']) / ncb_stats['std']) if ncb_stats['std'] > 0 else float('inf')

            # The support score is the difference between how well it fits one profile vs the other
            if is_clickbait_prediction:
                # For a clickbait prediction, we want features where it's a good fit for clickbait
                # and a bad fit for non-clickbait.
                support_score = z_score_ncb - z_score_cb
            else:
                # For a non-clickbait prediction, we want the reverse.
                support_score = z_score_cb - z_score_ncb

            feature_support_scores[feature_name] = support_score

        # Get the top N features with the highest support for the prediction
        top_features = sorted(feature_support_scores, key=feature_support_scores.get, reverse=True)[
                       :self.top_n_features]

        # Now, build the explanation results only for these top features
        results = {}
        for feature_name in top_features:
            value = article_features[feature_name]
            z_score_cb = abs(
                (value - self.clickbait_profile[feature_name]['mean']) / self.clickbait_profile[feature_name]['std']) if \
            self.clickbait_profile[feature_name]['std'] > 0 else float('inf')
            z_score_ncb = abs(
                (value - self.non_clickbait_profile[feature_name]['mean']) / self.non_clickbait_profile[feature_name][
                    'std']) if self.non_clickbait_profile[feature_name]['std'] > 0 else float('inf')

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