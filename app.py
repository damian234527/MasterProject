import logging
from flask import Flask, request, render_template, jsonify, abort
from main import ClickbaitAndSimilarityDetector, ArticleScraper
from results_analysis import ExplanationGenerator
from config import HEADLINE_CONTENT_CONFIG, HEADLINE_CONFIG, GENERAL_CONFIG

# Configure Flask app
app = Flask(__name__)

# --- Model Loading ---
print("=" * 50)
print("Loading models, please wait this may take a moment...")
detector = None
explainer = None
try:
    detector = ClickbaitAndSimilarityDetector(
        headline_model_type=HEADLINE_CONFIG.get("model_type", "logistic"),
        headline_model_path=HEADLINE_CONFIG.get("model_path"),
        headline_content_model_path=HEADLINE_CONTENT_CONFIG.get("model_path_default")[1],
        headline_content_model_type=HEADLINE_CONTENT_CONFIG.get("model_type")[1],
        headline_content_transformer=HEADLINE_CONTENT_CONFIG.get("model_name")
    )
    # MODIFIED: Pass paths for both feature and similarity statistics
    feature_stats_path = "data/clickbait17/feature_statistics.json"
    similarity_stats_path = "data/clickbait17/similarity_statistics.json"  # Ensure this file exists at this path
    explainer = ExplanationGenerator(feature_stats_path, similarity_stats_path)
    print("Models and feature analysis loaded successfully.")
    print("=" * 50)
except Exception as e:
    logging.error(
        f"FATAL: Could not load models or analysis tool. Please ensure they are trained and paths are correct. Error: {e}")


def _perform_analysis(url: str, post_text: str = ""):
    """
    A core helper function that performs the entire analysis pipeline.
    """
    # 1. Scrape Article
    print(f"Scraping URL: {url}")
    scraper = ArticleScraper(url)
    headline = scraper.get_headline()
    content = scraper.get_content()

    if not headline or not content:
        raise ValueError("Could not extract a headline or content from the URL.")

    # 2. Perform Analysis
    print(f"Analyzing Headline: '{headline}'")
    is_clickbait, clickbait_score = detector.detect_clickbait(headline)
    similarity_scores = detector.compare_similarity(headline, content, post=post_text, headline_score=clickbait_score)
    features = detector.extract_features(post=post_text, headline=headline, content=content)

    # 3. Package and return all raw results
    return {
        "headline": headline,
        "content": content,
        "is_clickbait": is_clickbait,
        "clickbait_score": clickbait_score,
        "similarity_scores": similarity_scores,
        "features": features
    }


# --- Web Interface Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page with a URL input form."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the form submission, calls analysis functions, and renders the results.
    """
    if not detector or not explainer:
        return "Error: Models are not loaded. Please check server logs.", 500

    url = request.form.get('url')
    post = request.form.get('post', '').strip()
    if not url:
        return "Error: No URL provided.", 400

    try:
        analysis_data = _perform_analysis(url, post_text=post)

        # Generate linguistic feature explanations
        explanations = explainer.generate_explanations(analysis_data["features"], analysis_data["clickbait_score"])

        # NEW: Generate percentile analysis for all metrics
        percentiles = explainer.generate_percentile_analysis(analysis_data)

        # Filter explanations if no post is provided
        post_provided = bool(post)
        if not post_provided:
            post_related_keywords = ["Post", "Jaccard", "Sentiment Difference"]
            filtered_explanations = {
                name: data for name, data in explanations.items()
                if not any(keyword in name for keyword in post_related_keywords)
            }
        else:
            filtered_explanations = explanations

        # Package final results for the template
        results = {
            "url": url,
            "headline": analysis_data["headline"],
            "post_text": post if post_provided else "N/A",
            "post_provided": post_provided,
            "is_clickbait": "CLICKBAIT" if analysis_data["is_clickbait"] else "NOT CLICKBAIT",
            "clickbait_score": f"{analysis_data['clickbait_score']:.4f}",
            "similarity_scores": analysis_data["similarity_scores"],
            "features": analysis_data["features"],
            "explanations": filtered_explanations,
            "percentiles": percentiles  # NEW: Add percentiles to the result object
        }
        return render_template('results.html', result=results)

    except Exception as e:
        logging.error(f"An error occurred during analysis for URL {url}: {e}")
        return f"An error occurred: {e}", 500


# --- API Route ---

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Handles API requests. Calls the core analysis function
    and returns the results as a JSON object.
    """
    if not detector:
        return jsonify({"error": "Models are not loaded. Please check server logs."}), 500

    if not request.json or 'url' not in request.json:
        abort(400, description="Bad Request: Missing 'url' in JSON payload.")

    url = request.json['url']
    post = request.json.get('post', '')

    try:
        analysis_data = _perform_analysis(url, post_text=post)

        results = {
            "url": url,
            "headline": analysis_data["headline"],
            "is_clickbait": analysis_data["is_clickbait"],
            "clickbait_score": analysis_data["clickbait_score"],
            "similarity_scores": analysis_data["similarity_scores"]
        }
        return jsonify(results)

    except Exception as e:
        logging.error(f"API Error for URL {url}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)