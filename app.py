import logging
from flask import Flask, request, render_template, jsonify, abort
from main import ClickbaitAndSimilarityDetector, ArticleScraper
from results_analysis import ExplanationGenerator
from config import HEADLINE_CONTENT_CONFIG, HEADLINE_CONFIG, GENERAL_CONFIG

# Configure Flask app
app = Flask(__name__)

# --- Model Loading ---
# Load the detector once when the server starts to avoid reloading on every request.
# This can take a few moments.
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
    explainer = ExplanationGenerator("data/clickbait17/feature_statistics.json")
    print("Models and feature analysis loaded successfully.")
    print("=" * 50)
except Exception as e:
    logging.error(
        f"FATAL: Could not load models or analysis tool. Please ensure they are trained and paths in config.py are correct. Error: {e}")


# --- Web Interface Routes ---

@app.route('/', methods=['GET'])
def index():
    """Renders the main page with a URL input form."""
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Handles the form submission from the web interface.
    Scrapes the URL, runs the analysis, and displays the results.
    """
    if not detector:
        return "Error: Models are not loaded. Please check server logs.", 500

    url = request.form.get('url')
    post = request.form.get('post', '').strip()
    if not url:
        return "Error: No URL provided.", 400

    try:
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
        similarity_scores = detector.compare_similarity(headline, content, post=post, headline_score=clickbait_score)
        features = detector.extract_features(post=post, headline=headline, content=content)

        # Generate the feature-by-feature explanation
        explanations = explainer.generate_explanations(features, clickbait_score)
        filtered_explanations = {}
        post_provided = bool(post)
        if not post_provided:
            # List of keywords that identify post-dependent features
            post_related_keywords = ["Post", "Jaccard", "Sentiment Difference"]
            for feature_name, explanation_data in explanations.items():
                # If the feature name contains any of the keywords, it's post-related.
                if not any(keyword in feature_name for keyword in post_related_keywords):
                    filtered_explanations[feature_name] = explanation_data
        else:
            # If a post was provided, show all explanations
            filtered_explanations = explanations

        # 3. Package Results
        results = {
            "url": url,
            "headline": headline,
            "post_text": post if post_provided else "N/A",
            "post_provided": post_provided,
            "is_clickbait": "CLICKBAIT" if is_clickbait else "NOT CLICKBAIT",
            "clickbait_score": f"{clickbait_score:.4f}",
            "similarity_scores": similarity_scores,
            "features": features,
            "explanations": filtered_explanations
        }
        return render_template('results.html', result=results)

    except Exception as e:
        logging.error(f"An error occurred during analysis for URL {url}: {e}")
        return f"An error occurred: {e}", 500


# --- API Route ---

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """
    Handles API requests. Expects a JSON payload with a 'url' key.
    Returns the analysis results as a JSON object.
    """
    if not detector:
        return jsonify({"error": "Models are not loaded. Please check server logs."}), 500

    if not request.json or 'url' not in request.json:
        abort(400, description="Bad Request: Missing 'url' in JSON payload.")

    url = request.json['url']

    try:
        # Scrape and analyze as in the web interface version
        scraper = ArticleScraper(url)
        headline = scraper.get_headline()
        content = scraper.get_content()
        if not headline or not content:
            raise ValueError("Could not extract a headline or content from the URL.")

        is_clickbait, clickbait_score = detector.detect_clickbait(headline)
        similarity_scores = detector.compare_similarity(headline, content, post="", headline_score=clickbait_score)

        results = {
            "url": url,
            "headline": headline,
            "is_clickbait": is_clickbait,
            "clickbait_score": clickbait_score,
            "similarity_scores": similarity_scores
        }
        return jsonify(results)

    except Exception as e:
        logging.error(f"API Error for URL {url}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Set debug=False for production use
    app.run(host='0.0.0.0', port=5000, debug=True)