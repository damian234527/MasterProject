"""A Flask web application for the Clickbait Detection and Analysis Tool.

This script launches a web server that provides a user interface for analyzing
articles. It includes routes for displaying the main page, handling analysis
requests from a form, and providing a JSON-based API for programmatic access.
"""
import logging
from flask import Flask, request, render_template, jsonify, abort
from main import ClickbaitAndSimilarityDetector, ArticleScraper
from results_analysis import ExplanationGenerator
from config import HEADLINE_CONTENT_CONFIG, HEADLINE_CONFIG, GENERAL_CONFIG

logger = logging.getLogger(__name__)

# Configure the Flask web application instance.
app = Flask(__name__)

# Initialize global variables for the detector and explainer.
# These are loaded once when the application starts to avoid reloading models
# on every request, which would be very inefficient.
logger.info("=" * 50)
logger.info("Loading models, please wait this may take a moment...")
detector = None
explainer = None
try:
    # Instantiate the main detector class with model configurations.
    detector = ClickbaitAndSimilarityDetector(
        headline_model_type=HEADLINE_CONFIG.get("model_type", "logistic"),
        headline_model_path=HEADLINE_CONFIG.get("model_path"),
        headline_content_model_path=HEADLINE_CONTENT_CONFIG.get("model_path_default")[1],
        headline_content_model_type=HEADLINE_CONTENT_CONFIG.get("model_type")[1],
        headline_content_transformer=HEADLINE_CONTENT_CONFIG.get("model_name")
    )
    # Define paths to pre-computed statistics files for generating explanations.
    feature_stats_path = "data/clickbait17/feature_statistics.json"
    similarity_stats_path = "data/clickbait17/similarity_statistics.json"
    explainer = ExplanationGenerator(feature_stats_path, similarity_stats_path)
    logger.info("Models and feature analysis loaded successfully.")
    logger.info("=" * 50)
except Exception as e:
    # Log a fatal error if model loading fails, as the app cannot function.
    logger.error(
        f"FATAL: Could not load models or analysis tool. Please ensure they are trained and paths are correct. Error: {e}")


def _perform_analysis(url: str, post_text: str = ""):
    """Performs the complete analysis pipeline for a given URL and post text.

    This helper function encapsulates the process of scraping an article,
    running the clickbait and similarity analyses, and extracting features.

    Args:
        url (str): The URL of the article to analyze.
        post_text (str): Optional text from a social media post associated
            with the article.

    Returns:
        A dictionary containing all the raw results from the analysis, including
        the headline, content, clickbait score, similarity scores, and a
        dictionary of extracted linguistic features.

    Raises:
        ValueError: If the scraper fails to extract a headline or content from
            the given URL.
    """
    # Step 1: Scrape the article from the web.
    logger.info(f"Scraping URL: {url}")
    scraper = ArticleScraper(url)
    headline = scraper.get_headline()
    content = scraper.get_content()

    # Raise an error if scraping was unsuccessful.
    if not headline or not content:
        raise ValueError("Could not extract a headline or content from the URL.")

    # Step 2: Perform the core analysis using the loaded models.
    logger.info(f"Analyzing Headline: '{headline}'")
    is_clickbait, clickbait_score = detector.detect_clickbait(headline)
    similarity_scores = detector.compare_similarity(headline, content, post=post_text, headline_score=clickbait_score)
    features = detector.extract_features(post=post_text, headline=headline, content=content)

    # Step 3: Package all results into a single dictionary and return.
    return {
        "headline": headline,
        "content": content,
        "is_clickbait": is_clickbait,
        "clickbait_score": clickbait_score,
        "similarity_scores": similarity_scores,
        "features": features
    }


@app.route('/', methods=['GET'])
def index():
    """Renders the main homepage with the URL input form.

    Returns:
        The rendered 'index.html' template.
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    """Handles form submissions from the main page to analyze an article.

    This function retrieves the URL and optional post text from the form,
    runs the full analysis pipeline, generates human-readable explanations
    and percentile data, and renders the results on the results page.

    Returns:
        The rendered 'results.html' template populated with analysis data,
        or an error message and HTTP status code if something goes wrong.
    """
    # Abort if the models failed to load on startup.
    if not detector or not explainer:
        return "Error: Models are not loaded. Please check server logs.", 500

    # Retrieve URL and post text from the submitted form.
    url = request.form.get('url')
    post = request.form.get('post', '').strip()
    if not url:
        return "Error: No URL provided.", 400

    try:
        # Perform the core analysis.
        analysis_data = _perform_analysis(url, post_text=post)

        # Generate human-readable explanations for the top contributing features.
        explanations = explainer.generate_explanations(analysis_data["features"], analysis_data["clickbait_score"])

        # Generate percentile analysis for all metrics to show how they compare
        # to typical clickbait and non-clickbait articles.
        percentiles = explainer.generate_percentile_analysis(analysis_data)

        # If no post text was provided, filter out explanations related to it.
        post_provided = bool(post)
        if not post_provided:
            post_related_keywords = ["Post", "Jaccard", "Sentiment Difference"]
            filtered_explanations = {
                name: data for name, data in explanations.items()
                if not any(keyword in name for keyword in post_related_keywords)
            }
        else:
            filtered_explanations = explanations

        # Package all data into a final 'results' dictionary for the template.
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
            "percentiles": percentiles
        }
        return render_template('results.html', result=results)

    except Exception as e:
        # Log any exceptions that occur during the process and return an error page.
        logger.error(f"An error occurred during analysis for URL {url}: {e}")
        return f"An error occurred: {e}", 500


@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Provides a JSON-based API endpoint for article analysis.

    This endpoint accepts a POST request with a JSON payload containing a 'url'
    and an optional 'post' key. It performs the analysis and returns the key
    results in JSON format.

    Returns:
        A JSON response containing the analysis results or an error object.
    """
    # Abort if the models failed to load on startup.
    if not detector:
        return jsonify({"error": "Models are not loaded. Please check server logs."}), 500

    # Validate that the request contains JSON and the required 'url' key.
    if not request.json or 'url' not in request.json:
        abort(400, description="Bad Request: Missing 'url' in JSON payload.")

    url = request.json['url']
    post = request.json.get('post', '')

    try:
        # Perform the core analysis.
        analysis_data = _perform_analysis(url, post_text=post)

        # Package the primary results for the API response.
        results = {
            "url": url,
            "headline": analysis_data["headline"],
            "is_clickbait": analysis_data["is_clickbait"],
            "clickbait_score": analysis_data["clickbait_score"],
            "similarity_scores": analysis_data["similarity_scores"]
        }
        return jsonify(results)

    except Exception as e:
        # Log any exceptions and return a JSON error response.
        logger.error(f"API Error for URL {url}: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask application.
    # 'debug=True' enables auto-reloading on code changes and provides
    # detailed error pages. Should be set to 'False' in a production environment.
    app.run(host='0.0.0.0', port=5000, debug=True)