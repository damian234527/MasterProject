"""Methods for computing similarity between headlines and content.

This module defines a standardized interface (`SimilarityMethod`) and several
implementations for calculating a similarity score between a headline/post
and the main content of an article. This includes classic NLP methods like
TF-IDF Cosine Similarity, modern methods using transformer embeddings, and
directly using a fine-tuned clickbait detection model's output as a score.
"""
import os.path
import sys
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Type, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache
import pandas as pd
from tqdm import tqdm
from config import GENERAL_CONFIG, HEADLINE_CONTENT_CONFIG
from headline_content_models import (
    ClickbaitModelBase,
    ClickbaitTransformer,
    ClickbaitFeatureEnhancedTransformer
)
from headline_content_evaluation import evaluate_clickbait_predictions
from data.clickbait17.clickbait17_utils import get_dataset_folder, combined_headline
from data.clickbait_news_detection_dataset import prepare_cnd_dataset
import logging_config
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimilarityMethod(ABC):
    """Abstract Base Class for all similarity calculation methods."""

    @abstractmethod
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Computes a similarity score between a headline and content.

        Args:
            headline (str): The article headline.
            content (str): The main content of the article.
            post (str, optional): Associated social media post text.
                Defaults to None.
            headline_score (float, optional): A pre-computed score from a
                headline-only classifier, required by some models (e.g., hybrid).
                Defaults to None.

        Returns:
            A float representing the similarity score.
        """
        pass


class CosineSimilarityTFIDF(SimilarityMethod):
    """Computes similarity using TF-IDF vectors and cosine similarity.

    Attributes:
        vectorizer (TfidfVectorizer): The scikit-learn TF-IDF vectorizer.
    """

    def __init__(self, stop_words: Optional[str] = "english"):
        """Initializes the TF-IDF vectorizer.

        Args:
            stop_words (Optional[str], optional): The stop words setting for
                the vectorizer (e.g., 'english'). Defaults to "english".
        """
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    @lru_cache(maxsize=128)
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Calculates the cosine similarity of TF-IDF vectors.

        Args:
            headline (str): The article headline.
            content (str): The main article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): Not used by this method.

        Returns:
            The cosine similarity score as a float.
        """
        # Combine the headline and post for a richer representation.
        headline = combined_headline(headline=headline, post=post)
        if not headline or not content:
            logger.warning("No headline or content provided.")
            return 0.0
        try:
            # Create TF-IDF vectors for the headline and content.
            vectors = self.vectorizer.fit_transform([headline, content])
            # Compute and return the cosine similarity between the two vectors.
            return float(sk_cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except ValueError:
            # Handle cases where the vocabulary might be empty.
            return 0.0


class TransformerEmbeddingSimilarity(SimilarityMethod):
    """Computes similarity using sentence embeddings from a transformer model.

    Attributes:
        model_name (str): The name of the transformer model to use.
        max_length (int): The maximum sequence length for the tokenizer.
        tokenizer (AutoTokenizer): The Hugging Face tokenizer instance.
        model (AutoModel): The Hugging Face model instance.
    """

    def __init__(self, model_name: str = HEADLINE_CONTENT_CONFIG["model_name"],
                 max_length: int = HEADLINE_CONTENT_CONFIG["length_max"]):
        """Initializes the transformer model and tokenizer.

        Args:
            model_name (str, optional): The name or path of the transformer
                model. Defaults to the value in `HEADLINE_CONTENT_CONFIG`.
            max_length (int, optional): The maximum tokenization length.
                Defaults to the value in `HEADLINE_CONTENT_CONFIG`.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        """Loads the tokenizer and model on first use."""
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(DEVICE).eval()

    @lru_cache(maxsize=128)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generates a mean-pooled embedding for a given text.

        Args:
            text (str): The input text to embed.

        Returns:
            A numpy array representing the text embedding, or None if the
            input text is empty.
        """
        if not text:
            logger.warning("No text provided.")
            return None
        self._load_model()
        # Tokenize the input text.
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        # Get model outputs.
        with torch.no_grad():
            outputs = self.model(**inputs)
        # Perform mean pooling using the attention mask to ignore padding tokens.
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled.cpu().numpy().flatten()

    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Calculates cosine similarity between transformer embeddings.

        Args:
            headline (str): The article headline.
            content (str): The main article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): Not used by this method.

        Returns:
            The cosine similarity score as a float.
        """
        headline = combined_headline(headline=headline, post=post)
        h_embed = self._get_embedding(headline)
        c_embed = self._get_embedding(content)
        if h_embed is None or c_embed is None:
            return 0.0
        norm_h, norm_c = np.linalg.norm(h_embed), np.linalg.norm(c_embed)
        if norm_h == 0 or norm_c == 0:
            return 0.0
        # Compute cosine similarity and clip the value to handle floating point errors.
        return float(np.clip(np.dot(h_embed, c_embed) / (norm_h * norm_c), -1.0, 1.0))


class ClickbaitModelScore(SimilarityMethod):
    """Uses a fine-tuned clickbait model's output directly as a score.

    This class acts as a wrapper around the models defined in
    `headline_content_models.py`, allowing them to be used as a
    `SimilarityMethod`.
    """
    CLICKBAIT_MODEL_CLASSES: Dict[str, Type[ClickbaitModelBase]] = {
        "standard": ClickbaitTransformer,
        "hybrid": ClickbaitFeatureEnhancedTransformer,
    }

    def __init__(self, model_type: str, model_name_or_path: Optional[str] = None, **kwargs):
        """Initializes and loads the specified clickbait model.

        Args:
            model_type (str): The type of model to use ('standard' or 'hybrid').
            model_name_or_path (Optional[str], optional): The name or path of
                the model to load. If a path exists, the model is loaded from
                it. Defaults to None.
            **kwargs: Additional arguments passed to the model's constructor.

        Raises:
            ValueError: If an unknown `model_type` is provided.
        """
        if model_type not in self.CLICKBAIT_MODEL_CLASSES:
            raise ValueError(f"Unknown model_type '{model_type}'")
        ModelClass = self.CLICKBAIT_MODEL_CLASSES[model_type]

        self.model = ModelClass(**kwargs)

        # Load a pre-trained model if a valid path is provided.
        if model_name_or_path and os.path.exists(model_name_or_path):
            self.model.load_model(model_name_or_path)
        elif model_name_or_path:
            logger.warning(f"Model path specified but not found: {model_name_or_path}. Using an untrained model.")

    @lru_cache(maxsize=128)
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Computes the score by calling the underlying model's predict method.

        Args:
            headline (str): The article headline.
            content (str): The main article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): A headline-only score, required
                for the 'hybrid' model.

        Returns:
            The raw output score from the model as a float.

        Raises:
            ValueError: If the 'hybrid' model is used without a `headline_score`.
        """
        if not self.model.model:
            logger.warning("Inner model for ClickbaitModelScore is not loaded.")
            return float('nan')
        try:
            # The hybrid model requires the headline_score as an additional feature.
            if isinstance(self.model, ClickbaitFeatureEnhancedTransformer):
                if headline_score is None:
                    raise ValueError("The hybrid model requires a headline_score, but none was provided.")
                return float(self.model.predict(post=post, headline=headline, content=content,
                                                headline_score=headline_score))
            else:
                # The standard model's prediction does not require the extra score.
                return float(self.model.predict(post=post, headline=headline, content=content))
        except Exception as e:
            logger.error(f"Prediction failed for headline '{headline[:30]}...': {e}", exc_info=True)
            return float('nan')


class SimilarityMethodEvaluator:
    """A wrapper for evaluating non-trainable `SimilarityMethod` instances.

    This class makes methods like `CosineSimilarityTFIDF` compatible with the
    evaluation framework used for the trainable models by providing a `test`
    method that computes predictions over a dataset.
    """

    def __init__(self, method: SimilarityMethod, model_type: str):
        """Initializes the evaluator.

        Args:
            method (SimilarityMethod): The similarity method instance to evaluate.
            model_type (str): A name for the model type being evaluated.

        Raises:
            TypeError: If the provided method is not a `SimilarityMethod`.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("The provided method must be an instance of SimilarityMethod.")
        self.method = method
        self.model_type = model_type

    def test(self, test_csv: str) -> Tuple[Dict[str, float], list]:
        """Runs the similarity method over a test set and evaluates the scores.

        Args:
            test_csv (str): The path to the test CSV file.

        Returns:
            A tuple containing a dictionary of evaluation metrics and a list
            of the generated prediction scores.
        """
        logging.info(f"--- Evaluating non-trainable model: {self.model_type} ---")
        time_start = time.perf_counter()
        try:
            df = pd.read_csv(test_csv).dropna(subset=["headline", "content", "clickbait_score"])
            if "post" not in df.columns:
                df["post"] = ""
            df["post"] = df["post"].fillna("")
            if df.empty:
                logging.warning("Warning: Test dataframe is empty.")
                return {}, []
        except FileNotFoundError:
            logging.error(f"Error: Test file not found at {test_csv}")
            return {}, []

        predictions = []
        true_labels = list(df["clickbait_score"])

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Predicting with {self.model_type}"):
            score = self.method.compute_score(row["headline"], row["content"], post=row["post"])
            predictions.append(score)

        metrics = evaluate_clickbait_predictions(
            y_true=true_labels,
            y_pred=predictions,
            verbose=False,
            time_start=time_start
        )

        if hasattr(self.method, 'model') and isinstance(self.method.model, torch.nn.Module):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metrics, predictions


class HeadlineContentSimilarity:
    """A high-level wrapper for using a chosen similarity method."""

    def __init__(self, method: SimilarityMethod):
        """Initializes the comparator with a specific similarity method.

        Args:
            method (SimilarityMethod): The similarity method instance to use.

        Raises:
            TypeError: If the method does not inherit from `SimilarityMethod`.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    def set_method(self, method: SimilarityMethod):
        """Allows changing the similarity method at runtime.

        Args:
            method (SimilarityMethod): The new similarity method to use.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    def compare(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        """Computes similarity by delegating to the current method.

        Args:
            headline (str): The article headline.
            content (str): The main article content.
            post (str, optional): Associated social media post text.
            headline_score (float, optional): Headline-only clickbait score.

        Returns:
            The computed similarity score as a float.
        """
        import time
        start = time.perf_counter()
        score = self.method.compute_score(headline, content, post=post, headline_score=headline_score)
        elapsed = time.perf_counter() - start
        return score


if __name__ == "__main__":
    from utils import set_seed
    set_seed(GENERAL_CONFIG["seed"])
    transformers = ["sentence-transformers/all-MiniLM-L6-v2"]
    # tets = ClickbaitModelScore(model_type="standard", model_name_or_path="./models/transformer_bert-base-uncased_bert-base-uncased_1745798398/best_model")
    # print(tets.model.test("./data/clickbait17/models/bert-base-uncased/clickbait17_test.csv"))

    for transformer in transformers:
        # prepare_cnd_dataset(input_csv_path="data/clickbait_news_detection/raw/train.csv", output_dir="data/clickbait_news_detection/", clickbait17_train_meta_path="data/clickbait17/models/default/clickbait17_train_features_metadata.json", tokenizer_name=transformer)
        directory = get_dataset_folder(transformer)
        standard = ClickbaitModelScore(model_type="standard", model_name_or_path=HEADLINE_CONTENT_CONFIG["model_name"], output_directory="models/tets")
        #standard.model.load_model("models/tets_sentence-transformers_all-MiniLM-L6-v2_2025_07_06_13_26_30/best_model")
        # standard.model.train(os.path.join(directory, "clickbait17_train.csv"), os.path.join(directory, "clickbait17_validation.csv"))
        standard.model.train(os.path.join(directory, "clickbait17_train.csv"), use_weighted_loss=True)
        # sampling_strategy="oversample",
        # use_weighted_loss=True
        standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #standard.model.test(os.path.join(directory, "clickbait17_test_no_post.csv"))


        #hybrid = ClickbaitModelScore(model_type="hybrid", model_name_or_path=HEADLINE_CONTENT_CONFIG["model_name"])
        #hybrid.model.load_model("models/hybrid/best_model")
        #hybrid.model.train(os.path.join(directory, "clickbait17_train_features.csv"), os.path.join(directory, "clickbait17_validation_features.csv"), use_weighted_loss=True)
        #hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))
        # hybrid.model.test(os.path.join(directory, "clickbait17_test_features_no_post.csv"))
        # hybrid.model.test("data/clickbait_news_detection/custom_dataset_test_features.csv")


        #standard.model.train(os.path.join(directory, "clickbait17_train.csv"), os.path.join(directory, "clickbait17_validation.csv"))
        #standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #hybrid.model.train(os.path.join(directory, "clickbait17_train_features.csv"), os.path.join(directory, "clickbait17_validation_features.csv"))

        # standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #print("HYBRID")
        #hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))