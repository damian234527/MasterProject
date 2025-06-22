import os.path
import sys

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
from data.clickbait17.clickbait17_utils import get_dataset_folder
import logging_config
import logging

logger = logging.getLogger(__name__)
seed = GENERAL_CONFIG["seed"]
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================
# Similarity Method Interface
# ============================================

class SimilarityMethod(ABC):
    @abstractmethod
    def compute_score(self, headline: str, content: str) -> float:
        pass

# ============================================
# Concrete Similarity Implementations
# ============================================

class CosineSimilarityTFIDF(SimilarityMethod):
    def __init__(self, stop_words: Optional[str] = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    @lru_cache(maxsize=128)
    def compute_score(self, headline: str, content: str) -> float:
        if not headline or not content:
            logger.warning("No headline or content provided.")
            return 0.0
        try:
            vectors = self.vectorizer.fit_transform([headline, content])
            return float(sk_cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except ValueError:
            return 0.0

class TransformerEmbeddingSimilarity(SimilarityMethod):

    def __init__(self, model_name: str = HEADLINE_CONTENT_CONFIG["model_name"], max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(DEVICE).eval()

    @lru_cache(maxsize=128)
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text:
            logger.warning("No text provided.")
            return None
        self._load_model()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        return mean_pooled.cpu().numpy().flatten()

    def compute_score(self, headline: str, content: str) -> float:
        h_embed = self._get_embedding(headline)
        c_embed = self._get_embedding(content)
        if h_embed is None or c_embed is None:
            return 0.0
        norm_h, norm_c = np.linalg.norm(h_embed), np.linalg.norm(c_embed)
        if norm_h == 0 or norm_c == 0:
            return 0.0
        return float(np.clip(np.dot(h_embed, c_embed) / (norm_h * norm_c), -1.0, 1.0))

class ClickbaitModelScore(SimilarityMethod):
    CLICKBAIT_MODEL_CLASSES: Dict[str, Type[ClickbaitModelBase]] = {
        "standard": ClickbaitTransformer,
        "hybrid": ClickbaitFeatureEnhancedTransformer,
    }

    def __init__(self, model_type: str, model_name_or_path: Optional[str] = None, **kwargs):
        if model_type not in self.CLICKBAIT_MODEL_CLASSES:
            raise ValueError(f"Unknown model_type '{model_type}'")
        ModelClass = self.CLICKBAIT_MODEL_CLASSES[model_type]
        self.model = ModelClass(model_name_or_path=model_name_or_path, **kwargs)

    @lru_cache(maxsize=128)
    def compute_score(self, headline: str, content: str) -> float:
        if not self.model.model:
            logger.warning("Inner model for ClickbaitModelScore is not loaded.")
            return float('nan')
        try:
            return float(self.model.predict(post=headline, headline=headline, content=content))
        except Exception:
            return float('nan')
# ============================================
# Wrapper for Evaluation
# ============================================

class SimilarityMethodEvaluator:
    """
    A wrapper to make non-trainable SimilarityMethod instances compatible
    with the evaluation framework used for trainable models.
    """
    def __init__(self, method: SimilarityMethod, model_type: str):
        """
        Initializes the evaluator.

        Args:
            method (SimilarityMethod): The similarity method instance to evaluate (e.g., CosineSimilarityTFIDF).
            model_type (str): A descriptive name for the method for reporting purposes.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("The provided method must be an instance of SimilarityMethod.")
        self.method = method
        self.model_type = model_type

    def test(self, test_csv: str) -> Tuple[Dict[str, float], list]:
        """
        Evaluates the similarity method on a test dataset.
        Mimics the .test() method of the ClickbaitModelBase class.

        Args:
            test_csv (str): Path to the test CSV file.

        Returns:
            A tuple containing:
            - A dictionary of calculated metrics.
            - A list of the raw prediction scores.
        """
        logging.info(f"--- Evaluating non-trainable model: {self.model_type} ---")
        try:
            df = pd.read_csv(test_csv).dropna(subset=["headline", "content", "clickbait_score"])
            if df.empty:
                logging.warning("Warning: Test dataframe is empty.")
                return {}, []
        except FileNotFoundError:
            logging.error(f"Error: Test file not found at {test_csv}")
            return {}, []

        predictions = []
        true_labels = list(df["clickbait_score"])

        # Use tqdm for a progress bar, as some embedding models can be slow.
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Predicting with {self.model_type}"):
            score = self.method.compute_score(row["headline"], row["content"])
            predictions.append(score)

        # Use the existing evaluation function to get all metrics
        # Set verbose=False as the main script will print the summary
        metrics = evaluate_clickbait_predictions(
            y_true=true_labels,
            y_pred=predictions,
            verbose=False
        )

        # Clear GPU cache if the underlying method used it (e.g., TransformerEmbeddingSimilarity)
        if hasattr(self.method, 'model') and isinstance(self.method.model, torch.nn.Module):
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()

        return metrics, predictions

# ============================================
# Main Comparator
# ============================================

class HeadlineContentSimilarity:
    def __init__(self, method: SimilarityMethod):
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    def set_method(self, method: SimilarityMethod):
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    def compare(self, headline: str, content: str) -> float:
        import time
        start = time.perf_counter()
        score = self.method.compute_score(headline, content)
        elapsed = time.perf_counter() - start
        logging.info(f"Comparison took {elapsed:.4f} seconds.")
        return score

# ============================================
# End of streamlined headline_content_similarity.py
# ============================================

if __name__ == "__main__":
    transformers = ["sentence-transformers/all-MiniLM-L6-v2"]
    # tets = ClickbaitModelScore(model_type="standard", model_name_or_path="./models/transformer_bert-base-uncased_bert-base-uncased_1745798398/best_model")
    # print(tets.model.test("./data/clickbait17/models/bert-base-uncased/clickbait17_test.csv"))
    for transformer in transformers:
        directory = get_dataset_folder(transformer)
        standard = ClickbaitModelScore(model_type="standard", model_name_or_path=HEADLINE_CONTENT_CONFIG["model_name"])
        standard = ClickbaitModelScore(model_type="standard", model_name_or_path="models/standard_sentence-transformers_all-MiniLM-L6-v2_1750613557/best_model")

        #standard.model.train(os.path.join(directory, "clickbait17_train.csv"),
        #                     os.path.join(directory, "clickbait17_validation.csv"))
        #logger.info("Standard")
        # standard.model.train(os.path.join(directory, "clickbait17_train.csv"), sampling_strategy="oversample")
        # sampling_strategy="oversample",
        # use_weighted_loss=True
        logger.info("")
        standard.model.test(os.path.join(directory, "clickbait17_test.csv"))

        # hybrid = ClickbaitModelScore(model_type="hybrid", model_name_or_path=transformer)
        # hybrid.model.load_model("./models/hybrid/best_model")
        # --------------------------------------------
        # UNTRAINED
        #print("UNTRAINED")
        #print("STANDARD: ")
        #standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #print("HYBRID: ")
        #hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))
        # --------------------------------------------
        # TRAINED
        #standard.model.train(os.path.join(directory, "clickbait17_train.csv"), os.path.join(directory, "clickbait17_validation.csv"))
        #standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #hybrid.model.train(os.path.join(directory, "clickbait17_train_features.csv"), os.path.join(directory, "clickbait17_validation_features.csv"))

        # standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #print("HYBRID")
        #hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))