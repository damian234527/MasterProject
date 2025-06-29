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


class SimilarityMethod(ABC):
    @abstractmethod
    # MODIFIED: Added optional 'headline_score' argument
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        pass


class CosineSimilarityTFIDF(SimilarityMethod):
    def __init__(self, stop_words: Optional[str] = "english"):
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    @lru_cache(maxsize=128)
    # MODIFIED: Signature updated to match the abstract base class
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
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

    # MODIFIED: Signature updated to match the abstract base class
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
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

        self.model = ModelClass(**kwargs)

        if model_name_or_path and os.path.exists(model_name_or_path):
            self.model.load_model(model_name_or_path)
        elif model_name_or_path:
            logger.warning(f"Model path specified but not found: {model_name_or_path}. Using an untrained model.")

    @lru_cache(maxsize=128)
    # MODIFIED: Signature updated to accept headline_score
    def compute_score(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        if not self.model.model:
            logger.warning("Inner model for ClickbaitModelScore is not loaded.")
            return float('nan')
        try:
            effective_post = post if post else headline

            # MODIFIED: If the model is the hybrid version, pass the headline_score to its predict method.
            if isinstance(self.model, ClickbaitFeatureEnhancedTransformer):
                if headline_score is None:
                    raise ValueError("The hybrid model requires a headline_score, but none was provided.")
                return float(self.model.predict(post=effective_post, headline=headline, content=content,
                                                headline_score=headline_score))
            else:
                # Standard model is called without the extra parameter.
                return float(self.model.predict(post=effective_post, headline=headline, content=content))

        except Exception as e:
            logger.error(f"Prediction failed for headline '{headline[:30]}...': {e}", exc_info=True)
            return float('nan')


class SimilarityMethodEvaluator:
    # ... (No changes to this class)
    """
    A wrapper to make non-trainable SimilarityMethod instances compatible
    with the evaluation framework used for trainable models.
    """

    def __init__(self, method: SimilarityMethod, model_type: str):
        if not isinstance(method, SimilarityMethod):
            raise TypeError("The provided method must be an instance of SimilarityMethod.")
        self.method = method
        self.model_type = model_type

    def test(self, test_csv: str) -> Tuple[Dict[str, float], list]:
        logging.info(f"--- Evaluating non-trainable model: {self.model_type} ---")
        try:
            df = pd.read_csv(test_csv).dropna(subset=["headline", "content", "clickbait_score"])
            if 'post' not in df.columns:  # Ensure post column exists
                df['post'] = ''
            df['post'] = df['post'].fillna(df['headline'])  # Fallback for missing posts
            if df.empty:
                logging.warning("Warning: Test dataframe is empty.")
                return {}, []
        except FileNotFoundError:
            logging.error(f"Error: Test file not found at {test_csv}")
            return {}, []

        predictions = []
        true_labels = list(df["clickbait_score"])

        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Predicting with {self.model_type}"):
            # This evaluation for non-trainable methods won't use the headline score by default,
            # as it requires a headline classifier. The main evaluation is in main.py.
            score = self.method.compute_score(row["headline"], row["content"], post=row["post"])
            predictions.append(score)

        metrics = evaluate_clickbait_predictions(
            y_true=true_labels,
            y_pred=predictions,
            verbose=False
        )

        if hasattr(self.method, 'model') and isinstance(self.method.model, torch.nn.Module):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return metrics, predictions


class HeadlineContentSimilarity:
    def __init__(self, method: SimilarityMethod):
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    def set_method(self, method: SimilarityMethod):
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Method must inherit from SimilarityMethod")
        self.method = method

    # MODIFIED: Added headline_score parameter
    def compare(self, headline: str, content: str, post: str = None, headline_score: float = None) -> float:
        import time
        start = time.perf_counter()
        # MODIFIED: Pass the score down to the specific similarity method implementation
        score = self.method.compute_score(headline, content, post=post, headline_score=headline_score)
        elapsed = time.perf_counter() - start
        return score


if __name__ == "__main__":
    transformers = ["sentence-transformers/all-MiniLM-L6-v2"]
    # tets = ClickbaitModelScore(model_type="standard", model_name_or_path="./models/transformer_bert-base-uncased_bert-base-uncased_1745798398/best_model")
    # print(tets.model.test("./data/clickbait17/models/bert-base-uncased/clickbait17_test.csv"))
    for transformer in transformers:
        directory = get_dataset_folder(transformer)
        # standard = ClickbaitModelScore(model_type="standard", model_name_or_path=HEADLINE_CONTENT_CONFIG["model_name"])
        #standard = ClickbaitModelScore(model_type="standard", model_name_or_path="models/standard_sentence-transformers_all-MiniLM-L6-v2_1750613557/best_model")

        # standard.model.train(os.path.join(directory, "clickbait17_train.csv"),
        #                     os.path.join(directory, "clickbait17_validation.csv"))
        #logger.info("Standard")
        # standard.model.train(os.path.join(directory, "clickbait17_train.csv"), sampling_strategy="oversample")
        # sampling_strategy="oversample",
        # use_weighted_loss=True
        #logger.info("")
        # standard.model.test(os.path.join(directory, "clickbait17_test.csv"))

        hybrid = ClickbaitModelScore(model_type="hybrid", model_name_or_path=HEADLINE_CONTENT_CONFIG["model_name"])
        #hybrid.model.load_model("models/hybrid/best_model")
        # --------------------------------------------
        # UNTRAINED
        #print("UNTRAINED")
        #print("STANDARD: ")
        #standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #print("HYBRID: ")
        hybrid.model.train(os.path.join(directory, "clickbait17_train_features.csv"))
        hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))
        # --------------------------------------------
        # TRAINED
        #standard.model.train(os.path.join(directory, "clickbait17_train.csv"), os.path.join(directory, "clickbait17_validation.csv"))
        #standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #hybrid.model.train(os.path.join(directory, "clickbait17_train_features.csv"), os.path.join(directory, "clickbait17_validation_features.csv"))

        # standard.model.test(os.path.join(directory, "clickbait17_test.csv"))
        #print("HYBRID")
        #hybrid.model.test(os.path.join(directory, "clickbait17_test_features.csv"))