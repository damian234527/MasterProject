import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Optional, Type, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from transformers import AutoTokenizer, AutoModel
from functools import lru_cache

from headline_content_models import (
    ClickbaitModelBase,
    ClickbaitTransformer,
    ClickbaitFeatureEnhancedTransformer
)

# Set seeds and deterministic behavior
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
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
            return 0.0
        try:
            vectors = self.vectorizer.fit_transform([headline, content])
            return float(sk_cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
        except ValueError:
            return 0.0

class TransformerEmbeddingSimilarity(SimilarityMethod):
    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512):
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
        "transformer": ClickbaitTransformer,
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
            return float('nan')
        try:
            return float(self.model.predict(headline, content))
        except Exception:
            return float('nan')

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
        print(f"Comparison took {elapsed:.4f} seconds.")
        return score

# ============================================
# End of streamlined headline_content_similarity.py
# ============================================

if __name__ == "__main__":
    tets = ClickbaitModelScore(model_type="transformer", model_name_or_path="./models/transformer_bert-base-uncased_bert-base-uncased_1745798398/best_model")
    print(tets.model.test("./data/clickbait17/models/bert-base-uncased/clickbait17_test.csv"))