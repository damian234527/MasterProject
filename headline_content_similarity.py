import numpy as np
import pandas as pd
import torch
from typing import List
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
from headline_content_transformer import ClickbaitTransformer  # <-- import your new transformer

class SimilarityMethod(ABC):
    """Abstract class for all methods of checking the similarity of headline and content of the article"""

    @abstractmethod
    def computeSimilarity(self, headline: str, content: str) -> float:
        pass


class CosineSimilarity(SimilarityMethod):
    """Cosine similarity using Term Frequency-Inverse Document Frequency (TF-IDF)"""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")

    def computeSimilarity(self, headline: str, content: str) -> float:
        vectors = self.vectorizer.fit_transform([headline, content])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        return float(similarity)


class TransformerSimilarity(SimilarityMethod):
    """Transformer-based similarity using sentence embeddings"""

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _embed(self, text: str) -> np.ndarray:
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        tokens = {key: value.to(self.device) for key, value in tokens.items()}
        with torch.no_grad():
            output = self.model(**tokens)
            embeddings = output.last_hidden_state.mean(dim=1)  # mean pooling
        return embeddings.cpu().numpy().flatten()

    def computeSimilarity(self, headline: str, content: str) -> float:
        headline_embed = self._embed(headline)
        content_embed = self._embed(content)
        similarity = np.dot(headline_embed, content_embed) / (
            np.linalg.norm(headline_embed) * np.linalg.norm(content_embed)
        )
        return float(similarity)


class ClickbaitTransformerSimilarity(SimilarityMethod):
    """Wrapper around ClickbaitTransformer for use in similarity interface"""

    def __init__(self, model_path: str):
        self.model = ClickbaitTransformer()
        self.model.load_model(model_path)

    def computeSimilarity(self, headline: str, content: str) -> float:
        score = self.model.predict(headline, content)
        # Optional: Normalize score to range [0, 1] if necessary
        return float(score)


class HeadlineContentSimilarity:
    """Compares headline to the content of the article"""

    def __init__(self, method: SimilarityMethod):
        self.method = method

    def setMethod(self, method: SimilarityMethod):
        """To change the method of calculating similarity"""
        self.method = method

    def compare(self, headline: str, content: str) -> float:
        """Returns similarity score from 0 to 1"""
        return self.method.computeSimilarity(headline, content)


if __name__ == "__main__":
    headline = "This headline is crazy!"
    content = "Here's the actual article body."

    method_cosine = CosineSimilarity()
    #method_transformer = TransformerSimilarity("distilbert-base-uncased")
    method_clickbait_transformer = ClickbaitTransformerSimilarity("bert-clickbait-regression/")

    comparator = HeadlineContentSimilarity(method_cosine)
    #print(f"Cosine TF-IDF similarity: {comparator.compare(headline, content):.4f}")

    #comparator.setMethod(method_transformer)
    #print(f"Embedding Transformer similarity: {comparator.compare(headline, content):.4f}")

    comparator.setMethod(method_clickbait_transformer)
    print(f"ClickbaitTransformer predicted score: {comparator.compare(headline, content):.4f}")
