import time
import os
import pandas as pd
import numpy as np
import torch
import re
from keras.src.ops import dtype
# from pipeline import model_name
from sympy import resultant
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from thinc.util import data_validation
from torch.utils.data import Dataset, random_split
from transformers import PreTrainedTokenizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from typing import Optional, Union, List, Dict

class Clickbait17Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, tokenizer: PreTrainedTokenizer, length_max: int = 512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.length_max = length_max

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict:
        row = self.data.iloc[index]
        encodings = self.tokenizer(row["post"], row["content"], truncation=True, padding="max_length", max_length=self.length_max, return_tensors="pt")
        item = {key: value.squeeze() for key, value in encodings.items()}
        item["labels"] = torch.tensor([row["clickbait_score"]], dtype=torch.float)
        return item

class FeatureExtractor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def extract_features(self, headline: str, content: str, similarity_score: float) -> Dict[str, float]:
        headline_words = headline.split()
        content_words = content.split()

        return {
            "headline_length_words": len(headline_words),
            "headline_length_chars": len(headline),
            "content_length_words": len(content_words),
            "content_length_chars": len(content),
            "headline_to_content_length_ratio": len(headline_words) / max(len(content_words), 1),
            "exclamation_count": headline.count("!"),
            "question_mark_count": headline.count("?"),
            "uppercase_ratio": sum(c.isupper() for c in headline) / max(len(headline), 1),
            "stopword_ratio_headline": sum(w.lower() in self.stop_words for w in headline_words) / max(len(headline_words), 1),
            "clickbait_word_count": len(re.findall(r"\b(shocking|unbelievable|you won't believe|top \d+|must see)\b", headline.lower())),
            "sentiment_diff": abs(self.sentiment_analyzer.polarity_scores(headline)["compound"] -
                                  self.sentiment_analyzer.polarity_scores(content)["compound"]),
            "similarity_score": similarity_score
        }

    def download_nltk_resources():
        import nltk
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
