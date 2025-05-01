import os
import pandas as pd
from transformers import AutoTokenizer
from clickbait17_preprocess import dataset17_create_csv
from clickbait17_dataset import Clickbait17FeatureAugmentedDataset  # replace with actual file name

# OLD; DELETE
subsets = ["train", "validation", "test"]
tokenizer_name = os.getenv("TOKENIZER_NAME", "bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

for subset in subsets:
    path = os.path.join(os.path.dirname(__file__), subset)
    df = dataset17_create_csv(subset)
    dataset = Clickbait17FeatureAugmentedDataset(df, tokenizer)
    dataset.save_with_features("clickbait17_" + str(subset) + "_features.csv")
