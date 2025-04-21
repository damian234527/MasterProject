import os
import time
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments
from typing import Union, List
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import string
from torch import nn

nltk.download("stopwords")
nltk.download("vader_lexicon")


class ClickbaitModelBase(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def _load_data(self, csv_path: str) -> Dataset:
        pass

    @abstractmethod
    def predict(self, headline: str, content: str) -> float:
        pass

    @abstractmethod
    def load_model(self, path: str):
        pass

    def train(self, train_csv: str, validation_csv: str):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def test(self, test_csv: str) -> List[float]:
        test_dataset = self._load_data(test_csv)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)

        self.model.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch.get("features")
                if features is not None:
                    features = features.to(self.device)
                label = batch["label"].to(self.device)

                if features is not None:
                    output = self.model(input_ids, attention_mask, features)
                else:
                    output = self.model(input_ids=input_ids, attention_mask=attention_mask).logits.squeeze()

                predictions.extend(output.cpu().numpy())
                labels.extend(label.cpu().numpy())

        mse = mean_squared_error(labels, predictions)
        print(f"MSE for test set: {mse:.4f}")
        return predictions


class ClickbaitTransformer(ClickbaitModelBase):
    def __init__(self, model_name: str = "bert-base-uncased", length_max: int = 512, batch_size: int = 64, epochs: int = 5, fp16: bool = True, output_directory: str = "TransformerOutput"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")
        self.length_max = length_max
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16 = fp16
        self.output_directory = output_directory + str(time.time())
        self.trainer = None

    def _load_data(self, csv_path: str) -> Dataset:
        from dataset_clickbait17 import Clickbait17Dataset
        df = pd.read_csv(csv_path).dropna()
        return Clickbait17Dataset(df, self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str) -> None:
        data_train = self._load_data(train_csv)
        data_validation = self._load_data(validation_csv)

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=.01,
            fp16=self.fp16,
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation
        )

        trainer.train()
        trainer.save_model(self.output_directory)
        self.tokenizer.save_pretrained(self.output_directory)
        self.trainer = trainer

    def predict(self, headline: str, content: str) -> float:
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        inputs = self.tokenizer(headline, content, return_tensors="pt", truncation=True, padding=True, max_length=self.length_max)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()
        return score

    def load_model(self, model_directory: str):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_directory)


class ClickbaitHybrid(ClickbaitModelBase):
    class HybridClickbaitModel(nn.Module):
        def __init__(self, transformer_name: str, num_features: int):
            super().__init__()
            self.bert = AutoModel.from_pretrained(transformer_name)
            self.dropout = nn.Dropout(0.3)
            self.regressor = nn.Sequential(
                nn.Linear(self.bert.config.hidden_size + num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

        def forward(self, input_ids, attention_mask, features):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls_output = outputs.last_hidden_state[:, 0, :]
            combined = torch.cat((cls_output, features), dim=1)
            x = self.dropout(combined)
            return self.regressor(x).squeeze()

    class ClickbaitHybridDataset(Dataset):
        def __init__(self, dataframe: pd.DataFrame, tokenizer, length_max: int = 512):
            self.tokenizer = tokenizer
            self.length_max = length_max
            self.data = dataframe.dropna().reset_index(drop=True)
            self.stopwords = set(stopwords.words("english"))
            self.sentiment = SentimentIntensityAnalyzer()

        def __len__(self):
            return len(self.data)

        def _extract_features(self, headline: str, content: str) -> List[float]:
            def count_punctuation(text):
                return sum([1 for char in text if char in string.punctuation])

            headline_len = len(headline.split())
            content_len = len(content.split())
            headline_punct = count_punctuation(headline)
            content_punct = count_punctuation(content)
            headline_sentiment = self.sentiment.polarity_scores(headline)["compound"]
            content_sentiment = self.sentiment.polarity_scores(content)["compound"]

            return [
                headline_len,
                content_len,
                headline_punct,
                content_punct,
                headline_sentiment,
                content_sentiment,
            ]

        def __getitem__(self, idx):
            row = self.data.iloc[idx]
            headline, content, label = row["headline"], row["content"], row["clickbait"]

            features = self._extract_features(headline, content)

            encoding = self.tokenizer(
                headline,
                content,
                truncation=True,
                padding="max_length",
                max_length=self.length_max,
                return_tensors="pt"
            )

            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "features": torch.tensor(features, dtype=torch.float),
                "label": torch.tensor(label, dtype=torch.float)
            }

    def __init__(self, model_name: str = "bert-base-uncased", length_max: int = 512, batch_size: int = 32, epochs: int = 5, lr: float = 2e-5):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.length_max = length_max
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.HybridClickbaitModel(transformer_name=model_name, num_features=6).to(self.device)

    def _load_data(self, csv_path: str) -> Dataset:
        df = pd.read_csv(csv_path)
        return self.ClickbaitHybridDataset(df, self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str):
        train_dataset = self._load_data(train_csv)
        val_dataset = self._load_data(validation_csv)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs} - Training Loss: {total_loss/len(train_loader):.4f}")

        os.makedirs("HybridOutput", exist_ok=True)
        torch.save(self.model.state_dict(), "HybridOutput/model.pt")

    def predict(self, headline: str, content: str) -> float:
        self.model.eval()
        example = pd.DataFrame([{"headline": headline, "content": content, "clickbait": 0.0}])
        dataset = self.ClickbaitHybridDataset(example, self.tokenizer, self.length_max)
        item = dataset[0]
        input_ids = item["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
        features = item["features"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features)
        return output.item()

    def load_model(self, path: str = "HybridOutput/model.pt"):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
