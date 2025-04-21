import time
import os
import pandas as pd
import numpy as np
import torch
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, PreTrainedTokenizer, PreTrainedModel
from dataset_clickbait17 import Clickbait17Dataset
from typing import Optional, Union, List, Dict

class ClickbaitTransformer:
    def __init__(self, model_name: str = "bert-base-uncased", length_max: int = 512, batch_size: int = 64, epochs: int = 5, fp16: bool = True, output_directory: str = "TransformerOutput"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, problem_type="regression")
        self.length_max = length_max
        self.batch_size = batch_size
        self.epochs = epochs
        self.fp16 = fp16
        self.trainer = None
        self.output_directory = output_directory + str(time.time())

    def _load_dataset(self, csv_path: str) -> Clickbait17Dataset:
        df = pd.read_csv(csv_path).dropna()
        return Clickbait17Dataset(df, self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str) -> None:
        data_train = self._load_dataset(train_csv)
        data_validation = self._load_dataset(validation_csv)

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

        def compute_metrics(predictions) -> dict:
            labels = predictions.label_ids.squeeze()
            predictions = predictions.predictions.squeeze()
            mse = mean_squared_error(labels, predictions)
            return {"mse": mse}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation
        )

        trainer.train()
        trainer.save_model(self.output_directory)
        self.tokenizer.save_pretrained(self.output_directory)

    def test(self, test_csv: str) -> Union[List[float], torch.Tensor]:
        if not self.trainer:
            self.trainer = Trainer(model=self.model)
        data_test = self._load_dataset(test_csv)
        result = self.trainer.predict(data_test)
        predictions = result.predictions.squeeze()
        labels = result.label_ids.squeeze()
        mse = mean_squared_error(labels, predictions)
        print(f"MSE for test set: {mse:.4f}")
        return predictions

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

    def load_model(self, model_directory: str) -> None:
        # self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_directory)



if __name__ == "__main__":
    data_path = "data/clickbait17"
    data_train_name = "clickbait17_train.csv"
    data_validation_name = "clickbait17_validation.csv"
    data_test_name = "clickbait17_test.csv"

    transformer = ClickbaitTransformer()
    # transformer.train(os.path.join(data_path, data_train_name), os.path.join(data_path, data_validation_name))


    model_path = "./bert-clickbait-regression/"
    transformer.load_model(model_path)
    # transformer.test(os.path.join(data_path, data_test_name))

    score = transformer.predict("This headline is crazy!", "Here's the actual article body.")
    print(f"Predicted score: {score:.4f}")