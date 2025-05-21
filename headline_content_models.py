import os
import time
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, PreTrainedModel
from typing import Union, List
import string
from torch import nn
from data.clickbait17.clickbait17_dataset import Clickbait17Dataset, Clickbait17FeatureAugmentedDataset
from headline_content_evaluation import evaluate_clickbait_predictions
from config import HEADLINE_CONTENT_CONFIG

class ClickbaitModelBase(ABC):
    """Abstract Base Class for Clickbait Detection Models."""
    def __init__(self, model_name: str, tokenizer_name: str, length_max: int, batch_size: int, epochs: int):
        """
        Initializes common attributes for clickbait models.

        Args:
            model_name (str): The name of the pre-trained transformer model.
            length_max (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            self.tokenizer = AutoTokenizer.from_pretrained(HEADLINE_CONTENT_CONFIG["tokenizer_name"])
        self.length_max = length_max
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None # To be initialized by subclasses

    @abstractmethod
    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data from a CSV file into a PyTorch Dataset."""
        pass

    @abstractmethod
    def train(self, train_csv: str, validation_csv: str):
        """Trains the model."""
        pass

    @abstractmethod
    def predict(self, post: str, headline: str, content: str) -> float:
        """Predicts the clickbait score for a single headline and content pair."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Loads a trained model from the specified path."""
        pass

    def test(self, test_csv: str) -> (float, List[float]):
        """
        Evaluates the model on a test dataset and returns MSE and predictions.

        Args:
            test_csv (str): Path to the test CSV file.

        Returns:
            tuple[float, List[float]]: A tuple containing the Mean Squared Error (MSE)
                                       on the test set and the list of predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        test_dataset = self._load_data(test_csv)
        # Use a default batch size if test_dataset is small to avoid errors
        effective_batch_size = min(self.batch_size, len(test_dataset))
        if effective_batch_size == 0:
             print("Warning: Test dataset is empty.")
             return float("nan"), []
        test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)

        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                # Move all tensor items in batch to the correct device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                features = batch.get("features") # Use .get() for optional features

                if features is not None:
                    features = features.to(self.device)
                    # Assumes model with features takes features as the third argument
                    output = self.model(input_ids, attention_mask, features)
                else:
                    # Assumes standard HF sequence classification model output structure
                    model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    output = model_output.logits.squeeze(-1) # Squeeze the last dimension

                predictions.extend(output.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        if not true_labels: # Handle empty dataset case after loop
            print("Warning: No data processed in test loop.")
            return float("nan"), []

        # mse = mean_squared_error(true_labels, predictions)
        # print(f"MSE for test set: {mse:.4f}")

        path = f"./results/{os.path.basename(self.model_name)}"
        metrics = evaluate_clickbait_predictions(true_labels, predictions, save_path=os.path.join(path, f"{self.model_name}_test_metrics.csv"))
        return metrics, predictions

# ============================================
# Subclass 1: Standard Transformer
# ============================================
class ClickbaitTransformer(ClickbaitModelBase):
    """Clickbait detection using a standard Transformer model (e.g., BERT)."""
    # model_name_default = os.getenv("MODEL_NAME", "bert-base-uncased") # Keep track of default base "google/bert_uncased_L-4_H-256_A-4" - smaller default example
    def __init__(self,
                 model_name_or_path: str = HEADLINE_CONTENT_CONFIG["model_name"],
                 tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"],
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"],
                 batch_size: int = HEADLINE_CONTENT_CONFIG["batch_size"],
                 epochs: int = HEADLINE_CONTENT_CONFIG["epochs"],
                 learning_rate = HEADLINE_CONTENT_CONFIG["learning_rate"],
                 weight_decay = HEADLINE_CONTENT_CONFIG["weight_decay"],
                 dropout_rate = HEADLINE_CONTENT_CONFIG["dropout_rate"],
                 fp16: bool = HEADLINE_CONTENT_CONFIG["fp16"],
                 output_directory: str = os.path.join(HEADLINE_CONTENT_CONFIG["output_directory"], "standard"),
                 **kwargs): # Allow passing extra args to from_pretrained
        """
        Initializes the standard Clickbait Transformer model.
        Can load base models or fine-tuned models from Hugging Face Hub or local path.

        Args:
            model_name_or_path (str): Name/path of the pre-trained transformer
                                      (e.g., "bert-base-uncased", "username/my-model", "./local_dir").
            length_max (int): Maximum sequence length.
            batch_size (int): Training/evaluation batch size.
            epochs (int): Number of training epochs.
            fp16 (bool): Whether to use mixed-precision training.
            output_directory (str): Directory to save model outputs and logs during training.
            **kwargs: Additional arguments passed to AutoModelForSequenceClassification.from_pretrained.
        """
        # Pass base class relevant args - model_name is used for tokenizer only here
        # The actual model loaded depends on model_name_or_path
        super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max, batch_size=batch_size, epochs=epochs)
        self.model_identifier = model_name_or_path # Store the identifier used
        self.test_run = kwargs.pop("test_run", False)
        try:
            print(f"Loading AutoModelForSequenceClassification from: {model_name_or_path}")
            # Initialize the specific model for sequence classification
            # This will load from Hub or local path directly
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
                **kwargs # Pass extra args like trust_remote_code if needed
            )
            self.model.to(self.device)  # Ensure model is on device after loading
        except OSError as e:
             model_default = HEADLINE_CONTENT_CONFIG["model_name"]
             print(f"Error loading model from {tokenizer_name}: {e}")
             print(f"Model initialization failed. self.model will be set as {model_default}.")
             self.model = model_default
        try:
            # Also reload tokenizer associated with the potentially fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
            print(f"Successfully loaded model and tokenizer from {tokenizer_name}")

        except OSError as e:
            # Attempt to load tokenizer with default base if model fails? Or fail completely?
            # Let's try loading tokenizer from the base name for basic functionality
            tokenizer_default = HEADLINE_CONTENT_CONFIG["tokenizer_name"]
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_default)
                print(f"Warning: Model loading failed, loaded tokenizer for {tokenizer_default}")
            except Exception as te:
                print(f"Error loading default tokenizer {tokenizer_default}: {te}")
                self.tokenizer = None  # Give up on tokenizer too

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.fp16 = fp16
        self.output_directory = f"{output_directory}_{self.model_identifier.replace("/", "_")}_{int(time.time())}"
        self.trainer = None

    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data using Clickbait17Dataset."""
        try:
            df = pd.read_csv(csv_path).dropna(subset=["post", "headline", "content", "clickbait_score"])
            if df.empty:
                 print(f"Warning: No valid data after dropping NaNs in {csv_path}")
            return Clickbait17Dataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {csv_path}: {e}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer, self.length_max)


    def train(self, train_csv: str, validation_csv: str) -> None:
        """Trains the model using the Hugging Face Trainer API."""
        data_train = self._load_data(train_csv)
        data_validation = self._load_data(validation_csv)

        if len(data_train) == 0 or len(data_validation) == 0:
            print("Error: Training or validation dataset is empty. Aborting training.")
            return

        training_args = TrainingArguments(
            output_dir="temp/model/standard/" if self.test_run else self.output_directory,
            evaluation_strategy="no" if self.test_run else "epoch",
            save_strategy="no" if self.test_run else "epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            fp16=self.fp16 and torch.cuda.is_available(), # Only enable fp16 if cuda is available
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=False if self.test_run else True,
            metric_for_best_model="eval_loss", # Regression task, use loss
            greater_is_better=False, # Lower loss is better
            report_to="none", # external reporting like wandb disabled

        )

        self.trainer = Trainer(
            model=self.model, # Already initialized in __init__
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation,
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            # No compute_metrics needed by default for Trainer's regression eval loss
        )

        print(f"Starting training for {self.epochs} epochs...")
        self.trainer.train()
        print("Training finished.")

        # Save the best model and tokenizer if not test run
        if not self.test_run:
            best_model_path = os.path.join(self.output_directory, "best_model")
            self.trainer.save_model(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            print(f"Best model saved to {best_model_path}")
            # Load the best model into self.model
            self.load_model(best_model_path)


    def predict(self, post: str, headline: str, content: str) -> float:
        """Predicts the clickbait score for a single headline/content pair."""
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()

        # Combine headline and content, let tokenizer handle truncation/padding
        # Using text pair encoding
        inputs = self.tokenizer(
            text=post,
            text_pair=content,
            return_tensors="pt",
            truncation="longest_first", # Truncate longest sequence first if needed
            padding=True,
            max_length=self.length_max
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Output for sequence classification is logits
            score = outputs.logits.squeeze().item()
        return score

    def load_model(self, model_path: str, **kwargs):
        """
        Loads a trained AutoModelForSequenceClassification model from a local directory.
        Typically used after saving with trainer.save_model() or model.save_pretrained().

        Args:
            model_path (str): Path to the directory containing the saved model files.
            **kwargs: Additional arguments for from_pretrained.
        """
        print(f"Loading model explicitly from local directory: {model_path}...")
        try:
            # This assumes model_path is a directory saved via save_pretrained
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs) # Load corresponding tokenizer
            self.model.to(self.device) # Move loaded model to device
            self.model_identifier = model_path # Update identifier
            print("Model loaded successfully from directory.")
        except OSError as e:
             print(f"Error loading model from directory {model_path}: {e}.")
             # Optionally reset model state
             # self.model = None
        except Exception as e:
            print(f"An unexpected error occurred during model loading from directory: {e}")
            # Optionally reset model state
            # self.model = None

# ============================================
# Subclass 2: Feature-Enhanced Transformer
# ============================================
class ClickbaitFeatureEnhancedTransformer(ClickbaitModelBase):
    """Clickbait detection using a Transformer enhanced with engineered features."""

    # Define the custom hybrid model architecture inside the class
    class HybridClickbaitModel(nn.Module):
        def __init__(self, transformer_name: str, num_features: int, dropout_rate: float = 0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(transformer_name)
            # Use config for hidden size to be more robust
            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(dropout_rate)

            self.feature_proj = nn.Sequential(
                nn.LayerNorm(num_features),
                nn.Linear(num_features, hidden_size),
                nn.ReLU()
            )

            # Simple regressor head
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size * 2, 128), # Combine BERT output and features
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2), # Optional extra dropout
                nn.Linear(128, 1)
            )

        def forward(self, input_ids, attention_mask, features):
            # Get embeddings from the base transformer model
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            try:
                cls_output = outputs.pooler_output
            except:
                print(f"outputs.pooler_output not available, using mean pooling instead")
                mask = attention_mask.unsqueeze(-1)
                sum_hidden = (outputs.last_hidden_state * mask).sum(1)
                avg_hidden = sum_hidden / mask.sum(1)
                cls_output = avg_hidden
            # Concatenate [CLS] output with the engineered features
            # combined = torch.cat((cls_output, features), dim=1)
            projected_features = self.feature_proj(features)
            combined = torch.cat((cls_output, projected_features), dim=1)
            # Apply dropout and pass through the regressor head
            x = self.dropout(combined)
            logits = self.regressor(x)
            # return logits.squeeze(-1)
            probs = torch.sigmoid(logits).squeeze(-1)  # map to [0, 1]
            return probs


    class HybridWrapperModel(PreTrainedModel):
        def __init__(self, hybrid_model, config):
            super().__init__(config)
            self.hybrid_model = hybrid_model

        def forward(self, input_ids=None, attention_mask=None, features=None, labels=None):
            logits = self.hybrid_model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = None
            if labels is not None:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits, labels)
            return {"loss": loss, "logits": logits}

    class HybridDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features):
            input_features = [{k: f[k] for k in ["input_ids", "attention_mask"]} for f in features]
            batch = self.tokenizer.pad(input_features, return_tensors="pt")
            batch["features"] = torch.stack([f["features"] for f in features])
            if "label" in features[0]:
                batch["labels"] = torch.tensor([f["label"] for f in features], dtype=torch.float)
            return batch

    NUM_FEATURES = 14 # TODO
    # model_name_default = os.getenv("MODEL_NAME", "bert-base-uncased")
    def __init__(self,
                 model_name_or_path: str = HEADLINE_CONTENT_CONFIG["model_name"],   # Base transformer name
                 tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"],
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"],
                 batch_size: int = HEADLINE_CONTENT_CONFIG["batch_size"],
                 epochs: int = HEADLINE_CONTENT_CONFIG["epochs"],
                 learning_rate: float = HEADLINE_CONTENT_CONFIG["learning_rate"],
                 weight_decay: float = HEADLINE_CONTENT_CONFIG["weight_decay"],
                 dropout_rate: float = HEADLINE_CONTENT_CONFIG["dropout_rate"],
                 output_directory: str = os.path.join(HEADLINE_CONTENT_CONFIG["output_directory"], "hybrid"),
                 **kwargs):
         # Use model_name for the base transformer in the hybrid model
         super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max, batch_size=batch_size, epochs=epochs)
         self.test_run = kwargs.pop("test_run", False)
         self.learning_rate = learning_rate
         self.weight_decay = weight_decay
         self.fp16 = HEADLINE_CONTENT_CONFIG["fp16"]
         self.output_directory = output_directory
         os.makedirs(self.output_directory, exist_ok=True)
         # Initialize the custom hybrid model using the specified base transformer name
         try:
             print(f"Initializing HybridClickbaitModel with base transformer: {model_name_or_path}")
             self.model = self.HybridClickbaitModel(
                 transformer_name=model_name_or_path, # Use the base model name here
                 num_features=self.NUM_FEATURES,
                 dropout_rate=dropout_rate
             ).to(self.device) # Move model to device during initialization
         except OSError as e:
             model_default = HEADLINE_CONTENT_CONFIG["model_name"]
             print(f"Error loading base transformer '{model_name_or_path}' for hybrid model: {e}. Using {model_default}")
             self.model = self.HybridClickbaitModel(
                 transformer_name=model_default, # Use the base model name here
                 num_features=self.NUM_FEATURES,
                 dropout_rate=dropout_rate).to(self.device)
         self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)

    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data using Clickbait17FeatureAugmentedDataset."""
        try:
             # Feature Augmented Dataset handles NaN internally, no need to drop here
             df = pd.read_csv(csv_path)
             return Clickbait17FeatureAugmentedDataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            return Clickbait17FeatureAugmentedDataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {csv_path}: {e}")
            return Clickbait17FeatureAugmentedDataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str):
        train_dataset = self._load_data(train_csv)
        val_dataset = self._load_data(validation_csv)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Error: Training or validation dataset is empty. Aborting training.")
            return

        wrapper_model = self.HybridWrapperModel(self.model, self.model.bert.config)

        training_args = TrainingArguments(
            output_dir="temp/model/hybrid/" if self.test_run else self.output_directory,
            evaluation_strategy="no" if self.test_run else "epoch",
            save_strategy="no" if self.test_run else "epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=False if self.test_run else True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.fp16 and torch.cuda.is_available(),
            report_to="none",
            max_grad_norm=1.0
        )

        self.trainer = Trainer(
            model=wrapper_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.HybridDataCollator(self.tokenizer),
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        self.trainer.train()

        if not self.test_run:
            best_model_path = os.path.join(self.output_directory, "best_model")
            self.trainer.save_model(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            self.load_model(os.path.join(best_model_path, "pytorch_model.bin"))

    def predict(self, post: str, headline: str, content: str) -> float:
        """Predicts the clickbait score using the hybrid model."""
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")
        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()
        # Need to create a temporary dataset/dataframe to run feature extraction
        # Include a dummy label (e.g., 0.0) as expected by the dataset
        temp_data = pd.DataFrame([{"post": post, "headline": headline, "content": content, "clickbait_score": 0.0}])
        # Use the same dataset class used for training to ensure consistent feature extraction
        temp_dataset = Clickbait17FeatureAugmentedDataset(temp_data, self.tokenizer, self.length_max)

        if len(temp_dataset) == 0:
            print("Warning: Could not process input for prediction.")
            return float("nan")

        item = temp_dataset[0] # Get the single processed item

        # Prepare tensors and move to device (add batch dimension)
        input_ids = item["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
        features = item["features"].unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features)

        return output.item() # Output is already squeezed by the model's forward pass

    def load_model(self, path: str = "models/hybrid/best_model.pt"):
        """Loads the trained model's state dictionary from a .pt file."""
        if self.model is None:
            print("Error: Model architecture not initialized.")
            return
        print(f"Loading model from {path}")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Failed to load model: {e}")
