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
import string
from torch import nn
from data.clickbait17.clickbait17_dataset import Clickbait17Dataset, Clickbait17FeatureAugmentedDataset
from headline_content_evaluation import evaluate_clickbait_predictions

class ClickbaitModelBase(ABC):
    """Abstract Base Class for Clickbait Detection Models."""
    def __init__(self, model_name: str, length_max: int, batch_size: int, epochs: int):
        """
        Initializes common attributes for clickbait models.

        Args:
            model_name (str): The name of the pre-trained transformer model.
            length_max (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
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
    def predict(self, headline: str, content: str) -> float:
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

        mse = mean_squared_error(true_labels, predictions)
        print(f"MSE for test set: {mse:.4f}")


        metrics = evaluate_clickbait_predictions(true_labels, predictions, save_path="test_metrics.csv")
        print(f"MSE old: {mse}")
        print(f"New metrics: \n{metrics}")
        return metrics, predictions


# ============================================
# Subclass 1: Standard Transformer
# ============================================
class ClickbaitTransformer(ClickbaitModelBase):
    """Clickbait detection using a standard Transformer model (e.g., BERT)."""
    # DEFAULT_BASE_MODEL = "bert-base-uncased" # Keep track of default base
    DEFAULT_BASE_MODEL = "google/bert_uncased_L-4_H-256_A-4" # Smaller default example

    def __init__(self,
                 # Changed model_name -> model_name_or_path
                 model_name_or_path: str = DEFAULT_BASE_MODEL,
                 length_max: int = 512,
                 batch_size: int = 64,
                 epochs: int = 3,
                 fp16: bool = True,
                 output_directory: str = "TransformerOutput",
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
        super().__init__(model_name=model_name_or_path, length_max=length_max, batch_size=batch_size, epochs=epochs)
        self.model_identifier = model_name_or_path # Store the identifier used

        try:
            print(f"Loading AutoModelForSequenceClassification from: {model_name_or_path}")
            # Initialize the specific model for sequence classification
            # This will load from Hub or local path directly
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                problem_type="regression",
                **kwargs # Pass extra args like trust_remote_code if needed
            )
            self.model.to(self.device) # Ensure model is on device after loading
            # Also reload tokenizer associated with the potentially fine-tuned model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
            print(f"Successfully loaded model and tokenizer from {model_name_or_path}")

        except OSError as e:
             print(f"Error loading model from {model_name_or_path}: {e}")
             print("Model initialization failed. self.model will be None.")
             self.model = None
             # Attempt to load tokenizer with default base if model fails? Or fail completely?
             # Let's try loading tokenizer from the base name for basic functionality
             try:
                 self.tokenizer = AutoTokenizer.from_pretrained(self.DEFAULT_BASE_MODEL)
                 print(f"Warning: Model loading failed, loaded tokenizer for {self.DEFAULT_BASE_MODEL}")
             except Exception as te:
                 print(f"Error loading default tokenizer {self.DEFAULT_BASE_MODEL}: {te}")
                 self.tokenizer = None # Give up on tokenizer too

        self.fp16 = fp16
        self.output_directory = f"{output_directory}_{self.model_identifier.replace("/", "_")}_{int(time.time())}"
        self.trainer = None

    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data using Clickbait17Dataset."""
        try:
            df = pd.read_csv(csv_path).dropna(subset=["headline", "content", "clickbait"])
            if df.empty:
                 print(f"Warning: No valid data after dropping NaNs in {csv_path}")
            return Clickbait17Dataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["headline", "content", "clickbait"]), self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {csv_path}: {e}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["headline", "content", "clickbait"]), self.tokenizer, self.length_max)


    def train(self, train_csv: str, validation_csv: str) -> None:
        """Trains the model using the Hugging Face Trainer API."""
        data_train = self._load_data(train_csv)
        data_validation = self._load_data(validation_csv)

        if len(data_train) == 0 or len(data_validation) == 0:
            print("Error: Training or validation dataset is empty. Aborting training.")
            return

        training_args = TrainingArguments(
            output_dir=self.output_directory,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=0.01,
            fp16=self.fp16 and torch.cuda.is_available(), # Only enable fp16 if cuda is available
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss", # Regression task, use loss
            greater_is_better=False, # Lower loss is better
            report_to="none" # Disable external reporting like wandb if not needed
        )

        self.trainer = Trainer(
            model=self.model, # Already initialized in __init__
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation,
            # No compute_metrics needed by default for Trainer's regression eval loss
        )

        print(f"Starting training for {self.epochs} epochs...")
        self.trainer.train()
        print("Training finished.")

        # Save the best model and tokenizer
        best_model_path = os.path.join(self.output_directory, "best_model")
        self.trainer.save_model(best_model_path)
        self.tokenizer.save_pretrained(best_model_path)
        print(f"Best model saved to {best_model_path}")
        # Load the best model into self.model
        self.load_model(best_model_path)


    def predict(self, headline: str, content: str) -> float:
        """Predicts the clickbait score for a single headline/content pair."""
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()

        # Combine headline and content, let tokenizer handle truncation/padding
        # Using text pair encoding
        inputs = self.tokenizer(
            text=headline,
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
            # Simple regressor head
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size + num_features, 128), # Combine BERT output and features
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2), # Optional extra dropout
                nn.Linear(128, 1)
            )

        def forward(self, input_ids, attention_mask, features):
            # Get embeddings from the base transformer model
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            # Use the [CLS] token's output representation
            cls_output = outputs.last_hidden_state[:, 0, :]
            # Concatenate [CLS] output with the engineered features
            combined = torch.cat((cls_output, features), dim=1)
            # Apply dropout and pass through the regressor head
            x = self.dropout(combined)
            logits = self.regressor(x)
            return logits.squeeze(-1) # Squeeze the last dimension

    # Fixed number of features expected by this implementation
    NUM_FEATURES = 11
    DEFAULT_BASE_MODEL = "google/bert_uncased_L-4_H-256_A-4" # Smaller default example

    def __init__(self,
                 model_name: str = DEFAULT_BASE_MODEL, # Base transformer name
                 length_max: int = 512,
                 batch_size: int = 32,
                 epochs: int = 5,
                 lr: float = 2e-5,
                 dropout_rate: float = 0.3,
                 output_dir: str = "HybridOutput"):
         # Use model_name for the base transformer in the hybrid model
         super().__init__(model_name, length_max, batch_size, epochs)
         self.lr = lr
         self.output_dir = output_dir
         os.makedirs(self.output_dir, exist_ok=True)

         # Initialize the custom hybrid model using the specified base transformer name
         try:
             print(f"Initializing HybridClickbaitModel with base transformer: {model_name}")
             self.model = self.HybridClickbaitModel(
                 transformer_name=model_name, # Use the base model name here
                 num_features=self.NUM_FEATURES,
                 dropout_rate=dropout_rate
             ).to(self.device) # Move model to device during initialization
         except OSError as e:
             print(f"Error loading base transformer '{model_name}' for hybrid model: {e}")
             self.model = None

    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data using Clickbait17FeatureAugmentedDataset."""
        try:
             # Feature Augmented Dataset handles NaN internally, no need to drop here
             df = pd.read_csv(csv_path)
             return Clickbait17FeatureAugmentedDataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            return Clickbait17FeatureAugmentedDataset(pd.DataFrame(columns=["headline", "content", "clickbait"]), self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {csv_path}: {e}")
            return Clickbait17FeatureAugmentedDataset(pd.DataFrame(columns=["headline", "content", "clickbait"]), self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str):
        """Trains the hybrid model using a custom PyTorch training loop."""
        train_dataset = self._load_data(train_csv)
        val_dataset = self._load_data(validation_csv)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
             print("Error: Training or validation dataset is empty. Aborting training.")
             return

        # Ensure batch sizes are not larger than datasets
        train_batch_size = min(self.batch_size, len(train_dataset))
        val_batch_size = min(self.batch_size, len(val_dataset))
        if train_batch_size == 0 or val_batch_size == 0:
             print("Error: Effective batch size is zero. Aborting training.")
             return

        train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=val_batch_size) # No shuffle for validation

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss() # Mean Squared Error for regression

        best_val_loss = float("inf")
        model_save_path = os.path.join(self.output_dir, "best_model.pt")

        print(f"Starting training for {self.epochs} epochs on device {self.device}...")

        for epoch in range(self.epochs):
            self.model.train()
            total_train_loss = 0
            for batch in train_loader:
                # Move batch items to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                features = batch["features"].to(self.device)
                labels = batch["label"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask, features)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validation phase
            self.model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    features = batch["features"].to(self.device)
                    labels = batch["label"].to(self.device)

                    outputs = self.model(input_ids, attention_mask, features)
                    loss = loss_fn(outputs, labels)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            print(f"Epoch {epoch+1}/{self.epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            # Save the model if validation loss improved
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_save_path)
                print(f"Validation loss improved. Model saved to {model_save_path}")

        print("Training finished.")
        # Load the best model state dict after training
        self.load_model(model_save_path)


    def predict(self, headline: str, content: str) -> float:
        """Predicts the clickbait score using the hybrid model."""
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        self.model.to(self.device) # Ensure model is on the correct device
        self.model.eval()

        # Need to create a temporary dataset/dataframe to run feature extraction
        # Include a dummy label (e.g., 0.0) as expected by the dataset
        temp_data = pd.DataFrame([{"headline": headline, "content": content, "clickbait": 0.0}])
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

    def load_model(self, path: str = "HybridOutput/best_model.pt"):
        """Loads the trained model's state dictionary from a .pt file."""
        # This method remains the way to load trained weights for the hybrid model
        if self.model is None:
             print("Error: Hybrid model architecture not initialized. Cannot load state dict.")
             return
        print(f"Loading hybrid model state dict from: {path}...")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            print("Hybrid model state dict loaded successfully.")
        except FileNotFoundError:
             print(f"Error: Model state dict file not found at {path}")
        except Exception as e:
             print(f"An error occurred loading the hybrid model state dict: {e}")
             # Optionally reset model state
             # self.model = None