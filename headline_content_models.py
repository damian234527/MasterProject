import os
import json
import time
import pandas as pd
import numpy as np
import torch
import tempfile
import shutil
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, \
    EarlyStoppingCallback, PreTrainedModel, AutoConfig, PretrainedConfig
from typing import Union, List
import string
from scipy.stats import boxcox
from torch import nn
from data.clickbait17.clickbait17_dataset import Clickbait17Dataset, Clickbait17FeatureAugmentedDataset
from headline_content_feature_extractor import FeatureExtractor
from headline_content_evaluation import evaluate_clickbait_predictions
from config import HEADLINE_CONTENT_CONFIG, GENERAL_CONFIG
from resampling import apply_sampling
from loss_functions import WeightedLossTrainer, calculate_class_weights
import logging_config
import logging

# Added imports for feature extraction
import textstat
import spacy
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

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
        self.model = None  # To be initialized by subclasses
        self.seed = GENERAL_CONFIG["seed"]

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
        start_time = time.perf_counter()
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        test_dataset = self._load_data(test_csv)
        # Use a default batch size if test_dataset is small to avoid errors
        effective_batch_size = min(self.batch_size, len(test_dataset))
        if effective_batch_size == 0:
            print("Warning: Test dataset is empty.")
            return float("nan"), []
        test_loader = DataLoader(
            test_dataset,
            batch_size=effective_batch_size,
            num_workers=8,
            pin_memory=True
        )

        self.model.to(self.device)  # Ensure model is on the correct device
        self.model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                # Move all tensor items in batch to the correct device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                features = batch.get("features")  # Use .get() for optional features

                if features is not None:
                    features = features.to(self.device)
                    # Assumes model with features takes features as the third argument
                    output = self.model(input_ids, attention_mask, features)
                else:
                    # Assumes standard HF sequence classification model output structure
                    model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    output = model_output.logits.squeeze(-1)  # Squeeze the last dimension

                predictions.extend(output.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        if not true_labels:  # Handle empty dataset case after loop
            print("Warning: No data processed in test loop.")
            return float("nan"), []

        # mse = mean_squared_error(true_labels, predictions)
        # print(f"MSE for test set: {mse:.4f}")

        path = f"./results/{os.path.basename(self.model_name)}"
        metrics = evaluate_clickbait_predictions(true_labels, predictions,
                                                 save_path=os.path.join(path, f"{self.model_name}_test_metrics.csv"))
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logging.info(f"Testing took {total_time:.4f} seconds.")

        return metrics, predictions


# ============================================
# Subclass 1: Standard Transformer
# ============================================
class ClickbaitTransformer(ClickbaitModelBase):
    """Clickbait detection using a standard Transformer model (e.g., BERT)."""

    # model_name_default = os.getenv("MODEL_NAME", "bert-base-uncased") # Keep track of default base "google/bert_uncased_L-4_H-256_A-4" - smaller default example
    def __init__(self,
                 model_name_or_path: str = HEADLINE_CONTENT_CONFIG["model_name"],
                 tokenizer_name: str = None,
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"],
                 batch_size: int = HEADLINE_CONTENT_CONFIG["batch_size"],
                 epochs: int = HEADLINE_CONTENT_CONFIG["epochs"],
                 learning_rate=HEADLINE_CONTENT_CONFIG["learning_rate"],
                 weight_decay=HEADLINE_CONTENT_CONFIG["weight_decay"],
                 dropout_rate=HEADLINE_CONTENT_CONFIG["dropout_rate"],
                 fp16: bool = HEADLINE_CONTENT_CONFIG["fp16"],
                 output_directory: str = os.path.join(HEADLINE_CONTENT_CONFIG["output_directory"], "standard"),
                 **kwargs):  # Allow passing extra args to from_pretrained
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
        tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name_or_path
        super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max,
                         batch_size=batch_size, epochs=epochs)
        self.model_identifier = model_name_or_path  # Store the identifier used
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
                **kwargs  # Pass extra args like trust_remote_code if needed
            )
            self.model.to(self.device)  # Ensure model is on device after loading
        except OSError as e:
            model_default = HEADLINE_CONTENT_CONFIG["model_name"]
            print(f"Error loading model from {tokenizer_name}: {e}")
            print(f"Model initialization failed. self.model will be set as {model_default}.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_default,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
                **kwargs  # Pass extra args like trust_remote_code if needed
            )
            self.model.to(self.device)  # Ensure model is on device after loading
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
        self.output_directory = f"{output_directory}_{self.model_identifier.replace("/", "_")}_{time.strftime("%Y_%m_%d_%H_%M_%S", time.gmtime())}"
        self.trainer = None

    def _load_data(self, data: Union[str, pd.DataFrame]) -> Dataset:
        """Loads data from a CSV file or DataFrame."""
        try:
            if isinstance(data, str):
                df = pd.read_csv(data).dropna(subset=["content", "clickbait_score"])
            else:
                df = data  # When a DataFrame

            if df.empty:
                print(f"Warning: No valid data after dropping NaNs in {data}")
            return Clickbait17Dataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {data}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]),
                                      self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {data}: {e}")
            # Return an empty dataset or raise an error
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]),
                                      self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str = None, sampling_strategy: str = None,
              use_weighted_loss: bool = False) -> None:
        """
        Trains the model using the Hugging Face Trainer API.

        Args:
            train_csv (str): Path to the training data.
            validation_csv (str, optional): Path to the validation data. Defaults to None.
            sampling_strategy (str, optional): Strategy for resampling data ('oversample' or 'undersample'). Defaults to None.
            use_weighted_loss (bool, optional): Whether to use a weighted loss function. Defaults to False.
        """
        # Loading and optional resample of training data
        try:
            df_train = pd.read_csv(train_csv).dropna(subset=["content", "clickbait_score"])
            # Apply sampling if a strategy is specified
            df_train = apply_sampling(df_train, sampling_strategy, self.seed)
        except FileNotFoundError:
            print(f"Error: Training file not found at {train_csv}")
            return

        data_train = self._load_data(df_train)
        data_validation = self.test(validation_csv) if validation_csv else None

        if len(data_train) == 0:
            print("Error: Training dataset is empty. Aborting training.")
            return

        # Optional weights for Trainer
        trainer_class = Trainer
        trainer_kwargs = {}

        if use_weighted_loss:
            # Calculate weights from the *original* training file
            class_weights = calculate_class_weights(train_csv)
            if class_weights:
                # If weights are available, switch to our custom trainer
                # and prepare its unique argument.
                trainer_class = WeightedLossTrainer
                trainer_kwargs['class_weights'] = class_weights

        # Definition of training arguments and initialisation of trainer
        training_args = TrainingArguments(
            output_dir="temp/model/standard/" if self.test_run else self.output_directory,
            evaluation_strategy="epoch" if not self.test_run and data_validation else "no",
            save_strategy="epoch" if not self.test_run and data_validation else "no",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            fp16=self.fp16 and torch.cuda.is_available(),  # Only enable fp16 if cuda is available
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=True if not self.test_run and data_validation else False,
            metric_for_best_model="eval_loss",  # Regression task, use loss
            greater_is_better=False,  # Lower loss is better
            report_to="none",  # external reporting like wandb disabled

        )

        self.trainer = trainer_class(
            model=self.model,  # Already initialized in __init__
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation,
            **trainer_kwargs
            # callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            # No compute_metrics needed by default for Trainer's regression eval loss
        )

        # Training start
        print(f"Starting training for {self.epochs} epochs...")
        self.trainer.train()
        print("Training finished.")

        # Best model instead of last one
        self.model = self.trainer.model
        self.model.to(self.device)

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

        self.model.to(self.device)  # Ensure model is on the correct device
        self.model.eval()

        combined_text = ""
        if post and headline:
            combined_text = f"{post}: {headline}"
        elif post:
            combined_text = post
        elif headline:
            combined_text = headline

        inputs = self.tokenizer(
            text=combined_text,
            text_pair=content,
            return_tensors="pt",
            truncation="longest_first",  # Truncate longest sequence first if needed
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
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)  # Load corresponding tokenizer
            self.model.to(self.device)  # Move loaded model to device
            self.model_identifier = model_path  # Update identifier
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
# Subclass 2: Feature-Enhanced Transformer (REVISED)
# ============================================
class ClickbaitFeatureEnhancedTransformer(ClickbaitModelBase):
    """Clickbait detection using a Transformer enhanced with engineered features."""

    # Inner HybridClickbaitModel - this is your actual custom architecture
    class HybridClickbaitModel(nn.Module):
        def __init__(self, transformer_name: str, num_features: int,
                     dropout_rate: float = HEADLINE_CONTENT_CONFIG["dropout_rate"]):
            super().__init__()
            self.bert = AutoModel.from_pretrained(transformer_name)  # Base transformer
            self.bert_config = self.bert.config  # Store base config
            bert_hidden_size = self.bert_config.hidden_size

            # Store config for saving/loading
            self.custom_config = {
                "transformer_name": transformer_name,
                "num_features": num_features,
                "dropout_rate": dropout_rate,
                "bert_hidden_size": bert_hidden_size,
                "fusion_strategy": "gated"  # Add new info about fusion
            }

            self.dropout = nn.Dropout(dropout_rate)
            self.feature_proj = nn.Linear(num_features, bert_hidden_size)
            self.leaky_relu = nn.LeakyReLU()
            # The regressor now takes the concatenated output of the text representation and projected features
            self.regressor = nn.Sequential(
                nn.Linear(bert_hidden_size * 2, bert_hidden_size),  # Project combined features down
                nn.LeakyReLU(),  # Apply non-linearity
                nn.Dropout(dropout_rate),  # Apply dropout for regularization
                nn.Linear(bert_hidden_size, 1)  # Final output layer
            )

        def forward(self, input_ids, attention_mask, features):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # Attention-weighted average of hidden states
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            text_representation = sum_hidden / sum_mask

            projected_features = self.feature_proj(features)

            # Concatenate the raw representations
            combined_representation = torch.cat((text_representation, projected_features), dim=1)

            # Pass the combined vector through the new, more powerful regressor.
            logits = self.regressor(combined_representation)

            return logits.squeeze(-1)

    # Wrapper for Hugging Face Trainer compatibility
    class HybridWrapperModel(PreTrainedModel):
        config_class = AutoConfig

        def __init__(self, config: PretrainedConfig, custom_hybrid_model_instance: nn.Module = None, **custom_kwargs):
            super().__init__(config)
            if custom_hybrid_model_instance:
                self.hybrid_model = custom_hybrid_model_instance
            else:
                self.hybrid_model = ClickbaitFeatureEnhancedTransformer.HybridClickbaitModel(
                    transformer_name=config.transformer_name_custom,
                    num_features=config.num_features_custom,
                    dropout_rate=config.dropout_rate_custom
                )

        def forward(self, input_ids=None, attention_mask=None, features=None, labels=None, **kwargs):
            logits = self.hybrid_model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = None
            if labels is not None:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits, labels.float())

            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(loss=loss, logits=logits)

    class HybridDataCollator:
        def __init__(self, tokenizer, max_length: int):
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __call__(self, features_list):
            input_features_dict = [{k: f[k] for k in ["input_ids", "attention_mask"]} for f in features_list]
            batch = self.tokenizer.pad(input_features_dict, return_tensors="pt", padding="max_length",
                                       max_length=min(self.max_length, self.tokenizer.model_max_length))
            batch["features"] = torch.stack([f["features"] for f in features_list])
            if "label" in features_list[0]:
                batch["labels"] = torch.tensor([f["label"] for f in features_list], dtype=torch.float)
            return batch

    def __init__(self,
                 model_name_or_path: str = HEADLINE_CONTENT_CONFIG["model_name"],
                 tokenizer_name: str = None,
                 num_features: int = 23,
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"],
                 batch_size: int = HEADLINE_CONTENT_CONFIG["batch_size"],
                 epochs: int = HEADLINE_CONTENT_CONFIG["epochs"],
                 learning_rate: float = HEADLINE_CONTENT_CONFIG["learning_rate"],
                 weight_decay: float = HEADLINE_CONTENT_CONFIG["weight_decay"],
                 dropout_rate: float = HEADLINE_CONTENT_CONFIG["dropout_rate"],
                 output_directory: str = os.path.join(HEADLINE_CONTENT_CONFIG["output_directory"], "hybrid"),
                 fp16: bool = HEADLINE_CONTENT_CONFIG["fp16"],
                 **kwargs):
        super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max,
                         batch_size=batch_size, epochs=epochs)
        self.test_run = kwargs.pop("test_run", False)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fp16 = fp16
        self.output_directory = output_directory
        self.num_features = num_features
        self.dropout_rate = dropout_rate

        self.feature_extractor = FeatureExtractor()
        # self.feature_means = None
        # self.feature_stds = None
        self.feature_median = None
        self.feature_iqr = None
        self.boxcox_lambdas = {}

        os.makedirs(self.output_directory, exist_ok=True)

        try:
            print(f"Initializing inner HybridClickbaitModel with base transformer: {model_name_or_path}")
            self.model = self.HybridClickbaitModel(
                transformer_name=model_name_or_path,
                num_features=self.num_features,
                dropout_rate=self.dropout_rate
            ).to(self.device)
        except OSError as e:
            model_default = HEADLINE_CONTENT_CONFIG["model_name"]
            print(f"Error loading base transformer '{model_name_or_path}' for hybrid model: {e}. Using {model_default}")
            self.model = self.HybridClickbaitModel(
                transformer_name=model_default,
                num_features=self.num_features,
                dropout_rate=self.dropout_rate).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                       **kwargs)
    def _load_data(self, csv_path: str) -> Clickbait17FeatureAugmentedDataset:
        try:
            if not os.path.exists(csv_path.replace(".csv", "_metadata.json")):
                raise FileNotFoundError(f"Metadata file not found for {csv_path}. Please run data preparation script.")
            return Clickbait17FeatureAugmentedDataset.from_feature_csv(csv_path, self.tokenizer, self.length_max)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return Clickbait17FeatureAugmentedDataset(
                pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer,
                self.length_max)
        except Exception as e:
            print(f"An unexpected error occurred while loading data from {csv_path}: {e}")
            return Clickbait17FeatureAugmentedDataset(
                pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer,
                self.length_max)

    def train(self, train_csv: str, validation_csv: str = None, sampling_strategy: str = None, use_weighted_loss: bool = False):
        """
        Trains the hybrid model with optional resampling and weighted loss.

        Args:
            train_csv (str): Path to the training data with pre-calculated features.
            validation_csv (str, optional): Path to the validation data.
            sampling_strategy (str, optional): 'oversample' or 'undersample'.
            use_weighted_loss (bool, optional): Whether to use weighted loss.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # --- RESAMPLING LOGIC ---
            training_data_path = train_csv
            if sampling_strategy:
                print(f"Applying '{sampling_strategy}' to training data...")
                try:
                    df_train = pd.read_csv(train_csv)
                except FileNotFoundError:
                    print(f"Error: Training file not found at {train_csv}")
                    shutil.rmtree(temp_dir)
                    return

                processed_df = apply_sampling(df_train, sampling_strategy, self.seed)
                temp_train_csv = os.path.join(temp_dir, "temp_train.csv")
                processed_df.to_csv(temp_train_csv, index=False)
                training_data_path = temp_train_csv

                # Copy metadata file to correspond with the temporary resampled csv
                original_meta_path = train_csv.replace(".csv", "_metadata.json")
                temp_meta_path = temp_train_csv.replace(".csv", "_metadata.json")
                if os.path.exists(original_meta_path):
                    shutil.copy(original_meta_path, temp_meta_path)
                else:
                    logging.warning(f"Metadata for {train_csv} not found. Normalization might fail.")

            # Load datasets
            train_dataset = self._load_data(training_data_path)
            val_dataset = self._load_data(validation_csv) if validation_csv else None

            if len(train_dataset) == 0:
                print("Error: Training dataset is empty. Aborting training.")
                return

            loaded_num_features = len(train_dataset.feature_median)
            if self.num_features != loaded_num_features:
                print(f"Warning: num_features in config ({self.num_features}) does not match data ({loaded_num_features}). Using {loaded_num_features}.")
                self.num_features = loaded_num_features

            self.feature_median = train_dataset.feature_median.to(self.device)
            self.feature_iqr = train_dataset.feature_iqr.to(self.device)
            if val_dataset:
                val_dataset.feature_median = self.feature_median
                val_dataset.feature_iqr = self.feature_iqr

            # Initialize the inner model, ensuring num_features is correct
            try:
                self.model = self.HybridClickbaitModel(
                    transformer_name=self.model_name,
                    num_features=self.num_features,
                    dropout_rate=self.dropout_rate
                ).to(self.device)
            except OSError as e:
                print(f"Error loading base transformer '{self.model_name}' for hybrid model: {e}.")
                raise e

            wrapper_config = self.model.bert_config
            wrapper_config.transformer_name_custom = self.model.custom_config["transformer_name"]
            wrapper_config.num_features_custom = self.model.custom_config["num_features"]
            wrapper_config.dropout_rate_custom = self.model.custom_config["dropout_rate"]
            # wrapper_config.feature_means = self.feature_means.cpu().tolist()
            # wrapper_config.feature_stds = self.feature_stds.cpu().tolist()
            wrapper_config.feature_median = self.feature_median.cpu().tolist()
            wrapper_config.feature_iqr = self.feature_iqr.cpu().tolist()

            train_meta_path = train_csv.replace(".csv", "_metadata.json")
            if os.path.exists(train_meta_path):
                with open(train_meta_path, "r") as f:
                    meta = json.load(f)
                    self.boxcox_lambdas = meta.get("boxcox_lambdas", {})
                    wrapper_config.boxcox_lambdas = self.boxcox_lambdas

            wrapper_model_for_trainer = self.HybridWrapperModel(
                config=wrapper_config,
                custom_hybrid_model_instance=self.model
            ).to(self.device)

            # --- WEIGHTED LOSS LOGIC ---
            trainer_class = Trainer
            trainer_kwargs = {}
            if use_weighted_loss:
                # Calculate weights from the *original* non-resampled training file
                class_weights = calculate_class_weights(train_csv)
                if class_weights:
                    trainer_class = WeightedLossTrainer
                    trainer_kwargs['class_weights'] = class_weights

            trainer_output_dir = os.path.join(self.output_directory, "trainer_checkpoints")
            training_args = TrainingArguments(
                output_dir=trainer_output_dir,
                evaluation_strategy="epoch" if not self.test_run and val_dataset else "no",
                save_strategy="epoch" if not self.test_run and val_dataset else "no",
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                num_train_epochs=self.epochs,
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
                logging_dir=os.path.join(self.output_directory, "logs"),
                load_best_model_at_end=True if not self.test_run and val_dataset else False,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.fp16 and torch.cuda.is_available(),
                report_to="none",
                max_grad_norm=1.0
            )

            self.trainer = trainer_class(
                model=wrapper_model_for_trainer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.HybridDataCollator(self.tokenizer, self.length_max),
                **trainer_kwargs
            )
            print(f"Starting hybrid model training. Output directory: {self.output_directory}")
            self.trainer.train()
            print("Hybrid model training finished.")

            if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                self.model = self.trainer.model.hybrid_model
                self.model.to(self.device)
                print("Updated self.model with weights from the trained trainer.model.")
            else:
                print("Warning: self.trainer.model is not available. self.model may not have updated weights.")

            final_model_save_path = os.path.join(self.output_directory, "best_model")
            if not self.test_run:
                config_to_save = self.trainer.model.config
                print("Adding feature normalization stats to model config...")
                # config_to_save.feature_means = self.feature_means.cpu().tolist()
                # config_to_save.feature_stds = self.feature_stds.cpu().tolist()
                config_to_save.feature_median = self.feature_median.cpu().tolist()
                config_to_save.feature_iqr = self.feature_iqr.cpu().tolist()

                print(f"Saving final best model to: {final_model_save_path}")
                self.trainer.save_model(final_model_save_path)
                self.tokenizer.save_pretrained(final_model_save_path)
                print(f"Final best model and tokenizer saved to {final_model_save_path}.")

        finally:
            # Clean up the temporary directory and its contents
            shutil.rmtree(temp_dir)


    def predict(self, post: str, headline: str, content: str, headline_score: float) -> float:
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        if self.feature_median is None or self.feature_iqr is None:
            raise ValueError("Feature normalization stats are not available. Train or load a model first.")

        self.model.to(self.device)
        self.model.eval()

        # Step 1: Get base 22 features and append the provided headline score.
        base_features = self.feature_extractor.extract(post, headline, content, as_dict=False)
        final_features = base_features + [headline_score]

        # Apply Box-Cox transformation before scaling
        if self.boxcox_lambdas:
            feature_names = self.feature_extractor.feature_names + ["headline_score"]
            for i, feature_name in enumerate(feature_names):
                # The feature names in the lambda dict are "f1", "f2", etc.
                feature_key = f"f{i+1}"
                if feature_key in self.boxcox_lambdas:
                    # Check for positivity before applying
                    if final_features[i] > 0:
                        final_features[i] = boxcox(final_features[i], lmbda=self.boxcox_lambdas[feature_key])

        features_tensor = torch.tensor(final_features, dtype=torch.float).to(self.device)

        # Step 2: Normalize the features using Robust Scaler
        iqr_safe = self.feature_iqr.clone()
        iqr_safe[self.feature_iqr < 1e-5] = 1.0
        features_normalised = (features_tensor - self.feature_median) / iqr_safe
        features_tensor = features_normalised.unsqueeze(0)  # Add batch dimension

        # Step 3: Combine text and tokenize
        combined_text = ""
        if post and headline:
            combined_text = f"{post}: {headline}"
        elif post:
            combined_text = post
        elif headline:
            combined_text = headline

        inputs = self.tokenizer(
            text=combined_text,
            text_pair=content,
            return_tensors="pt",
            truncation="longest_first",
            padding=True,
            max_length=self.length_max
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Step 4: Get prediction
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features_tensor)

        return output.item()

    # REVISED: load_model now restores feature normalization stats
    def load_model(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.join(self.output_directory, "best_model")
        if not os.path.isdir(model_path):
            print(f"Error: Model directory not found at {model_path}")
            return

        print(f"Loading Hybrid model from directory: {model_path}")
        try:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"Error: config.json not found in {model_path}. Cannot load model.")
                return
            wrapper_config = PretrainedConfig.from_json_file(config_path)

            self.model = self.HybridClickbaitModel(
                transformer_name=wrapper_config.transformer_name_custom,
                num_features=wrapper_config.num_features_custom,
                dropout_rate=wrapper_config.dropout_rate_custom
            )

            model_weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                model_weights_path = os.path.join(model_path, "model.safetensors")

            if "safetensors" in model_weights_path:
                from safetensors.torch import load_file
                full_state_dict = load_file(model_weights_path, device=self.device.type)
            else:
                full_state_dict = torch.load(model_weights_path, map_location=self.device)

            inner_model_state_dict = {
                k.replace("hybrid_model.", "", 1): v
                for k, v in full_state_dict.items() if k.startswith("hybrid_model.")
            }
            self.model.load_state_dict(inner_model_state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"Hybrid model (inner) loaded successfully from {model_path} and assigned to self.model.")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Tokenizer loaded from {model_path}.")

            # NEW: Load normalization stats from the config
            if hasattr(wrapper_config, "feature_median") and hasattr(wrapper_config, "feature_iqr"):
                self.feature_median = torch.tensor(wrapper_config.feature_median, dtype=torch.float).to(self.device)
                self.feature_iqr = torch.tensor(wrapper_config.feature_iqr, dtype=torch.float).to(self.device)
                print("Feature normalization stats (median/IQR) loaded from model config.")
            else:
                print(
                    "Warning: Feature normalization stats (mean/std) not found in config. The 'predict' method will fail.")
            if hasattr(wrapper_config, "boxcox_lambdas"):
                self.boxcox_lambdas = wrapper_config.boxcox_lambdas
                print("Box-Cox lambdas loaded from model config.")
            else:
                print("Info: No Box-Cox lambdas found in model config.")
                self.boxcox_lambdas = {}

        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")