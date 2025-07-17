"""Core PyTorch models for headline-content clickbait analysis.

This module defines the neural network architectures and training logic for
analyzing the relationship between a social media post, an article headline,
and the article's content. It provides an abstract base class `ClickbaitModelBase`
and two main implementations:
- `ClickbaitTransformer`: A standard sequence classification model using a
  pre-trained transformer like BERT.
- `ClickbaitFeatureEnhancedTransformer`: A hybrid model that combines a
  transformer with a set of engineered linguistic features for improved
  performance.
"""
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
from data.clickbait17.clickbait17_utils import combined_headline
from headline_content_feature_extractor import FeatureExtractor
from headline_content_evaluation import evaluate_clickbait_predictions
from config import HEADLINE_CONTENT_CONFIG, GENERAL_CONFIG
from resampling import apply_sampling
from loss_functions import WeightedLossTrainer, calculate_class_weights
import logging_config
import logging
import textstat
import spacy
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

logger = logging.getLogger(__name__)

class ClickbaitModelBase(ABC):
    """Abstract Base Class for Clickbait Detection Models.

    This class defines the common interface for all headline-content models,
    including methods for training, prediction, testing, and data loading.
    Subclasses must implement the abstract methods.
    """

    def __init__(self, model_name: str, tokenizer_name: str, length_max: int, batch_size: int, epochs: int):
        """Initializes common attributes for all clickbait models.

        Args:
            model_name (str): The name or path of the base transformer model.
            tokenizer_name (str): The name or path of the tokenizer.
            length_max (int): The maximum sequence length for tokenization.
            batch_size (int): The batch size for training and evaluation.
            epochs (int): The number of training epochs.
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
        self.model = None
        self.seed = GENERAL_CONFIG["seed"]

    @abstractmethod
    def _load_data(self, csv_path: str) -> Dataset:
        """Loads data from a CSV file into a PyTorch Dataset object."""
        pass

    @abstractmethod
    def train(self, train_csv: str, validation_csv: str):
        """Trains the model on the provided data."""
        pass

    @abstractmethod
    def predict(self, post: str, headline: str, content: str) -> float:
        """Predicts the clickbait score for a single data instance."""
        pass

    @abstractmethod
    def load_model(self, path: str):
        """Loads a trained model from a specified path."""
        pass

    def test(self, test_csv: str) -> (float, List[float]):
        """Evaluates the model on a test dataset.

        This method processes a test CSV file, generates predictions for each
        row, and computes a set of evaluation metrics (e.g., MSE, F1, AUC).
        It also handles and reports any rows that cause prediction errors.

        Args:
            test_csv (str): The path to the test CSV file.

        Returns:
            A tuple where the first element is a dictionary of evaluation
            metrics and the second element is a list of all predictions made.
        """
        start_time = time.perf_counter()
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        test_dataset = self._load_data(test_csv)
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

        self.model.to(self.device)
        self.model.eval()
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in test_loader:
                # Move all tensors in the batch to the correct device.
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                features = batch.get("features")

                # Handle both standard and feature-enhanced models.
                if features is not None:
                    features = features.to(self.device)
                    output = self.model(input_ids, attention_mask, features)
                else:
                    model_output = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    output = model_output.logits.squeeze(-1)

                predictions.extend(output.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        if not true_labels:
            print("Warning: No data processed in test loop.")
            return float("nan"), []

        true_labels_np = np.array(true_labels)
        predictions_np = np.array(predictions)

        # Identify and report any rows that result in NaN or infinity predictions.
        nan_indices = np.where(~np.isfinite(predictions_np))[0]
        if len(nan_indices) > 0:
            logger.error(f"FATAL: Found {len(nan_indices)} rows that resulted in NaN/inf predictions.")
            problematic_df = test_dataset.data.iloc[nan_indices]
            logger.error("The following input rows are causing the model to fail:")
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                logger.error(f"\n{problematic_df}")

            feature_cols = [col for col in problematic_df.columns if col.startswith('f')]
            if feature_cols:
                nan_in_features = problematic_df[feature_cols].isnull().sum()
                if nan_in_features.sum() > 0:
                    logger.error("NaN values were found in these feature columns of the problematic rows:")
                    logger.error(nan_in_features[nan_in_features > 0])
                else:
                    logger.info(
                        "No NaN values found in the feature columns. Issue may be from normalization.")

        # Clean the data by removing invalid entries before evaluation.
        valid_indices_mask = np.isfinite(true_labels_np) & np.isfinite(predictions_np)
        num_removed = len(true_labels_np) - np.sum(valid_indices_mask)
        if num_removed > 0:
            logger.warning(f"Removing {num_removed} rows with NaN/inf values from metrics calculation.")

        true_labels_clean = true_labels_np[valid_indices_mask]
        predictions_clean = predictions_np[valid_indices_mask]

        if len(true_labels_clean) == 0:
            logger.error("After removing invalid values, the evaluation set is empty. Cannot compute metrics.")
            return {}, predictions

        # Evaluate the predictions using the cleaned data.
        path = f"./results/{os.path.basename(self.model_name)}"
        metrics = evaluate_clickbait_predictions(true_labels_clean, predictions_clean,
                                                 save_path=os.path.join(path,
                                                                        f"{os.path.basename(self.model_name)}_test_metrics.csv"), time_start=start_time)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.info(f"Testing took {total_time:.4f} seconds.")

        return metrics, predictions


class ClickbaitTransformer(ClickbaitModelBase):
    """A standard transformer model for clickbait regression.

    This class wraps a Hugging Face `AutoModelForSequenceClassification` model
    for the task of predicting a continuous clickbait score. It uses the
    Hugging Face `Trainer` API for training.
    """

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
                 **kwargs):
        """Initializes the standard Clickbait Transformer model.

        This can load a base model from the Hugging Face Hub (e.g., 'bert-base-uncased')
        or a fine-tuned model from a local directory.

        Args:
            model_name_or_path (str): The name or path of the transformer model.
            tokenizer_name (str, optional): The name or path of the tokenizer. If
                None, it defaults to `model_name_or_path`.
            length_max (int): Maximum sequence length for tokenization.
            batch_size (int): Batch size for training and evaluation.
            epochs (int): Number of training epochs.
            learning_rate (float): The learning rate for the optimizer.
            weight_decay (float): The weight decay for the optimizer.
            dropout_rate (float): The dropout rate for the model's classifier.
            fp16 (bool): Whether to use 16-bit floating point precision for training.
            output_directory (str): The directory to save model checkpoints and logs.
            **kwargs: Additional arguments passed to `from_pretrained`.
        """
        tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name_or_path
        super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max,
                         batch_size=batch_size, epochs=epochs)
        self.model_identifier = model_name_or_path
        self.test_run = kwargs.pop("test_run", False)
        try:
            print(f"Loading AutoModelForSequenceClassification from: {model_name_or_path}")
            # Initialize the model for regression with a single output label.
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
                **kwargs
            )
            self.model.to(self.device)
        except OSError as e:
            model_default = HEADLINE_CONTENT_CONFIG["model_name"]
            print(f"Error loading model from {tokenizer_name}: {e}")
            print(f"Model initialization failed. self.model will be set as {model_default}.")
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_default,
                num_labels=1,
                problem_type="regression",
                ignore_mismatched_sizes=True,
                **kwargs
            )
            self.model.to(self.device)
        try:
            # Load the tokenizer associated with the model.
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **kwargs)
            print(f"Successfully loaded model and tokenizer from {tokenizer_name}")
        except OSError as e:
            # Fallback to a default tokenizer if loading fails.
            tokenizer_default = HEADLINE_CONTENT_CONFIG["tokenizer_name"]
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_default)
                print(f"Warning: Model loading failed, loaded tokenizer for {tokenizer_default}")
            except Exception as te:
                print(f"Error loading default tokenizer {tokenizer_default}: {te}")
                self.tokenizer = None

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.fp16 = fp16
        self.output_directory = f"{output_directory}_{self.model_identifier.replace('/', '_')}_{time.strftime('%Y_%m_%d_%H_%M_%S', time.gmtime())}"
        self.trainer = None

    def _load_data(self, data: Union[str, pd.DataFrame]) -> Dataset:
        """Loads data from a CSV path or a pandas DataFrame.

        Args:
            data (Union[str, pd.DataFrame]): The path to the CSV file or a
                DataFrame.

        Returns:
            A `Clickbait17Dataset` instance.
        """
        try:
            if isinstance(data, str):
                df = pd.read_csv(data).dropna(subset=["content", "clickbait_score"])
            else:
                df = data

            if df.empty:
                print(f"Warning: No valid data after dropping NaNs in {data}")
            return Clickbait17Dataset(df, self.tokenizer, self.length_max)
        except FileNotFoundError:
            print(f"Error: File not found at {data}")
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]),
                                      self.tokenizer, self.length_max)
        except Exception as e:
            print(f"Error loading data from {data}: {e}")
            return Clickbait17Dataset(pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]),
                                      self.tokenizer, self.length_max)

    def train(self, train_csv: str, validation_csv: str = None, sampling_strategy: str = None,
              use_weighted_loss: bool = False) -> None:
        """Trains the model using the Hugging Face Trainer API.

        Args:
            train_csv (str): The path to the training data CSV.
            validation_csv (str, optional): The path to the validation data CSV.
                Defaults to None.
            sampling_strategy (str, optional): A resampling strategy to apply
                ('oversample' or 'undersample'). Defaults to None.
            use_weighted_loss (bool, optional): Whether to use a custom weighted
                loss function to handle class imbalance. Defaults to False.
        """
        try:
            df_train = pd.read_csv(train_csv).dropna(subset=["content", "clickbait_score"])
            # Apply resampling to the training data if specified.
            df_train = apply_sampling(df_train, sampling_strategy, self.seed)
        except FileNotFoundError:
            print(f"Error: Training file not found at {train_csv}")
            return

        data_train = self._load_data(df_train)
        data_validation = self._load_data(validation_csv) if validation_csv else None

        if len(data_train) == 0:
            print("Error: Training dataset is empty. Aborting training.")
            return

        # Select the Trainer class based on whether weighted loss is used.
        trainer_class = Trainer
        trainer_kwargs = {}
        if use_weighted_loss:
            class_weights = calculate_class_weights(train_csv)
            if class_weights:
                trainer_class = WeightedLossTrainer
                trainer_kwargs['class_weights'] = class_weights

        # Configure training arguments for the Trainer.
        training_args = TrainingArguments(
            output_dir="temp/model/standard/" if self.test_run else self.output_directory,
            evaluation_strategy="epoch" if not self.test_run and data_validation else "no",
            save_strategy="epoch" if not self.test_run and data_validation else "no",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            fp16=self.fp16 and torch.cuda.is_available(),
            logging_dir=os.path.join(self.output_directory, "logs"),
            load_best_model_at_end=True if not self.test_run and data_validation else False,
            save_total_limit=1,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to="none",
            seed=self.seed

        )

        self.trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=data_train,
            eval_dataset=data_validation,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if data_validation else None,
            **trainer_kwargs
        )

        # Start the training process.
        print(f"Starting training for {self.epochs} epochs...")
        self.trainer.train()
        print("Training finished.")

        try:
            log_history = self.trainer.state.log_history
            train_loss = [log['loss'] for log in log_history if 'loss' in log]
            validation_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

            if train_loss:
                logging.info("Complete Training Loss History:")
                logging.info(train_loss)

            if validation_loss:
                logging.info("Complete Validation Loss History:")
                logging.info(validation_loss)
        except Exception as e:
            logger.info(f"Unable to retrieve Loss History: {e}")

        # Replace the current model with the best model found during training.
        self.model = self.trainer.model
        self.model.to(self.device)

        # Save the best model and its tokenizer.
        if not self.test_run:
            best_model_path = os.path.join(self.output_directory, "best_model")
            self.trainer.save_model(best_model_path)
            self.tokenizer.save_pretrained(best_model_path)
            print(f"Best model saved to {best_model_path}")
            self.load_model(best_model_path)

    def predict(self, post: str, headline: str, content: str) -> float:
        """Predicts the clickbait score for a single instance.

        Args:
            post (str): The social media post text.
            headline (str): The article headline.
            content (str): The article content.

        Returns:
            The predicted clickbait score as a float.
        """
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")

        self.model.to(self.device)
        self.model.eval()

        combined_text = combined_headline(headline=headline, post=post)

        inputs = self.tokenizer(
            text=combined_text,
            text_pair=content,
            return_tensors="pt",
            truncation="longest_first",
            padding=True,
            max_length=self.length_max
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            score = outputs.logits.squeeze().item()
        return score

    def load_model(self, model_path: str, **kwargs):
        """Loads a trained model from a local directory.

        Args:
            model_path (str): The path to the directory containing the saved model.
            **kwargs: Additional arguments for `from_pretrained`.
        """
        print(f"Loading model explicitly from local directory: {model_path}...")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path, **kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            self.model.to(self.device)
            self.model_identifier = model_path
            print("Model loaded successfully from directory.")
        except OSError as e:
            print(f"Error loading model from directory {model_path}: {e}.")
        except Exception as e:
            print(f"An unexpected error occurred during model loading from directory: {e}")


class ClickbaitFeatureEnhancedTransformer(ClickbaitModelBase):
    """A hybrid model combining a transformer with engineered features.

    This class defines a custom neural network architecture that fuses the
    text representations from a transformer model with a set of pre-extracted
    linguistic features. It includes custom wrapper classes to ensure
    compatibility with the Hugging Face `Trainer` API.
    """

    class HybridClickbaitModel(nn.Module):
        """The core neural network architecture for the hybrid model."""

        def __init__(self, transformer_name: str, num_features: int,
                     dropout_rate: float = HEADLINE_CONTENT_CONFIG["dropout_rate"]):
            super().__init__()
            self.bert = AutoModel.from_pretrained(transformer_name)
            self.bert_config = self.bert.config
            bert_hidden_size = self.bert_config.hidden_size

            self.custom_config = {
                "transformer_name": transformer_name,
                "num_features": num_features,
                "dropout_rate": dropout_rate,
                "bert_hidden_size": bert_hidden_size,
                "fusion_strategy": "gated"
            }

            self.dropout = nn.Dropout(dropout_rate)
            self.feature_proj = nn.Linear(num_features, bert_hidden_size)
            self.leaky_relu = nn.LeakyReLU()
            # A regressor to process the concatenated text and feature representations.
            self.regressor = nn.Sequential(
                nn.Linear(bert_hidden_size * 2, bert_hidden_size),
                nn.LeakyReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(bert_hidden_size, 1),
                nn.Sigmoid()
            )

        def forward(self, input_ids, attention_mask, features):
            """Defines the forward pass of the model."""
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_state = outputs.last_hidden_state

            # Use attention-weighted averaging to get a single text representation.
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_hidden = torch.sum(last_hidden_state * attention_mask_expanded, 1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(1), min=1e-9)
            text_representation = sum_hidden / sum_mask

            # Project engineered features to the same dimension as the text representation.
            projected_features = self.feature_proj(features)

            # Concatenate the text and feature vectors.
            combined_representation = torch.cat((text_representation, projected_features), dim=1)

            # Pass the combined vector through the final regressor.
            logits = self.regressor(combined_representation)

            return logits.squeeze(-1)

    class HybridWrapperModel(PreTrainedModel):
        """A wrapper to make the custom model compatible with the Trainer API."""
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
            """Defines the forward pass for the wrapper model."""
            logits = self.hybrid_model(input_ids=input_ids, attention_mask=attention_mask, features=features)
            loss = None
            if labels is not None:
                loss_fn = nn.MSELoss()
                loss = loss_fn(logits, labels.float())

            from transformers.modeling_outputs import SequenceClassifierOutput
            return SequenceClassifierOutput(loss=loss, logits=logits)

    class HybridDataCollator:
        """A custom data collator for batching text and feature data."""

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
            print(f"Error loading base transformer '{model_name_or_path}'. Using {model_default}")
            self.model = self.HybridClickbaitModel(
                transformer_name=model_default,
                num_features=self.num_features,
                dropout_rate=self.dropout_rate).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                       **kwargs)

    def _load_data(self, csv_path: str) -> Clickbait17FeatureAugmentedDataset:
        """Loads data from a CSV with pre-calculated features.

        Args:
            csv_path (str): Path to the CSV file.

        Returns:
            A `Clickbait17FeatureAugmentedDataset` instance.
        """
        try:
            if not os.path.exists(csv_path.replace(".csv", "_metadata.json")):
                raise FileNotFoundError(f"Metadata file for {csv_path} not found.")
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

    def train(self, train_csv: str, validation_csv: str = None, sampling_strategy: str = None,
              use_weighted_loss: bool = False):
        """Trains the hybrid model.

        Args:
            train_csv (str): Path to the training CSV with features.
            validation_csv (str, optional): Path to the validation CSV.
            sampling_strategy (str, optional): Resampling strategy ('oversample' or 'undersample').
            use_weighted_loss (bool, optional): Whether to use weighted loss.
        """
        temp_dir = tempfile.mkdtemp()
        try:
            # Apply resampling if specified.
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
                original_meta_path = train_csv.replace(".csv", "_metadata.json")
                temp_meta_path = temp_train_csv.replace(".csv", "_metadata.json")
                if os.path.exists(original_meta_path):
                    shutil.copy(original_meta_path, temp_meta_path)
                else:
                    logger.warning(f"Metadata for {train_csv} not found. Normalization might fail.")

            # Load datasets and normalization statistics.
            train_dataset = self._load_data(training_data_path)
            val_dataset = self._load_data(validation_csv) if validation_csv else None
            if len(train_dataset) == 0:
                print("Error: Training dataset is empty. Aborting training.")
                return
            loaded_num_features = len(train_dataset.feature_median)
            if self.num_features != loaded_num_features:
                print(
                    f"Warning: num_features in config ({self.num_features}) != data ({loaded_num_features}). Using {loaded_num_features}.")
                self.num_features = loaded_num_features

            self.feature_median = train_dataset.feature_median.to(self.device)
            self.feature_iqr = train_dataset.feature_iqr.to(self.device)
            if val_dataset:
                val_dataset.feature_median = self.feature_median
                val_dataset.feature_iqr = self.feature_iqr

            # Initialize the model and its wrapper for the Trainer.
            self.model = self.HybridClickbaitModel(
                transformer_name=self.model_name, num_features=self.num_features, dropout_rate=self.dropout_rate
            ).to(self.device)
            wrapper_config = self.model.bert_config
            wrapper_config.transformer_name_custom = self.model.custom_config["transformer_name"]
            wrapper_config.num_features_custom = self.model.custom_config["num_features"]
            wrapper_config.dropout_rate_custom = self.model.custom_config["dropout_rate"]
            wrapper_config.feature_median = self.feature_median.cpu().tolist()
            wrapper_config.feature_iqr = self.feature_iqr.cpu().tolist()
            train_meta_path = train_csv.replace(".csv", "_metadata.json")
            if os.path.exists(train_meta_path):
                with open(train_meta_path, "r") as f:
                    meta = json.load(f)
                    self.boxcox_lambdas = meta.get("boxcox_lambdas", {})
                    wrapper_config.boxcox_lambdas = self.boxcox_lambdas

            wrapper_model_for_trainer = self.HybridWrapperModel(
                config=wrapper_config, custom_hybrid_model_instance=self.model
            ).to(self.device)

            # Set up weighted loss if enabled.
            trainer_class = Trainer
            trainer_kwargs = {}
            if use_weighted_loss:
                class_weights = calculate_class_weights(train_csv)
                if class_weights:
                    trainer_class = WeightedLossTrainer
                    trainer_kwargs['class_weights'] = class_weights

            # Configure and run the Trainer.
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
                save_total_limit=1,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                fp16=self.fp16 and torch.cuda.is_available(),
                report_to="none",
                max_grad_norm=1.0,
                seed=self.seed
            )
            self.trainer = trainer_class(
                model=wrapper_model_for_trainer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                data_collator=self.HybridDataCollator(self.tokenizer, self.length_max),
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  if val_dataset else None,
                **trainer_kwargs
            )
            print(f"Starting hybrid model training. Output directory: {self.output_directory}")
            self.trainer.train()
            try:
                if self.trainer:
                    log_history = self.trainer.state.log_history
                    train_loss = [log['loss'] for log in log_history if 'loss' in log]
                    validation_loss = [log['eval_loss'] for log in log_history if 'eval_loss' in log]

                    if train_loss:
                        logger.info("Complete Hybrid Model Training Loss History:")
                        logger.info(train_loss)

                    if validation_loss:
                        logger.info("Complete Hybrid Model Validation Loss History:")
                        logger.info(validation_loss)
            except Exception as e:
                logger.info(f"Unable to retrieve Loss History: {e}")
            print("Hybrid model training finished.")

            if hasattr(self.trainer, 'model') and self.trainer.model is not None:
                self.model = self.trainer.model.hybrid_model
                self.model.to(self.device)
                print("Updated self.model with weights from the trained trainer.model.")
            else:
                print("Warning: self.trainer.model not available. self.model may not have updated weights.")

            # Save the final best model.
            final_model_save_path = os.path.join(self.output_directory, "best_model")
            if not self.test_run:
                config_to_save = self.trainer.model.config
                print("Adding feature normalization stats to model config...")
                config_to_save.feature_median = self.feature_median.cpu().tolist()
                config_to_save.feature_iqr = self.feature_iqr.cpu().tolist()
                print(f"Saving final best model to: {final_model_save_path}")
                self.trainer.save_model(final_model_save_path)
                self.tokenizer.save_pretrained(final_model_save_path)
                print(f"Final best model and tokenizer saved to {final_model_save_path}.")
        finally:
            shutil.rmtree(temp_dir)

    def predict(self, post: str, headline: str, content: str, headline_score: float) -> float:
        """Predicts the clickbait score for a single instance using the hybrid model.

        Args:
            post (str): The social media post text.
            headline (str): The article headline.
            content (str): The article content.
            headline_score (float): A pre-computed clickbait score from a
                headline-only classifier. This is used as an input feature.

        Returns:
            The predicted clickbait score as a float.
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        if self.feature_median is None or self.feature_iqr is None:
            raise ValueError("Feature normalization stats not available. Train/load a model first.")

        self.model.to(self.device)
        self.model.eval()

        # Extract base features and append the headline score.
        base_features = self.feature_extractor.extract(post, headline, content, as_dict=False)
        final_features = base_features + [headline_score]

        # Apply Box-Cox transformation if lambdas are available.
        if self.boxcox_lambdas:
            feature_names = self.feature_extractor.feature_names + ["headline_score"]
            for i, feature_name in enumerate(feature_names):
                feature_key = f"f{i + 1}"
                if feature_key in self.boxcox_lambdas:
                    if final_features[i] > 0:
                        final_features[i] = boxcox(final_features[i], lmbda=self.boxcox_lambdas[feature_key])
        features_tensor = torch.tensor(final_features, dtype=torch.float).to(self.device)

        # Normalize features using Robust Scaler stats (median/IQR).
        iqr_safe = self.feature_iqr.clone()
        iqr_safe[self.feature_iqr < 1e-5] = 1.0
        features_normalised = (features_tensor - self.feature_median) / iqr_safe
        features_tensor = features_normalised.unsqueeze(0)

        # Tokenize text inputs.
        combined_text = combined_headline(headline=headline, post=post)
        inputs = self.tokenizer(
            text=combined_text, text_pair=content, return_tensors="pt", truncation="longest_first",
            padding=True, max_length=self.length_max
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Get the prediction from the model.
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features_tensor)

        return output.item()

    def load_model(self, model_path: str = None):
        """Loads a trained hybrid model from a directory.

        This method restores the model's weights, its tokenizer, and the
        feature normalization statistics required for prediction.

        Args:
            model_path (str, optional): Path to the directory containing the
                saved model. Defaults to the 'best_model' in the instance's
                output directory.
        """
        if model_path is None:
            model_path = os.path.join(self.output_directory, "best_model")
        if not os.path.isdir(model_path):
            print(f"Error: Model directory not found at {model_path}")
            return

        print(f"Loading Hybrid model from directory: {model_path}")
        try:
            # Load the model configuration.
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                print(f"Error: config.json not found in {model_path}. Cannot load model.")
                return
            wrapper_config = PretrainedConfig.from_json_file(config_path)

            # Re-instantiate the inner model architecture.
            self.model = self.HybridClickbaitModel(
                transformer_name=wrapper_config.transformer_name_custom,
                num_features=wrapper_config.num_features_custom,
                dropout_rate=wrapper_config.dropout_rate_custom
            )

            # Load the saved model weights.
            model_weights_path = os.path.join(model_path, "pytorch_model.bin")
            if os.path.exists(os.path.join(model_path, "model.safetensors")):
                model_weights_path = os.path.join(model_path, "model.safetensors")
            if "safetensors" in model_weights_path:
                from safetensors.torch import load_file
                full_state_dict = load_file(model_weights_path, device=self.device.type)
            else:
                full_state_dict = torch.load(model_weights_path, map_location=self.device)

            # Filter the state dictionary to only include weights for the inner model.
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

            # Load normalization statistics from the configuration file.
            if hasattr(wrapper_config, "feature_median") and hasattr(wrapper_config, "feature_iqr"):
                self.feature_median = torch.tensor(wrapper_config.feature_median, dtype=torch.float).to(self.device)
                self.feature_iqr = torch.tensor(wrapper_config.feature_iqr, dtype=torch.float).to(self.device)
                print("Feature normalization stats (median/IQR) loaded from model config.")
            else:
                print("Warning: Feature normalization stats not found. 'predict' will fail.")
            if hasattr(wrapper_config, "boxcox_lambdas"):
                self.boxcox_lambdas = wrapper_config.boxcox_lambdas
                print("Box-Cox lambdas loaded from model config.")
            else:
                print("Info: No Box-Cox lambdas found in model config.")
                self.boxcox_lambdas = {}
        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")