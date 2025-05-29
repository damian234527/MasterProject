import os
import time
import pandas as pd
import numpy as np
import torch
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, PreTrainedModel, AutoConfig, PretrainedConfig
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

    # Inner HybridClickbaitModel - this is your actual custom architecture
    class HybridClickbaitModel(nn.Module):
        def __init__(self, transformer_name: str, num_features: int, dropout_rate: float = 0.3):
            super().__init__()
            self.bert = AutoModel.from_pretrained(transformer_name)  # Base transformer
            self.bert_config = self.bert.config  # Store base config for later
            hidden_size = self.bert_config.hidden_size

            # Store config for saving/loading
            self.custom_config = {
                "transformer_name": transformer_name,
                "num_features": num_features,
                "dropout_rate": dropout_rate,
                "bert_hidden_size": hidden_size
            }

            self.dropout = nn.Dropout(dropout_rate)
            self.feature_proj = nn.Sequential(
                nn.LayerNorm(num_features),
                nn.Linear(num_features, hidden_size),  # Project features
                nn.ReLU()
            )
            self.regressor = nn.Sequential(
                nn.Linear(hidden_size * 2, 128),  # Combined BERT output and projected features
                nn.ReLU(),
                nn.Dropout(dropout_rate / 2),
                nn.Linear(128, 1)
            )

        def forward(self, input_ids, attention_mask, features):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            try:
                cls_output = outputs.pooler_output
            except AttributeError:
                mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
                sum_hidden = (outputs.last_hidden_state * mask).sum(1)
                avg_hidden = sum_hidden / torch.clamp(mask.sum(1), min=1e-9)
                cls_output = avg_hidden

            projected_features = self.feature_proj(features)
            combined = torch.cat((cls_output, projected_features), dim=1)
            x = self.dropout(combined)
            logits = self.regressor(x)
            return logits.squeeze(-1)  # Return raw logits for regression

    # Wrapper for Hugging Face Trainer compatibility
    class HybridWrapperModel(PreTrainedModel):
        # Use the base transformer's config for the PreTrainedModel parent class
        # This makes it easier for Trainer to handle some aspects.
        config_class = AutoConfig  # Allows loading various transformer configs

        def __init__(self, config: PretrainedConfig, custom_hybrid_model_instance: nn.Module = None, **custom_kwargs):
            super().__init__(config)
            # If custom_hybrid_model_instance is provided, use it.
            # Otherwise, initialize from config (useful for from_pretrained)
            if custom_hybrid_model_instance:
                self.hybrid_model = custom_hybrid_model_instance
            else:
                # Initialize HybridClickbaitModel from stored custom_kwargs or config attributes
                # This requires transformer_name, num_features, dropout_rate to be in config
                # or passed via custom_kwargs.
                self.hybrid_model = ClickbaitFeatureEnhancedTransformer.HybridClickbaitModel(
                    transformer_name=config.transformer_name_custom,  # Expect these in the config
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

    # Data Collator remains the same
    class HybridDataCollator:
        def __init__(self, tokenizer):
            self.tokenizer = tokenizer

        def __call__(self, features_list):  # Renamed 'features' to 'features_list' to avoid confusion
            input_features_dict = [{k: f[k] for k in ["input_ids", "attention_mask"]} for f in features_list]
            batch = self.tokenizer.pad(input_features_dict, return_tensors="pt", padding="max_length",
                                       max_length=self.tokenizer.model_max_length)
            batch["features"] = torch.stack([f["features"] for f in features_list])
            if "label" in features_list[0]:
                batch["labels"] = torch.tensor([f["label"] for f in features_list], dtype=torch.float)
            return batch

    # NUM_FEATURES can be passed or determined from data
    # model_name_default = os.getenv("MODEL_NAME", "bert-base-uncased")
    def __init__(self,
                 model_name_or_path: str = HEADLINE_CONTENT_CONFIG["model_name"],  # Base transformer name
                 tokenizer_name: str = HEADLINE_CONTENT_CONFIG["tokenizer_name"],
                 num_features: int = 14,  # Default, should be derived from dataset ideally
                 length_max: int = HEADLINE_CONTENT_CONFIG["length_max"],
                 batch_size: int = HEADLINE_CONTENT_CONFIG["batch_size"],
                 epochs: int = HEADLINE_CONTENT_CONFIG["epochs"],
                 learning_rate: float = HEADLINE_CONTENT_CONFIG["learning_rate"],
                 weight_decay: float = HEADLINE_CONTENT_CONFIG["weight_decay"],
                 dropout_rate: float = HEADLINE_CONTENT_CONFIG["dropout_rate"],
                 output_directory: str = os.path.join(HEADLINE_CONTENT_CONFIG["output_directory"], "hybrid"),  #
                 fp16: bool = HEADLINE_CONTENT_CONFIG["fp16"],  # Added fp16 from config
                 **kwargs):
        super().__init__(model_name=model_name_or_path, tokenizer_name=tokenizer_name, length_max=length_max,
                         batch_size=batch_size, epochs=epochs)
        self.test_run = kwargs.pop("test_run", False)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.fp16 = fp16  # Store fp16
        self.output_directory = output_directory  # Use the argument
        self.num_features = num_features  # Store num_features

        os.makedirs(self.output_directory, exist_ok=True)

        try:
            print(f"Initializing inner HybridClickbaitModel with base transformer: {model_name_or_path}")
            # This is self.model, the inner custom nn.Module
            self.model = self.HybridClickbaitModel(
                transformer_name=model_name_or_path,
                num_features=self.num_features,
                dropout_rate=dropout_rate
            ).to(self.device)
        except OSError as e:
            model_default = HEADLINE_CONTENT_CONFIG["model_name"]  #
            print(f"Error loading base transformer '{model_name_or_path}' for hybrid model: {e}. Using {model_default}")
            self.model = self.HybridClickbaitModel(
                transformer_name=model_default,
                num_features=self.num_features,
                dropout_rate=dropout_rate).to(self.device)

        # Tokenizer should match the base transformer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name if tokenizer_name else model_name_or_path,
                                                       **kwargs)

    def _load_data(self, csv_path: str):  # Return type hint: -> Dataset
        # ... (same as your existing _load_data for hybrid) ...
        # Make sure it returns Clickbait17FeatureAugmentedDataset
        try:
            df = pd.read_csv(csv_path)
            # Pass num_features if your dataset needs it, or let it be inferred/fixed
            return Clickbait17FeatureAugmentedDataset(df, self.tokenizer, self.length_max)  #
        except FileNotFoundError:
            print(f"Error: File not found at {csv_path}")
            return Clickbait17FeatureAugmentedDataset(
                pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer,
                self.length_max)
        except Exception as e:
            print(f"Error loading data from {csv_path}: {e}")
            return Clickbait17FeatureAugmentedDataset(
                pd.DataFrame(columns=["post", "headline", "content", "clickbait_score"]), self.tokenizer,
                self.length_max)

    def train(self, train_csv: str, validation_csv: str):
        train_dataset = self._load_data(train_csv)
        val_dataset = self._load_data(validation_csv)

        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("Error: Training or validation dataset is empty. Aborting training.")
            return

        # Prepare config for HybridWrapperModel
        # It should be the config of the base transformer, with custom attributes added.
        # self.model here is the inner HybridClickbaitModel
        wrapper_config = self.model.bert_config  # Get the base transformer's config from inner model

        # Add custom attributes needed by HybridWrapperModel's __init__ (for from_pretrained)
        wrapper_config.transformer_name_custom = self.model.custom_config["transformer_name"]
        wrapper_config.num_features_custom = self.model.custom_config["num_features"]
        wrapper_config.dropout_rate_custom = self.model.custom_config["dropout_rate"]

        # Instantiate the wrapper, passing the inner model instance and its derived config
        wrapper_model_for_trainer = self.HybridWrapperModel(
            config=wrapper_config,
            custom_hybrid_model_instance=self.model  # Pass the already initialized inner model
        ).to(self.device)

        # Define output directory for Trainer's own checkpoints, distinct from final model save path
        trainer_output_dir = os.path.join(self.output_directory, "trainer_checkpoints")

        training_args = TrainingArguments(
            output_dir=trainer_output_dir,  # Trainer saves its own checkpoints here
            evaluation_strategy="no" if self.test_run else "epoch",
            save_strategy="no" if self.test_run else "epoch",
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            logging_dir=os.path.join(self.output_directory, "logs"),  # Log to a logs subfolder
            load_best_model_at_end=False if self.test_run else True,  # Important!
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            fp16=self.fp16 and torch.cuda.is_available(),
            report_to="none",
            max_grad_norm=1.0  #
        )

        self.trainer = Trainer(
            model=wrapper_model_for_trainer,  # Train the wrapper
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=self.HybridDataCollator(self.tokenizer),
        )

        print(f"Starting hybrid model training. Output directory: {self.output_directory}")
        self.trainer.train()
        print("Hybrid model training finished.")

        # After training, if load_best_model_at_end=True, self.trainer.model is the best one.
        # Update self.model (the inner HybridClickbaitModel) with the weights from the trained wrapper.
        if hasattr(self.trainer, 'model') and self.trainer.model is not None:
            self.model = self.trainer.model.hybrid_model  # Get the inner model from the trained wrapper
            self.model.to(self.device)  # Ensure it's on the correct device
            print("Updated self.model with weights from the trained trainer.model.")
        else:
            print("Warning: self.trainer.model is not available. self.model may not have updated weights.")
            # This case should ideally not happen if training completes.

        # Define a clear path for the final "best model" to be saved for inference
        # This is where users will load the model from later.
        final_model_save_path = os.path.join(self.output_directory, "best_model")

        if not self.test_run:  # Only save the "best_model" artifact for full runs
            print(f"Saving final best model to: {final_model_save_path}")
            # self.trainer.save_model will save the HybridWrapperModel (which includes HybridClickbaitModel)
            self.trainer.save_model(final_model_save_path)
            # Save the tokenizer alongside the model for easy reloading
            self.tokenizer.save_pretrained(final_model_save_path)
            print(f"Final best model and tokenizer saved to {final_model_save_path}.")
        elif self.test_run:  # For test runs, you might save to a different, temporary path if needed
            test_run_save_path = os.path.join(self.output_directory, "test_run_model")
            print(f"Saving test run model to: {test_run_save_path}")
            self.trainer.save_model(test_run_save_path)
            self.tokenizer.save_pretrained(test_run_save_path)
            print(f"Test run model and tokenizer saved.")
            # self.model is already updated from self.trainer.model

    def predict(self, post: str, headline: str, content: str) -> float:  #
        # ... (predict method remains largely the same, ensure self.model is used correctly) ...
        if self.model is None:
            raise ValueError("Model has not been loaded or trained yet.")
        self.model.to(self.device)
        self.model.eval()

        temp_data = pd.DataFrame([{"post": post, "headline": headline, "content": content, "clickbait_score": 0.0}])
        # Assuming Clickbait17FeatureAugmentedDataset is correctly imported and set up
        # It needs access to tokenizer and length_max, potentially means/stds for normalization
        # For prediction, ensure normalization uses stats from training or is not applied if model trained without it.
        # The current _load_data doesn't set up means/stds for normalization on the fly for new instances.
        # This is a separate concern about feature consistency in prediction.
        # For now, let's assume features are extracted as during training.
        temp_dataset = self._load_data_for_prediction(temp_data)  # You might need a separate method for this

        if len(temp_dataset) == 0:
            print("Warning: Could not process input for prediction.")
            return float("nan")

        item = temp_dataset[0]

        input_ids = item["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = item["attention_mask"].unsqueeze(0).to(self.device)
        features_tensor = item["features"].unsqueeze(0).to(self.device)  # Renamed for clarity
        with torch.no_grad():
            output = self.model(input_ids, attention_mask, features_tensor)
        return output.item()

    # Helper for predict, might need to adjust feature normalization handling
    def _load_data_for_prediction(self, dataframe: pd.DataFrame):
        # This is a simplified version. Ideally, use the same feature scaling (mean/std)
        # as used during training. This might require loading those stats.
        return Clickbait17FeatureAugmentedDataset(dataframe, self.tokenizer, self.length_max)

    def load_model(self, model_path: str = None):
        """
        Loads the Hybrid model for inference.
        'model_path' should be the directory where HybridWrapperModel was saved.
        """
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

            # --- MODIFIED WEIGHTS LOADING ---
            model_weights_safetensors_path = os.path.join(model_path, "model.safetensors")
            model_weights_bin_path = os.path.join(model_path, "pytorch_model.bin")
            full_state_dict = None

            if os.path.exists(model_weights_safetensors_path):
                try:
                    from safetensors.torch import load_file as load_safetensors_file
                    print(f"Loading weights from safetensors file: {model_weights_safetensors_path}")
                    # Pass device as a string 'cuda' or 'cpu'
                    device_type_str = self.device.type if self.device.type != 'mps' else 'cpu'  # safetensors map_location is limited
                    full_state_dict = load_safetensors_file(model_weights_safetensors_path, device=device_type_str)
                except ImportError:
                    print("Warning: 'model.safetensors' found, but the 'safetensors' library is not installed. "
                          "Attempting to load 'pytorch_model.bin'. "
                          "Please install with: pip install safetensors")
                except Exception as e:
                    print(f"Error loading 'model.safetensors': {e}. Attempting to load 'pytorch_model.bin'.")

            if full_state_dict is None and os.path.exists(model_weights_bin_path):
                print(f"Loading weights from '.bin' file: {model_weights_bin_path}")
                try:
                    full_state_dict = torch.load(model_weights_bin_path, map_location=self.device)
                except Exception as e:
                    print(f"Error loading 'pytorch_model.bin': {e}")
                    # Consider returning or raising if this also fails

            if full_state_dict is None:
                print(f"Error: Neither 'model.safetensors' nor 'pytorch_model.bin' found in {model_path}, "
                      "or an error occurred during loading. Cannot load weights.")
                return
            # --- END OF MODIFIED WEIGHTS LOADING ---

            inner_model_state_dict = {
                k.replace("hybrid_model.", "", 1): v
                for k, v in full_state_dict.items()
                if k.startswith("hybrid_model.")
            }

            if not inner_model_state_dict:
                print(f"Warning: No keys found with 'hybrid_model.' prefix in the state_dict from {model_path}.")
                print(
                    "Attempting to load the full state_dict directly into the inner model. This might fail or be incorrect.")
                self.model.load_state_dict(full_state_dict, strict=False)
            else:
                self.model.load_state_dict(inner_model_state_dict)

            self.model.to(self.device)
            self.model.eval()
            print(f"Hybrid model (inner) loaded successfully from {model_path} and assigned to self.model.")

            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            print(f"Tokenizer loaded from {model_path}.")

        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")

        except Exception as e:
            print(f"Failed to load model from {model_path}: {e}")
            # self.model might remain None or in an inconsistent state.
