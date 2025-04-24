import numpy as np
import pandas as pd
import os
import torch
from typing import List, Type, Dict, Optional
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from transformers import AutoTokenizer, AutoModel

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # if using multi-GPU

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --- Import the refactored base and specific clickbait models ---
# Ensure this file exists and contains the refactored classes
try:
    from headline_content_models import (
        ClickbaitModelBase,
        ClickbaitTransformer,
        ClickbaitFeatureEnhancedTransformer
    )
except ImportError:
    print("Error: Could not import from clickbait_models.py.")
    print("Please ensure the file exists and contains the required classes:")
    print("ClickbaitModelBase, ClickbaitTransformer, ClickbaitFeatureEnhancedTransformer")
    # Define dummy classes to allow the rest of the script to be parsed
    class ClickbaitModelBase: pass
    class ClickbaitTransformer(ClickbaitModelBase): pass
    class ClickbaitFeatureEnhancedTransformer(ClickbaitModelBase): pass

# ============================================
# Similarity Method Interface and Implementations
# ============================================

class SimilarityMethod(ABC):
    """
    Abstract Base Class for methods comparing headline and content similarity
    or related scores (like clickbaitiness).
    """

    @abstractmethod
    def compute_score(self, headline: str, content: str) -> float:
        """
        Computes a similarity or related score between the headline and content.

        Args:
            headline (str): The headline text.
            content (str): The article content text.

        Returns:
            float: The computed score (e.g., cosine similarity, clickbait score).
                   The range might vary depending on the method (e.g., [0, 1] for
                   cosine similarity, potentially different for clickbait models).
        """
        pass


class CosineSimilarityTFIDF(SimilarityMethod):
    """
    Computes cosine similarity using Term Frequency-Inverse Document Frequency (TF-IDF).
    Score range: [0, 1], where 1 means identical TF-IDF vectors.
    """

    def __init__(self, stop_words: Optional[str] = "english"):
        """
        Initializes the TF-IDF vectorizer.

        Args:
            stop_words (Optional[str]): The stop words configuration for TfidfVectorizer.
                                        Defaults to "english". Can be None.
        """
        # Initialize TfidfVectorizer here to reuse it
        self.vectorizer = TfidfVectorizer(stop_words=stop_words)

    def compute_score(self, headline: str, content: str) -> float:
        """
        Computes TF-IDF vectors and their cosine similarity.

        Args:
            headline (str): The headline text.
            content (str): The article content text.

        Returns:
            float: The cosine similarity score [0, 1]. Returns 0.0 if inputs are empty
                   or cannot be vectorized.
        """
        # Handle potential empty inputs
        if not headline or not content:
            return 0.0
        try:
            # Fit and transform the texts
            vectors = self.vectorizer.fit_transform([headline, content])
            # Compute cosine similarity between the two vectors
            # vectors[0:1] keeps it as a sparse matrix slice
            similarity = sk_cosine_similarity(vectors[0:1], vectors[1:2])
            # Extract the single similarity value
            return float(similarity[0][0])
        except ValueError:
            # May happen if texts contain only stop words after preprocessing
            print("Warning: Could not compute TF-IDF similarity (perhaps due to only stop words?). Returning 0.0.")
            return 0.0


class TransformerEmbeddingSimilarity(SimilarityMethod):
    """
    Computes cosine similarity based on sentence embeddings from a transformer model.
    Uses mean pooling of the last hidden state.
    Score range: Approximately [-1, 1], often closer to [0, 1] for similar texts.
                 1 indicates identical embedding vectors.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # A common choice for embeddings

    def __init__(self, model_name: str = DEFAULT_MODEL, max_length: int = 512):
        """
        Initializes the tokenizer and model for embeddings.

        Args:
            model_name (str): The name of the pre-trained transformer model
                              (ideally one suited for sentence embeddings).
            max_length (int): Maximum sequence length for the tokenizer.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval() # Set model to evaluation mode
            print(f"TransformerEmbeddingSimilarity: Loaded model '{self.model_name}' on {self.device}.")
        except OSError as e:
            print(f"Error loading transformer model '{self.model_name}': {e}")
            print("Falling back to default behavior or raising error...")
            # Depending on desired robustness, you might raise an error here
            # or allow the object to be created but fail during compute_score.
            self.model = None
            self.tokenizer = None


    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Helper function to get the mean-pooled embedding for a text."""
        if not self.model or not self.tokenizer or not text:
            return None
        try:
            # Tokenize the text
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True, # Pad to max_length or max in batch (if batching)
                max_length=self.max_length
            )
            # Move tensors to the correct device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            # Get model output without computing gradients
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Perform mean pooling: average token embeddings across sequence length dimension
            # outputs.last_hidden_state shape: (batch_size, sequence_length, hidden_size)
            # We need to mask padding tokens before averaging for better representation
            attention_mask = inputs['attention_mask']
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9) # Avoid division by zero
            mean_pooled = sum_embeddings / sum_mask

            # Move result to CPU and convert to numpy array
            return mean_pooled.cpu().numpy().flatten()

        except Exception as e:
            print(f"Error generating embedding for text: '{text[:50]}...': {e}")
            return None

    def compute_score(self, headline: str, content: str) -> float:
        """
        Computes embeddings for headline and content and returns their cosine similarity.

        Args:
            headline (str): The headline text.
            content (str): The article content text.

        Returns:
            float: The cosine similarity score. Returns 0.0 if embeddings cannot be computed
                   or if norms are zero.
        """
        if not self.model:
             print("Error: Transformer model not loaded. Cannot compute similarity.")
             return 0.0

        headline_embed = self._get_embedding(headline)
        content_embed = self._get_embedding(content)

        if headline_embed is None or content_embed is None:
            print("Warning: Could not compute embeddings for headline or content. Returning 0.0.")
            return 0.0

        # Calculate cosine similarity using numpy
        norm_headline = np.linalg.norm(headline_embed)
        norm_content = np.linalg.norm(content_embed)

        if norm_headline == 0 or norm_content == 0:
            # Handle cases where embedding norm is zero (e.g., empty processed text)
             print("Warning: Zero norm for headline or content embedding. Returning 0.0.")
             return 0.0

        similarity = np.dot(headline_embed, content_embed) / (norm_headline * norm_content)

        # Clip similarity to [-1, 1] range due to potential floating point inaccuracies
        return float(np.clip(similarity, -1.0, 1.0))


class ClickbaitModelScore(SimilarityMethod):
    """
    Wrapper to use a Clickbait Detection model as a "similarity" method.
    Can load standard transformers from Hugging Face Hub/local path,
    or hybrid models by loading a state dict from a local .pt file.
    """
    CLICKBAIT_MODEL_CLASSES: Dict[str, Type[ClickbaitModelBase]] = {
        "transformer": ClickbaitTransformer,
        "hybrid": ClickbaitFeatureEnhancedTransformer,
    }

    def __init__(self,
                 model_type: str,
                 model_name_or_path: Optional[str] = None, # Renamed from model_path
                 **kwargs):
        """
        Initializes the wrapper with a specific clickbait model type.

        Args:
            model_type (str): Type of model ("transformer" or "hybrid").
            model_name_or_path (Optional[str]):
                - For "transformer": Hugging Face Hub ID or local directory path.
                  If None, defaults to the base model specified in ClickbaitTransformer
                  (e.g., "bert-base-uncased") with a WARNING that it's not fine-tuned.
                - For "hybrid": Path to the local .pt state dict file. If None,
                  the model is initialized with base weights but requires loading
                  a state dict later to be useful (issues a WARNING).
            **kwargs: Additional keyword arguments passed to the underlying
                      clickbait model's constructor (e.g., base model_name for hybrid,
                      batch_size, length_max).

        Raises:
            ValueError: If model_type is invalid or initialization/loading fails critically.
        """
        self.model_type = model_type
        self.model_identifier = model_name_or_path # Store for reference
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_type not in self.CLICKBAIT_MODEL_CLASSES:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Available types: {list(self.CLICKBAIT_MODEL_CLASSES.keys())}")

        model_cls = self.CLICKBAIT_MODEL_CLASSES[model_type]
        self.clickbait_model: Optional[ClickbaitModelBase] = None # Initialize as None

        try:
            if model_type == "transformer":
                # identifier_to_load = model_name_or_path
                identifier_to_load = "bert-base-uncased"
                if identifier_to_load is None:
                    # Default to base model if no identifier provided
                    identifier_to_load = model_cls.DEFAULT_BASE_MODEL # Get default from class
                    print("\n" + "="*60)
                    print(f"WARNING: No 'model_name_or_path' provided for transformer.")
                    print(f"Initializing with default BASE model: '{identifier_to_load}'.")
                    print("This model is NOT fine-tuned for clickbait detection.")
                    print("Predictions may not be meaningful without fine-tuning or loading a specific checkpoint.")
                    print("="*60 + "\n")

                # Pass identifier directly to constructor, it handles Hub/local loading
                # Also pass kwargs like batch_size, length_max if provided
                self.clickbait_model = model_cls(model_name_or_path=identifier_to_load, **kwargs)
                # No explicit load_model call needed here for transformer

            elif model_type == "hybrid":
                # For hybrid, model_name_or_path is the path to the state dict (.pt file)
                state_dict_path = model_name_or_path
                # Get base model name from kwargs or use default from class
                base_model_name = kwargs.get("model_name", model_cls.DEFAULT_BASE_MODEL)
                print(f"Initializing hybrid model with base: '{base_model_name}'")

                # Instantiate with base model name (and other kwargs)
                self.clickbait_model = model_cls(model_name=base_model_name, **kwargs)

                if state_dict_path:
                     if os.path.exists(state_dict_path) and state_dict_path.endswith(".pt"):
                         print(f"Attempting to load hybrid state dict from: {state_dict_path}")
                         self.clickbait_model.load_model(state_dict_path)
                     else:
                         print(f"Warning: Provided path '{state_dict_path}' does not exist or is not a .pt file. Hybrid model weights not loaded.")
                         print("The hybrid model will use its base transformer weights only.")
                else:
                     # No state dict path provided
                     print("\n" + "="*60)
                     print(f"WARNING: No state dict path ('model_name_or_path') provided for hybrid model.")
                     print(f"Initialized with base transformer '{base_model_name}' weights only.")
                     print("The model requires loading a trained state dict (.pt file) via ")
                     print("`load_model()` or providing the path during init to be effective.")
                     print("="*60 + "\n")

            # Final check and eval mode setting
            if self.clickbait_model and self.clickbait_model.model:
                self.clickbait_model.model.eval()
            elif self.clickbait_model is None or self.clickbait_model.model is None:
                 # Raise error if model failed to init completely
                 raise ValueError(f"Failed to initialize underlying model for type '{model_type}'. Check logs.")


        except Exception as e:
            print(f"Error setting up clickbait model '{model_type}' with identifier '{model_name_or_path}': {e}")
            # Optionally re-raise or handle more gracefully
            raise ValueError(f"Failed to set up clickbait model '{model_type}'.") from e

    def compute_score(self, headline: str, content: str) -> float:
        """Uses the loaded clickbait model to predict the score."""
        if self.clickbait_model is None or self.clickbait_model.model is None:
             print(f"Error: Clickbait model '{self.model_type}' (id: {self.model_identifier}) not ready for prediction.")
             return float('nan')
        # Check if model is hybrid and seems untrained (optional check)
        # if self.model_type == 'hybrid' and not _check_if_hybrid_seems_loaded(self.clickbait_model.model):
        #      print("Warning: Hybrid model may not have loaded trained weights. Prediction might be inaccurate.")

        try:
            score = self.clickbait_model.predict(headline, content)
            return float(score)
        except Exception as e:
            print(f"Error during prediction with model '{self.model_type}': {e}")
            return float('nan')

# ============================================
# Main Comparator Class
# ============================================

class HeadlineContentSimilarity:
    """
    Provides an interface to compare headline and content using a selected method.
    """

    def __init__(self, method: SimilarityMethod):
        """
        Initializes the comparator with a specific similarity method.

        Args:
            method (SimilarityMethod): An instance of a class implementing SimilarityMethod.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Provided method must be an instance of SimilarityMethod")
        self.method = method
        print(f"Initialized HeadlineContentSimilarity with method: {type(method).__name__}")

    def set_method(self, method: SimilarityMethod):
        """
        Changes the method used for calculating the score.

        Args:
            method (SimilarityMethod): An instance of a class implementing SimilarityMethod.
        """
        if not isinstance(method, SimilarityMethod):
            raise TypeError("Provided method must be an instance of SimilarityMethod")
        print(f"Changing similarity method to: {type(method).__name__}")
        self.method = method

    def compare(self, headline: str, content: str) -> float:
        """
        Compares the headline and content using the currently set method.

        Args:
            headline (str): The headline text.
            content (str): The article content text.

        Returns:
            float: The score computed by the current method.
        """
        if not headline or not content:
             print("Warning: Headline or content is empty.")
             # Return a default value, maybe dependent on the method's expected range
             return 0.0 # Or float('nan')

        print(f"Comparing using method: {type(self.method).__name__}")
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        if isinstance(self.method, (TransformerEmbeddingSimilarity, ClickbaitModelScore)) and \
           self.method.device.type == 'cuda':
            # Time GPU execution if applicable
            start_time.record()
            score = self.method.compute_score(headline, content)
            end_time.record()
            torch.cuda.synchronize() # Wait for the events to complete
            elapsed_time = start_time.elapsed_time(end_time) / 1000.0 # Time in seconds
            print(f"Computation took {elapsed_time:.4f} seconds on GPU.")
        else:
             # Time CPU execution
             import time
             start = time.perf_counter()
             score = self.method.compute_score(headline, content)
             end = time.perf_counter()
             elapsed_time = end - start
             print(f"Computation took {elapsed_time:.4f} seconds on CPU.")

        return score


# ============================================
# Example Usage
# ============================================
if __name__ == "__main__":
    print("\n--- Running Headline Content Similarity Examples ---")

    headline_example = "You Won't Believe What Happened Next! The Secret They Don't Want You to Know!"
    content_example = ("Following the recent developments, sources close to the committee revealed "
                       "that the proposal faced significant hurdles...") # Truncated for brevity
    headline_example = "This headline is crazy!"
    content_example = "Here's the actual article body."

    # --- Method 1: TF-IDF Cosine Similarity ---
    # ... (no changes needed) ...

    try:
        print("\n--- Method: TF-IDF Cosine Similarity ---")
        method_cosine = CosineSimilarityTFIDF()
        comparator = HeadlineContentSimilarity(method_cosine)
        similarity_cosine = comparator.compare(headline_example, content_example)
        print(f"TF-IDF Cosine Similarity Score: {similarity_cosine:.4f}")
    except Exception as e:
        print(f"Error running TF-IDF example: {e}")

    # --- Method 2: Transformer Embedding Similarity ---
    # ... (no changes needed - already uses default from Hub) ...

    try:
        print("\n--- Method: Transformer Embedding Similarity ---")
        method_transformer_embed = TransformerEmbeddingSimilarity() # Uses default Hub model
        comparator.set_method(method_transformer_embed)
        similarity_embed = comparator.compare(headline_example, content_example)
        print(f"Transformer Embedding Similarity Score: {similarity_embed:.4f}")
    except Exception as e:
        print(f"Error running Transformer Embedding example: {e}")

    # --- Method 3a: Clickbait Model Score (Default BASE Transformer - NO PATH) ---
    try:
        print("\n--- Method: Clickbait Model Score (DEFAULT BASE Transformer) ---")
        # Instantiate WITHOUT providing model_name_or_path
        # This will trigger the WARNING about using a non-fine-tuned base model
        method_clickbait_default = ClickbaitModelScore(
            model_type="transformer"
            # No model_name_or_path provided
        )
        comparator.set_method(method_clickbait_default)
        score_clickbait_default = comparator.compare(headline_example, content_example)
        print(f"Clickbait DEFAULT BASE Transformer Score: {score_clickbait_default:.4f} (WARNING: Untrained prediction)")
    except (ValueError, NameError, ImportError) as e:
        print(f"Could not run Clickbait Default Transformer example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred running Clickbait Default Transformer example: {e}")


    # --- Method 3b: Clickbait Model Score (SPECIFIC Transformer - Path/Hub ID) ---
    # OPTION A: Load fine-tuned from Hub (if available)
    # hub_model_id = "username/my-finetuned-clickbait-transformer" # EXAMPLE ID
    # OPTION B: Load from local path (directory saved via save_pretrained)
    path_to_transformer_model = "./bert-clickbait-regression" # EXAMPLE LOCAL PATH

    # --- CHOOSE ONE IDENTIFIER ---
    transformer_identifier = path_to_transformer_model # Or use hub_model_id
    # transformer_identifier = hub_model_id

    try:
        print("\n--- Method: Clickbait Model Score (Specific Transformer) ---")
        if "path/to/your" in transformer_identifier or "username/" in transformer_identifier:
             print(f"INFO: Using placeholder/example identifier: '{transformer_identifier}'. Update if needed.")
             # In a real scenario, you might raise an error if it's still a placeholder

        method_clickbait_specific = ClickbaitModelScore(
            model_type="transformer",
            model_name_or_path=transformer_identifier
        )
        comparator.set_method(method_clickbait_specific)
        score_clickbait_specific = comparator.compare(headline_example, content_example)
        print(f"Clickbait Specific Transformer ({transformer_identifier}) Score: {score_clickbait_specific:.4f}")

    except (FileNotFoundError, OSError, ValueError, NameError, ImportError) as e:
        print(f"Could not run Specific Clickbait Transformer example: {e}")
        print(f"Ensure 'clickbait_models.py' is available and '{transformer_identifier}' is valid.")
    except Exception as e:
        print(f"An unexpected error occurred running Specific Clickbait Transformer example: {e}")



    """
    # --- Method 4: Clickbait Model Score (Feature-Enhanced Hybrid - REQUIRES PATH) ---
    path_to_hybrid_model_pt = "/path/to/your/hybrid_model_output_dir/best_model.pt" # EXAMPLE .pt PATH

    try:
        print("\n--- Method: Clickbait Model Score (Feature-Enhanced Hybrid) ---")
        if "path/to/your" in path_to_hybrid_model_pt:
            print(f"INFO: Using placeholder path: '{path_to_hybrid_model_pt}'. Please update.")
            # Consider raising error if still placeholder in production

        # Hybrid model requires the path to the .pt file
        method_clickbait_hybrid = ClickbaitModelScore(
            model_type="hybrid",
            model_name_or_path=path_to_hybrid_model_pt, # Path to state dict
            # Optionally specify base model if different from default
            # model_name="bert-base-uncased"
        )
        comparator.set_method(method_clickbait_hybrid)
        score_clickbait_hybrid = comparator.compare(headline_example, content_example)
        print(f"Clickbait Hybrid Model ({path_to_hybrid_model_pt}) Score: {score_clickbait_hybrid:.4f}")

    except (FileNotFoundError, ValueError, NameError, ImportError) as e:
         print(f"Could not run Clickbait Hybrid example: {e}")
         print("Ensure 'clickbait_models.py' is available and")
         print(f"'{path_to_hybrid_model_pt}' is a valid path to a trained .pt state dict.")
    except Exception as e:
        print(f"An unexpected error occurred running Clickbait Hybrid example: {e}")
    """
    # --- Example showing Hybrid initialization WITHOUT path (triggers warning) ---

    try:
        print("\n--- Method: Clickbait Model Score (DEFAULT BASE Hybrid - NO PATH) ---")
        method_clickbait_hybrid_default = ClickbaitModelScore(
            model_type="hybrid"
            # No model_name_or_path provided
            # Optionally specify base: model_name="distilbert-base-uncased"
        )
        comparator.set_method(method_clickbait_hybrid_default)
        score_hybrid_default = comparator.compare(headline_example, content_example)
        print(f"Clickbait DEFAULT BASE Hybrid Score: {score_hybrid_default:.4f} (WARNING: Untrained prediction)")
    except (ValueError, NameError, ImportError) as e:
        print(f"Could not run default Clickbait Hybrid example: {e}")
    except Exception as e:
        print(f"An unexpected error occurred running default Clickbait Hybrid example: {e}")


    print("\n--- Headline Content Similarity Examples Finished ---")