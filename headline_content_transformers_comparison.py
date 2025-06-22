import os
import pandas as pd
from datetime import datetime
import torch # For torch.cuda.empty_cache
from headline_content_models import (
    ClickbaitTransformer,
    ClickbaitFeatureEnhancedTransformer
)
from headline_content_similarity import (
    CosineSimilarityTFIDF,
    TransformerEmbeddingSimilarity,
    SimilarityMethodEvaluator
)
from data.clickbait17.clickbait17_prepare import prepare_clickbait17_datasets, dataset_check
from data.clickbait17.clickbait17_utils import get_basic_csv_paths, get_feature_csv_paths, get_safe_name
from config import HEADLINE_CONTENT_CONFIG, DATASETS_CONFIG, HEADLINE_CONTENT_MODELS_PRETRAINED
import logging

logger = logging.getLogger(__name__)


# --- Configuration for Fast Testing ---

# Parameters for quick runs
FAST_TEST_EPOCHS = 1
FAST_TEST_BATCH_SIZE = HEADLINE_CONTENT_CONFIG.get("batch_size", 8) # Use existing config or a default

# Attempt to get the dataset name from config, otherwise use a default
DATASET_BASE_NAME = DATASETS_CONFIG.get("dataset_headline_content_name", "clickbait17-validation-1.0")
# --- End of Configuration ---


def run_fast_test(
    model_configurations: list,
    test_standard_transformer: bool = True,
    test_hybrid_transformer: bool = False, # Set to True to also test feature-enhanced models
    test_simple_methods: bool = True
):
    """
    Runs a fast test for a list of transformer model configurations
    and other non-trainable similarity methods.
    """
    logging.info("\n--- Starting Fast Test of Headline-Content Transformers ---")
    output_root_directory = "models/fast_test_outputs"
    os.makedirs(output_root_directory, exist_ok=True)
    all_results = []

    if test_standard_transformer or test_hybrid_transformer:
        for model_config_entry in model_configurations:
            model_name_for_hf = ""
            tokenizer_name_for_hf = ""

            if isinstance(model_config_entry, str):
                model_name_for_hf = model_config_entry
                tokenizer_name_for_hf = model_config_entry
            elif isinstance(model_config_entry, dict):
                model_name_for_hf = model_config_entry['model_name']
                tokenizer_name_for_hf = model_config_entry.get('tokenizer_name', model_name_for_hf)
            else:
                logging.warning(f"Skipping invalid model configuration: {model_config_entry}")
                continue

            logging.info(f"\n>>> Processing Model: {model_name_for_hf} (Tokenizer: {tokenizer_name_for_hf})")

            # 1. Prepare Datasets
            # The prepare_clickbait17_datasets in your train.py is called without arguments.
            # This implies it might use a default tokenizer or prepare for all.
            # For testing multiple tokenizers, ensure this step correctly provides data
            # for the *current* tokenizer_name_for_hf.
            # If prepare_clickbait17_datasets needs to be tokenizer-specific, you might need to modify it
            # or ensure data is pre-generated for all tokenizers in MODELS_TO_TEST.
            try:
                logging.info(f"Ensuring datasets are prepared for tokenizer: {tokenizer_name_for_hf}...")
                # Calling as in your train.py
                prepare_clickbait17_datasets(tokenizer_name=tokenizer_name_for_hf)
            except Exception as e:
                logging.warning(f"Warning: Dataset preparation step encountered an issue for {tokenizer_name_for_hf}: {e}")
                # Depending on implementation, paths might still be found if pre-generated

            train_csv_basic_path, val_csv_basic_path = get_basic_csv_paths(tokenizer_name_for_hf) #
            train_csv_features_path, val_csv_features_path = get_feature_csv_paths(tokenizer_name_for_hf) #

            # Generate a unique name for output files/directories for this specific run
            sanitized_model_name = get_safe_name(model_name_for_hf)
            sanitized_tokenizer_name = model_name_for_hf(tokenizer_name_for_hf)
            run_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

            # 2. Test Standard ClickbaitTransformer
            if test_standard_transformer:
                if not os.path.exists(train_csv_basic_path) or not os.path.exists(val_csv_basic_path):
                    logging.warning(f"Basic dataset CSVs not found for {tokenizer_name_for_hf}. Skipping standard model test.")
                    all_results.append({
                        "model_type": "standard", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "skipped (dataset missing)",
                        "NMSE": float('nan'), "PR-AUC": float('nan') # Ensure consistent columns
                    })
                else:
                    logging.info(f"\n--- Testing ClickbaitTransformer (Standard) for {model_name_for_hf} ---")
                    current_run_output_dir = os.path.join(output_root_directory, f"transformer_{sanitized_model_name}_{sanitized_tokenizer_name}_{run_timestamp}")
                    try:
                        standard_transformer = ClickbaitTransformer(
                            model_name_or_path=model_name_for_hf,
                            tokenizer_name=tokenizer_name_for_hf,
                            epochs=FAST_TEST_EPOCHS,
                            batch_size=FAST_TEST_BATCH_SIZE,
                            output_directory=current_run_output_dir,
                            test_run=True  # Crucial: This modifies TrainingArguments for faster, minimal training
                        )
                        logging.info("Training standard model (fast mode)...")
                        standard_transformer.train(train_csv_basic_path, val_csv_basic_path)
                        logging.info("Evaluating standard model...")
                        # The .test() method is defined in ClickbaitModelBase
                        metrics, _ = standard_transformer.test(val_csv_basic_path)
                        logging.info(f"Metrics for {model_name_for_hf} (Standard): {metrics}")
                        all_results.append({
                            "model_type": "standard", "model_name": model_name_for_hf,
                            "tokenizer_name": tokenizer_name_for_hf, "status": "success",
                            **metrics  # Assumes metrics dict contains NMSE, PR-AUC etc.
                        })
                        del standard_transformer # Free memory
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    except Exception as e:
                        logging.error(f"ERROR testing standard transformer {model_name_for_hf}: {e}")
                        all_results.append({
                            "model_type": "standard", "model_name": model_name_for_hf,
                            "tokenizer_name": tokenizer_name_for_hf, "status": "error", "error_message": str(e),
                            "NMSE": float('nan'), "PR-AUC": float('nan')
                        })

            # 3. Test Hybrid ClickbaitFeatureEnhancedTransformer
            if test_hybrid_transformer:
                if not os.path.exists(train_csv_features_path) or not os.path.exists(val_csv_features_path):
                    logging.warning(f"Feature-augmented dataset CSVs not found for {tokenizer_name_for_hf}. Skipping hybrid model test.")
                    all_results.append({
                        "model_type": "hybrid", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "skipped (dataset missing)",
                        "NMSE": float('nan'), "PR-AUC": float('nan')
                    })
                else:
                    logging.info(f"\n--- Testing ClickbaitFeatureEnhancedTransformer (Hybrid) for {model_name_for_hf} ---")
                    current_run_output_dir = os.path.join(output_root_directory, f"hybrid_{sanitized_model_name}_{sanitized_tokenizer_name}_{run_timestamp}")
                    try:
                        # Note: Constructor for ClickbaitFeatureEnhancedTransformer uses model_name_or_path for its base transformer
                        hybrid_transformer = ClickbaitFeatureEnhancedTransformer(
                            model_name_or_path=model_name_for_hf, # This is the base transformer for the hybrid architecture
                            tokenizer_name=tokenizer_name_for_hf,
                            epochs=FAST_TEST_EPOCHS,
                            batch_size=FAST_TEST_BATCH_SIZE,
                            output_directory=current_run_output_dir, # Parameter name is output_directory
                            test_run=True # Crucial for fast training
                        )
                        logging.info("Training hybrid model (fast mode)...")
                        hybrid_transformer.train(train_csv_features_path, val_csv_features_path)
                        logging.info("Evaluating hybrid model...")
                        metrics, _ = hybrid_transformer.test(val_csv_features_path)
                        logging.info(f"Metrics for {model_name_for_hf} (Hybrid): {metrics}")
                        all_results.append({
                            "model_type": "hybrid", "model_name": model_name_for_hf,
                            "tokenizer_name": tokenizer_name_for_hf, "status": "success",
                            **metrics
                        })
                        del hybrid_transformer # Free memory
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                    except Exception as e:
                        logging.error(f"ERROR testing hybrid transformer {model_name_for_hf}: {e}")
                        all_results.append({
                            "model_type": "hybrid", "model_name": model_name_for_hf,
                            "tokenizer_name": tokenizer_name_for_hf, "status": "error", "error_message": str(e),
                            "NMSE": float('nan'), "PR-AUC": float('nan')
                        })

    # 4. Test simple methods
    if test_simple_methods:
        logging.info("\n\n--- Testing Non-Trainable Similarity Methods ---")

        # Define the methods to test
        # We can test multiple embedding models by creating different instances
        methods_to_test = {
            "TF-IDF Cosine Similarity": CosineSimilarityTFIDF(),
            "Embedding Sim (MiniLM-L6-v2)": TransformerEmbeddingSimilarity(model_name="sentence-transformers/all-MiniLM-L6-v2"),
            # "Embedding Sim (BERT-base)": TransformerEmbeddingSimilarity(model_name="bert-base-uncased"),
        }

        # We need a dataset to test on. We can use the one prepared for the default tokenizer.
        default_tokenizer = HEADLINE_CONTENT_CONFIG.get("tokenizer_name", "bert-base-uncased")
        logging.info(f"\nUsing dataset prepared for '{default_tokenizer}' for non-trainable method evaluation.")
        dataset_check(default_tokenizer) # Ensures the data is available
        _, test_csv_basic_path = get_basic_csv_paths(default_tokenizer)

        if not os.path.exists(test_csv_basic_path):
            logging.warning(f"Basic test CSV not found at {test_csv_basic_path}. Cannot run simple method tests.")
        else:
            for name, method_instance in methods_to_test.items():
                try:
                    # Use our new wrapper to evaluate the method
                    evaluator = SimilarityMethodEvaluator(method=method_instance, model_type=name)
                    metrics, _ = evaluator.test(test_csv_basic_path)

                    logging.info(f"Metrics for {name}: {metrics}")
                    all_results.append({
                        "model_type": name,
                        "model_name": name, # Using name for both fields for clarity in the report
                        "tokenizer_name": "N/A", # Not applicable
                        "status": "success",
                        **metrics # Add all returned metrics to the results
                    })

                except Exception as e:
                    logging.error(f"ERROR testing {name}: {e}")
                    all_results.append({
                        "model_type": name,
                        "model_name": name,
                        "tokenizer_name": "N/A",
                        "status": "error", "error_message": str(e),
                    })

    # 5. Save all results to a CSV file
    if not all_results:
        logging.warning("\nNo results were generated. Skipping summary file creation.")
        return

    results_df = pd.DataFrame(all_results)
    # Reorder columns for better readability
    fixed_cols = ["model_type", "model_name", "tokenizer_name", "status"]
    metric_cols = sorted([c for c in results_df.columns if c not in fixed_cols and c != "error_message"])
    error_col = ["error_message"] if "error_message" in results_df.columns else []
    results_df = results_df[fixed_cols + metric_cols + error_col]

    summary_file_path = os.path.join(output_root_directory,
                                     f"fast_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(summary_file_path, index=False)
    logging.info(f"\n--- Model Testing Finished ---")
    logging.info(f"Summary of all results saved to: {summary_file_path}")
    logging.info("\nFinal Results Summary:")
    logging.info(results_df.to_markdown(index=False))


if __name__ == "__main__":
    transformers_tested = HEADLINE_CONTENT_MODELS_PRETRAINED
    run_fast_test(
        transformers_tested,
        test_standard_transformer=False,
        test_hybrid_transformer=False,
        test_simple_methods=True
    )