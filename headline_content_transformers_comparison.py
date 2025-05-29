import os
import pandas as pd
from datetime import datetime
import torch # For torch.cuda.empty_cache

# Ensure your project modules are accessible
# Add project root to PYTHONPATH if running from a different directory, or adjust imports
from headline_content_models import (
    ClickbaitTransformer,
    ClickbaitFeatureEnhancedTransformer
)
from data.clickbait17.clickbait17_prepare import prepare_clickbait17_datasets
from data.clickbait17.clickbait17_utils import get_basic_csv_paths, get_feature_csv_paths, get_safe_name
from config import HEADLINE_CONTENT_CONFIG, DATASETS_CONFIG, HEADLINE_CONTENT_MODELS_PRETRAINED

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
    test_hybrid_transformer: bool = False # Set to True to also test feature-enhanced models
):
    """
    Runs a fast test for a list of transformer model configurations.
    """
    print("\n--- Starting Fast Test of Headline-Content Transformers ---")
    output_root_directory = "models/fast_test_outputs"
    os.makedirs(output_root_directory, exist_ok=True)
    all_results = []

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
            print(f"Skipping invalid model configuration: {model_config_entry}")
            continue

        print(f"\n>>> Processing Model: {model_name_for_hf} (Tokenizer: {tokenizer_name_for_hf})")

        # 1. Prepare Datasets
        # The prepare_clickbait17_datasets in your train.py is called without arguments.
        # This implies it might use a default tokenizer or prepare for all.
        # For testing multiple tokenizers, ensure this step correctly provides data
        # for the *current* tokenizer_name_for_hf.
        # If prepare_clickbait17_datasets needs to be tokenizer-specific, you might need to modify it
        # or ensure data is pre-generated for all tokenizers in MODELS_TO_TEST.
        try:
            print(f"Ensuring datasets are prepared for tokenizer: {tokenizer_name_for_hf}...")
            # Calling as in your train.py
            prepare_clickbait17_datasets(tokenizer_name=tokenizer_name_for_hf)
        except Exception as e:
            print(f"Warning: Dataset preparation step encountered an issue for {tokenizer_name_for_hf}: {e}")
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
                print(f"Basic dataset CSVs not found for {tokenizer_name_for_hf}. Skipping standard model test.")
                all_results.append({
                    "model_type": "standard", "model_name": model_name_for_hf,
                    "tokenizer_name": tokenizer_name_for_hf, "status": "skipped (dataset missing)",
                    "NMSE": float('nan'), "PR-AUC": float('nan') # Ensure consistent columns
                })
            else:
                print(f"\n--- Testing ClickbaitTransformer (Standard) for {model_name_for_hf} ---")
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
                    print("Training standard model (fast mode)...")
                    standard_transformer.train(train_csv_basic_path, val_csv_basic_path)
                    print("Evaluating standard model...")
                    # The .test() method is defined in ClickbaitModelBase
                    metrics, _ = standard_transformer.test(val_csv_basic_path)
                    print(f"Metrics for {model_name_for_hf} (Standard): {metrics}")
                    all_results.append({
                        "model_type": "standard", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "success",
                        **metrics  # Assumes metrics dict contains NMSE, PR-AUC etc.
                    })
                    del standard_transformer # Free memory
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e:
                    print(f"ERROR testing standard transformer {model_name_for_hf}: {e}")
                    all_results.append({
                        "model_type": "standard", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "error", "error_message": str(e),
                        "NMSE": float('nan'), "PR-AUC": float('nan')
                    })

        # 3. Test Hybrid ClickbaitFeatureEnhancedTransformer
        if test_hybrid_transformer:
            if not os.path.exists(train_csv_features_path) or not os.path.exists(val_csv_features_path):
                print(f"Feature-augmented dataset CSVs not found for {tokenizer_name_for_hf}. Skipping hybrid model test.")
                all_results.append({
                    "model_type": "hybrid", "model_name": model_name_for_hf,
                    "tokenizer_name": tokenizer_name_for_hf, "status": "skipped (dataset missing)",
                    "NMSE": float('nan'), "PR-AUC": float('nan')
                })
            else:
                print(f"\n--- Testing ClickbaitFeatureEnhancedTransformer (Hybrid) for {model_name_for_hf} ---")
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
                    print("Training hybrid model (fast mode)...")
                    hybrid_transformer.train(train_csv_features_path, val_csv_features_path)
                    print("Evaluating hybrid model...")
                    metrics, _ = hybrid_transformer.test(val_csv_features_path)
                    print(f"Metrics for {model_name_for_hf} (Hybrid): {metrics}")
                    all_results.append({
                        "model_type": "hybrid", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "success",
                        **metrics
                    })
                    del hybrid_transformer # Free memory
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                except Exception as e:
                    print(f"ERROR testing hybrid transformer {model_name_for_hf}: {e}")
                    all_results.append({
                        "model_type": "hybrid", "model_name": model_name_for_hf,
                        "tokenizer_name": tokenizer_name_for_hf, "status": "error", "error_message": str(e),
                        "NMSE": float('nan'), "PR-AUC": float('nan')
                    })

    # 4. Save all results to a CSV file
    results_df = pd.DataFrame(all_results)
    summary_file_path = os.path.join(output_root_directory, f"fast_test_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    results_df.to_csv(summary_file_path, index=False)
    print(f"\n--- Fast Testing Finished ---")
    print(f"Summary of results saved to: {summary_file_path}")
    print(results_df)


if __name__ == "__main__":
    transformers_tested = HEADLINE_CONTENT_MODELS_PRETRAINED
    run_fast_test(
        transformers_tested,
        test_standard_transformer=True,
        test_hybrid_transformer=False  # Change to True if you also want to screen hybrid models
    )