"""Preprocesses the raw JSONL data from the Clickbait17 dataset.

This script reads the 'instances.jsonl' and 'truth.jsonl' files, merges them,
selects relevant columns, and performs text cleaning. The final preprocessed
data is saved as a CSV file.
"""
import os
import json
import csv
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def dataset17_create_csv(json_path: str = None):
    """Creates a CSV file from the raw Clickbait17 JSONL data.

    Args:
        json_path: The path to the directory containing 'instances.jsonl' and
            'truth.jsonl'. If not provided, it defaults to the directory
            of this script.

    Returns:
        A pandas DataFrame containing the merged and preprocessed data.
    """
    if not json_path:
        json_path = os.path.dirname(__file__)
    json_instances = "instances.jsonl"
    json_truth = "truth.jsonl"

    # Construct path to the JSONL file and read them
    path_instances = os.path.join(json_path, json_instances)
    path_truth = os.path.join(json_path, json_truth)
    df_instances = pd.read_json(path_instances, lines=True)

    # Apply clean_text to relevant text columns
    for col in ['postText', 'targetTitle', 'targetParagraphs', 'targetDescription', 'targetKeywords']:
        if col in df_instances.columns:
            df_instances[col] = df_instances[col].apply(clean_text)

    df_truth = pd.read_json(path_truth, lines=True)
    # Merge the instances and truth data on their common 'id'
    df_merged = pd.merge(df_instances, df_truth, on="id", how="inner")
    # Select a subset of columns and rename them for clarity
    df_merged = df_merged[["postText", "targetTitle", "targetParagraphs", "truthMean"]] # "targetDescription", "targetKeywords" had too many missing values
    df_merged = df_merged.rename(columns={"postText": "post", "targetTitle": "headline", "targetParagraphs": "content", "truthMean": "clickbait_score"})

    # Ensure content column is not empty after cleaning, fillna with an empty string
    df_merged['content'] = df_merged['content'].fillna('')

    # Save the processed DataFrame to a new CSV file
    output_csv_path = "clickbait17_" + os.path.basename(os.path.normpath(json_path)) + ".csv"
    df_merged.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Successfully created CSV file at: {output_csv_path}")
    return df_merged

def clean_text(val):
    """Cleans a text string by removing special characters and extra whitespace.

    Args:
        val: The input value, which can be a string or a list of strings.

    Returns:
        The cleaned text as a single string, or the original value if it's
        not a string.
    """
    # If the value is a list of paragraphs, join them into a single text block
    if isinstance(val, list):
        val = "\n".join(str(item).strip() for item in val if item)
    # If the value is not a string (e.g., NaN), return it as is
    elif not isinstance(val, str):
        return val

    # Removing special characters and extra whitespace
    val = re.sub(r"[^\w\s.,:;!?@#&()'\"%-]", "", val)
    # val = val.replace('\n', '__NEWLINE__')
    val = re.sub(r"\s+", " ", val)
    # val = val.replace('__NEWLINE__', '\n')
    return val.strip()
