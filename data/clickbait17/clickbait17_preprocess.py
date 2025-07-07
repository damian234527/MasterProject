"""Preprocesses the raw JSONL data from the Clickbait17 dataset.

This script contains functions to read the original 'instances.jsonl' and
'truth.jsonl' files, merge them based on their common ID, select and rename
relevant columns, and perform basic text cleaning. The final, clean, and merged
data is saved as a single CSV file, ready for further processing.
"""
import os
import json
import csv
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)


def dataset17_create_csv(json_path: str = None):
    """Creates a preprocessed CSV file from the raw Clickbait17 JSONL data.

    This function reads 'instances.jsonl' (containing the text data) and
    'truth.jsonl' (containing the clickbait scores), cleans the text fields,
    merges the data, and saves it to a structured CSV file.

    Args:
        json_path (str, optional): The path to the directory containing the
            raw .jsonl files. If not provided, it defaults to the directory
            of this script.

    Returns:
        A pandas DataFrame containing the merged and preprocessed data.
    """
    if not json_path:
        json_path = os.path.dirname(__file__)
    json_instances = "instances.jsonl"
    json_truth = "truth.jsonl"

    # Construct full paths to the JSONL files and read them into DataFrames.
    path_instances = os.path.join(json_path, json_instances)
    path_truth = os.path.join(json_path, json_truth)
    df_instances = pd.read_json(path_instances, lines=True)

    # Apply the text cleaning function to all relevant text columns.
    for col in ['postText', 'targetTitle', 'targetParagraphs', 'targetDescription', 'targetKeywords']:
        if col in df_instances.columns:
            df_instances[col] = df_instances[col].apply(clean_text)

    df_truth = pd.read_json(path_truth, lines=True)
    # Merge the instances and truth DataFrames on their common 'id' column.
    df_merged = pd.merge(df_instances, df_truth, on="id", how="inner")
    # Select a subset of the most relevant columns and rename them for clarity.
    df_merged = df_merged[["postText", "targetTitle", "targetParagraphs", "truthMean"]]
    df_merged = df_merged.rename(columns={
        "postText": "post",
        "targetTitle": "headline",
        "targetParagraphs": "content",
        "truthMean": "clickbait_score"
    })

    # Ensure text columns are not null and are of string type.
    df_merged["post"] = df_merged["post"].fillna("")
    df_merged["content"] = df_merged["content"].fillna("")

    # Save the processed DataFrame to a new CSV file.
    output_csv_path = "clickbait17_" + os.path.basename(os.path.normpath(json_path)) + ".csv"
    df_merged.to_csv(output_csv_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Successfully created CSV file at: {output_csv_path}")
    return df_merged


def clean_text(val):
    """Cleans a text string by removing special characters and extra whitespace.

    This function can handle both single strings and lists of strings (which
    are common in the 'targetParagraphs' field of the original data).

    Args:
        val: The input value, which can be a string, a list of strings, or NaN.

    Returns:
        The cleaned text as a single string, or the original value if it is
        not a string (e.g., NaN).
    """
    # If the value is a list of paragraphs, join them into a single text block.
    if isinstance(val, list):
        val = " ".join(str(item).strip() for item in val if item)
    # If the value is not a string (e.g., NaN), return it as is to be handled later.
    elif not isinstance(val, str):
        return val

    # Standardize quotes and remove unwanted characters and excess whitespace.
    val = val.replace('"', "'")
    val = re.sub(r"[^\w\s.,:;!?@#&()\'%-]", "", val)
    val = re.sub(r"\s+", " ", val)
    return val.strip()