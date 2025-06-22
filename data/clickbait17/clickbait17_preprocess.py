import os
import json
import csv
import pandas as pd
import re
import logging

logger = logging.getLogger(__name__)

def dataset17_create_csv(json_path: str = None):
    if not json_path:
        json_path = os.path.dirname(__file__)
    json_instances = "instances.jsonl"
    json_truth = "truth.jsonl"

    path_instances = os.path.join(json_path, json_instances)
    path_truth = os.path.join(json_path, json_truth)
    df_instances = pd.read_json(path_instances, lines=True)
    # Apply clean_text to all relevant text columns
    for col in ['postText', 'targetTitle', 'targetParagraphs', 'targetDescription', 'targetKeywords']:
        if col in df_instances.columns:
            df_instances[col] = df_instances[col].apply(clean_text)

    df_truth = pd.read_json(path_truth, lines=True)
    df_merged = pd.merge(df_instances, df_truth, on="id", how="inner")
    df_merged = df_merged[["postText", "targetTitle", "targetParagraphs", "truthMedian"]] # "targetDescription", "targetKeywords" had too many missing values
    df_merged = df_merged.rename(columns={"postText": "post", "targetTitle": "headline", "targetParagraphs": "content", "truthMedian": "clickbait_score"})

    # Ensure content column is not empty after cleaning, fillna with an empty string
    df_merged['content'] = df_merged['content'].fillna('')

    df_merged.to_csv("clickbait17_" + os.path.basename(os.path.normpath(json_path)) + ".csv", index=False, quoting=csv.QUOTE_ALL)
    return df_merged

def clean_text(val):
    if isinstance(val, list):
        # FIX: Join paragraphs with a newline to preserve structure
        val = "\n".join(str(item).strip() for item in val if item)
    elif not isinstance(val, str):
        # Return non-string values (like NaN) as is, to be handled later
        return val

    # Removing special characters
    val = re.sub(r"[^\w\s.,:;!?@#&()'\"%-]", "", val)
    # val = val.replace('\n', '__NEWLINE__')
    val = re.sub(r"\s+", " ", val)
    # val = val.replace('__NEWLINE__', '\n')
    return val.strip()