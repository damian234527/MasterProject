import os
import json
import csv
import pandas as pd
import re

def dataset17_create_csv(json_path: str = None):
    if not json_path:
        json_path = os.path.dirname(__file__)
        print(json_path)
    json_instances = "instances.jsonl"
    json_truth = "truth.jsonl"

    path_instances = os.path.join(json_path, json_instances)
    path_truth = os.path.join(json_path, json_truth)
    print(path_instances)
    df_instances = pd.read_json(path_instances, lines=True)
    df_instances = df_instances.applymap(clean_text)
    df_truth = pd.read_json(path_truth, lines=True)
    df_merged = pd.merge(df_instances, df_truth, on="id", how="inner")
    df_merged = df_merged[["postText", "targetTitle", "targetParagraphs", "truthMedian"]] # "targetDescription", "targetKeywords" had too many missing values
    df_merged = df_merged.rename(columns={"postText": "post", "targetTitle": "headline", "targetParagraphs": "content", "truthMedian": "clickbait_score"})

    df_merged.to_csv("clickbait17_" + os.path.basename(os.path.normpath(json_path)) + ".csv", index=False, quoting=csv.QUOTE_ALL)
    return df_merged

def clean_text(val):
    if isinstance(val, list):
        val = " ".join(str(item) for item in val)
    elif not isinstance(val, str):
        return val

    # Removing special characters
    val = re.sub(r"[^\w\s.,:;!?@#&()'\"%-]", "", val)   # Keeping basic symbols
    val = re.sub(r"\s+", " ", val)                      # Deleting multiple spaces
    return val.strip()
