import os
import json
import csv
import pandas as pd
import re

def data_load(json_path: str = None):
    if not json_path:
        json_path = os.getcwd()
    json_instances = "instances.jsonl"
    json_truth = "truth.jsonl"

    path_instances = os.path.join(json_path, json_instances)
    path_truth = os.path.join(json_path, json_truth)
    df_instances = pd.read_json(os.path.join(json_path, json_instances), lines=True)
    df_instances = df_instances.applymap(clean_text)
    df_truth = pd.read_json(os.path.join(json_path, json_truth), lines=True)
    df_merged = pd.merge(df_instances, df_truth, on="id", how="inner")
    df_merged = df_merged[["postText", "targetTitle", "targetDescription", "targetKeywords", "targetParagraphs", "truthMedian"]]
    df_merged.to_csv("merged.csv", index=False, quoting=csv.QUOTE_ALL)

def clean_text(val):
    if isinstance(val, list):
        val = " ".join(str(item) for item in val)
    elif not isinstance(val, str):
        return val

    # Removing special characters
    val = re.sub(r"[^\w\s.,:;!?@#&()'\"%-]", "", val)   # Keeping basic symbols
    val = re.sub(r"\s+", " ", val)                      # Deleting multiple spaces
    return val.strip()

if __name__ == "__main__":
    path_train = "train"
    path_test = "test"
    path_validation = "validation"
    path = os.path.join(os.getcwd(), path_train)
    data_load(path)
