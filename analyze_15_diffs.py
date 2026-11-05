import json
from pathlib import Path
import shutil
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
SUB_DIR = PROJECT_ROOT / "submissions"
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
DOWNLOADS_DIR = Path("C:/Users/hp/Downloads")

test_df = pd.read_csv(DATA_DIR / "test.csv")
train_df = pd.read_csv(DATA_DIR / "train.csv")
champ_df = pd.read_csv(SUB_DIR / "blend_m15_recalibrated_composite_triblend_448_submission.csv")
quad_df = pd.read_csv(SUB_DIR / "blend_rank1_locked_quad_champion_submission.csv")
cons_df = pd.read_csv(SUB_DIR / "blend_rank1_locked_conservative_champion_submission.csv")

with open(ART_DIR / "taxonomy_classes.json", "r") as f:
    tax_classes = json.load(f)

# Find question topic map
q_to_topic = {r["question_id"]: r["topic_name"] for _, r in train_df.iterrows()}

# Examine the 15 differing rows between Quad and Cons
diff_rows = [i for i in range(len(quad_df)) if quad_df.iloc[i]["root_code"] != cons_df.iloc[i]["root_code"]]

print(f"Analyzing the 15 differing rows between both 0.54716 champions:")
for i in diff_rows:
    qid = test_df.iloc[i]["question_id"]
    topic = q_to_topic.get(qid, "Unknown")
    c_trip = champ_df.iloc[i]["root_code"]
    q_trip = quad_df.iloc[i]["root_code"]
    k_trip = cons_df.iloc[i]["root_code"]
    print(f"Row {i:4d} | Topic: {topic[:20]:20s} | Champ: {c_trip} | Quad: {q_trip} | Cons: {k_trip}")

