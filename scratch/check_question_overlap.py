import json
from pathlib import Path
from typing import cast
import pandas as pd

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))

print("Auditing existing data/splits.json for question leakage across folds:")
for f in splits:
    val_q = set(train_df.iloc[f["val_indices"]]["question_id"])
    trn_q = set(train_df.iloc[f["train_indices"]]["question_id"])
    overlap = val_q & trn_q
    print(f"Fold {f['fold']}: val rows={len(f['val_indices'])}, val questions={len(val_q)}, OVERLAPPING questions with train={len(overlap)} ({len(overlap)/len(val_q):.1%})")

