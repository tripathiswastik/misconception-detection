"""Audit question-level misconception consistency in 5-fold cross validation.

We measure:
For rows in the validation set whose question_id appeared in the training fold:
1. Is the true root_code in the set of root_codes seen for that question in the training fold?
2. How often is it NONE vs Misconception?
3. If we restrict/boost candidate misconceptions to only those observed for that question in the training fold, what happens to MAP@3?
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    total_val_rows = 0
    val_rows_with_q_in_train = 0
    gt_in_train_q_set = 0
    gt_misc_in_train_q_set = 0
    gt_misc_count = 0

    for f_idx, fold in enumerate(splits):
        tr = fold["train_indices"]
        va = fold["val_indices"]
        tr_df = train_df.iloc[tr]
        va_df = train_df.iloc[va]

        # Map question_id -> set of all codes (including NONE or non-NONE)
        q_all_codes: Dict[str, Set[str]] = {}
        q_misc_codes: Dict[str, Set[str]] = {}
        for q, grp in tr_df.groupby("question_id"):
            codes = set(grp["root_code"].fillna("NONE").unique())
            q_all_codes[q] = codes
            q_misc_codes[q] = {c for c in codes if c != "NONE"}

        for _, row in va_df.iterrows():
            total_val_rows += 1
            qid = row["question_id"]
            gt = row["root_code"] if pd.notna(row["root_code"]) else "NONE"

            if qid in q_all_codes:
                val_rows_with_q_in_train += 1
                if gt in q_all_codes[qid]:
                    gt_in_train_q_set += 1

                if gt != "NONE":
                    gt_misc_count += 1
                    if gt in q_misc_codes[qid]:
                        gt_misc_in_train_q_set += 1

    print(f"Total validation rows across 5 folds: {total_val_rows}")
    print(f"Validation rows where question appeared in train: {val_rows_with_q_in_train} ({val_rows_with_q_in_train/total_val_rows:.2%})")
    print(f"Ground truth code in train question set (all codes including NONE): {gt_in_train_q_set} / {val_rows_with_q_in_train} ({gt_in_train_q_set/val_rows_with_q_in_train:.2%})")
    print(f"When GT is a Misconception: in train question misc set: {gt_misc_in_train_q_set} / {gt_misc_count} ({gt_misc_in_train_q_set/gt_misc_count:.2%})")


if __name__ == "__main__":
    main()
