"""Design and Benchmark Question-Held-Out Split for Rare-Class Pseudo-Labeling Pilot.

Protocol:
1. GroupKFold on question_id (5 folds).
   Ensures zero question overlap between train and validation.
2. Identify rare classes (<= 10 train samples) and evaluate baseline OOF MAP@3 on these rare classes.
3. Define the Teacher Consensus Filter for pseudo-labeling:
   - Margin requirement (top_p - second_p >= delta).
   - Multi-teacher consensus (e.g. M15 + ArcFace agree).
   - Question prior consistency (predicted class must be admissible for the question topic).
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("QUESTION-HELD-OUT VALIDATION SETUP FOR RARE-CLASS PILOT")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    unl_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "unlabelled.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    # GroupKFold on question_id
    gkf = GroupKFold(n_splits=5)
    groups = train_df["question_id"].values

    counts = Counter(train_df["root_code"].dropna())
    rare_10 = set(c for c, cnt in counts.items() if cnt <= 10)
    print(f"Total labeled rows: {len(train_df)}")
    print(f"Unique questions in Train: {train_df['question_id'].nunique()}")
    print(f"Classes with <= 10 train samples: {len(rare_10)}")

    # Check fold distribution
    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups)):
        val_df = train_df.iloc[val_idx]
        val_questions = set(val_df["question_id"])
        trn_questions = set(train_df.iloc[trn_idx]["question_id"])
        assert len(val_questions & trn_questions) == 0, f"Fold {fold} has question leakage!"
        val_rare = val_df[val_df["root_code"].isin(rare_10)]
        print(f"Fold {fold+1}: {len(trn_idx)} trn rows, {len(val_idx)} val rows | {len(val_questions)} val questions | {len(val_rare)} rare-class val rows")

    # Evaluate Baseline Champion OOF on Question-Held-Out Folds
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Score full OOF
    top3_all = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_m15, axis=1)]
    base_map3 = calculate_map3(y_true, top3_all)
    print(f"\nM15 Base Global MAP@3 (Fold-Local OOF): {base_map3:.5f}")

    # Score on rare-class rows only
    rare_mask = np.array([c in rare_10 for c in y_true])
    rare_y_true = [y_true[i] for i in range(len(y_true)) if rare_mask[i]]
    rare_top3 = [top3_all[i] for i in range(len(y_true)) if rare_mask[i]]
    rare_map3 = calculate_map3(rare_y_true, rare_top3)
    print(f"M15 Base MAP@3 on Rare-Class Rows ({sum(rare_mask)} rows): {rare_map3:.5f}")

    # Inspect Unlabelled Traces for questions tied to rare classes
    train_q_to_rare = {}
    for _, row in train_df.iterrows():
        c = str(row["root_code"])
        if c in rare_10:
            qid = str(row["question_id"])
            train_q_to_rare.setdefault(qid, set()).add(c)

    unl_rare_q = unl_df[unl_df["question_id"].isin(train_q_to_rare.keys())]
    print(f"\nUnlabelled Pool for Rare-Class Questions: {len(unl_rare_q)} traces across {unl_rare_q['question_id'].nunique()} questions")

    # Group by question to see candidate batch sizes
    q_counts = unl_rare_q["question_id"].value_counts()
    print("Top 5 questions with largest unlabelled pools for rare classes:")
    for qid, count in q_counts.head(5).items():
        associated_rare = train_q_to_rare.get(qid, set())
        print(f"  Question {qid}: {count} unlabelled traces | Rare class(es): {associated_rare}")

    print("=" * 80)

if __name__ == "__main__":
    main()

