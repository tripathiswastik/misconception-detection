"""Audit Rare Taxonomy Classes in Train, Test, and Unlabelled.

Questions:
1. Which classes have <= 10 train samples?
2. Which test questions are associated with those rare classes?
3. What is our current model predicting on those test rows?
"""
from collections import Counter
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager

def main():
    print("=" * 80, flush=True)
    print("AUDITING RARE TAXONOMY CLASSES IN TRAIN & TEST", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    unl_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "unlabelled.csv"))

    counts = Counter(train_df["root_code"].dropna())
    tax_codes = [c for c in classes if c != "NONE"]

    rare_10 = [c for c in tax_codes if counts.get(c, 0) <= 10]
    rare_5 = [c for c in tax_codes if counts.get(c, 0) <= 5]
    zero_train = [c for c in tax_codes if counts.get(c, 0) == 0]

    print(f"Total Taxonomy Classes (excluding NONE): {len(tax_codes)}")
    print(f"Classes with <= 10 train samples: {len(rare_10)} / {len(tax_codes)} ({len(rare_10)/len(tax_codes):.1%})")
    print(f"Classes with <= 5 train samples : {len(rare_5)} / {len(tax_codes)} ({len(rare_5)/len(tax_codes):.1%})")
    print(f"Classes with 0 train samples    : {len(zero_train)} -> {zero_train}")

    # Check question association: which questions in train produced these rare classes?
    q_to_rare = {}
    for _, row in train_df.iterrows():
        c = str(row["root_code"])
        if c in rare_10:
            qid = str(row["question_id"])
            if qid not in q_to_rare:
                q_to_rare[qid] = []
            q_to_rare[qid].append(c)

    print(f"\nUnique questions in Train associated with rare classes: {len(q_to_rare)}")

    # How many test rows belong to these rare-associated questions?
    test_rare_rows = test_df[test_df["question_id"].isin(q_to_rare.keys())]
    print(f"Test rows belonging to rare-associated questions: {len(test_rare_rows)} / {len(test_df)} ({len(test_rare_rows)/len(test_df):.1%})")

    # What does our Peak SOTA 550 model predict on these test rows?
    s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s550_test_rare = s550.iloc[test_rare_rows.index]

    r1_rare_test = [p.split()[0] for p in s550_test_rare["root_code"]]
    print(f"\nSOTA 550 Rank 1 predictions on these {len(test_rare_rows)} test rows:")
    print(f"  Predicted NONE: {sum(1 for x in r1_rare_test if x == 'NONE')} ({sum(1 for x in r1_rare_test if x == 'NONE')/len(test_rare_rows):.1%})")
    print(f"  Predicted Rare Class: {sum(1 for x in r1_rare_test if x in rare_10)} ({sum(1 for x in r1_rare_test if x in rare_10)/len(test_rare_rows):.1%})")
    print(f"  Predicted Other Class: {sum(1 for x in r1_rare_test if x != 'NONE' and x not in rare_10)} ({sum(1 for x in r1_rare_test if x != 'NONE' and x not in rare_10)/len(test_rare_rows):.1%})")

    # Unlabelled traces available for these rare questions
    unl_rare_traces = unl_df[unl_df["question_id"].isin(q_to_rare.keys())]
    print(f"\nUnlabelled student traces available on these rare questions: {len(unl_rare_traces)}")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

