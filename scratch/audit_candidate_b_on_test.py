"""Audit Candidate B Domain Gating Policy on Test Set."""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, cast
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
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    # Training question label distributions
    q_label_dist: Dict[str, Counter] = {
        q: Counter(grp["root_code"].fillna("NONE").values)
        for q, grp in train_df.groupby("question_id")
    }

    # Load test submissions for S550 and S554
    s550_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s554_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"))

    # Also load test binary error predictions and probability matrices if available
    # Check test artifacts
    print(f"Test size: {len(test_df)}")
    diff_mask = s550_sub["root_code"] != s554_sub["root_code"]
    diff_indices = np.where(diff_mask)[0]
    print(f"Number of test rows where S550 != S554: {len(diff_indices)} / {len(test_df)} ({len(diff_indices)/len(test_df):.2%})")

    # Let's see how many differ at Rank 1
    r1_550 = [row.split()[0] for row in s550_sub["root_code"]]
    r1_554 = [row.split()[0] for row in s554_sub["root_code"]]
    r1_diff = [i for i in range(len(test_df)) if r1_550[i] != r1_554[i]]
    print(f"Number of test rows where Rank 1 differs: {len(r1_diff)} / {len(test_df)} ({len(r1_diff)/len(test_df):.2%})")

    # Check row 3264
    if 3264 in diff_indices or 3264 in r1_diff:
        print(f"Row 3264 in diff: S550={s550_sub.iloc[3264]['root_code']} | S554={s554_sub.iloc[3264]['root_code']}")
    else:
        print(f"Row 3264 is identical between S550 and S554: {s550_sub.iloc[3264]['root_code']}")

    # Let's inspect which files exist in artifacts for test
    test_files = list(ART_DIR.glob("*test*"))
    print("\nTest files in artifacts:")
    for f in test_files:
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
