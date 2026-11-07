"""Audit Question Misconception Distribution and Multiplicity in train.csv."""
import sys
from collections import Counter
from pathlib import Path
from typing import cast
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

print("=" * 80)
print("QUESTION MISCONCEPTION DISTRIBUTION AUDIT")
print("=" * 80)
print(f"Total Train Rows: {len(train_df)}")
print(f"Total Unique Train Questions: {train_df['question_id'].nunique()}")
print(f"Avg Traces per Question in Train: {len(train_df) / train_df['question_id'].nunique():.2f}")

print(f"\nTotal Test Rows: {len(test_df)}")
print(f"Total Unique Test Questions: {test_df['question_id'].nunique()}")
print(f"Avg Traces per Question in Test: {len(test_df) / test_df['question_id'].nunique():.2f}")

# Overlap
tr_q = set(train_df["question_id"].unique())
te_q = set(test_df["question_id"].unique())
overlap_q = tr_q.intersection(te_q)
print(f"\nQuestions in Test also in Train: {len(overlap_q)} / {len(te_q)} ({len(overlap_q) / len(te_q):.2%})")
te_overlap_rows = test_df[test_df["question_id"].isin(tr_q)]
print(f"Test Rows with Question in Train: {len(te_overlap_rows)} / {len(test_df)} ({len(te_overlap_rows) / len(test_df):.2%})")

# Per Question Misconception Variety in Train
q_misc_counts = {}
for q, grp in train_df.groupby("question_id"):
    miscs = set(grp["root_code"].dropna().unique())
    q_misc_counts[q] = len(miscs)

var_dist = Counter(q_misc_counts.values())
print("\nUnique Non-NONE Misconceptions observed per Question in Train:")
for num_miscs, count in sorted(var_dist.items()):
    print(f"  {num_miscs} unique misconceptions: {count} questions ({count / len(q_misc_counts):.1%})")

# Unlabelled Traces Overlap
unlab_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "unlabelled.csv"))
print(f"\nTotal Unlabelled Rows: {len(unlab_df)}")
print(f"Total Unique Unlabelled Questions: {unlab_df['question_id'].nunique()}")
unlab_q = set(unlab_df["question_id"].unique())
print(f"Questions in Unlabelled also in Train: {len(unlab_q.intersection(tr_q))} / {len(unlab_q)} ({len(unlab_q.intersection(tr_q)) / len(unlab_q):.2%})")
print(f"Unlabelled Rows with Question in Train: {len(unlab_df[unlab_df['question_id'].isin(tr_q)])} / {len(unlab_df)} ({len(unlab_df[unlab_df['question_id'].isin(tr_q)]) / len(unlab_df):.2%})")
print(f"Questions in Test also in Unlabelled: {len(te_q.intersection(unlab_q))} / {len(te_q)} ({len(te_q.intersection(unlab_q)) / len(te_q):.2%})")
print("=" * 80)

