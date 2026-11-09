import pandas as pd
from collections import Counter
from pathlib import Path
from typing import cast

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
tax_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "taxonomy_v1.csv"))

counts = Counter(train_df["root_code"].dropna())
n_clean = sum(1 for x in train_df["root_code"] if x == "NONE" or pd.isna(x))
n_misc = len(train_df) - n_clean

print(f"Total train traces: {len(train_df)}")
print(f"  Clean (NONE): {n_clean} ({n_clean/len(train_df):.1%})")
print(f"  Misconceptions: {n_misc} ({n_misc/len(train_df):.1%})")
print(f"  Distinct active classes in train: {len(counts)}")

tax_codes = set(tax_df["id"].unique())
print(f"Total taxonomy classes: {len(tax_codes)}")

rare_10 = {c: cnt for c, cnt in counts.items() if cnt <= 10}
rare_5 = {c: cnt for c, cnt in counts.items() if cnt <= 5}
zero_train = tax_codes - set(counts.keys())

print(f"\nClasses with <= 10 train samples: {len(rare_10)} classes ({sum(rare_10.values())} total train samples)")
print(f"Classes with <= 5 train samples:  {len(rare_5)} classes ({sum(rare_5.values())} total train samples)")
print(f"Taxonomy classes with 0 train samples: {len(zero_train)} classes -> {sorted(list(zero_train))}")

# Check test and unlabelled question overlap
test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
unl_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "unlabelled.csv"))

train_q = set(train_df["question_id"].unique())
test_q = set(test_df["question_id"].unique())
unl_q = set(unl_df["question_id"].unique())

print(f"\nQuestion Overlap Analysis:")
print(f"  Unique Questions in Train: {len(train_q)}")
print(f"  Unique Questions in Test:  {len(test_q)}")
print(f"  Unique Questions in Unl:   {len(unl_q)}")
print(f"  Test questions in Train:   {len(test_q & train_q)} / {len(test_q)} ({len(test_q & train_q)/len(test_q):.1%})")
print(f"  Test questions in Unl:     {len(test_q & unl_q)} / {len(test_q)} ({len(test_q & unl_q)/len(test_q):.1%})")
print(f"  Unlabelled traces on Test questions: {len(unl_df[unl_df['question_id'].isin(test_q)])} / {len(unl_df)}")
