from collections import Counter
from pathlib import Path
from typing import cast
import pandas as pd

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
s554 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"))

r1_diff = [i for i in range(len(s550)) if s550.iloc[i]["root_code"].split()[0] != s554.iloc[i]["root_code"].split()[0]]

# Train questions distribution
q_train_dist = {}
for qid, grp in train_df.groupby("question_id"):
    counts = Counter(grp["root_code"].fillna("NONE"))
    q_train_dist[qid] = counts

print(f"Audit of the 21 Diverging Rows between S550 and S554:")
print("=" * 110)
print(f"{'Row':4s} | {'Trace ID':12s} | {'Question ID':36s} | {'S550 Rank 1':11s} | {'S554 Rank 1':11s} | {'Train Question Distribution'}")
print("-" * 110)

for i in r1_diff:
    t_row = test_df.iloc[i]
    qid = t_row["question_id"]
    tid = t_row["trace_id"]
    p550 = s550.iloc[i]["root_code"].split()[0]
    p554 = s554.iloc[i]["root_code"].split()[0]
    dist = dict(q_train_dist.get(qid, {"UNSEEN": 1}))
    print(f"{i:4d} | {tid:12s} | {qid:36s} | {p550:11s} | {p554:11s} | {dist}")
