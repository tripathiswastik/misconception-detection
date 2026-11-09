from pathlib import Path
from typing import cast
import pandas as pd

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))

s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
s551 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv"))

flips = [i for i in range(len(s550)) if s550.iloc[i]["root_code"] != s551.iloc[i]["root_code"]]
print(f"Golden Flips count: {len(flips)}")
print("Golden Flips indices:", flips)

for i in flips:
    qid = test_df.iloc[i]["question_id"]
    train_q_codes = train_df[train_df["question_id"] == qid]["root_code"].dropna().unique().tolist()
    print(f"Row {i:4d} (qid={qid}): Base='{s551.iloc[i]['root_code']}' -> SOTA550='{s550.iloc[i]['root_code']}' | Train codes for question: {train_q_codes}")

# Now inspect row 3264
print("\n" + "=" * 50)
print("INSPECTING ROW 3264:")
r3264 = test_df.iloc[3264]
print("Row 3264 details:")
for col in test_df.columns:
    print(f"  {col}: {r3264[col]}")
qid_3264 = r3264["question_id"]
train_matches = train_df[train_df["question_id"] == qid_3264]
print(f"Train matches for question {qid_3264}: {len(train_matches)} rows")
print(train_matches[["question_id", "root_code", "image_file"]].head(10))

print(f"\nSOTA 550 prediction on row 3264: '{s550.iloc[3264]['root_code']}'")
print(f"SOTA 551 prediction on row 3264: '{s551.iloc[3264]['root_code']}'")
