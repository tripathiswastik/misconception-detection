import pandas as pd
from pathlib import Path

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

test_df = pd.read_csv(DATA_DIR / "test.csv")
train_df = pd.read_csv(DATA_DIR / "train.csv")

candidates = [418, 809, 2882, 3086]
for idx in candidates:
    r = test_df.iloc[idx]
    qid = r["question_id"]
    train_q = train_df[train_df["question_id"] == qid]
    print(f"\n--- Row {idx} (qid={qid}) ---")
    print(f"Topic: {r['topic_name']} | Grade: {r['grade']}")
    print(f"Question: {r['question_text']}")
    print(f"Train matches ({len(train_q)} rows):")
    print(train_q[["root_code", "image_file"]])
