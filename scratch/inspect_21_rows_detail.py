from pathlib import Path
from typing import cast
import pandas as pd

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
s554 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"))

r1_diff = [i for i in range(len(s550)) if s550.iloc[i]["root_code"].split()[0] != s554.iloc[i]["root_code"].split()[0]]

print("Detailed Student Work & Text for 21 Diverging Rows:\n")
for i in r1_diff:
    t_row = test_df.iloc[i]
    p550 = s550.iloc[i]["root_code"]
    p554 = s554.iloc[i]["root_code"]
    q_text = str(t_row.get("question_text", ""))[:80]
    std_ans = str(t_row.get("student_answer", ""))
    ocr = str(t_row.get("transcription_steps", ""))[:100]
    print(f"Row {i:4d} | Trace: {t_row['trace_id']}")
    print(f"  S550 Pred : {p550}")
    print(f"  S554 Pred : {p554}")
    print(f"  Question  : {q_text}...")
    print(f"  StudentAns: {std_ans}")
    print(f"  OCR Steps : {ocr}")
    print("-" * 80)
