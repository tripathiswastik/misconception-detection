from pathlib import Path
from typing import cast
import pandas as pd

SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")
s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
s554 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"))

r1_diff = [i for i in range(len(s550)) if s550.iloc[i]["root_code"].split()[0] != s554.iloc[i]["root_code"].split()[0]]
print(f"Total Rank 1 differences between S550 and S554: {len(r1_diff)}")
print(f"Indices: {r1_diff}\n")

print(f"{'Row':5s} | {'SOTA 550 (0.55049)':30s} | {'SOTA 554 (0.55049)':30s}")
print("-" * 75)
for i in r1_diff:
    print(f"{i:5d} | {s550.iloc[i]['root_code']:30s} | {s554.iloc[i]['root_code']:30s}")
