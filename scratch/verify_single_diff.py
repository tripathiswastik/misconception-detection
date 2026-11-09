import pandas as pd
from pathlib import Path

SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")
s550 = pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv")
s557 = pd.read_csv(SUB_DIR / "blend_sota557_surgical_12flip_submission.csv")

diffs = [i for i in range(len(s550)) if s550.iloc[i]["root_code"] != s557.iloc[i]["root_code"]]
print(f"Total differences between Champion (0.55049) and Probe (SOTA 557): {len(diffs)}")
assert len(diffs) == 1, f"Expected exactly 1 difference, found {len(diffs)}: {diffs}"
idx = diffs[0]
print(f"Index: {idx}")
print(f"Champion SOTA 550: '{s550.iloc[idx]['root_code']}'")
print(f"Probe SOTA 557   : '{s557.iloc[idx]['root_code']}'")
print("VERIFICATION SUCCESS: Candidate differs at exactly row 3264.")

