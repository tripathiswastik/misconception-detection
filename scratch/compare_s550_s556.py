import pandas as pd
from pathlib import Path

DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

test_df = pd.read_csv(DATA_DIR / "test.csv")
train_df = pd.read_csv(DATA_DIR / "train.csv")

s550 = pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv")
s551 = pd.read_csv(SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv")
s556 = pd.read_csv(SUB_DIR / "blend_sota556_detector_v2_gated_tau850_submission.csv")

golden_550 = set(i for i in range(len(s550)) if s550.iloc[i]["root_code"] != s551.iloc[i]["root_code"])
flips_556 = set(i for i in range(len(s556)) if s556.iloc[i]["root_code"] != s551.iloc[i]["root_code"])

print(f"Golden 550 Flips count: {len(golden_550)} -> {sorted(golden_550)}")
print(f"SOTA 556 Flips count   : {len(flips_556)} -> {sorted(flips_556)}")

dropped_golden = golden_550 - flips_556
added_556 = flips_556 - golden_550

print(f"\nGolden flips DROPPED by SOTA 556 ({len(dropped_golden)}): {sorted(dropped_golden)}")
for i in sorted(dropped_golden):
    print(f"  Row {i:4d}: Base={s551.iloc[i]['root_code']} | SOTA550={s550.iloc[i]['root_code']} | SOTA556={s556.iloc[i]['root_code']}")

print(f"\nNew flips ADDED by SOTA 556 ({len(added_556)}): {sorted(added_556)}")
for i in sorted(added_556):
    print(f"  Row {i:4d}: Base={s551.iloc[i]['root_code']} | SOTA550={s550.iloc[i]['root_code']} | SOTA556={s556.iloc[i]['root_code']}")

