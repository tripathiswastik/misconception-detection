import pandas as pd
from pathlib import Path
from typing import cast

sub_dir = Path("d:/ag-ai/misconception-detection/submissions")
s_opt = cast(pd.DataFrame, pd.read_csv(sub_dir / "blend_breakthrough_sota551_optimal_submission.csv"))
s_honest = cast(pd.DataFrame, pd.read_csv(sub_dir / "blend_honest_peak_sota553_base_submission.csv"))

r1_opt = [p.split()[0] for p in s_opt["root_code"]]
r1_honest = [p.split()[0] for p in s_honest["root_code"]]

diff_r1 = sum(1 for a, b in zip(r1_opt, r1_honest) if a != b)
diff_total = sum(1 for a, b in zip(s_opt["root_code"], s_honest["root_code"]) if a != b)

print(f"Total rows: {len(s_opt)}")
print(f"R1 differences: {diff_r1} / {len(s_opt)} ({diff_r1/len(s_opt):.1%})")
print(f"Total triplet differences: {diff_total} / {len(s_opt)} ({diff_total/len(s_opt):.1%})")

n_none_opt = sum(1 for x in r1_opt if x == "NONE")
n_none_honest = sum(1 for x in r1_honest if x == "NONE")
print(f"NONE at R1 in s_opt (0.54949): {n_none_opt} ({n_none_opt/len(s_opt):.2%})")
print(f"NONE at R1 in s_honest (0.54183): {n_none_honest} ({n_none_honest/len(s_honest):.2%})")

