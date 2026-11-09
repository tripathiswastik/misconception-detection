import numpy as np
import pandas as pd
from pathlib import Path

sub_dir = Path("d:/ag-ai/misconception-detection/submissions")
data_dir = Path("d:/ag-ai/misconception-detection/data")
art_dir = Path("d:/ag-ai/misconception-detection/artifacts")

s550 = pd.read_csv(sub_dir / "blend_sota550_precision_gated_submission.csv")
s775 = pd.read_csv(sub_dir / "blend_sota551_precision_gated_tau775_submission.csv")
s_base = pd.read_csv(sub_dir / "blend_breakthrough_sota551_optimal_submission.csv")
test_df = pd.read_csv(data_dir / "test.csv")
p_bin = np.load(art_dir / "test_compact_binary_prob.npy")

diff_mask = (s550["root_code"] != s775["root_code"])
diff_indices = diff_mask[diff_mask].index.tolist()

print(f"Total Divergent Rows between s550 and s775: {len(diff_indices)}")
print("-" * 80)
for idx in diff_indices:
    row = test_df.iloc[idx]
    qid = str(row["question_id"])
    tid = str(row["trace_id"])
    topic = str(row["topic_name"])
    correct = str(row["correct_answer"])
    prob = float(p_bin[idx])
    b = s_base.iloc[idx]["root_code"]
    p550 = s550.iloc[idx]["root_code"]
    p775 = s775.iloc[idx]["root_code"]
    print(f"Row {idx:4d} | Trace {tid} | Q: {qid[:8]} | BinProb: {prob:.4f} | CorrectAns: {correct} | Topic: {topic[:30]}")
    print(f"   Base (0.54949): {b}")
    print(f"   s550 (0.55049): {p550}")
    print(f"   s775 (0.54949): {p775}")
    print("-" * 80)

