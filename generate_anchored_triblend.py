"""Generate Anchored Calibrated Tri-Blend Submission.

Combines:
- M15 Balanced Fusion (90%)
- ArcFace-620 (5%): high-purity rare-class embeddings
- Calibrated ArcFace-1465 (5%): high-diversity 44-class embeddings with Bayesian prior correction

Enforces the Golden NONE Anchor Invariant:
- Any trace where the Champion (0.54483) had NONE at Rank 1 is guaranteed to retain NONE at Rank 1.
- Tri-blend probabilities optimize Rank 2 and Rank 3 for those traces.
- For all traces where a misconception was identified at Rank 1, the tri-blend ranks all 76 classes naturally.
"""

import json
from pathlib import Path
import shutil
from typing import List

import numpy as np
import pandas as pd

art = Path("d:/ag-ai/misconception-detection/artifacts")
subs = Path("d:/ag-ai/misconception-detection/submissions")
data = Path("d:/ag-ai/misconception-detection/data")

with open(art / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax: List[str] = json.load(f)
none_idx = tax.index("NONE")

test_df = pd.read_csv(data / "test.csv")
test_m15 = np.load(art / "test_m15_balanced_fusion.npy")
arc_620 = np.load(art / "test_arcface_expanded_pseudo.npy")
arc_1465 = np.load(art / "test_arcface_expanded_1465.npy")
s_champ = pd.read_csv(subs / "blend_m15_expanded_pseudo_submission.csv")

# Calibrate 1465 NONE prior
cal_1465 = np.copy(arc_1465)
cal_1465[:, none_idx] *= 1.4
cal_1465 = cal_1465 / cal_1465.sum(axis=1, keepdims=True)

# Tri-blend: 90% M15 + 5% Arc620 + 5% Calibrated Arc1465
p_tri = 0.90 * test_m15 + 0.05 * arc_620 + 0.05 * cal_1465
champ_top1 = [r.split()[0] for r in s_champ["root_code"]]

new_preds: List[str] = []
for i in range(3498):
    row_probs = np.copy(p_tri[i])
    if champ_top1[i] == "NONE":
        row_probs[none_idx] = -1e9
        top_misc = np.argsort(-row_probs)[:2]
        pred_codes = ["NONE", tax[top_misc[0]], tax[top_misc[1]]]
    else:
        top3 = np.argsort(-row_probs)[:3]
        pred_codes = [tax[idx] for idx in top3]
    new_preds.append(" ".join(pred_codes))

sub_df = pd.DataFrame({
    "trace_id": test_df["trace_id"],
    "root_code": new_preds,
})

out_path = subs / "blend_m15_anchored_triblend_submission.csv"
sub_df.to_csv(out_path, index=False)
print(f"Saved candidate submission to: {out_path}")

# Copy to Downloads
dl_path = Path("C:/Users/hp/Downloads/blend_m15_anchored_triblend_submission.csv")
shutil.copyfile(out_path, dl_path)
print(f"Copied to Downloads: {dl_path}")

# Verification Gate
assert len(sub_df) == 3498, f"Expected 3498 rows, got {len(sub_df)}"
assert list(sub_df.columns) == ["trace_id", "root_code"], f"Wrong columns: {sub_df.columns}"
assert sub_df["root_code"].isnull().sum() == 0, "Null values found"
for i, code_str in enumerate(sub_df["root_code"]):
    codes = code_str.split()
    assert len(codes) == 3, f"Row {i} does not have 3 codes: {code_str}"
    assert len(set(codes)) == 3, f"Row {i} has duplicates: {code_str}"
    for c in codes:
        assert c in tax, f"Unknown code: {c}"

top1_count = sum(1 for c in sub_df["root_code"] if c.startswith("NONE"))
diff_count = sum(1 for a, b in zip(sub_df["root_code"], s_champ["root_code"]) if a != b)

print("All submission verification checks PASSED successfully!")
print(f"Top-1 NONE count: {top1_count} / 3498 ({top1_count/3498:.2%})")
print(f"Differences vs 0.54483 Champion: {diff_count} / 3498 ({diff_count/3498:.2%})")

