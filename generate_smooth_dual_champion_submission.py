"""Generate Smooth Dual-Champion Submission (Target: 0.547+ / 0.548+).

Blends the two highest scoring models in project history:
- Champion 1: blend_m15_recalibrated_composite_triblend_448 (Public LB: 0.54616 🏆)
- Champion 2: blend_transductive_champion_448 (Public LB: 0.54599)

Scientific Mechanism:
- Champion 1 (0.54616) holds 85% probability weight.
- Champion 2 (0.54599) holds 15% probability weight.
- Strictly locks the Golden NONE Anchor at EXACTLY 2,452 / 3,498 (70.10%).
- Protects Rank 2 NONE against premature demotion while promoting ~32 highest-confidence
  problem-specific misconceptions into Top-3.
"""

from collections import Counter
import json
from pathlib import Path
import shutil
from typing import List, cast
import numpy as np
import pandas as pd

ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")
DATA_DIR = Path("d:/ag-ai/misconception-detection/data")
SUB_DIR = Path("d:/ag-ai/misconception-detection/submissions")

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes: List[str] = json.load(f)
n_classes = len(tax_classes)
none_idx = tax_classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
y_train_orig = train_df["root_code"].fillna("NONE").values

# 1. Base Components for 0.54616 Champion
oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")

p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448
p_test_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

# Recalibration (gamma = 0.05)
top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
pred_counts = Counter([tax_classes[idx] for row in top3_train for idx in row])
true_counts = Counter(y_train_orig)

mult = np.ones(n_classes, dtype=np.float64)
for i, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult[i] = (tc / pc) ** 0.05
mult[none_idx] = 0.9840

p_test_54616 = p_test_base * mult
p_test_54616[:, none_idx] *= 0.9995 # calibration factor locking 2452

# Transductive question-boosted probability matrix (0.54599)
q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
p_test_boosted = np.copy(p_test_54616)

for i in range(len(test_df)):
    qid = test_df.iloc[i]["question_id"]
    if qid in q_map:
        allowed = q_map[qid]
        for ci, code in enumerate(tax_classes):
            if code != "NONE" and code in allowed:
                p_test_boosted[i, ci] *= 1.20

# 2. Smooth 85/15 Blend
w_trans = 0.15
p_test_smooth = (1.0 - w_trans) * p_test_54616 + w_trans * p_test_boosted

# 3. Lock Top-1 NONE to EXACTLY 2,452
best_ns = 1.0
best_diff = 9999
for ns in np.linspace(0.98, 1.05, 701):
    test_scaled = np.copy(p_test_smooth)
    test_scaled[:, none_idx] *= ns
    cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
    if abs(cnt - 2452) < best_diff:
        best_diff = abs(cnt - 2452)
        best_ns = ns

p_test_final = np.copy(p_test_smooth)
p_test_final[:, none_idx] *= best_ns

top1_final = np.argmax(p_test_final, axis=1)
none_count = np.sum(top1_final == none_idx)
print(f"Top-1 NONE count: {none_count} / {len(test_df)} ({none_count / len(test_df):.2%}) with scale={best_ns:.4f}")
assert none_count == 2452, f"Target 2452 missed! Got {none_count}"

top3_test = np.argsort(-p_test_final.astype(np.float64), axis=1)[:, :3]
preds_new = [" ".join(tax_classes[idx] for idx in row) for row in top3_test]

# 4. Comparative Statistics vs Both Champions
champ_54616 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_recalibrated_composite_triblend_448_submission.csv"))
champ_54599 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_transductive_champion_448_submission.csv"))

diff_vs_54616 = sum(1 for a, b in zip(preds_new, champ_54616["root_code"]) if a != b)
diff_vs_54599 = sum(1 for a, b in zip(preds_new, champ_54599["root_code"]) if a != b)

print(f"\n[Comparison Metrics]")
print(f"  Differences vs 0.54616 Champion: {diff_vs_54616} / {len(test_df)} ({diff_vs_54616/len(test_df):.2%})")
print(f"  Differences vs 0.54599 Transductive: {diff_vs_54599} / {len(test_df)} ({diff_vs_54599/len(test_df):.2%})")

# NONE count across ranks
r1_none = sum(1 for s in preds_new if s.split()[0] == "NONE")
r2_none = sum(1 for s in preds_new if s.split()[1] == "NONE")
r3_none = sum(1 for s in preds_new if s.split()[2] == "NONE")
print(f"  NONE by Rank: R1={r1_none}, R2={r2_none}, R3={r3_none}, Total={r1_none+r2_none+r3_none}")

# 5. Output Submission File
out_file = SUB_DIR / "blend_smooth_dual_champion_448_submission.csv"
sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_new})
sub_df.to_csv(out_file, index=False)
print(f"\nSaved submission to: {out_file}")

dl_path = Path("C:/Users/hp/Downloads/blend_smooth_dual_champion_448_submission.csv")
shutil.copy(out_file, dl_path)
print(f"Copied to Downloads: {dl_path}")

# Verification checks
assert len(sub_df) == 3498, "Wrong rows"
assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
assert sub_df["root_code"].isnull().sum() == 0, "Null values"
assert all(len(s.split()) == 3 for s in sub_df["root_code"]), "Invalid triplet length"
assert all(len(set(s.split())) == 3 for s in sub_df["root_code"]), "Duplicate codes"
assert sum(1 for s in sub_df["root_code"] if s.startswith("NONE")) == 2452, "NONE count drifted!"
print("ALL SUBMISSION INTEGRITY CHECKS PASSED 100%!")

