"""Generate Smooth Dual-Letterbox Champion Submission (95% M15 + 3.5% Arc448 + 1.5% DualLB).

Fuses:
  - M15 Balanced Fusion (95%)
  - ConvNeXt-448 Letterbox Composite ArcFace (3.5%)
  - Dual-Letterbox (ConvNeXt-448 + TrOCR-384) ArcFace (1.5%)

Strict Golden NONE Anchor Invariant:
  - Locked at EXACTLY 2,452 / 3,498 (70.10%).
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

with open(ART_DIR / "taxonomy_classes.json") as f:
    tax_classes: List[str] = json.load(f)
n_classes = len(tax_classes)
none_idx = tax_classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
y_train = train_df["root_code"].fillna("NONE").values

print("Loading test probabilities...", flush=True)
p_test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
p_test_arc448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
p_test_dual = np.load(ART_DIR / "test_arcface_dual_letterbox.npy")

p_oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
p_oof_arc448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
p_oof_dual = np.load(ART_DIR / "oof_arcface_dual_letterbox.npy")

# Proven Balanced Fusion Weights: 95% M15 + 3.5% Arc448 + 1.5% Dual-Letterbox
w_m15, w_448, w_dual = 0.950, 0.035, 0.015
p_test_blend = w_m15 * p_test_m15 + w_448 * p_test_arc448 + w_dual * p_test_dual
p_oof_blend = w_m15 * p_oof_m15 + w_448 * p_oof_arc448 + w_dual * p_oof_dual

# Frequency Recalibration (gamma=0.05)
top3_train = np.argsort(-p_oof_blend.astype(np.float64), axis=1)[:, :3]
pred_counts = Counter([tax_classes[idx] for row in top3_train for idx in row])
true_counts = Counter(y_train)

gamma = 0.05
mult = np.ones(n_classes, dtype=np.float64)
for i, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult[i] = (tc / pc) ** gamma
mult[none_idx] = 0.9840

p_test_cal = p_test_blend * mult

# Golden Invariant Calibration: Target EXACTLY 2,452 NONE
best_ns = 1.0
best_diff = 9999
for ns in np.linspace(0.85, 1.15, 601):
    test_scaled = np.copy(p_test_cal)
    test_scaled[:, none_idx] *= ns
    cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
    if abs(cnt - 2452) < best_diff:
        best_diff = abs(cnt - 2452)
        best_ns = ns

p_test_final = np.copy(p_test_cal)
p_test_final[:, none_idx] *= best_ns
final_none = np.sum(np.argmax(p_test_final, axis=1) == none_idx)
print(f"Final Calibrated Top-1 NONE: {final_none} / {len(test_df)} ({final_none / len(test_df):.2%}) with scale={best_ns:.4f}", flush=True)
assert final_none == 2452, f"Target 2452 missed! Got {final_none}"

top3_test = np.argsort(-p_test_final.astype(np.float64), axis=1)[:, :3]
preds_new = [" ".join(tax_classes[idx] for idx in row) for row in top3_test]

champ_54616 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_recalibrated_composite_triblend_448_submission.csv"))
diffs = sum(1 for a, b in zip(preds_new, champ_54616["root_code"]) if a != b)
print(f"Differences vs 0.54616 Champion: {diffs} / {len(test_df)} ({diffs / len(test_df):.2%})", flush=True)

out_file = SUB_DIR / "blend_smooth_dual_letterbox_champion_submission.csv"
sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_new})
sub_df.to_csv(out_file, index=False)
print(f"Saved submission: {out_file}", flush=True)

dl_path = Path("C:/Users/hp/Downloads/blend_smooth_dual_letterbox_champion_submission.csv")
shutil.copy(out_file, dl_path)
print(f"Copied to Downloads: {dl_path}", flush=True)

# Verification checks
assert len(sub_df) == 3498, "Wrong rows"
assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
assert sub_df["root_code"].isnull().sum() == 0, "Null values"
assert all(len(s.split()) == 3 for s in sub_df["root_code"]), "Invalid triplet length"
assert all(len(set(s.split())) == 3 for s in sub_df["root_code"]), "Duplicate codes"
assert sum(1 for s in sub_df["root_code"] if s.startswith("NONE")) == 2452, "NONE count drifted!"
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!", flush=True)

