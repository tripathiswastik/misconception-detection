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

def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> float:
    top3_indices = np.argsort(-y_probs.astype(np.float64), axis=1)[:, :3]
    scores = []
    for true_label, top3 in zip(y_true, top3_indices):
        preds = [tax_classes[idx] for idx in top3]
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))

# Model 1: 0.54566 Champion (Pseudo ArcFace 224 + M15)
oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
oof_arc_pseudo = np.load(ART_DIR / "oof_arcface_expanded_pseudo.npy")
test_arc_pseudo = np.load(ART_DIR / "test_arcface_expanded_pseudo.npy")

p_oof_1 = 0.90 * oof_m15 + 0.10 * oof_arc_pseudo
p_test_1 = 0.90 * test_m15 + 0.10 * test_arc_pseudo

top3_train_1 = np.argsort(-p_oof_1.astype(np.float64), axis=1)[:, :3]
pred_counts_1 = Counter([tax_classes[idx] for row in top3_train_1 for idx in row])
true_counts = Counter(y_train_orig)

mult_1 = np.ones(n_classes, dtype=np.float64)
for i, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts_1.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult_1[i] = (tc / pc) ** 0.05
mult_1[none_idx] = 0.9500

p_oof_champ1 = p_oof_1 * mult_1
p_test_champ1 = p_test_1 * mult_1

# Model 2: 0.54616 Champion (Composite 448 Tri-Blend)
oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")

p_oof_2 = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448
p_test_2 = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

top3_train_2 = np.argsort(-p_oof_2.astype(np.float64), axis=1)[:, :3]
pred_counts_2 = Counter([tax_classes[idx] for row in top3_train_2 for idx in row])

mult_2 = np.ones(n_classes, dtype=np.float64)
for i, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts_2.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult_2[i] = (tc / pc) ** 0.05
mult_2[none_idx] = 0.9840

p_oof_champ2 = p_oof_2 * mult_2
p_test_champ2 = p_test_2 * mult_2

score1 = compute_map3(y_train_orig, p_oof_champ1)
score2 = compute_map3(y_train_orig, p_oof_champ2)
print(f"Champion 1 (LB 0.54566) OOF MAP@3: {score1:.5f}")
print(f"Champion 2 (LB 0.54616) OOF MAP@3: {score2:.5f}")

# Sweep ensemble weights between Champion 1 and Champion 2
print("\n--- Sweeping Dual-Champion Consensus Blend ---")
best_w = 0.5
best_score = 0.0
for w in np.linspace(0.0, 1.0, 21):
    blend_oof = w * p_oof_champ2 + (1.0 - w) * p_oof_champ1
    score = compute_map3(y_train_orig, blend_oof)
    print(f"  Weight {w:.2f} Champ2 (0.54616) + {1-w:.2f} Champ1 (0.54566): OOF MAP@3 = {score:.5f}")
    if score > best_score:
        best_score = score
        best_w = w

print(f"\nOptimal Dual-Champion Weight: {best_w:.2f} Champ2 + {1-best_w:.2f} Champ1 (OOF MAP@3: {best_score:.5f})")

# Generate Dual-Champion Consensus Test Submission
p_test_dual_champ = best_w * p_test_champ2 + (1.0 - best_w) * p_test_champ1

# Lock Top-1 NONE at EXACTLY 2,452
best_ns = 1.0
best_diff = 9999
for ns in np.linspace(0.95, 1.05, 201):
    test_scaled = np.copy(p_test_dual_champ)
    test_scaled[:, none_idx] *= ns
    cnt = np.sum(np.argmax(test_scaled, axis=1) == none_idx)
    if abs(cnt - 2452) < best_diff:
        best_diff = abs(cnt - 2452)
        best_ns = ns

p_test_final = np.copy(p_test_dual_champ)
p_test_final[:, none_idx] *= best_ns
final_none = np.sum(np.argmax(p_test_final, axis=1) == none_idx)
print(f"Locked Top-1 NONE count: {final_none} / {len(test_df)} ({final_none / len(test_df):.2%}) with scale={best_ns:.4f}")
assert final_none == 2452, f"Target 2452 missed! Got {final_none}"

top3_test = np.argsort(-p_test_final.astype(np.float64), axis=1)[:, :3]
preds_new = [" ".join(tax_classes[idx] for idx in row) for row in top3_test]

s_champ_54616 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_m15_recalibrated_composite_triblend_448_submission.csv"))
diffs = sum(1 for a, b in zip(preds_new, s_champ_54616["root_code"]) if a != b)
print(f"Differences vs 0.54616 Active Champion: {diffs} / {len(test_df)} ({diffs / len(test_df):.2%})")

out_file = SUB_DIR / "blend_dual_champion_consensus_448_submission.csv"
sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_new})
sub_df.to_csv(out_file, index=False)
print(f"\nSaved submission: {out_file}")

dl_path = Path("C:/Users/hp/Downloads/blend_dual_champion_consensus_448_submission.csv")
shutil.copy(out_file, dl_path)
print(f"Copied to: {dl_path}")

assert len(sub_df) == 3498, "Wrong rows"
assert list(sub_df.columns) == ["trace_id", "root_code"], "Wrong columns"
assert sub_df["root_code"].isnull().sum() == 0, "Null values"
assert all(len(s.split()) == 3 for s in sub_df["root_code"]), "Invalid triplet length"
assert all(len(set(s.split())) == 3 for s in sub_df["root_code"]), "Duplicate codes"
print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")

