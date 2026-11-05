import json
from pathlib import Path
from collections import Counter
from typing import List, cast
import numpy as np
import pandas as pd

ART_DIR = Path("d:/ag-ai/misconception-detection/artifacts")
DATA_DIR = Path("d:/ag-ai/misconception-detection/data")

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes: List[str] = json.load(f)
n_classes = len(tax_classes)
none_idx = tax_classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").values

with open(DATA_DIR / "splits.json") as f:
    splits = json.load(f)["folds"]

# Load Champion 0.54616 Base Components
oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")

p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

# Pre-compute recalibration
top3_base = np.argsort(-p_base.astype(np.float64), axis=1)[:, :3]
pred_counts = Counter([tax_classes[idx] for row in top3_base for idx in row])
true_counts = Counter(y_true)

mult = np.ones(n_classes, dtype=np.float64)
for ci, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult[ci] = (tc / pc) ** 0.05
mult[none_idx] = 0.9840

p_champ_cal = p_base * mult

def compute_map3(y: np.ndarray, p: np.ndarray) -> float:
    top3 = np.argsort(-p.astype(np.float64), axis=1)[:, :3]
    scores = []
    for true_label, row in zip(y, top3):
        preds = [tax_classes[idx] for idx in row]
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))

base_score = compute_map3(y_true, p_champ_cal)
print(f"Active Champion 0.54616 OOF MAP@3: {base_score:.5f}", flush=True)

# Pre-build boolean indicator masks for each sample per fold
# q_mask[i, c] = True if class c was observed on train question for sample i
# top_mask[i, c] = True if class c was observed on train topic for sample i
q_mask = np.zeros((len(train_df), n_classes), dtype=bool)
top_mask = np.zeros((len(train_df), n_classes), dtype=bool)

for fold in splits:
    tr_df = train_df.iloc[fold["train_indices"]]
    val_indices = fold["val_indices"]
    
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_df.groupby("question_id")}
    top_map = {top: set(grp["root_code"].dropna()) for top, grp in tr_df.groupby("topic_name")}
    
    for vi in val_indices:
        row = train_df.iloc[vi]
        q_allowed = q_map.get(row["question_id"], set())
        top_allowed = top_map.get(row["topic_name"], set())
        
        for ci, code in enumerate(tax_classes):
            if code == "NONE":
                continue
            if code in q_allowed:
                q_mask[vi, ci] = True
            if code in top_allowed:
                top_mask[vi, ci] = True

non_none_mask = np.ones(n_classes, dtype=bool)
non_none_mask[none_idx] = False

print("Built vectorized masks. Sweeping boost and dampening parameters...", flush=True)

best_score = base_score
best_params = (0.0, 0.0)

for q_boost in np.linspace(0.0, 0.8, 9):
    for t_damp in np.linspace(0.0, 0.6, 7):
        p_test = np.copy(p_champ_cal)
        
        if q_boost > 0:
            p_test[q_mask] *= (1.0 + q_boost)
        if t_damp > 0:
            # Dampen classes that are NOT in topic and NOT NONE
            invalid_top_mask = (~top_mask) & non_none_mask
            p_test[invalid_top_mask] *= (1.0 - t_damp)
            
        score = compute_map3(y_true, p_test)
        if score > best_score:
            best_score = score
            best_params = (float(q_boost), float(t_damp))
            print(f"  --> NEW BEST: q_boost={q_boost:.2f}, t_damp={t_damp:.2f} | OOF MAP@3 = {score:.5f} (+{score - base_score:.5f})", flush=True)

print(f"\nOptimal Transductive Priors: q_boost={best_params[0]:.2f}, t_damp={best_params[1]:.2f} -> OOF MAP@3: {best_score:.5f}", flush=True)

