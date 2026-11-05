import json
import numpy as np
import pandas as pd
from collections import Counter

ART_DIR = "d:/ag-ai/misconception-detection/artifacts"
DATA_DIR = "d:/ag-ai/misconception-detection/data"

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
y_true = train_df["root_code"].fillna("NONE").values

with open(f"{ART_DIR}/taxonomy_classes.json") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")

oof_m15 = np.load(f"{ART_DIR}/oof_m15_balanced_fusion.npy")
oof_dual = np.load(f"{ART_DIR}/oof_arcface_dualhead_448_soft.npy")
oof_comp = np.load(f"{ART_DIR}/oof_arcface_composite_448.npy")

p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

# Recalibration
top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
pred_counts = Counter([tax_classes[idx] for row in top3_train for idx in row])
true_counts = Counter(y_true)

mult = np.ones(len(tax_classes), dtype=np.float64)
for i, c in enumerate(tax_classes):
    if c != "NONE":
        pc = max(pred_counts.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult[i] = (tc / pc) ** 0.05
mult[none_idx] = 0.9840

p_oof_cal = p_oof_base * mult

def compute_map3(preds_list, y_true):
    scores = []
    for true_label, top3 in zip(y_true, preds_list):
        if true_label == top3[0]:
            scores.append(1.0)
        elif true_label == top3[1]:
            scores.append(0.5)
        elif true_label == top3[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    return float(np.mean(scores))

# Baseline standard top3
top3_base = [
    [tax_classes[idx] for idx in row]
    for row in np.argsort(-p_oof_cal.astype(np.float64), axis=1)[:, :3]
]
score_base = compute_map3(top3_base, y_true)
print(f"Standard OOF MAP@3: {score_base:.5f}")

# Experiment: When Rank 1 is a Misconception, suppress or evict NONE from Rank 2/3
for none_penalty in [1.0, 0.8, 0.5, 0.3, 0.1, 0.0]:
    preds_exp = []
    for i in range(len(y_true)):
        p = np.copy(p_oof_cal[i])
        r1_idx = int(np.argmax(p))
        if r1_idx != none_idx:
            # Rank 1 is a misconception! Penalize NONE for R2 & R3
            p_cand = np.copy(p)
            p_cand[r1_idx] = -1.0 # Mask R1
            p_cand[none_idx] *= none_penalty # Penalize NONE in R2/R3
            top2_rest = np.argsort(-p_cand)[:2]
            preds_exp.append([tax_classes[r1_idx], tax_classes[top2_rest[0]], tax_classes[top2_rest[1]]])
        else:
            # Rank 1 is NONE
            p_cand = np.copy(p)
            p_cand[none_idx] = -1.0
            top2_rest = np.argsort(-p_cand)[:2]
            preds_exp.append(["NONE", tax_classes[top2_rest[0]], tax_classes[top2_rest[1]]])
            
    score_exp = compute_map3(preds_exp, y_true)
    print(f"NONE Penalty={none_penalty:.2f} | OOF MAP@3: {score_exp:.5f} | Delta: {score_exp - score_base:+.5f}")

