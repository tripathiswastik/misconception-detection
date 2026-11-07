"""Analyze False Negative Misconception Rows (True Misconception, but NONE at R1 and 0 points scored)."""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager

tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
classes = tax.all_codes(include_none=True)
none_idx = classes.index("NONE")
n_classes = len(classes)

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
n_samples = len(train_df)

# Load SOTA 0.54949 probabilities
oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
true_counts = Counter(y_true)
mult = np.ones(n_classes, dtype=np.float64)
for i, c in enumerate(classes):
    if c != "NONE":
        pc = max(pred_counts.get(c, 1), 1)
        tc = max(true_counts.get(c, 1), 1)
        mult[i] = (tc / pc) ** 0.05
mult[none_idx] = 0.9840
p_oof_cal = p_oof_base * mult
p_oof_cal[:, none_idx] *= 0.9995

p_oof_boosted = p_oof_cal.copy()
for fold in splits:
    tr = fold["train_indices"]
    va = fold["val_indices"]
    tr_df = train_df.iloc[tr]
    q_allowed: Dict[str, Set[str]] = {}
    for q, grp in tr_df.groupby("question_id"):
        q_allowed[q] = set(grp["root_code"].dropna().unique())
    for i in va:
        qid = train_df.iloc[i]["question_id"]
        allowed_q = q_allowed.get(qid, set())
        for ci in range(n_classes):
            if ci != none_idx and classes[ci] in allowed_q:
                p_oof_boosted[i, ci] *= 1.20

oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
p_oof_arc3500_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult

oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
oof_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
oof_mn = np.load(ART_DIR / "oof_finetuned_vision.npy")
oof_th = np.load(ART_DIR / "oof_text.npy")
oof_ts = np.load(ART_DIR / "oof_semantic_text.npy")
oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
oof_arc = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

def sharp(probs, t):
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

p_oof_clean = (
    0.6448 * oof_prior +
    0.0839 * sharp(oof_th, 0.6) +
    0.0401 * sharp(oof_ts, 0.6) +
    0.0458 * oof_cn +
    0.0454 * oof_swin +
    0.0483 * oof_mn +
    0.0455 * oof_trocr +
    0.0462 * sharp(oof_arc, 0.8)
)

p_champ = 0.70 * p_oof_cal + 0.25 * p_oof_boosted + 0.05 * p_oof_arc3500_cal
p_final = 0.96 * p_champ + 0.04 * p_oof_clean

row_idx = np.arange(n_samples)
r1 = np.argmax(p_final, axis=1)
cand = p_final.copy()
cand[row_idx, r1] = -1.0
cand[r1 != none_idx, none_idx] *= 0.82
top2 = np.argsort(-cand, axis=1)[:, :2]
triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]

# Find where true label is in top 10
ranks = []
for i in range(n_samples):
    y = y_true[i]
    # Full ranking of classes
    full_rank = np.argsort(-p_final[i])
    for r, idx in enumerate(full_rank):
        if classes[idx] == y:
            ranks.append(r + 1)
            break

ranks = np.array(ranks)
print("=" * 80)
print("TRUE LABEL RANK DISTRIBUTION ACROSS ALL 1,500 ROWS:")
print("=" * 80)
for k in range(1, 11):
    cnt = np.sum(ranks == k)
    cum = np.sum(ranks <= k)
    print(f"  Rank {k:2d}: {cnt:4d} rows ({cnt / n_samples:.2%}) | Cumulative <= {k:2d}: {cum:4d} ({cum / n_samples:.2%})")

print(f"  Rank > 10: {np.sum(ranks > 10):4d} rows ({np.sum(ranks > 10) / n_samples:.2%})")

# Specifically for Misconceptions (y != NONE)
misc_mask = np.array([y != "NONE" for y in y_true])
misc_ranks = ranks[misc_mask]
n_misc = np.sum(misc_mask)
print("\n" + "=" * 80)
print(f"TRUE LABEL RANK DISTRIBUTION ON TRUE MISCONCEPTIONS (N={n_misc}):")
print("=" * 80)
for k in range(1, 11):
    cnt = np.sum(misc_ranks == k)
    cum = np.sum(misc_ranks <= k)
    print(f"  Rank {k:2d}: {cnt:4d} rows ({cnt / n_misc:.2%}) | Cumulative <= {k:2d}: {cum:4d} ({cum / n_misc:.2%})")
print(f"  Rank > 10: {np.sum(misc_ranks > 10):4d} rows ({np.sum(misc_ranks > 10) / n_misc:.2%})")

# Look at Rank 4, 5, 6
print("\n" + "=" * 80)
print(f"OPPORTUNITY POOL: MISCONCEPTIONS RANKED AT 4, 5, 6 (JUST OUTSIDE TOP 3)")
print("=" * 80)
r456_cnt = np.sum((misc_ranks >= 4) & (misc_ranks <= 6))
print(f"True Misconceptions at Ranks 4-6: {r456_cnt} rows ({r456_cnt / n_misc:.2%})")
print(f"If promoted to Rank 2/3: +{r456_cnt * 0.40 / n_samples:.5f} to +{r456_cnt * 0.50 / n_samples:.5f} MAP@3 boost!")
print("=" * 80)
