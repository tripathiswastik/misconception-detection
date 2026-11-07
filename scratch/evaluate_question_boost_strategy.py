"""Evaluate Rank-1 Locked vs Gated Question Boost in 5-Fold Cross Validation.

Questions to answer:
1. If Rank 1 is 100% locked to baseline champion, and alpha_q is applied ONLY to Rank 2 & 3:
   What is the MAP@3 delta? (Pure upside test).
2. If alpha_q is applied to all ranks, but only allowed to flip Rank 1 if P(c)*alpha_q > threshold:
   What is the optimal threshold?
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def score_triplets(y_true: List[str], triplets: List[List[str]]) -> float:
    scores = []
    for true_label, row in zip(y_true, triplets):
        score = 0.0
        for r, pred in enumerate(row[:3]):
            if pred == true_label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    from collections import Counter
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

    row_idx = np.arange(n_samples)
    r1_base = np.argmax(p_oof_cal, axis=1)
    cand_base = p_oof_cal.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]
    base_triplets = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, base_triplets)
    print(f"Honest Baseline OOF MAP@3: {base_score:.5f}")

    # Build question match mask
    mask_q_boost = np.zeros((n_samples, n_classes), dtype=bool)
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
                    mask_q_boost[i, ci] = True

    print("\n--- TEST 1: RANK-1 STRICTLY LOCKED, alpha_q ON RANK 2 & 3 ONLY ---")
    for aq in [1.00, 1.10, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 5.00]:
        mult_matrix = np.ones((n_samples, n_classes), dtype=np.float64)
        mult_matrix[mask_q_boost] = aq
        p_oof_boosted = p_oof_cal * mult_matrix

        # Lock Rank 1 to r1_base!
        cand = p_oof_boosted.copy()
        cand[row_idx, r1_base] = -1.0
        cand[r1_base != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]

        triplets = [[classes[r1_base[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        score = score_triplets(y_true, triplets)
        r2_diff = sum(1 for a, b in zip(triplets, base_triplets) if a[1] != b[1])
        r3_diff = sum(1 for a, b in zip(triplets, base_triplets) if a[2] != b[2])
        print(f"alpha_q = {aq:4.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R2 changes: {r2_diff}, R3 changes: {r3_diff})")

    print("\n--- TEST 2: ALL RANKS WITH alpha_q = 1.10 and 1.25 ---")
    for aq in [1.05, 1.10, 1.15, 1.20, 1.25]:
        mult_matrix = np.ones((n_samples, n_classes), dtype=np.float64)
        mult_matrix[mask_q_boost] = aq
        p_oof_boosted = p_oof_cal * mult_matrix

        r1 = np.argmax(p_oof_boosted, axis=1)
        cand = p_oof_boosted.copy()
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]

        triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        score = score_triplets(y_true, triplets)
        r1_flips = sum(1 for a, b in zip(r1, r1_base) if a != b)
        print(f"alpha_q = {aq:4.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R1 flips: {r1_flips})")


if __name__ == "__main__":
    main()

