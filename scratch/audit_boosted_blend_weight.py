"""Evaluate exact blend weight of p_boosted in 5-fold CV.

In 0.54899:
p_champ_ensemble = 0.85 * p_base + 0.10 * p_boosted (1.20) + 0.05 * p_arc3500

We test varying w_boosted in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70] where:
p_champ_ensemble = (0.95 - w_boosted) * p_base + w_boosted * p_boosted (1.20) + 0.05 * p_arc3500
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
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

    train_df = pd.read_csv(DATA_DIR / "train.csv")
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

    # Build fold-local question boosted matrix (1.20)
    p_oof_boosted_120 = p_oof_cal.copy()
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
                    p_oof_boosted_120[i, ci] *= 1.20

    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    p_oof_arc3500_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult

    row_idx = np.arange(n_samples)

    # 0.54899 baseline in OOF (w_boosted = 0.10)
    p_baseline = 0.85 * p_oof_cal + 0.10 * p_oof_boosted_120 + 0.05 * p_oof_arc3500_cal
    r1_base = np.argmax(p_baseline, axis=1)
    cand_base = p_baseline.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]
    triplets_base = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    score_baseline = score_triplets(y_true, triplets_base)
    print(f"Active Peak Baseline (w_boost = 0.10) OOF MAP@3: {score_baseline:.5f}")

    w_boost_vals = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    best_score = score_baseline
    best_w = 0.10

    print("\nSweeping w_boosted:")
    print("-" * 80)
    for wb in w_boost_vals:
        w_base = 0.95 - wb
        p_ens = w_base * p_oof_cal + wb * p_oof_boosted_120 + 0.05 * p_oof_arc3500_cal

        r1 = np.argmax(p_ens, axis=1)
        cand = p_ens.copy()
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]

        triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        score = score_triplets(y_true, triplets)
        flips = sum(1 for a, b in zip(r1, r1_base) if a != b)

        flag = " *" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_w = wb
        print(f"w_boosted = {wb:4.2f} (w_base = {w_base:4.2f}) | MAP@3: {score:.5f} (Delta: {score - score_baseline:+.5f}, R1 flips: {flips:2d}){flag}")

    print("=" * 80)
    print(f"Optimal w_boosted: {best_w:.2f} (Peak MAP@3: {best_score:.5f}, Delta: {best_score - score_baseline:+.5f})")


if __name__ == "__main__":
    main()

