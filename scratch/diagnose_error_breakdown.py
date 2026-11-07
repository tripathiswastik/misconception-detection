"""Diagnose Comprehensive Error Breakdown on 5-Fold OOF Predictions.

Breaks down all 1,500 training samples into:
1. Exact Rank 1 Matches (Score = 1.0)
2. Rank 2 Matches (Score = 0.50)
3. Rank 3 Matches (Score = 0.3333)
4. Misses (Score = 0.0):
   a. True NONE predicted as Misconception (False Positive)
   b. True Misconception predicted as NONE (False Negative / Under-calling)
   c. True Misconception predicted as Wrong Misconception (Misclassification)
      - Is the true misconception seen for this question in training?
      - Is it a rare class (N <= 10)?
"""
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

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


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
    q_allowed_map: Dict[str, Set[str]] = {}
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
            q_allowed_map[f"{i}_{qid}"] = allowed_q
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

    # Categorize predictions
    r1_correct = 0
    r2_correct = 0
    r3_correct = 0
    misses = 0

    fp_none = 0  # True NONE, predicted misc
    fn_misc_all_none = 0  # True misc, top 3 has only NONE and wrong miscs, or NONE at R1
    fn_misc_r1_none = 0  # True misc, predicted NONE at R1
    misc_confused = 0  # True misc, predicted wrong misc at R1
    miss_rare = 0  # Missed where true count <= 10
    miss_unseen_in_q = 0  # Missed where true misc never seen for this question

    for i in range(n_samples):
        y = y_true[i]
        trip = triplets[i]
        qid = train_df.iloc[i]["question_id"]
        allowed = q_allowed_map.get(f"{i}_{qid}", set())

        if trip[0] == y:
            r1_correct += 1
        elif trip[1] == y:
            r2_correct += 1
        elif trip[2] == y:
            r3_correct += 1
        else:
            misses += 1
            if y == "NONE":
                fp_none += 1
            else:
                if trip[0] == "NONE":
                    fn_misc_r1_none += 1
                else:
                    misc_confused += 1

                if true_counts[y] <= 10:
                    miss_rare += 1
                if y not in allowed:
                    miss_unseen_in_q += 1

    map3 = (r1_correct * 1.0 + r2_correct * 0.5 + r3_correct * (1.0 / 3.0)) / n_samples

    print("=" * 80)
    print("COMPREHENSIVE OOF ERROR DIAGNOSIS (SOTA PEAK 0.54122 / 0.54949 ARCHITECTURE)")
    print("=" * 80)
    print(f"Total Samples          : {n_samples}")
    print(f"OOF MAP@3              : {map3:.5f}")
    print(f"Rank 1 Correct         : {r1_correct:4d} / {n_samples} ({r1_correct / n_samples:.2%}) -> Points: {r1_correct * 1.0:.1f}")
    print(f"Rank 2 Correct         : {r2_correct:4d} / {n_samples} ({r2_correct / n_samples:.2%}) -> Points: {r2_correct * 0.5:.1f}")
    print(f"Rank 3 Correct         : {r3_correct:4d} / {n_samples} ({r3_correct / n_samples:.2%}) -> Points: {r3_correct / 3.0:.1f}")
    print(f"Zero Points (Misses)   : {misses:4d} / {n_samples} ({misses / n_samples:.2%}) -> LOST POINTS: {misses * 1.0:.1f} max")
    print("-" * 80)
    print("DETAILED BREAKDOWN OF ZERO-POINT MISSES:")
    print(f"  1. True NONE called as Misconception (False Positives)     : {fp_none:3d} rows (lost 1.0 pts each)")
    print(f"  2. True Misconception with NONE at Rank 1 (False Negatives) : {fn_misc_r1_none:3d} rows (lost 0.50 pts if at R2, or 1.0 pts)")
    print(f"  3. True Misconception confused with other Misconception     : {misc_confused:3d} rows")
    print(f"     - Of which True Class is Rare (N_train <= 10)           : {miss_rare:3d} rows ({miss_rare / (fn_misc_r1_none + misc_confused):.1%})")
    print(f"     - Of which True Class was Unseen for question in train  : {miss_unseen_in_q:3d} rows")
    print("=" * 80)


if __name__ == "__main__":
    main()

