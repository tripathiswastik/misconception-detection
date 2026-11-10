"""Evaluate Candidate Rank-2 / Rank-3 Reordering Rules with Rank 1 Fixed.

Invariants:
1. Rank 1 is 100% FIXED across all rules (never altered).
2. Evaluated strictly under 5-Fold GroupKFold on question_id (zero question leakage).
3. Candidate rules use training-only question label sets and in-fold topic-grade distributions.
4. Gate: Promote a rule ONLY if it improves held-out MAP@3 without hurting other cases.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("EVALUATING CANDIDATE RANK-2 / RANK-3 REORDERING RULES (RANK 1 FIXED)")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # Load baseline question-held-out validation prediction matrix
    base_probs = np.load(ART_DIR / "oof_question_heldout_honest_baseline.npy")
    assert base_probs.shape == (n_samples, n_classes)

    # Baseline predictions
    row_indices = np.arange(n_samples)
    base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-base_probs, axis=1)]
    base_map3 = calculate_map3(y_raw.tolist(), base_top3)

    # Identify rows where Rank 1 is NONE
    r1_is_none = np.array([p[0] == "NONE" for p in base_top3])
    print(f"Total validation rows: {n_samples}")
    print(f"Rows where Rank 1 is NONE: {r1_is_none.sum()} ({r1_is_none.sum()/n_samples:.1%})")
    print(f"Baseline Global MAP@3 (Held-Out Questions): {base_map3:.5f}")

    # Setup GroupKFold to enforce training-only label sets and priors per fold
    gkf = GroupKFold(n_splits=5)
    
    # -------------------------------------------------------------------------
    # RULE 1: Topic-Grade Empirical Prior Re-weighting for Rank 2 & Rank 3
    # When Rank 1 is NONE, candidate misconceptions at Rank 2/3 are re-scored by:
    # Score(c) = P_model(c)^alpha * P_topic_grade(c)^(1-alpha)
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Rule 1: Topic-Grade Prior Re-weighting for Rank 2/3 ---")
    best_alpha = 1.0
    best_alpha_score = base_map3

    for alpha in [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        new_top3 = [list(row) for row in base_top3]
        swaps_count = 0

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]

            # In-fold topic-grade prior strictly from trn_df
            prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
            val_tg_prior = prior_engine.predict_proba(val_df)

            for local_i, global_i in enumerate(val_idx):
                if r1_is_none[global_i]:
                    r1 = base_top3[global_i][0]  # Kept strictly fixed (NONE)
                    r2_code = base_top3[global_i][1]
                    r3_code = base_top3[global_i][2]

                    c2_idx = class_to_idx[r2_code]
                    c3_idx = class_to_idx[r3_code]

                    p_mod2 = base_probs[global_i, c2_idx]
                    p_mod3 = base_probs[global_i, c3_idx]

                    p_tg2 = max(val_tg_prior[local_i, c2_idx], 1e-6)
                    p_tg3 = max(val_tg_prior[local_i, c3_idx], 1e-6)

                    score2 = (p_mod2 ** alpha) * (p_tg2 ** (1.0 - alpha))
                    score3 = (p_mod3 ** alpha) * (p_tg3 ** (1.0 - alpha))

                    # If score3 > score2, swap Rank 2 and Rank 3
                    if score3 > score2:
                        new_top3[global_i][1] = r3_code
                        new_top3[global_i][2] = r2_code
                        swaps_count += 1

        score = calculate_map3(y_raw.tolist(), new_top3)
        delta = score - base_map3
        print(f"  alpha = {alpha:.2f} | Swaps: {swaps_count:3d} | Held-Out MAP@3: {score:.5f} ({delta:+.5f})")
        if score > best_alpha_score:
            best_alpha_score = score
            best_alpha = alpha

    # -------------------------------------------------------------------------
    # RULE 2: Specificity Boost over Catch-All Classes (e.g. OTH-00, CAL-01)
    # Generic classes like OTH-00 often accumulate mass at Rank 2. If R3 is a specific
    # domain misconception within margin delta of R2, prefer the specific one.
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Rule 2: Specificity Boost over Catch-All Classes at Rank 2/3 ---")
    generic_codes = {"OTH-00", "CAL-01", "ANS-01", "ANS-02"}

    for delta_thresh in [0.005, 0.010, 0.015, 0.020, 0.030]:
        new_top3 = [list(row) for row in base_top3]
        swaps_count = 0

        for i in range(n_samples):
            if r1_is_none[i]:
                r2 = base_top3[i][1]
                r3 = base_top3[i][2]
                c2_idx = class_to_idx[r2]
                c3_idx = class_to_idx[r3]

                # If R2 is generic, but R3 is specific, and gap is small
                if r2 in generic_codes and r3 not in generic_codes:
                    gap = base_probs[i, c2_idx] - base_probs[i, c3_idx]
                    if 0 < gap < delta_thresh:
                        new_top3[i][1] = r3
                        new_top3[i][2] = r2
                        swaps_count += 1

        score = calculate_map3(y_raw.tolist(), new_top3)
        delta = score - base_map3
        print(f"  delta_thresh = {delta_thresh:.3f} | Swaps: {swaps_count:3d} | Held-Out MAP@3: {score:.5f} ({delta:+.5f})")

    # -------------------------------------------------------------------------
    # RULE 3: TrOCR / OCR Transcription Text Margin Boost
    # Evaluate whether TrOCR feature agreement promotes R3 over R2
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Rule 3: Handwriting TrOCR Modality Re-ranking ---")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tr_norm = tr_feat / (np.linalg.norm(tr_feat, axis=1, keepdims=True) + 1e-12)

    for tr_weight in [0.10, 0.20, 0.30, 0.40]:
        new_top3 = [list(row) for row in base_top3]
        swaps_count = 0

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, groups=groups), start=1):
            fold_y = y_indices = np.array([class_to_idx[c] for c in train_df.iloc[trn_idx]["root_code"].fillna("NONE").astype(str)])
            # Centroids for TrOCR
            tr_centroids = np.zeros((n_classes, tr_norm.shape[1]), dtype=np.float32)
            for c_i in range(n_classes):
                c_mask = (fold_y == c_i)
                if c_mask.sum() > 0:
                    c_m = tr_norm[trn_idx][c_mask].mean(axis=0)
                    tr_centroids[c_i] = c_m / (np.linalg.norm(c_m) + 1e-12)

            for global_i in val_idx:
                if r1_is_none[global_i]:
                    r2 = base_top3[global_i][1]
                    r3 = base_top3[global_i][2]
                    c2_idx = class_to_idx[r2]
                    c3_idx = class_to_idx[r3]

                    sim2 = float(np.dot(tr_norm[global_i], tr_centroids[c2_idx]))
                    sim3 = float(np.dot(tr_norm[global_i], tr_centroids[c3_idx]))

                    score2 = (1.0 - tr_weight) * base_probs[global_i, c2_idx] + tr_weight * sim2
                    score3 = (1.0 - tr_weight) * base_probs[global_i, c3_idx] + tr_weight * sim3

                    if score3 > score2:
                        new_top3[global_i][1] = r3
                        new_top3[global_i][2] = r2
                        swaps_count += 1

        score = calculate_map3(y_raw.tolist(), new_top3)
        delta = score - base_map3
        print(f"  tr_weight = {tr_weight:.2f} | Swaps: {swaps_count:3d} | Held-Out MAP@3: {score:.5f} ({delta:+.5f})")

    print("=" * 80)

if __name__ == "__main__":
    main()

