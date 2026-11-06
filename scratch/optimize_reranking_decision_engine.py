"""Optimize Theoretical MAP@3 Reranking Decision Engine on Leak-Free OOF.

Objective:
Test whether calibrating P(c* | Error) against P(NONE) systematically recovers
the 345 true misconception rows currently stuck at Rank 2 or Rank 3 behind NONE.

We evaluate across the canonical 5-fold CV splits:
1. Baseline argmax MAP@3
2. Threshold-gated promotion:
   c* is promoted to Rank 1 if:
   P(c*) / P_sum(misconceptions) * P(is_error) > lambda * P(NONE)
3. Grid search lambda in [0.30, 1.00] using Nested CV (tune lambda on train folds, evaluate on val fold).
4. Measure Rank 1 hit rate, Error MAP@3, NONE MAP@3, and overall MAP@3.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.evaluation.cross_validator import ModelCrossValidator

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


def run_reranking_optimization():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Load probability matrix
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    # Baseline probability blend
    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Load Binary Error Detector OOF probabilities
    oof_binary = np.load(ART_DIR / "oof_binary_error_prob.npy")

    print("=" * 80)
    print("THEORETICAL MAP@3 RERANKING OPTIMIZATION - 5-FOLD NESTED CV")
    print("=" * 80)

    # 1. Baseline Evaluation (Standard Argmax)
    base_top3 = []
    for i in range(len(p_oof_base)):
        top3_idx = np.argsort(-p_oof_base[i])[:3]
        base_top3.append([classes[idx] for idx in top3_idx])
    base_map3 = score_triplets(y_true, base_top3)
    print(f"Baseline Standard Argmax OOF MAP@3: {base_map3:.5f}")

    # 2. Decision Engine Ranking Function
    def apply_decision_engine(
        probs: np.ndarray,
        p_error: np.ndarray,
        lambda_thresh: float,
        secondary_beta: float = 0.82
    ) -> List[List[str]]:
        n = len(probs)
        triplets: List[List[str]] = []
        for i in range(n):
            row_p = probs[i].copy()
            p_none = row_p[none_idx]

            # Top candidate misconception
            row_p_nom = row_p.copy()
            row_p_nom[none_idx] = -1.0
            top_cand_idx = int(np.argmax(row_p_nom))
            p_cand = row_p_nom[top_cand_idx]

            sum_cand = np.sum(row_p[row_p > 0]) - p_none
            cond_cand = p_cand / max(sum_cand, 1e-12)

            # Test decision rule: promote c* to Rank 1 if Gain > 0
            expected_gain = p_error[i] * cond_cand - lambda_thresh * p_none

            if expected_gain > 0 and top_cand_idx != none_idx:
                r1_idx = top_cand_idx
                # Recalibrate remaining candidates
                row_p[r1_idx] = -1.0
                row_p[none_idx] *= secondary_beta
                rem_top2 = np.argsort(-row_p)[:2]
                r2_idx, r3_idx = rem_top2[0], rem_top2[1]
            else:
                # Normal argmax
                top3_idx = np.argsort(-row_p)[:3]
                r1_idx, r2_idx, r3_idx = top3_idx[0], top3_idx[1], top3_idx[2]

            triplets.append([classes[r1_idx], classes[r2_idx], classes[r3_idx]])
        return triplets

    # 3. Nested 5-Fold Parameter Optimization
    candidate_lambdas = np.linspace(0.20, 1.20, 21)
    nested_val_scores = []
    nested_lambdas = []

    for fold_idx, f in enumerate(splits, start=1):
        tr = f["train_indices"]
        va = f["val_indices"]
        y_tr = [y_true[i] for i in tr]
        y_va = [y_true[i] for i in va]

        best_lam = 1.0
        best_tr_score = -1.0

        for lam in candidate_lambdas:
            tr_triplets = apply_decision_engine(p_oof_base[tr], oof_binary[tr], lambda_thresh=lam)
            s_tr = score_triplets(y_tr, tr_triplets)
            if s_tr > best_tr_score:
                best_tr_score = s_tr
                best_lam = lam

        nested_lambdas.append(best_lam)
        va_triplets = apply_decision_engine(p_oof_base[va], oof_binary[va], lambda_thresh=best_lam)
        s_va = score_triplets(y_va, va_triplets)
        nested_val_scores.append(s_va)
        print(f"Fold {fold_idx} -> Best Train Lambda: {best_lam:.2f} | Val MAP@3: {s_va:.5f} (Delta vs base: {s_va - score_triplets(y_va, [base_top3[i] for i in va]):+.5f})")

    mean_val = float(np.mean(nested_val_scores))
    print(f"\nNested CV MAP@3: {mean_val:.5f} (Baseline: {base_map3:.5f}, Delta: {mean_val - base_map3:+.5f})")
    print(f"Selected Lambdas: {nested_lambdas} (Mean: {np.mean(nested_lambdas):.2f})")

    # Evaluate on full OOF with optimal lambda
    opt_lambda = float(np.mean(nested_lambdas))
    full_triplets = apply_decision_engine(p_oof_base, oof_binary, lambda_thresh=opt_lambda)
    full_score = score_triplets(y_true, full_triplets)
    print(f"Full OOF MAP@3 with Lambda={opt_lambda:.2f}: {full_score:.5f}")


if __name__ == "__main__":
    run_reranking_optimization()

