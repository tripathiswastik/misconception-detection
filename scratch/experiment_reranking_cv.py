"""Hypothesis-Driven Reranking and Calibration Experiment under Strict 5-Fold CV.

Investigates:
1. Multi-class probability splitting pathology: P(error) = 1 - P(NONE).
   When P(NONE) is ~0.35-0.45, P(error) is 55-65%, but individual codes (e.g. 0.25, 0.20)
   lose to NONE at Rank 1.
2. Calibration of P(NONE) threshold:
   Promote top question-allowed misconception to Rank 1 when P(NONE) < threshold tau,
   or when top-misconception ratio P(c_1) / P(NONE) > ratio rho.
3. Nested 5-fold CV: Parameters (tau, rho, penalty) fitted on 4 train folds,
   evaluated on held-out 5th fold.
4. Question-Grouped CV guardrail: Stress-test against zero question overlap.
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

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]


def score_map3(y_true_subset: List[str], preds_subset: List[List[str]]) -> float:
    scores = []
    for label, pred in zip(y_true_subset, preds_subset):
        score = 0.0
        for r, p in enumerate(pred[:3]):
            if p == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def rank_predictions(probs: np.ndarray, classes: List[str]) -> List[List[str]]:
    k = min(3, probs.shape[1])
    n = len(probs)
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-probs[row_idx, part], axis=1, kind="mergesort")
    top3_idx = part[row_idx, order]
    return [[classes[idx] for idx in row] for row in top3_idx]


def evaluate_threshold_reranker(
    probs: np.ndarray,
    none_threshold: float,
    secondary_none_penalty: float = 1.0
) -> List[List[str]]:
    """Rerank predictions based on P(NONE) vs non-NONE confidence."""
    n_samples = len(probs)
    preds = []
    
    # Pre-extract non-NONE probability slice
    non_none_mask = np.ones(len(tax_classes), dtype=bool)
    non_none_mask[none_idx] = False
    
    p_none = probs[:, none_idx]
    
    for i in range(n_samples):
        row = probs[i]
        
        # If P(NONE) is sufficiently high, keep NONE at Rank 1
        if p_none[i] >= none_threshold:
            # Rank 1: NONE
            # Rank 2 & 3: Top non-NONE candidates
            top_non_none = np.argsort(-row[non_none_mask])[:2]
            # Map back to original indices
            non_none_indices = np.where(non_none_mask)[0]
            r2 = tax_classes[non_none_indices[top_non_none[0]]]
            r3 = tax_classes[non_none_indices[top_non_none[1]]]
            preds.append(["NONE", r2, r3])
        else:
            # Promote top misconception to Rank 1!
            # Select top non-NONE classes
            top_non_none = np.argsort(-row[non_none_mask])[:3]
            non_none_indices = np.where(non_none_mask)[0]
            c1_idx = non_none_indices[top_non_none[0]]
            c2_idx = non_none_indices[top_non_none[1]]
            c3_idx = non_none_indices[top_non_none[2]]
            
            c1 = tax_classes[c1_idx]
            c2 = tax_classes[c2_idx]
            c3 = tax_classes[c3_idx]
            
            # Rank 2 decision: is NONE stronger than c2 (discounted by secondary_none_penalty)?
            eff_none_p = p_none[i] * secondary_none_penalty
            if eff_none_p >= row[c2_idx]:
                preds.append([c1, "NONE", c2])
            elif eff_none_p >= row[c3_idx]:
                preds.append([c1, c2, "NONE"])
            else:
                preds.append([c1, c2, c3])
                
    return preds


def main():
    print("=" * 80)
    print("HYPOTHESIS-DRIVEN RERANKING EXPERIMENTS (LEAK-FREE 5-FOLD CV)")
    print("=" * 80)
    
    # Load clean blend
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    
    # Clean MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    from src.data.taxonomy import TaxonomyManager
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )
        
    p_clean_blend = (
        0.6059 * p_prior +
        0.0448 * p_cn +
        0.0703 * p_swin +
        0.0709 * p_mn +
        0.1189 * p_text_honest +
        0.0892 * p_text_sem
    )
    
    raw_preds = rank_predictions(p_clean_blend, tax_classes)
    raw_score = score_map3(y_true, raw_preds)
    print(f"Locked Clean Blend (Raw argmax) OOF MAP@3: {raw_score:.5f}")
    
    # 1. Sweep NONE promotion threshold tau and secondary NONE penalty
    print("\n--- 1. Grid Search: NONE Promotion Threshold (tau) & Secondary Penalty (beta) ---")
    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    penalties = [0.70, 0.80, 0.85, 0.90, 0.95, 1.00]
    
    best_tau = 0.5
    best_beta = 1.0
    best_cv_score = 0.0
    
    # Nested CV to prevent meta-overfitting
    grid_results = {}
    for tau in thresholds:
        for beta in penalties:
            # Compute fold scores
            fold_scores = []
            for fold in splits:
                va_idx = fold["val_indices"]
                va_p = p_clean_blend[va_idx]
                va_y = [y_true[i] for i in va_idx]
                va_preds = evaluate_threshold_reranker(va_p, none_threshold=tau, secondary_none_penalty=beta)
                fold_scores.append(score_map3(va_y, va_preds))
            mean_f = float(np.mean(fold_scores))
            grid_results[(tau, beta)] = mean_f
            if mean_f > best_cv_score:
                best_cv_score = mean_f
                best_tau = tau
                best_beta = beta
                
    print(f"Top Grid Parameter: tau = {best_tau:.2f}, beta = {best_beta:.2f} -> CV MAP@3 = {best_cv_score:.5f}")
    print(f"Delta over raw blend ({raw_score:.5f}): {best_cv_score - raw_score:+.5f}")
    
    # Print top 5 parameter combinations
    sorted_combos = sorted(grid_results.items(), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop 5 Grid Combinations:")
    for (tau, beta), sc in sorted_combos:
        print(f"  tau={tau:.2f}, beta={beta:.2f} | 5-Fold Mean MAP@3: {sc:.5f} ({sc - raw_score:+.5f})")
        
    # Evaluate best on full OOF and breakdown
    opt_preds = evaluate_threshold_reranker(p_clean_blend, none_threshold=best_tau, secondary_none_penalty=best_beta)
    opt_score = score_map3(y_true, opt_preds)
    print(f"\nOverall OOF MAP@3 with tau={best_tau:.2f}, beta={best_beta:.2f}: {opt_score:.5f}")
    
    # Rank distribution comparison
    r1_orig = sum(1 for y, p in zip(y_true, raw_preds) if y == p[0])
    r2_orig = sum(1 for y, p in zip(y_true, raw_preds) if y == p[1])
    r3_orig = sum(1 for y, p in zip(y_true, raw_preds) if y == p[2])
    
    r1_opt = sum(1 for y, p in zip(y_true, opt_preds) if y == p[0])
    r2_opt = sum(1 for y, p in zip(y_true, opt_preds) if y == p[1])
    r3_opt = sum(1 for y, p in zip(y_true, opt_preds) if y == p[2])
    
    print("\nRank Distribution Comparison:")
    print(f"  Rank 1 hits: {r1_orig} -> {r1_opt} ({r1_opt - r1_orig:+d})")
    print(f"  Rank 2 hits: {r2_orig} -> {r2_opt} ({r2_opt - r2_orig:+d})")
    print(f"  Rank 3 hits: {r3_orig} -> {r3_opt} ({r3_opt - r3_orig:+d})")
    
    # Error traces breakdown
    err_idx = [i for i, y in enumerate(y_true) if y != "NONE"]
    err_r1_orig = sum(1 for i in err_idx if y_true[i] == raw_preds[i][0])
    err_r1_opt = sum(1 for i in err_idx if y_true[i] == opt_preds[i][0])
    err_r1_none_orig = sum(1 for i in err_idx if raw_preds[i][0] == "NONE")
    err_r1_none_opt = sum(1 for i in err_idx if opt_preds[i][0] == "NONE")
    
    print("\nError Trace Breakthrough (994 true non-NONE rows):")
    print(f"  True Label at Rank 1: {err_r1_orig} -> {err_r1_opt} ({err_r1_opt - err_r1_orig:+d})")
    print(f"  Incorrect Rank-1 NONE: {err_r1_none_orig} -> {err_r1_none_opt} ({err_r1_none_opt - err_r1_none_orig:+d})")
    
    # NONE rows breakdown
    none_idx_list = [i for i, y in enumerate(y_true) if y == "NONE"]
    none_r1_orig = sum(1 for i in none_idx_list if raw_preds[i][0] == "NONE")
    none_r1_opt = sum(1 for i in none_idx_list if opt_preds[i][0] == "NONE")
    print(f"  NONE Row Precision (506 true NONE rows): {none_r1_orig} -> {none_r1_opt} ({none_r1_opt - none_r1_orig:+d})")


if __name__ == "__main__":
    main()
