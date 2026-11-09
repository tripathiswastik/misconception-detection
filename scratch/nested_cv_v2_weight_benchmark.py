"""Strict Leak-Free Nested CV Benchmark for ArcFace V2 vs V1 Full Pipeline.

Protocol:
1. Outer 5 folds (from data/splits.json):
   - Outer Train: 1,200 samples
   - Outer Holdout (Test): 300 samples (UNTOUCHED during selection)
2. Inner 4 folds on Outer Train (1,200 samples):
   - Strict in-fold frequency calibration (mult fitted ONLY on inner train labels)
   - In-fold question allowed set (derived ONLY from inner train questions)
   - Transductive 1.20x boost
   - Clean multimodal blend (w_clean = 0.04)
   - Decision engine secondary beta (beta = 0.82)
   - Precision gating (tau = 0.780, eps = 0.040)
   - Evaluates weights w_v2 in [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
   - Selects w*_k = argmax InnerMAP3(w)
3. Evaluates selected w*_k and fixed comparisons ONCE on the untouched outer holdout fold (300 samples).
4. Aggregates true leak-free out-of-fold predictions across all 1,500 samples.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def run_pipeline(
    base_probs: np.ndarray,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    clean_probs: np.ndarray,
    bin_error_probs: np.ndarray,
    train_df: pd.DataFrame,
    classes: List[str],
    none_idx: int,
    tau: float = 0.780,
    eps: float = 0.040,
    beta: float = 0.82
) -> List[List[str]]:
    """Runs the complete leak-free compound pipeline on eval_indices fitted strictly on train_indices."""
    n_classes = len(classes)
    n_eval = len(eval_indices)

    # 1. Frequency Calibration Multiplier fitted STRICTLY on train_indices
    top3_tr = np.argsort(-base_probs[train_indices].astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_labels_tr = train_df.iloc[train_indices]["root_code"].fillna("NONE").astype(str).tolist()
    true_counts = Counter(true_labels_tr)

    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_eval_cal = base_probs[eval_indices] * mult
    p_eval_cal[:, none_idx] *= 0.9995

    # 2. Transductive Question Conditioning derived STRICTLY from train_indices
    tr_sub_df = train_df.iloc[train_indices]
    q_map: Dict[str, Set[str]] = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}

    p_eval_boosted = np.copy(p_eval_cal)
    eval_df = train_df.iloc[eval_indices].reset_index(drop=True)
    for i in range(n_eval):
        qid = eval_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_eval_boosted[i, ci] *= 1.20

    # Compound Foundation Ensemble
    p_eval_champ = 0.70 * p_eval_cal + 0.30 * p_eval_boosted

    # 3. Clean Multimodal Blend (w_clean = 0.04)
    p_eval_final = 0.96 * p_eval_champ + 0.04 * clean_probs[eval_indices]

    # 4. Decision Engine Ranking with secondary beta
    cand = p_eval_final.copy()
    row_idx = np.arange(n_eval)
    r1 = np.argmax(cand, axis=1)
    cand[row_idx, r1] = -1.0
    cand[r1 != none_idx, none_idx] *= beta
    top2 = np.argsort(-cand, axis=1)[:, :2]
    preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_eval)]

    # 5. Precision Gating (tau = 0.780, eps = 0.040)
    for i in range(n_eval):
        if preds[i][0] == "NONE" and bin_error_probs[eval_indices[i]] >= tau:
            row_p = p_eval_final[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            gap = p_none_val - row_p[top_misc_idx]

            if 0 < gap < eps:
                qid = eval_df.iloc[i]["question_id"]
                if top_misc_code in q_map.get(qid, set()):
                    row_p[top_misc_idx] = -1.0
                    second_misc_idx = int(np.argmax(row_p))
                    second_misc_code = classes[second_misc_idx]
                    preds[i] = [top_misc_code, "NONE", second_misc_code]

    return preds

def main():
    print("=" * 80, flush=True)
    print("STRICT LEAK-FREE NESTED CV BENCHMARK (V1 vs V2 WEIGHT SWEEP)", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # Load All OOF Components
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    # Clean Multimodal Matrix on OOF
    oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    oof_th = np.load(ART_DIR / "honest_oof_text.npy")
    oof_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_sw = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_ft = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_tr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_clean_arc = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    oof_clean = (
        0.6448 * oof_prior +
        0.0839 * sharp(oof_th, 0.6) +
        0.0401 * sharp(oof_ts, 0.6) +
        0.0458 * oof_cn +
        0.0454 * oof_sw +
        0.0483 * oof_ft +
        0.0455 * oof_tr +
        0.0462 * sharp(oof_clean_arc, 0.8)
    )

    oof_bin = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    # Full V1 Formula: 0.95 * M15 + 0.02 * dual + 0.03 * comp
    p_base_v1 = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    # Weights to sweep for V2
    weights_v2 = [0.00, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    n_outer_folds = len(splits)

    # Storage for Outer Fold Predictions
    outer_preds_v1: List[List[str]] = [[] for _ in range(n_samples)]
    outer_preds_nested_v2: List[List[str]] = [[] for _ in range(n_samples)]
    outer_preds_fixed: Dict[float, List[List[str]]] = {w: [[] for _ in range(n_samples)] for w in weights_v2}

    outer_selected_weights: List[float] = []

    for fold_idx, outer_fold in enumerate(splits, start=1):
        outer_tr_idx = np.array(outer_fold["train_indices"], dtype=np.int32)
        outer_va_idx = np.array(outer_fold["val_indices"], dtype=np.int32)

        print(f"\n--- OUTER FOLD {fold_idx}/{n_outer_folds} (Train: {len(outer_tr_idx)}, Holdout: {len(outer_va_idx)}) ---", flush=True)

        # -------------------------------------------------------------
        # STEP 1: INNER CV on outer_tr_idx (1,200 samples)
        # -------------------------------------------------------------
        # Create 4 inner folds using StratifiedKFold on outer_tr_idx
        y_outer_tr = [y_true[i] for i in outer_tr_idx]
        # Group rare classes with < 4 samples into 'RARE' for stratification stability
        cnt_tr = Counter(y_outer_tr)
        strat_y = [y if cnt_tr[y] >= 4 else "RARE" for y in y_outer_tr]
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + fold_idx)
        inner_splits = list(skf.split(outer_tr_idx, strat_y))

        inner_scores: Dict[float, float] = {}

        for w in weights_v2:
            p_base_candidate = (1.0 - w) * oof_m15 + w * oof_v2
            inner_fold_scores: List[float] = []

            for in_tr_local, in_va_local in inner_splits:
                in_tr_idx = outer_tr_idx[in_tr_local]
                in_va_idx = outer_tr_idx[in_va_local]

                preds_inner_va = run_pipeline(
                    base_probs=p_base_candidate,
                    train_indices=in_tr_idx,
                    eval_indices=in_va_idx,
                    clean_probs=oof_clean,
                    bin_error_probs=oof_bin,
                    train_df=train_df,
                    classes=classes,
                    none_idx=none_idx
                )
                y_true_inner_va = [y_true[i] for i in in_va_idx]
                score_in = calculate_map3(y_true_inner_va, preds_inner_va)
                inner_fold_scores.append(score_in)

            mean_inner_score = float(np.mean(inner_fold_scores))
            inner_scores[w] = mean_inner_score

        # Select best weight on inner CV
        best_w = max(inner_scores.keys(), key=lambda w: inner_scores[w])
        outer_selected_weights.append(best_w)

        print(f"  Inner CV Scores: " + ", ".join([f"w={w:.2f}:{inner_scores[w]:.5f}" for w in [0.00, 0.05, 0.10, 0.12, 0.15, 0.18, 0.20]]), flush=True)
        print(f"  => Selected Best Weight for Outer Fold {fold_idx}: w* = {best_w:.2f} (Inner MAP@3: {inner_scores[best_w]:.5f})", flush=True)

        # -------------------------------------------------------------
        # STEP 2: EVALUATE ON UNTOUCHED OUTER HOLDOUT FOLD (300 samples)
        # -------------------------------------------------------------
        # 1. Baseline Full SOTA 550 V1 Recipe
        preds_outer_v1 = run_pipeline(
            base_probs=p_base_v1,
            train_indices=outer_tr_idx,
            eval_indices=outer_va_idx,
            clean_probs=oof_clean,
            bin_error_probs=oof_bin,
            train_df=train_df,
            classes=classes,
            none_idx=none_idx
        )
        for local_i, global_i in enumerate(outer_va_idx):
            outer_preds_v1[global_i] = preds_outer_v1[local_i]

        y_true_outer = [y_true[i] for i in outer_va_idx]
        score_v1_outer = calculate_map3(y_true_outer, preds_outer_v1)

        # 2. Adaptive Nested CV Selected Weight (w* selected strictly from inner folds)
        p_base_nested = (1.0 - best_w) * oof_m15 + best_w * oof_v2
        preds_outer_nested = run_pipeline(
            base_probs=p_base_nested,
            train_indices=outer_tr_idx,
            eval_indices=outer_va_idx,
            clean_probs=oof_clean,
            bin_error_probs=oof_bin,
            train_df=train_df,
            classes=classes,
            none_idx=none_idx
        )
        for local_i, global_i in enumerate(outer_va_idx):
            outer_preds_nested_v2[global_i] = preds_outer_nested[local_i]

        score_nested_outer = calculate_map3(y_true_outer, preds_outer_nested)
        print(f"  Outer Holdout Result: V1={score_v1_outer:.5f} | Nested V2 (w*={best_w:.2f})={score_nested_outer:.5f} (Delta: {score_nested_outer - score_v1_outer:+.5f})", flush=True)

        # 3. Also evaluate fixed candidate weights for full transparency across holdouts
        for w in weights_v2:
            p_base_fixed = (1.0 - w) * oof_m15 + w * oof_v2
            preds_outer_fixed = run_pipeline(
                base_probs=p_base_fixed,
                train_indices=outer_tr_idx,
                eval_indices=outer_va_idx,
                clean_probs=oof_clean,
                bin_error_probs=oof_bin,
                train_df=train_df,
                classes=classes,
                none_idx=none_idx
            )
            for local_i, global_i in enumerate(outer_va_idx):
                outer_preds_fixed[w][global_i] = preds_outer_fixed[local_i]

    # -------------------------------------------------------------
    # OVERALL AGGREGATED LEAK-FREE RESULTS (1,500 samples)
    # -------------------------------------------------------------
    print("\n" + "=" * 80, flush=True)
    print("FINAL 100% LEAK-FREE NESTED CV RESULTS (1,500 UNTOUCHED SAMPLES):", flush=True)
    print("=" * 80, flush=True)

    overall_v1_map3 = calculate_map3(y_true, outer_preds_v1)
    overall_nested_map3 = calculate_map3(y_true, outer_preds_nested_v2)

    print(f"Selected Inner Weights Across 5 Folds: {outer_selected_weights}", flush=True)
    print(f"\n1. Full SOTA 550 V1 Recipe (Leak-Free Outer OOF)   : {overall_v1_map3:.5f}", flush=True)
    print(f"2. Adaptive Nested CV Selected V2 (Outer OOF)       : {overall_nested_map3:.5f} (Delta: {overall_nested_map3 - overall_v1_map3:+.5f})", flush=True)

    print("\n3. Fixed Weight Across All Outer Holdouts (Leak-Free Multiplier & Boost):", flush=True)
    for w in weights_v2:
        score_w = calculate_map3(y_true, outer_preds_fixed[w])
        print(f"   Fixed w_v2 = {w:.2f} -> Outer OOF MAP@3: {score_w:.5f} (Delta vs V1: {score_w - overall_v1_map3:+.5f})", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

