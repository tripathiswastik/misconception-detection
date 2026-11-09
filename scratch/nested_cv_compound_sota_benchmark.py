"""Test Compound SOTA 550 Recipe with ArcFace V2 in Nested CV.

In SOTA 550:
1. Base: 0.95 * M15 + 0.02 * dual448 + 0.03 * comp448
2. Calibrated: base * mult
3. Boosted: cal * 1.20 (question conditioned)
4. ArcFace Term: (0.95 * M15 + 0.05 * ArcFace) * mult
5. Compound Champion: (0.95 - w_boosted) * cal + w_boosted * boosted + 0.05 * arc_term
6. Clean Multimodal Blend: 0.96 * champ + 0.04 * clean
7. Decision Engine: beta = 0.82
8. Precision Gating: tau = 0.780, eps = 0.040

We test substituting ArcFace V2 for ArcFace V1 in term 4, and sweeping the ArcFace weight (0.02, 0.05, 0.08, 0.10, 0.12, 0.15) inside strict leak-free Nested CV.
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

def run_compound_sota_pipeline(
    oof_m15: np.ndarray,
    oof_dual: np.ndarray,
    oof_comp: np.ndarray,
    oof_arc: np.ndarray,
    w_arc: float,
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    clean_probs: np.ndarray,
    bin_error_probs: np.ndarray,
    train_df: pd.DataFrame,
    classes: List[str],
    none_idx: int,
    w_boosted: float = 0.25,
    tau: float = 0.780,
    eps: float = 0.040,
    beta: float = 0.82
) -> List[List[str]]:
    n_classes = len(classes)
    n_eval = len(eval_indices)

    # Base: 0.95 * M15 + 0.02 * dual + 0.03 * comp
    p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    # 1. Frequency Calibration Multiplier fitted STRICTLY on train_indices
    top3_tr = np.argsort(-p_base[train_indices].astype(np.float64), axis=1)[:, :3]
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

    p_eval_cal = p_base[eval_indices] * mult
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

    # 3. ArcFace Term
    p_eval_arc_cal = ((1.0 - w_arc) * oof_m15[eval_indices] + w_arc * oof_arc[eval_indices]) * mult

    w_base = 0.95 - w_boosted  # 0.70
    p_eval_champ = w_base * p_eval_cal + w_boosted * p_eval_boosted + 0.05 * p_eval_arc_cal

    # 4. Clean Multimodal Blend (w_clean = 0.04)
    p_eval_final = 0.96 * p_eval_champ + 0.04 * clean_probs[eval_indices]

    # 5. Decision Engine Ranking with secondary beta
    cand = p_eval_final.copy()
    row_idx = np.arange(n_eval)
    r1 = np.argmax(cand, axis=1)
    cand[row_idx, r1] = -1.0
    cand[r1 != none_idx, none_idx] *= beta
    top2 = np.argsort(-cand, axis=1)[:, :2]
    preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_eval)]

    # 6. Precision Gating (tau = 0.780, eps = 0.040)
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
    print("COMPOUND SOTA 550 RECIPE: STRICT NESTED CV COMPARISON (V1 vs V2)", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

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
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    # Candidate ArcFace weights inside the compound term:
    candidate_w = [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

    preds_v1_baseline: List[List[str]] = [[] for _ in range(n_samples)]
    preds_v2_at_v1_weight: List[List[str]] = [[] for _ in range(n_samples)]
    preds_v2_nested_opt: List[List[str]] = [[] for _ in range(n_samples)]
    preds_v2_fixed: Dict[float, List[List[str]]] = {w: [[] for _ in range(n_samples)] for w in candidate_w}

    selected_inner_w: List[float] = []

    for fold_idx, outer_fold in enumerate(splits, start=1):
        outer_tr_idx = np.array(outer_fold["train_indices"], dtype=np.int32)
        outer_va_idx = np.array(outer_fold["val_indices"], dtype=np.int32)

        print(f"\n--- OUTER FOLD {fold_idx}/5 (Holdout: {len(outer_va_idx)}) ---", flush=True)

        # 1. Inner CV on Outer Train (1,200) to select w_arc for V2
        y_outer_tr = [y_true[i] for i in outer_tr_idx]
        cnt_tr = Counter(y_outer_tr)
        strat_y = [y if cnt_tr[y] >= 4 else "RARE" for y in y_outer_tr]
        skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42 + fold_idx)
        inner_splits = list(skf.split(outer_tr_idx, strat_y))

        inner_scores: Dict[float, float] = {}
        for w in candidate_w:
            fold_scores = []
            for in_tr_local, in_va_local in inner_splits:
                in_tr_idx = outer_tr_idx[in_tr_local]
                in_va_idx = outer_tr_idx[in_va_local]

                preds_in = run_compound_sota_pipeline(
                    oof_m15=oof_m15,
                    oof_dual=oof_dual,
                    oof_comp=oof_comp,
                    oof_arc=oof_v2,
                    w_arc=w,
                    train_indices=in_tr_idx,
                    eval_indices=in_va_idx,
                    clean_probs=oof_clean,
                    bin_error_probs=oof_bin,
                    train_df=train_df,
                    classes=classes,
                    none_idx=none_idx
                )
                score_in = calculate_map3([y_true[i] for i in in_va_idx], preds_in)
                fold_scores.append(score_in)
            inner_scores[w] = float(np.mean(fold_scores))

        best_w = max(inner_scores.keys(), key=lambda w: inner_scores[w])
        selected_inner_w.append(best_w)
        print(f"  Inner CV Scores for V2: " + ", ".join([f"w={w:.2f}:{inner_scores[w]:.5f}" for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]]), flush=True)
        print(f"  => Selected Best Inner Weight: w* = {best_w:.2f} (Inner MAP@3: {inner_scores[best_w]:.5f})", flush=True)

        # 2. Evaluate on Untouched Outer Holdout
        # V1 Baseline (w_arc = 0.05 with V1 ArcFace)
        p_v1 = run_compound_sota_pipeline(
            oof_m15=oof_m15,
            oof_dual=oof_dual,
            oof_comp=oof_comp,
            oof_arc=oof_v1,
            w_arc=0.05,
            train_indices=outer_tr_idx,
            eval_indices=outer_va_idx,
            clean_probs=oof_clean,
            bin_error_probs=oof_bin,
            train_df=train_df,
            classes=classes,
            none_idx=none_idx
        )
        for local_i, global_i in enumerate(outer_va_idx):
            preds_v1_baseline[global_i] = p_v1[local_i]
        score_v1 = calculate_map3([y_true[i] for i in outer_va_idx], p_v1)

        # V2 at same baseline weight (w_arc = 0.05 with V2 ArcFace)
        p_v2_base = run_compound_sota_pipeline(
            oof_m15=oof_m15,
            oof_dual=oof_dual,
            oof_comp=oof_comp,
            oof_arc=oof_v2,
            w_arc=0.05,
            train_indices=outer_tr_idx,
            eval_indices=outer_va_idx,
            clean_probs=oof_clean,
            bin_error_probs=oof_bin,
            train_df=train_df,
            classes=classes,
            none_idx=none_idx
        )
        for local_i, global_i in enumerate(outer_va_idx):
            preds_v2_at_v1_weight[global_i] = p_v2_base[local_i]
        score_v2_base = calculate_map3([y_true[i] for i in outer_va_idx], p_v2_base)

        # V2 at nested selected weight w*
        p_v2_nested = run_compound_sota_pipeline(
            oof_m15=oof_m15,
            oof_dual=oof_dual,
            oof_comp=oof_comp,
            oof_arc=oof_v2,
            w_arc=best_w,
            train_indices=outer_tr_idx,
            eval_indices=outer_va_idx,
            clean_probs=oof_clean,
            bin_error_probs=oof_bin,
            train_df=train_df,
            classes=classes,
            none_idx=none_idx
        )
        for local_i, global_i in enumerate(outer_va_idx):
            preds_v2_nested_opt[global_i] = p_v2_nested[local_i]
        score_v2_nested = calculate_map3([y_true[i] for i in outer_va_idx], p_v2_nested)

        print(f"  Outer Holdout: V1={score_v1:.5f} | V2(w=0.05)={score_v2_base:.5f} | Nested V2(w*={best_w:.2f})={score_v2_nested:.5f} (Delta vs V1: {score_v2_nested - score_v1:+.5f})", flush=True)

        for w in candidate_w:
            p_v2_fix = run_compound_sota_pipeline(
                oof_m15=oof_m15,
                oof_dual=oof_dual,
                oof_comp=oof_comp,
                oof_arc=oof_v2,
                w_arc=w,
                train_indices=outer_tr_idx,
                eval_indices=outer_va_idx,
                clean_probs=oof_clean,
                bin_error_probs=oof_bin,
                train_df=train_df,
                classes=classes,
                none_idx=none_idx
            )
            for local_i, global_i in enumerate(outer_va_idx):
                preds_v2_fixed[w][global_i] = p_v2_fix[local_i]

    print("\n" + "=" * 80, flush=True)
    print("FINAL 100% LEAK-FREE NESTED CV RESULTS ON COMPOUND SOTA PIPELINE (1,500 ROWS):", flush=True)
    print("=" * 80, flush=True)

    tot_v1 = calculate_map3(y_true, preds_v1_baseline)
    tot_v2_base = calculate_map3(y_true, preds_v2_at_v1_weight)
    tot_v2_nested = calculate_map3(y_true, preds_v2_nested_opt)

    print(f"Selected Inner Weights Across 5 Folds: {selected_inner_w}", flush=True)
    print(f"\n1. Full SOTA 550 V1 Pipeline (Outer OOF)          : {tot_v1:.5f}", flush=True)
    print(f"2. Drop-in ArcFace V2 (w=0.05, Outer OOF)        : {tot_v2_base:.5f} (Delta: {tot_v2_base - tot_v1:+.5f})", flush=True)
    print(f"3. Nested CV Selected V2 (w*, Outer OOF)         : {tot_v2_nested:.5f} (Delta: {tot_v2_nested - tot_v1:+.5f})", flush=True)

    print("\n4. Fixed ArcFace V2 Weight Across Untouched Outer Holdouts:", flush=True)
    for w in candidate_w:
        score_w = calculate_map3(y_true, preds_v2_fixed[w])
        print(f"   w_arc = {w:.2f} -> Outer OOF MAP@3: {score_w:.5f} (Delta vs V1: {score_w - tot_v1:+.5f})", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

