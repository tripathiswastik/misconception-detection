"""Test NONE Probability Temperature / Scaling for Rank-1 Selection in 5-Fold CV.

Investigates:
When deciding Rank 1, NONE currently wins because 75 misconception classes split the remaining probability.
What if we calibrate NONE probability for Rank 1 selection with scaling factor alpha in [0.80, 1.00]?
Evaluates strictly across the 5 canonical folds with zero leakage.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

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

def main():
    print("=" * 80, flush=True)
    print("EVALUATING NONE RANK-1 SCALING (ALPHA) IN 5-FOLD CV", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_base_v1 = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    oof_clean = (
        0.6448 * np.load(ART_DIR / "best_adaptive_prior_oof.npy") +
        0.0839 * sharp(np.load(ART_DIR / "honest_oof_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "oof_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "oof_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "oof_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "oof_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "oof_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "oof_clean_arcface_2328.npy"), 0.8)
    )

    oof_bin = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    # Precompute outer holdout probabilities with leak-free multiplier and question boost
    oof_final_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_maps = []

    for fold in splits:
        tr_idx = np.array(fold["train_indices"], dtype=np.int32)
        va_idx = np.array(fold["val_indices"], dtype=np.int32)
        n_eval = len(va_idx)

        top3_tr = np.argsort(-p_base_v1[tr_idx].astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
        true_labels_tr = train_df.iloc[tr_idx]["root_code"].fillna("NONE").astype(str).tolist()
        true_counts = Counter(true_labels_tr)

        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult[i] = (tc / pc) ** 0.05
        mult[none_idx] = 0.9840

        p_eval_cal = p_base_v1[va_idx] * mult
        p_eval_cal[:, none_idx] *= 0.9995

        tr_sub_df = train_df.iloc[tr_idx]
        q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}

        p_eval_boosted = np.copy(p_eval_cal)
        eval_df = train_df.iloc[va_idx].reset_index(drop=True)
        for i in range(n_eval):
            qid = eval_df.iloc[i]["question_id"]
            if qid in q_map:
                allowed = q_map[qid]
                for ci, code in enumerate(classes):
                    if code != "NONE" and code in allowed:
                        p_eval_boosted[i, ci] *= 1.20

        p_eval_arc_cal = (0.95 * oof_m15[va_idx] + 0.05 * oof_v1[va_idx]) * mult
        p_eval_champ = 0.70 * p_eval_cal + 0.25 * p_eval_boosted + 0.05 * p_eval_arc_cal
        oof_final_probs[va_idx] = 0.96 * p_eval_champ + 0.04 * oof_clean[va_idx]

    # Now sweep alpha on NONE for Rank 1 decision
    print(f"{'alpha (NONE scale)':20s} | {'Leak-Free OOF MAP@3':22s} | {'Delta vs Baseline':20s}")
    print("-" * 68)

    base_score = 0.0
    for alpha in [1.00, 0.995, 0.990, 0.985, 0.980, 0.975, 0.970, 0.960, 0.950, 0.930, 0.900]:
        outer_preds = [[] for _ in range(n_samples)]
        for fold in splits:
            va_idx = np.array(fold["val_indices"], dtype=np.int32)
            eval_df = train_df.iloc[va_idx].reset_index(drop=True)
            tr_sub_df = train_df.iloc[fold["train_indices"]]
            q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}

            cand = oof_final_probs[va_idx].copy()
            # Apply alpha ONLY to NONE for Rank 1 selection
            cand_r1 = cand.copy()
            cand_r1[:, none_idx] *= alpha
            r1 = np.argmax(cand_r1, axis=1)

            # Re-rank remaining positions with secondary beta = 0.82
            row_idx = np.arange(len(va_idx))
            cand[row_idx, r1] = -1.0
            cand[r1 != none_idx, none_idx] *= 0.82
            top2 = np.argsort(-cand, axis=1)[:, :2]
            preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(len(va_idx))]

            # Gating
            for i in range(len(va_idx)):
                if preds[i][0] == "NONE" and oof_bin[va_idx[i]] >= 0.780:
                    row_p = oof_final_probs[va_idx[i]].copy()
                    p_none_val = row_p[none_idx]
                    row_p[none_idx] = -1.0
                    top_misc_idx = int(np.argmax(row_p))
                    top_misc_code = classes[top_misc_idx]
                    gap = p_none_val - row_p[top_misc_idx]

                    if 0 < gap < 0.040:
                        qid = eval_df.iloc[i]["question_id"]
                        if top_misc_code in q_map.get(qid, set()):
                            row_p[top_misc_idx] = -1.0
                            second_misc_idx = int(np.argmax(row_p))
                            second_misc_code = classes[second_misc_idx]
                            preds[i] = [top_misc_code, "NONE", second_misc_code]

            for local_i, global_i in enumerate(va_idx):
                outer_preds[global_i] = preds[local_i]

        score = calculate_map3(y_true, outer_preds)
        if alpha == 1.00:
            base_score = score
        delta = score - base_score
        print(f"alpha = {alpha:6.3f}       | {score:22.5f} | {delta:+20.5f}")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

