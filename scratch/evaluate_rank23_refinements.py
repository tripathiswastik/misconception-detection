"""Evaluate Rank 2 & Rank 3 Refinements on Locked-Rank-1 Champion under 5-Fold CV.

Strictly benchmarks:
1. Baseline: Champion Rank 1 Locked + Champion Rank 2/3 (beta=0.85) -> active 0.54799 LB
2. Candidate A: Champion Rank 1 Locked + Champion Rank 2/3 (beta=0.80)
3. Candidate B: Champion Rank 1 Locked + Fused (0.95 Champ + 0.05 Clean Multimodal, beta=0.85)
4. Candidate C: Champion Rank 1 Locked + Fused (0.95 Champ + 0.05 Clean Multimodal, beta=0.80)
5. Candidate D: Champion Rank 1 Locked + Fused (0.90 Champ + 0.10 Clean Multimodal, beta=0.80)
6. Guardrail stress test on Question-Grouped CV across all candidates!
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")
n_classes = len(tax_classes)

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]


def score_triplets(y_true_subset: List[str], preds_subset: List[List[str]]) -> float:
    scores = []
    for label, pred in zip(y_true_subset, preds_subset):
        score = 0.0
        for r, p in enumerate(pred[:3]):
            if p == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def generate_locked_r1_preds(
    r1_indices: np.ndarray,
    candidate_probs: np.ndarray,
    classes: List[str],
    beta_none: float = 0.85
) -> List[List[str]]:
    n = len(r1_indices)
    cand_matrix = candidate_probs.copy()
    row_idx = np.arange(n)

    # Mask Rank 1
    cand_matrix[row_idx, r1_indices] = -1.0

    # Discount secondary NONE if Rank 1 is non-NONE
    is_non_none = (r1_indices != none_idx)
    cand_matrix[is_non_none, none_idx] *= beta_none

    # Select top 2
    part2 = np.argpartition(-cand_matrix, kth=1, axis=1)[:, :2]
    p2 = cand_matrix[row_idx[:, None], part2]
    ord2 = np.argsort(-p2, axis=1, kind="mergesort")
    top2 = part2[row_idx[:, None], ord2]

    return [
        [classes[r1_indices[i]], classes[top2[i, 0]], classes[top2[i, 1]]]
        for i in range(n)
    ]


def main():
    print("=" * 80)
    print("RANK 2 & 3 REFINEMENT BENCHMARK (5-FOLD CV + QUESTION-GROUPED GUARDRAIL)")
    print("=" * 80)

    # 1. Champion OOF & Rank 1
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    mult = np.ones(n_classes, dtype=np.float64)
    mult[none_idx] = 0.9840
    p_champ_cal = p_champ * mult
    p_champ_cal[:, none_idx] *= 0.9995

    # Locked Rank 1 indices from Champion
    r1_locked = np.argmax(p_champ_cal, axis=1)

    # 2. Clean Multimodal Ensemble
    from src.data.taxonomy import TaxonomyManager
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")

    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_th = np.load(ART_DIR / "honest_oof_text.npy")
    p_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    p_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )

    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_th_sharp = apply_t(p_th, 0.6)
    p_ts_sharp = apply_t(p_ts, 0.6)

    p_clean = (
        0.58 * p_prior +
        0.05 * p_cn +
        0.07 * p_swin +
        0.07 * p_mn +
        0.04 * p_trocr +
        0.11 * p_th_sharp +
        0.08 * p_ts_sharp
    )

    # Candidate probability configurations for Rank 2 & 3
    configs = {
        "Baseline (Champ alone, beta=0.85)": (p_champ_cal, 0.85),
        "Cand A: Champ alone, beta=0.80": (p_champ_cal, 0.80),
        "Cand B: Fused 95/05, beta=0.85": (0.95 * p_champ_cal + 0.05 * p_clean, 0.85),
        "Cand C: Fused 95/05, beta=0.80": (0.95 * p_champ_cal + 0.05 * p_clean, 0.80),
        "Cand D: Fused 90/10, beta=0.80": (0.90 * p_champ_cal + 0.10 * p_clean, 0.80),
        "Cand E: Fused 92/08, beta=0.82": (0.92 * p_champ_cal + 0.08 * p_clean, 0.82),
    }

    gkf = GroupKFold(n_splits=5)
    g_folds = list(gkf.split(train_df, y_true, groups=train_df["question_id"].values))

    base_s = None
    results = {}

    print(f"\n{'Configuration':<35} | {'Strat OOF':<10} | {'5-Fold Mean':<12} | {'Grouped OOF':<12} | {'Delta vs Base'}")
    print("-" * 85)

    for name, (prob_mat, beta_val) in configs.items():
        all_preds = generate_locked_r1_preds(r1_locked, prob_mat, tax_classes, beta_none=beta_val)
        strat_score = score_triplets(y_true, all_preds)

        # 5-fold stratified fold scores
        fold_scores = []
        for f in splits:
            va = f["val_indices"]
            f_preds = [all_preds[i] for i in va]
            fold_scores.append(score_triplets([y_true[i] for i in va], f_preds))
        strat_mean = float(np.mean(fold_scores))

        # Question-grouped guardrail score
        group_fold_scores = []
        for tr, va in g_folds:
            g_preds = [all_preds[i] for i in va]
            group_fold_scores.append(score_triplets([y_true[i] for i in va], g_preds))
        grouped_mean = float(np.mean(group_fold_scores))

        if base_s is None:
            base_s = strat_score

        delta = strat_score - base_s
        print(f"{name:<35} | {strat_score:<10.5f} | {strat_mean:<12.5f} | {grouped_mean:<12.5f} | {delta:+.5f}")
        results[name] = {
            "strat_oof": strat_score,
            "strat_mean": strat_mean,
            "grouped_mean": grouped_mean,
            "delta": delta
        }

    print("=" * 85)


if __name__ == "__main__":
    main()

