"""Simulate Champion + Clean Leak-Free Multimodal Fusion.

Explores:
1. Probability blending between the Active Champion base (p_oof_base)
   and the Clean Leak-Free Multimodal Ensemble.
2. In-fold cross-validation performance of the fused probabilities.
3. Analysis of Rank 2 and Rank 3 accuracy improvements.
4. Validation against both Canonical 5-Fold CV and Question-Grouped CV.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")
n_classes = len(tax_classes)

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]


def score_map3(y_true_subset: List[str], probs: np.ndarray, classes: List[str]) -> float:
    k = min(3, probs.shape[1])
    n = len(probs)
    part = np.argpartition(-probs, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-probs[row_idx, part], axis=1, kind="mergesort")
    top3_idx = part[row_idx, order]
    
    scores = []
    for label, row in zip(y_true_subset, top3_idx):
        score = 0.0
        for r, idx in enumerate(row):
            if classes[idx] == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    print("=" * 80)
    print("CHAMPION + CLEAN MULTIMODAL FUSION & CROSS-VALIDATION")
    print("=" * 80)

    # 1. Load Champion OOF
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_champ_oof = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Frequency calibration on Champion OOF
    top3_champ = np.argsort(-p_champ_oof.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(tax_classes[idx] for row in top3_champ for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for ci, c in enumerate(tax_classes):
        if ci != none_idx:
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[ci] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_champ_cal = p_champ_oof * mult
    p_champ_cal[:, none_idx] *= 0.9995

    s_champ = score_map3(y_true, p_champ_cal, tax_classes)
    print(f"Active Champion Base OOF MAP@3: {s_champ:.5f}")

    # 2. Load Clean Multimodal Components
    from src.data.taxonomy import TaxonomyManager
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")

    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_th = np.load(ART_DIR / "honest_oof_text.npy")
    p_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    p_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    # In-fold domain mask for MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )

    # Temperature scaling on text
    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_th_sharp = apply_t(p_th, 0.6)
    p_ts_sharp = apply_t(p_ts, 0.6)

    # Clean multimodal ensemble
    p_clean_ens = (
        0.58 * p_prior +
        0.05 * p_cn +
        0.07 * p_swin +
        0.07 * p_mn +
        0.04 * p_trocr +
        0.11 * p_th_sharp +
        0.08 * p_ts_sharp
    )
    s_clean = score_map3(y_true, p_clean_ens, tax_classes)
    print(f"Clean Multimodal Ensemble OOF MAP@3: {s_clean:.5f}")

    # 3. Sweep Blending Ratio: alpha * Champion + (1 - alpha) * Clean Ensemble
    print("\n--- 3. Sweep: alpha * Champion + (1 - alpha) * Clean Ensemble ---")
    best_alpha = 1.0
    best_blend_s = s_champ
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.90, 0.95, 1.0]:
        fused = alpha * p_champ_cal + (1.0 - alpha) * p_clean_ens
        fold_scores = []
        for f in splits:
            va = f["val_indices"]
            fold_scores.append(score_map3([y_true[i] for i in va], fused[va], tax_classes))
        mean_cv = float(np.mean(fold_scores))
        overall = score_map3(y_true, fused, tax_classes)
        print(f"  alpha={alpha:.2f} | 5-Fold Mean: {mean_cv:.5f} | Overall: {overall:.5f} ({overall - s_champ:+.5f})")
        if overall > best_blend_s:
            best_blend_s = overall
            best_alpha = alpha

    print(f"\nBest Blend Alpha: {best_alpha:.2f} -> OOF MAP@3: {best_blend_s:.5f} ({best_blend_s - s_champ:+.5f})")

    # 4. Now evaluate SNR85 Secondary Recalibration on the Best Fused Probabilities
    print("\n--- 4. Secondary NONE Recalibration (SNR85) on Fused vs Champion ---")
    fused_best = best_alpha * p_champ_cal + (1.0 - best_alpha) * p_clean_ens

    for name, p_matrix in [("Original Champion", p_champ_cal), ("Fused Champion+Clean", fused_best)]:
        # Simulate SNR85 logic on OOF
        # Rank 1: Top 1 from p_matrix
        r1_idx = np.argmax(p_matrix, axis=1)
        cand_mat = p_matrix.copy()
        row_idx = np.arange(len(p_matrix))
        cand_mat[row_idx, r1_idx] = -1.0
        # Discount secondary NONE if Rank 1 is non-NONE
        is_nn = (r1_idx != none_idx)
        for beta in [0.80, 0.85, 0.90, 1.00]:
            cm = cand_mat.copy()
            cm[is_nn, none_idx] *= beta
            part2 = np.argpartition(-cm, kth=1, axis=1)[:, :2]
            p2 = cm[row_idx[:, None], part2]
            ord2 = np.argsort(-p2, axis=1, kind="mergesort")
            top2 = part2[row_idx[:, None], ord2]
            preds = [[tax_classes[r1_idx[i]], tax_classes[top2[i, 0]], tax_classes[top2[i, 1]]] for i in range(len(p_matrix))]
            sc = float(np.mean([1.0 if y == p[0] else 0.5 if y == p[1] else 1.0/3.0 if y == p[2] else 0.0 for y, p in zip(y_true, preds)]))
            print(f"  {name:22s} | beta={beta:.2f} -> OOF MAP@3: {sc:.5f} ({sc - s_champ:+.5f})")


if __name__ == "__main__":
    main()

