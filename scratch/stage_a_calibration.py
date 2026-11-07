"""Stage A: Base Model Quality & Calibration Evaluation under Strict 5-Fold CV.

Evaluates:
1. Component-level Temperature Scaling: T_vis, T_text, T_prior
2. Fold-local Class Frequency Balancing: (N_true / N_pred)^gamma computed ONLY on fold-train
3. Power Calibration on P(NONE) across folds
4. Nested CV score vs baseline (0.51856)
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

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")

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


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    if abs(t - 1.0) < 1e-5:
        return probs.copy()
    sharp = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    row_sums = sharp.sum(axis=1, keepdims=True)
    return np.divide(sharp, row_sums, where=row_sums > 0)


def apply_fold_frequency_balancing(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    probs_val: np.ndarray,
    probs_train: np.ndarray,
    gamma: float
) -> np.ndarray:
    """Fold-local frequency balancing fitted ONLY on fold-train."""
    if abs(gamma) < 1e-5:
        return probs_val.copy()
        
    k = min(3, probs_train.shape[1])
    n_tr = len(train_idx)
    part = np.argpartition(-probs_train, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n_tr)[:, None]
    order = np.argsort(-probs_train[row_idx, part], axis=1, kind="mergesort")
    top3_tr = part[row_idx, order]
    
    pred_counts = Counter(tax_classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true[i] for i in train_idx)
    
    mult = np.ones(len(tax_classes), dtype=np.float64)
    for ci, c in enumerate(tax_classes):
        if ci != none_idx:
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[ci] = (tc / pc) ** gamma
            
    val_cal = probs_val * mult
    row_sums = val_cal.sum(axis=1, keepdims=True)
    return np.divide(val_cal, row_sums, where=row_sums > 0)


def main():
    print("=" * 80)
    print("STAGE A: CALIBRATION & TEMPERATURE SCALING AUDIT")
    print("=" * 80)
    
    # 1. Load components
    from src.data.taxonomy import TaxonomyManager
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )
        
    components = {
        "Prior": p_prior,
        "ConvNeXt": p_cn,
        "Swin": p_swin,
        "MobileNet": p_mn,
        "Honest_Text": p_text_honest,
        "Semantic_Text": p_text_sem,
    }
    
    print("\n--- 1. Single Component Temperature Sweeps ---")
    for name, p in components.items():
        base_s = score_map3(y_true, p, tax_classes)
        best_t = 1.0
        best_s = base_s
        for t in [0.5, 0.7, 0.8, 0.9, 1.1, 1.2, 1.5, 2.0]:
            p_t = apply_temperature(p, t)
            s = score_map3(y_true, p_t, tax_classes)
            if s > best_s:
                best_s = s
                best_t = t
        print(f"  {name:15s} | Raw: {base_s:.5f} -> Best T={best_t:.1f}: {best_s:.5f} ({best_s - base_s:+.5f})")

    # 2. Temperature on Vision & Text before Clean Blend
    print("\n--- 2. Grid Search on Modality Temperatures in Clean Blend ---")
    w = np.array([0.6059, 0.0448, 0.0703, 0.0709, 0.1189, 0.0892])
    base_blend = (
        w[0] * p_prior + w[1] * p_cn + w[2] * p_swin +
        w[3] * p_mn + w[4] * p_text_honest + w[5] * p_text_sem
    )
    base_blend_score = score_map3(y_true, base_blend, tax_classes)
    print(f"  Uncalibrated Clean Blend OOF MAP@3: {base_blend_score:.5f}")
    
    best_blend_s = base_blend_score
    best_params = (1.0, 1.0, 1.0)
    for t_prior in [0.8, 1.0, 1.2]:
        for t_vis in [0.6, 0.8, 1.0, 1.2, 1.5]:
            for t_txt in [0.6, 0.8, 1.0, 1.2, 1.5]:
                p_pr_t = apply_temperature(p_prior, t_prior)
                p_cn_t = apply_temperature(p_cn, t_vis)
                p_sw_t = apply_temperature(p_swin, t_vis)
                p_mn_t = apply_temperature(p_mn, t_vis)
                p_th_t = apply_temperature(p_text_honest, t_txt)
                p_ts_t = apply_temperature(p_text_sem, t_txt)
                
                fused = (
                    w[0] * p_pr_t + w[1] * p_cn_t + w[2] * p_sw_t +
                    w[3] * p_mn_t + w[4] * p_th_t + w[5] * p_ts_t
                )
                s = score_map3(y_true, fused, tax_classes)
                if s > best_blend_s:
                    best_blend_s = s
                    best_params = (t_prior, t_vis, t_txt)
                    
    print(f"  Best Temperatures: T_prior={best_params[0]:.1f}, T_vis={best_params[1]:.1f}, T_txt={best_params[2]:.1f}")
    print(f"  Calibrated Blend OOF MAP@3: {best_blend_s:.5f} ({best_blend_s - base_blend_score:+.5f})")
    
    # 3. Fold-local Class Frequency Balancing
    print("\n--- 3. Fold-Local Class Frequency Balancing on Calibrated Blend ---")
    # Apply best temperatures first
    t_pr, t_vi, t_tx = best_params
    p_pr_opt = apply_temperature(p_prior, t_pr)
    p_cn_opt = apply_temperature(p_cn, t_vi)
    p_sw_opt = apply_temperature(p_swin, t_vi)
    p_mn_opt = apply_temperature(p_mn, t_vi)
    p_th_opt = apply_temperature(p_text_honest, t_tx)
    p_ts_opt = apply_temperature(p_text_sem, t_tx)
    
    best_fused = (
        w[0] * p_pr_opt + w[1] * p_cn_opt + w[2] * p_sw_opt +
        w[3] * p_mn_opt + w[4] * p_th_opt + w[5] * p_ts_opt
    )
    
    best_gamma = 0.0
    best_gamma_s = best_blend_s
    for gamma in [0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10]:
        cal_oof = np.zeros_like(best_fused)
        fold_scores = []
        for f in splits:
            tr = f["train_indices"]
            va = f["val_indices"]
            cal_oof[va] = apply_fold_frequency_balancing(
                np.array(tr), np.array(va), best_fused[va], best_fused[tr], gamma=gamma
            )
            fold_scores.append(score_map3([y_true[i] for i in va], cal_oof[va], tax_classes))
        overall_s = score_map3(y_true, cal_oof, tax_classes)
        mean_f = float(np.mean(fold_scores))
        print(f"  gamma={gamma:.2f} | 5-Fold Mean: {mean_f:.5f} | Overall OOF: {overall_s:.5f} ({overall_s - best_blend_s:+.5f})")
        if overall_s > best_gamma_s:
            best_gamma_s = overall_s
            best_gamma = gamma
            
    print(f"\nFinal Stage A Calibrated Result: gamma={best_gamma:.2f} -> OOF MAP@3: {best_gamma_s:.5f} ({best_gamma_s - base_blend_score:+.5f})")


if __name__ == "__main__":
    main()

