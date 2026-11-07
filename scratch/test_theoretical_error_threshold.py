"""Test Theoretically Optimal MAP@3 Error Promotion Threshold under Strict 5-Fold CV.

Formula derived from MAP@3 expectation:
Expected Gain of promoting c* to Rank 1 over NONE:
  Gain = P(Error) * P(c* | Error) - P(NONE)
  Promote to Rank 1 iff Gain > threshold (theoretical default: 0.0)

Evaluates strictly inside 5-fold CV across splits.json and GroupKFold guardrail.
"""

import json
import sys
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


def theoretical_ranker(
    probs: np.ndarray,
    classes: List[str],
    gain_margin: float = 0.0,
    secondary_none_penalty: float = 0.85
) -> List[List[str]]:
    """Rank predictions using the exact MAP@3 expected gain formula."""
    n = len(probs)
    preds = []
    non_none_mask = np.ones(len(classes), dtype=bool)
    non_none_mask[none_idx] = False
    non_none_indices = np.where(non_none_mask)[0]

    for i in range(n):
        row = probs[i]
        p_none = row[none_idx]
        p_error = 1.0 - p_none

        # Top non-NONE candidates
        top_nn_local = np.argsort(-row[non_none_mask])[:3]
        c1_idx = non_none_indices[top_nn_local[0]]
        c2_idx = non_none_indices[top_nn_local[1]]
        c3_idx = non_none_indices[top_nn_local[2]]

        c1 = classes[c1_idx]
        c2 = classes[c2_idx]
        c3 = classes[c3_idx]

        # P(c1 | Error)
        p_c1_given_err = row[c1_idx] / max(1e-12, p_error)

        # Theoretical Gain of putting c1 at Rank 1 over NONE at Rank 1
        gain = p_error * p_c1_given_err - p_none

        if gain > gain_margin:
            # Promote c1 to Rank 1!
            # Rank 2 decision: compare secondary NONE vs c2
            eff_none = p_none * secondary_none_penalty
            if eff_none >= row[c2_idx]:
                preds.append([c1, "NONE", c2])
            elif eff_none >= row[c3_idx]:
                preds.append([c1, c2, "NONE"])
            else:
                preds.append([c1, c2, c3])
        else:
            # NONE at Rank 1
            preds.append(["NONE", c1, c2])

    return preds


def main():
    print("=" * 80)
    print("THEORETICAL MAP@3 ERROR PROMOTION THRESHOLD AUDIT")
    print("=" * 80)

    # 1. Load Champion probabilities
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    mult = np.ones(n_classes, dtype=np.float64)
    mult[none_idx] = 0.9840
    p_champ_cal = p_champ * mult
    p_champ_cal[:, none_idx] *= 0.9995

    # 2. Load Clean Multimodal Ensemble
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

    fused_probs = 0.90 * p_champ_cal + 0.10 * p_clean

    # Raw baseline score
    k = 3
    n = len(fused_probs)
    part = np.argpartition(-fused_probs, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-fused_probs[row_idx, part], axis=1, kind="mergesort")
    raw_preds = [[tax_classes[idx] for idx in row] for row in part[row_idx, order]]
    raw_score = score_triplets(y_true, raw_preds)
    print(f"Base Fused Argmax OOF MAP@3: {raw_score:.5f}")

    # Sweep gain_margin around theoretical 0.0
    print("\n--- 1. Sweep Gain Margin in 5-Fold Stratified CV ---")
    best_margin = 0.0
    best_beta = 0.85
    best_cv = 0.0

    for margin in [-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]:
        for beta in [0.75, 0.80, 0.85, 0.90, 1.00]:
            fold_scores = []
            for f in splits:
                va = f["val_indices"]
                va_preds = theoretical_ranker(
                    fused_probs[va], tax_classes, gain_margin=margin, secondary_none_penalty=beta
                )
                fold_scores.append(score_triplets([y_true[i] for i in va], va_preds))
            mean_cv = float(np.mean(fold_scores))
            if mean_cv > best_cv:
                best_cv = mean_cv
                best_margin = margin
                best_beta = beta

    print(f"Top Parameters: gain_margin = {best_margin:+.2f}, secondary_beta = {best_beta:.2f}")
    print(f"5-Fold CV MAP@3: {best_cv:.5f} ({best_cv - raw_score:+.5f})")

    # Evaluate on full OOF
    opt_preds = theoretical_ranker(fused_probs, tax_classes, gain_margin=best_margin, secondary_none_penalty=best_beta)
    overall_s = score_triplets(y_true, opt_preds)
    print(f"Overall OOF MAP@3: {overall_s:.5f} ({overall_s - raw_score:+.5f})")

    # Guardrail: Question-Grouped CV
    gkf = GroupKFold(n_splits=5)
    g_folds = list(gkf.split(train_df, y_true, groups=train_df["question_id"].values))
    g_scores = []
    for tr, va in g_folds:
        va_preds = theoretical_ranker(fused_probs[va], tax_classes, gain_margin=best_margin, secondary_none_penalty=best_beta)
        g_scores.append(score_triplets([y_true[i] for i in va], va_preds))
    print(f"Question-Grouped CV Guardrail MAP@3: {np.mean(g_scores):.5f} +/- {np.std(g_scores, ddof=1):.5f}")

    # Rank Distribution
    r1 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[0])
    r2 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[1])
    r3 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[2])

    err_idx = [i for i, y in enumerate(y_true) if y != "NONE"]
    err_r1 = sum(1 for i in err_idx if y_true[i] == opt_preds[i][0])
    err_r1_none = sum(1 for i in err_idx if opt_preds[i][0] == "NONE")

    none_idx_list = [i for i, y in enumerate(y_true) if y == "NONE"]
    none_r1 = sum(1 for i in none_idx_list if opt_preds[i][0] == "NONE")

    print("\n[Detailed Diagnostics]")
    print(f"  Rank Distribution : Rank 1={r1:4d} ({(r1/1500)*100:.1f}%), Rank 2={r2:4d}, Rank 3={r3:4d}")
    print(f"  Error Traces (994): True at Rank 1={err_r1:4d} ({(err_r1/994)*100:.1f}%), Incorrect Rank 1 NONE={err_r1_none:4d}")
    print(f"  NONE Traces (506) : Correct at Rank 1={none_r1:4d} ({(none_r1/506)*100:.1f}%)")


if __name__ == "__main__":
    main()

