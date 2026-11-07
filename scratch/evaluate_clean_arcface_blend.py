"""Evaluate Clean Multimodal ArcFace (2,328-d) in Leak-Free Nested CV.

Benchmarks:
1. Addition of oof_clean_arcface_2328.npy to the clean multimodal ensemble.
2. 5-Fold Nested CV to determine optimal weights.
3. Theoretical MAP@3 decision engine & secondary NONE recalibration.
4. Comparison vs 0.51856 baseline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from scipy.optimize import minimize  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.evaluation.cross_validator import ModelCrossValidator
from src.evaluation.leak_free_cv import fast_fold_domain_mask

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def score_subset(y_sub: List[str], p_sub: np.ndarray, classes: List[str]) -> float:
    k = min(3, p_sub.shape[1])
    n = len(p_sub)
    part = np.argpartition(-p_sub, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-p_sub[row_idx, part], axis=1, kind="mergesort")
    top3_idx = part[row_idx, order]
    scores = []
    for label, row in zip(y_sub, top3_idx):
        score = 0.0
        for r, idx in enumerate(row):
            if classes[idx] == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    cv = ModelCrossValidator(taxonomy=tax, use_canonical_splits=True)

    print("=" * 80)
    print("CLEAN ARCFACE 2328-D MULTIMODAL FUSION AUDIT")
    print("=" * 80)

    # 1. Load components
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_th = np.load(ART_DIR / "honest_oof_text.npy")
    p_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    p_arc_clean = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    # In-fold domain mask for MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], classes, tax=tax, damping=0.85
        )

    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_th_sharp = apply_t(p_th, 0.6)
    p_ts_sharp = apply_t(p_ts, 0.6)
    p_arc_sharp = apply_t(p_arc_clean, 0.8)

    comp_keys = [
        "Prior",
        "Honest_Text_Sharp",
        "Semantic_Text_Sharp",
        "ConvNeXt",
        "Swin",
        "MobileNet",
        "TrOCR",
        "Clean_ArcFace_2328"
    ]
    matrices = [
        p_prior,
        p_th_sharp,
        p_ts_sharp,
        p_cn,
        p_swin,
        p_mn,
        p_trocr,
        p_arc_sharp
    ]

    print("\n--- 1. Standalone Performance ---")
    res_arc = cv.evaluate_probabilities(p_arc_clean, train_df, model_name="Clean_ArcFace_2328", classes=classes)
    print(f"Clean ArcFace 2328-d Standalone OOF MAP@3: {res_arc.overall_oof_map3:.5f}")

    # 2. Nested Cross-Validation for Optimal Weights
    print("\n--- 2. Nested 5-Fold Convex Optimization ---")
    nested_val_scores = []
    nested_weights = []

    for fold_idx, f in enumerate(splits, start=1):
        tr = f["train_indices"]
        va = f["val_indices"]
        y_tr = [y_true[i] for i in tr]
        y_va = [y_true[i] for i in va]

        tr_mat = [m[tr] for m in matrices]
        va_mat = [m[va] for m in matrices]

        def loss_fn(weights):
            w = np.exp(weights) / np.sum(np.exp(weights))
            blended = np.zeros_like(tr_mat[0])
            for i in range(len(w)):
                blended += w[i] * tr_mat[i]
            return -score_subset(y_tr, blended, classes)

        init_w = np.zeros(len(comp_keys))
        res = minimize(loss_fn, init_w, method="Powell", options={"maxiter": 250})
        opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
        nested_weights.append(opt_w)

        va_blended = np.zeros_like(va_mat[0])
        for i in range(len(opt_w)):
            va_blended += opt_w[i] * va_mat[i]
        va_score = score_subset(y_va, va_blended, classes)
        nested_val_scores.append(va_score)
        print(f"  Fold {fold_idx} Nested CV MAP@3: {va_score:.5f}")

    mean_nested = float(np.mean(nested_val_scores))
    std_nested = float(np.std(nested_val_scores, ddof=1))
    print(f"\nMean Nested CV MAP@3: {mean_nested:.5f} +/- {std_nested:.5f}")

    avg_w = np.mean(nested_weights, axis=0)
    print("\n--- 3. Final Average Weights across 5 Folds ---")
    for k, wt in zip(comp_keys, avg_w):
        print(f"  {k:20s}: {wt*100:5.2f}%")

    final_oof = np.zeros_like(matrices[0])
    for i in range(len(avg_w)):
        final_oof += avg_w[i] * matrices[i]

    final_res = cv.evaluate_probabilities(final_oof, train_df, model_name="Clean_Multimodal_8Comp_Blend", classes=classes)
    print(final_res.diagnostic_report())


if __name__ == "__main__":
    main()

