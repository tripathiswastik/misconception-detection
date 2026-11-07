"""Evaluate All Candidate Components & Representations under Leak-Free CV.

Benchmarks:
1. TrOCR Probe (oof_trocr_vision.npy)
2. Perceptual Strokes probe / representation
3. Semantic Text probe
4. Visual Foundation models
5. Full multimodal nested convex optimization combining all clean modalities!
"""

import json
import sys
from pathlib import Path
from typing import cast
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


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    cv = ModelCrossValidator(taxonomy=tax, use_canonical_splits=True)

    print("=" * 80)
    print("CANDIDATE REPRESENTATIONS & MULTIMODAL FUSION AUDIT")
    print("=" * 80)

    # Load components
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    # In-fold domain mask for MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], classes, tax=tax, damping=0.85
        )

    # Apply Stage A text temperature calibration (T=0.6)
    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_text_honest_sharp = apply_t(p_text_honest, 0.6)
    p_text_sem_sharp = apply_t(p_text_sem, 0.6)
    p_trocr_sharp = apply_t(p_trocr, 0.8)

    comp_dict = {
        "Prior": p_prior,
        "Honest_Text": p_text_honest,
        "Honest_Text_Sharp": p_text_honest_sharp,
        "Semantic_Text": p_text_sem,
        "Semantic_Text_Sharp": p_text_sem_sharp,
        "ConvNeXt": p_cn,
        "Swin": p_swin,
        "MobileNet": p_mn,
        "TrOCR_Vision": p_trocr,
        "TrOCR_Vision_Sharp": p_trocr_sharp,
    }

    print("\n--- 1. Standalone Candidate Performance ---")
    for name, probs in comp_dict.items():
        res = cv.evaluate_probabilities(probs, train_df, model_name=name, classes=classes)
        err_score = res.diagnostics.error_map3 if res.diagnostics is not None else 0.0
        print(f"  {name:20s} | MAP@3: {res.overall_oof_map3:.5f} | Error MAP@3: {err_score:.5f}")

    # 2. Multimodal Fusion with TrOCR
    print("\n--- 2. Nested 5-Fold Convex Optimization: 7-Component Clean Ensemble ---")
    # Components to fuse
    active_keys = [
        "Prior",
        "Honest_Text_Sharp",
        "Semantic_Text_Sharp",
        "ConvNeXt",
        "Swin",
        "MobileNet",
        "TrOCR_Vision_Sharp"
    ]
    matrices = [comp_dict[k] for k in active_keys]

    nested_val_scores = []
    nested_weights = []

    def score_subset(y_sub, p_sub):
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
            return -score_subset(y_tr, blended)

        init_w = np.zeros(len(active_keys))
        res = minimize(loss_fn, init_w, method="Powell", options={"maxiter": 250})
        opt_w = np.exp(res.x) / np.sum(np.exp(res.x))
        nested_weights.append(opt_w)

        va_blended = np.zeros_like(va_mat[0])
        for i in range(len(opt_w)):
            va_blended += opt_w[i] * va_mat[i]
        va_score = score_subset(y_va, va_blended)
        nested_val_scores.append(va_score)
        print(f"  Fold {fold_idx} Nested CV MAP@3: {va_score:.5f}")
        for k, wt in zip(active_keys, opt_w):
            print(f"    {k:20s}: {wt:.4f}")

    mean_nested = float(np.mean(nested_val_scores))
    std_nested = float(np.std(nested_val_scores, ddof=1))
    print(f"\nMean Nested CV MAP@3: {mean_nested:.5f} +/- {std_nested:.5f}")

    avg_w = np.mean(nested_weights, axis=0)
    print("\n--- 3. Final Average Weights across 5 Folds ---")
    for k, wt in zip(active_keys, avg_w):
        print(f"  {k:20s}: {wt*100:5.2f}%")

    final_oof = np.zeros_like(matrices[0])
    for i in range(len(avg_w)):
        final_oof += avg_w[i] * matrices[i]
    final_res = cv.evaluate_probabilities(final_oof, train_df, model_name="7-Component_TrOCR_Clean_Blend", classes=classes)
    print(final_res.diagnostic_report())


if __name__ == "__main__":
    main()

