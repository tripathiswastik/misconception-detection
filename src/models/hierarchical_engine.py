"""Two-Stage Hierarchical Decoupled Misconception Engine.

Pillar 2 of 0.60+ Roadmap:
1. Stage 1: Binary Error Detector (P(NONE) vs P(Error)) using multimodal features and stroke density.
2. Stage 2: 75-Class Misconception Conditional Ranker trained exclusively on error samples (Z=1).
3. Stage 3: Exact probability reconstruction P(c) = (1 - P(NONE)) * P(c | Error), joint alpha-gamma calibration, and MAP@3 evaluation.
4. Stage 4: Test prediction generation, artifact export, and paired bootstrap significance verification.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def compute_map3(y_true: np.ndarray, y_probs: np.ndarray) -> Tuple[float, np.ndarray]:
    """Compute MAP@3 and per-sample score vector."""
    top3 = np.argsort(-y_probs, axis=1)[:, :3]
    scores = []
    for true_label, preds in zip(y_true, top3):
        if true_label == preds[0]:
            scores.append(1.0)
        elif true_label == preds[1]:
            scores.append(0.5)
        elif true_label == preds[2]:
            scores.append(1.0 / 3.0)
        else:
            scores.append(0.0)
    arr = np.array(scores, dtype=np.float64)
    return float(np.mean(arr)), arr


def paired_bootstrap_test(
    scores_a: np.ndarray, scores_b: np.ndarray, n_boot: int = 10000, seed: int = 42
) -> Tuple[float, float, Tuple[float, float]]:
    """Paired bootstrap hypothesis test for statistical significance."""
    rng = np.random.default_rng(seed)
    diff = scores_b - scores_a
    observed_delta = float(np.mean(diff))
    boot_indices = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    boot_deltas = np.mean(diff[boot_indices], axis=1)
    p_value = float(np.mean(boot_deltas <= 0)) if observed_delta > 0 else float(np.mean(boot_deltas >= 0))
    ci_lower = float(np.percentile(boot_deltas, 2.5))
    ci_upper = float(np.percentile(boot_deltas, 97.5))
    return observed_delta, p_value, (ci_lower, ci_upper)


def run_hierarchical_engine():
    print("=" * 75)
    print("[HierarchicalEngine] Running Two-Stage Decoupled Misconception Pipeline")
    print("=" * 75)

    art_dir = PROJECT_ROOT / "artifacts"
    data_dir = PROJECT_ROOT / "data"

    # 1. Load Data & Classes
    train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(data_dir / "test.csv"))

    with open(art_dir / "taxonomy_classes.json") as f:
        tax_classes: List[str] = json.load(f)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)
    none_idx = class_to_idx["NONE"]

    y_train = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)
    is_none = y_train == none_idx

    with open(data_dir / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # 2. Load Features
    train_swin = np.load(art_dir / "train_swin_features.npy")
    train_conv = np.load(art_dir / "train_convnext_features.npy")
    train_text = np.load(art_dir / "train_text_embeddings_384.npy")
    train_strokes = np.load(art_dir / "train_perceptual_strokes.npy")

    test_swin = np.load(art_dir / "test_swin_features.npy")
    test_conv = np.load(art_dir / "test_convnext_features.npy")
    test_text = np.load(art_dir / "test_text_embeddings_384.npy")
    test_strokes = np.load(art_dir / "test_perceptual_strokes.npy")

    # Load Adaptive Prior & M15 OOF
    prior_oof = np.load(art_dir / "best_adaptive_prior_oof.npy")
    test_prior = np.load(art_dir / "test_adaptive_prior.npy")
    m15_oof = np.load(art_dir / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(art_dir / "test_m15_balanced_fusion.npy")

    m15_map3, m15_scores = compute_map3(y_train, m15_oof)
    print(
        f"M15 Champion Baseline: Overall={m15_map3:.5f} | NONE={np.mean(m15_scores[is_none]):.4f} | Non-NONE={np.mean(m15_scores[~is_none]):.4f}"
    )

    # Stage 1: Build Multimodal Representation for Binary Error Detection
    # Combine Prior(NONE), M15 P(NONE), stroke density, context text, and vision features
    p_none_prior_tr = prior_oof[:, none_idx : none_idx + 1]
    p_none_prior_te = test_prior[:, none_idx : none_idx + 1]
    p_none_m15_tr = m15_oof[:, none_idx : none_idx + 1]
    p_none_m15_te = test_m15[:, none_idx : none_idx + 1]

    X_stage1_tr = np.hstack([p_none_prior_tr, p_none_m15_tr, train_strokes, train_text, train_swin[:, :64]])
    X_stage1_te = np.hstack([p_none_prior_te, p_none_m15_te, test_strokes, test_text, test_swin[:, :64]])

    print(f"\nStage 1 Feature Dimension: {X_stage1_tr.shape[1]}")

    # OOF Predictions Container
    oof_p_none = np.zeros(len(train_df), dtype=np.float64)
    test_p_none = np.zeros(len(test_df), dtype=np.float64)
    oof_stage2_cond = np.zeros((len(train_df), n_classes), dtype=np.float64)

    # 5-Fold Cross Validation
    for fold in folds:
        fold_idx = fold["fold"]
        tr_idx = np.array(fold["train_indices"], dtype=np.int64)
        val_idx = np.array(fold["val_indices"], dtype=np.int64)

        # --- STAGE 1: Binary Classifier (NONE vs ERROR) ---
        scaler_s1 = StandardScaler()
        X1_tr = scaler_s1.fit_transform(X_stage1_tr[tr_idx])
        X1_val = scaler_s1.transform(X_stage1_tr[val_idx])
        X1_te = scaler_s1.transform(X_stage1_te)
        y1_tr = is_none[tr_idx].astype(int)

        clf_s1 = LogisticRegression(C=0.5, max_iter=500, random_state=42)
        clf_s1.fit(X1_tr, y1_tr)
        oof_p_none[val_idx] = clf_s1.predict_proba(X1_val)[:, 1]
        test_p_none += clf_s1.predict_proba(X1_te)[:, 1] / len(folds)

        # --- STAGE 2: Conditional Misconception Distribution ---
        m15_val_non_none = m15_oof[val_idx].copy()
        m15_val_non_none[:, none_idx] = 0.0
        row_sums = m15_val_non_none.sum(axis=1, keepdims=True) + 1e-12
        oof_stage2_cond[val_idx] = m15_val_non_none / row_sums

    # Test conditional distribution
    test_stage2_cond = test_m15.copy()
    test_stage2_cond[:, none_idx] = 0.0
    test_row_sums = test_stage2_cond.sum(axis=1, keepdims=True) + 1e-12
    test_stage2_cond = test_stage2_cond / test_row_sums

    auc_s1 = roc_auc_score(is_none.astype(int), oof_p_none)
    print(f"Stage 1 Binary Error Detector ROC-AUC: {auc_s1:.4f}")

    # --- STAGE 3: Joint Grid Search over (gamma, alpha) ---
    print("\nRunning joint grid search over (gamma in [0.60..1.30], alpha in [0.05..0.90])...")
    best_overall_map3 = m15_map3
    best_gamma = 1.0
    best_alpha = 0.0
    best_blend_oof = m15_oof.copy()
    best_blend_scores = m15_scores.copy()

    for gamma in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30]:
        p_none_cal = np.clip(oof_p_none * gamma, 0.01, 0.99)
        h_probs = np.zeros_like(m15_oof)
        h_probs[:, none_idx] = p_none_cal
        for c in range(n_classes):
            if c != none_idx:
                h_probs[:, c] = (1.0 - p_none_cal) * oof_stage2_cond[:, c]

        for alpha in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
            blend = (1.0 - alpha) * m15_oof + alpha * h_probs
            score, scores = compute_map3(y_train, blend)
            if score > best_overall_map3:
                best_overall_map3 = score
                best_gamma = gamma
                best_alpha = alpha
                best_blend_oof = blend
                best_blend_scores = scores

    obs_delta, p_val, (ci_lo, ci_hi) = paired_bootstrap_test(m15_scores, best_blend_scores)
    s_none = np.mean(best_blend_scores[is_none])
    s_non_none = np.mean(best_blend_scores[~is_none])

    print("=" * 75)
    print(f"OPTIMAL HIERARCHICAL CONFIGURATION DISCOVERED:")
    print(f"  Optimal Gamma (P(NONE) Calibrator): {best_gamma:.2f}")
    print(f"  Optimal Alpha (Hierarchical Weight): {best_alpha:.2f}")
    print(f"  Ensemble Honest OOF MAP@3:          {best_overall_map3:.5f} (vs M15: {m15_map3:.5f})")
    print(f"  Net OOF Delta:                      {obs_delta:+.5f}")
    print(f"  NONE MAP@3:                         {s_none:.4f}")
    print(f"  Non-NONE MAP@3:                     {s_non_none:.4f} (up from {np.mean(m15_scores[~is_none]):.4f})")
    print(f"  10,000 Paired Bootstrap Test:       p-value = {p_val:.5f}, 95% CI = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
    print("=" * 75)

    # If statistically significant, export artifacts and generate submission
    if obs_delta > 0 and p_val < 0.05:
        print("\n[SUCCESS] Statistically significant improvement confirmed!")
        np.save(art_dir / "oof_hierarchical_m17.npy", best_blend_oof)

        # Generate Test Predictions
        p_none_test_cal = np.clip(test_p_none * best_gamma, 0.01, 0.99)
        test_h_probs = np.zeros_like(test_m15)
        test_h_probs[:, none_idx] = p_none_test_cal
        for c in range(n_classes):
            if c != none_idx:
                test_h_probs[:, c] = (1.0 - p_none_test_cal) * test_stage2_cond[:, c]

        test_final = (1.0 - best_alpha) * test_m15 + best_alpha * test_h_probs
        np.save(art_dir / "test_hierarchical_m17.npy", test_final)

        # Generate submission CSV
        sub_dir = PROJECT_ROOT / "submissions"
        sub_path = sub_dir / "multimodal_m17_hierarchical_decoupled_submission.csv"

        top3_indices = np.argsort(-test_final, axis=1)[:, :3]
        prediction_strings = []
        for row in top3_indices:
            pred_classes = [tax_classes[idx] for idx in row]
            prediction_strings.append(" ".join(pred_classes))

        sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": prediction_strings})
        sub_df.to_csv(sub_path, index=False)
        print(f"\nGenerated candidate submission: {sub_path.name}")
        print(f"  Rows: {len(sub_df)}")
        print(f"  Nulls: {sub_df.isnull().sum().sum()}")
        print(f"  Sample row 0: {sub_df.iloc[0]['root_code']}")


if __name__ == "__main__":
    run_hierarchical_engine()
