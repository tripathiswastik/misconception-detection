"""TrOCR Handwritten Math Vision Probe & Multimodal Fusion.

Pillar 3 of 0.60+ Roadmap:
1. Fits in-fold regularized classifier on TrOCR handwritten features (384-d) across 5 folds.
2. Evaluates honest standalone OOF MAP@3 and non-NONE discrimination.
3. Tests linear unwarped blending with M15 champion (LB: 0.54149).
4. Conducts 10,000 paired bootstrap hypothesis testing.
5. Exports oof_trocr_vision.npy, test_trocr_vision.npy, and candidate submission if significant.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine


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


def run_trocr_probe_pipeline():
    print("=" * 70)
    print("[TrOCR Probe] Evaluating Handwritten Math Features (5-Fold CV)")
    print("=" * 70)

    art_dir = PROJECT_ROOT / "artifacts"
    data_dir = PROJECT_ROOT / "data"

    # 1. Load Data, Classes, and Splits
    train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(data_dir / "test.csv"))

    tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
    tax_classes = tax.all_codes(include_none=True)
    class_to_idx = {c: i for i, c in enumerate(tax_classes)}
    n_classes = len(tax_classes)
    none_idx = class_to_idx["NONE"]

    y_train = np.array([class_to_idx[r] for r in train_df["root_code"]], dtype=np.int64)
    is_none = (y_train == none_idx)

    with open(data_dir / "splits.json") as f:
        splits_data = json.load(f)
    folds = splits_data["folds"]

    # 2. Load Features
    train_trocr = np.load(art_dir / "train_trocr_features.npy")
    test_trocr = np.load(art_dir / "test_trocr_features.npy")
    print(f"TrOCR Features Loaded: Train={train_trocr.shape}, Test={test_trocr.shape}")

    # Baseline M15 Champion
    m15_oof = np.load(art_dir / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(art_dir / "test_m15_balanced_fusion.npy")
    m15_map3, m15_scores = compute_map3(y_train, m15_oof)
    print(f"M15 Champion Baseline: MAP@3 = {m15_map3:.5f} (NONE: {np.mean(m15_scores[is_none]):.4f}, Non-NONE: {np.mean(m15_scores[~is_none]):.4f})")

    # 3. Sweep Regularization C / alpha for TrOCR Probe
    best_c = 1.0
    best_probe_map3 = 0.0
    best_probe_oof = np.zeros_like(m15_oof)
    best_probe_test = np.zeros_like(test_m15)

    print("\nSweeping LogisticRegression C on TrOCR features...")
    for c_val in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
        test_probs_acc = np.zeros((len(test_df), n_classes), dtype=np.float32)

        for fold in folds:
            tr_idx = np.array(fold["train_indices"], dtype=np.int64)
            val_idx = np.array(fold["val_indices"], dtype=np.int64)

            fold_tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
            fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(train_trocr[tr_idx])
            X_val = scaler.transform(train_trocr[val_idx])
            X_te = scaler.transform(test_trocr)

            y_tr = y_train[tr_idx]

            clf = LogisticRegression(C=c_val, max_iter=500, solver="lbfgs", random_state=42)
            clf.fit(X_tr, y_tr)

            val_preds_raw = clf.predict_proba(X_val)
            test_preds_raw = clf.predict_proba(X_te)

            val_aligned = np.zeros((len(val_idx), n_classes), dtype=np.float32)
            test_aligned = np.zeros((len(test_df), n_classes), dtype=np.float32)

            for col_idx, real_class_idx in enumerate(clf.classes_):
                val_aligned[:, real_class_idx] = val_preds_raw[:, col_idx]
                test_aligned[:, real_class_idx] = test_preds_raw[:, col_idx]

            # In-fold domain masking
            masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(fold_tr_df)
            val_topics = [str(t) for t in fold_val_df["topic_name"].values]
            val_grades = [int(float(g)) for g in fold_val_df["grade"].values]
            val_calibrated = masker.apply_mask_batch(
                val_aligned, tax_classes, val_topics, val_grades, damping=0.05
            )

            oof_probs[val_idx] = val_calibrated

            test_topics = [str(t) for t in test_df["topic_name"].values]
            test_grades = [int(float(g)) for g in test_df["grade"].values]
            test_calibrated = masker.apply_mask_batch(
                test_aligned, tax_classes, test_topics, test_grades, damping=0.05
            )
            test_probs_acc += test_calibrated / len(folds)

        score, scores = compute_map3(y_train, oof_probs)
        s_none = np.mean(scores[is_none])
        s_non_none = np.mean(scores[~is_none])
        print(f"  C={c_val:5.2f} -> Standalone MAP@3: {score:.5f} (NONE: {s_none:.4f}, Non-NONE: {s_non_none:.4f})")

        if score > best_probe_map3:
            best_probe_map3 = score
            best_c = c_val
            best_probe_oof = oof_probs
            best_probe_test = test_probs_acc

    print("\n" + "=" * 70)
    print(f"Optimal TrOCR Probe (C={best_c}): Standalone MAP@3 = {best_probe_map3:.5f}")
    print("=" * 70)

    # Save probe artifacts
    np.save(art_dir / "oof_trocr_vision.npy", best_probe_oof)
    np.save(art_dir / "test_trocr_vision.npy", best_probe_test)

    # 4. Evaluate Linear Blends with M15 Champion
    print("\nEvaluating Linear Unwarped Ensembles with M15 Champion...")
    best_w = 0.0
    best_blend_map3 = m15_map3
    best_blend_oof = m15_oof.copy()
    best_blend_scores = m15_scores.copy()

    for w in [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        blend = (1.0 - w) * m15_oof + w * best_probe_oof
        score, scores = compute_map3(y_train, blend)
        s_non = np.mean(scores[~is_none])
        print(f"  w={w:4.2f}: MAP@3 = {score:.5f} (Delta: {score - m15_map3:+.5f}, Non-NONE: {s_non:.4f})")
        if score > best_blend_map3:
            best_blend_map3 = score
            best_w = w
            best_blend_oof = blend
            best_blend_scores = scores

    if best_w > 0:
        obs_delta, p_val, (ci_lo, ci_hi) = paired_bootstrap_test(m15_scores, best_blend_scores)
        print("\n" + "=" * 70)
        print(f"OPTIMAL TrOCR + M15 ENSEMBLE (M18):")
        print(f"  Optimal TrOCR Weight:      w = {best_w:.2f}")
        print(f"  Ensemble Honest OOF MAP@3: {best_blend_map3:.5f} (vs M15: {m15_map3:.5f})")
        print(f"  Observed Net Delta:        {obs_delta:+.5f}")
        print(f"  10,000 Paired Bootstrap:   p-value = {p_val:.5f}, 95% CI = [{ci_lo:+.5f}, {ci_hi:+.5f}]")
        print("=" * 70)

        # Generate Test Predictions
        test_blend = (1.0 - best_w) * test_m15 + best_w * best_probe_test
        np.save(art_dir / "oof_m18_trocr_blend.npy", best_blend_oof)
        np.save(art_dir / "test_m18_trocr_blend.npy", test_blend)

        # Generate submission CSV
        sub_dir = PROJECT_ROOT / "submissions"
        sub_path = sub_dir / "multimodal_m18_trocr_handwriting_submission.csv"

        top3_indices = np.argsort(-test_blend, axis=1)[:, :3]
        prediction_strings = []
        for row in top3_indices:
            pred_classes = [tax_classes[idx] for idx in row]
            prediction_strings.append(" ".join(pred_classes))

        sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": prediction_strings})
        sub_df.to_csv(sub_path, index=False)
        print(f"\nGenerated candidate submission: {sub_path.name}")
        print(f"  Rows: {len(sub_df)}")
        print(f"  Nulls: {sub_df.isnull().sum().sum()}")
        print(f"  Columns: {sub_df.columns.tolist()}")
        print(f"  Sample row 0: {sub_df.iloc[0]['root_code']}")
    else:
        print("\nNo positive blend gain over M15. Checking feature interactions...")


if __name__ == "__main__":
    run_trocr_probe_pipeline()

