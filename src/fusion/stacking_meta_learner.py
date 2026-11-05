"""In-Fold Stacking Meta-Learner & Paired Bootstrap Significance Testing.

Compares candidate stacking meta-learners against M10 Dual-Vision SOTA:
1. Baseline: M10 Tuned Prior (70%) + Dual-Vision (20%) + Context Text (10%)
2. Candidate A: Class-conditioned Ridge Meta-Learner (regularized multi-output)
3. Candidate B: Multi-class L2-regularized Logistic Regression Stacker
4. Candidate C: Segmented Modality Weight Optimizer (Seen vs. Unseen Question Blends)

Uses paired bootstrap significance testing (10,000 resamples on per-row deltas)
to decide whether any candidate statistically beats M10 (p < 0.05).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.diagnoser.question_prior import QuestionPriorEngine
from src.metrics.map3 import calculate_map3
from src.metrics.bootstrap import paired_bootstrap_test, print_bootstrap_report


def score_probs(y_true: List[str], probs: np.ndarray, classes: List[str]) -> float:
    top_indices = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    top_preds = [[classes[idx] for idx in row] for row in top_indices]
    return calculate_map3(y_true, top_preds)


def run_stacking_study() -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Stacking Study] In-Fold Meta-Learning & Paired Bootstrap")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_train = len(train_df)
    n_test = len(test_df)
    y_true_str = [str(c) for c in train_df["root_code"]]
    y_true_idx = np.array([class_to_idx[c] for c in y_true_str], dtype=np.int32)

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 1. Load / Compute In-Fold Base Modalities
    print("[1/5] Loading and computing base modality matrices...")
    oof_vis_cn = np.load(PROJECT_ROOT / "artifacts" / "oof_convnext_vision.npy")
    test_vis_cn = np.load(PROJECT_ROOT / "artifacts" / "test_convnext_vision.npy")

    oof_vis_mn_raw = np.load(PROJECT_ROOT / "artifacts" / "oof_finetuned_vision.npy")
    test_vis_mn_raw = np.load(PROJECT_ROOT / "artifacts" / "test_finetuned_vision.npy")

    oof_text = np.load(PROJECT_ROOT / "artifacts" / "honest_oof_text.npy")
    test_text = np.load(PROJECT_ROOT / "artifacts" / "test_text.npy")

    # In-fold domain masking for MobileNetV3
    oof_vis_mn = np.zeros_like(oof_vis_mn_raw)
    for tr_idx, va_idx in folds:
        fold_tr = train_df.iloc[tr_idx]
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(fold_tr)
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(float(g)) for g in train_df.iloc[va_idx]["grade"]]
        oof_vis_mn[va_idx] = masker.apply_mask_batch(oof_vis_mn_raw[va_idx], classes, t_va, g_va, damping=0.85)

    # Test domain masking for MobileNetV3
    full_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    test_topics = [str(t) for t in test_df["topic_name"]]
    test_grades = [int(float(g)) for g in test_df["grade"]]
    test_vis_mn = full_masker.apply_mask_batch(test_vis_mn_raw, classes, test_topics, test_grades, damping=0.85)

    # In-fold Question Prior (m_q = 10.0)
    oof_prior = np.zeros((n_train, n_classes), dtype=np.float32)
    test_prior = np.zeros((n_test, n_classes), dtype=np.float32)

    for tr_idx, va_idx in folds:
        fold_tr = train_df.iloc[tr_idx]
        fold_va = train_df.iloc[va_idx]
        prior_engine = QuestionPriorEngine(classes=classes, smoothing_question=10.0, smoothing_topic_grade=5.0)
        prior_engine.fit(fold_tr)
        p_va, _ = prior_engine.predict_proba(fold_va)
        oof_prior[va_idx] = p_va

        p_test_fold, _ = prior_engine.predict_proba(test_df)
        test_prior += p_test_fold / len(folds)

    # Dual-Vision combination
    oof_dual_vis = 0.5 * oof_vis_cn + 0.5 * oof_vis_mn
    test_dual_vis = 0.5 * test_vis_cn + 0.5 * test_vis_mn

    # 2. Evaluate Baseline M10
    oof_m10 = 0.70 * oof_prior + 0.20 * oof_dual_vis + 0.10 * oof_text
    map3_m10 = score_probs(y_true_str, oof_m10, classes)
    print(f"\n[Baseline M10] Honest 5-Fold OOF MAP@3: {map3_m10:.4f}")

    prior_alone_map3 = score_probs(y_true_str, oof_prior, classes)
    print(f"[Baseline Prior Alone] (m_q=10.0) Honest OOF MAP@3: {prior_alone_map3:.4f}")

    # 3. Candidate A: Multi-output Ridge Meta-Learner
    print("\n[2/5] Training In-Fold Ridge Meta-Learner...")
    # Features: Concatenate probabilities [Prior, DualVis, Text] (76 * 3 = 228 features)
    oof_features = np.hstack([oof_prior, oof_dual_vis, oof_text])
    test_features = np.hstack([test_prior, test_dual_vis, test_text])

    oof_ridge_probs = np.zeros((n_train, n_classes), dtype=np.float32)
    test_ridge_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    # One-hot target matrix for multi-output Ridge
    Y_train_onehot = np.zeros((n_train, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_true_idx):
        Y_train_onehot[i, idx] = 1.0

    best_alpha = 100.0  # Strong regularization to prevent overfitting on 1500 samples
    for tr_idx, va_idx in folds:
        X_tr = oof_features[tr_idx]
        Y_tr = Y_train_onehot[tr_idx]
        X_va = oof_features[va_idx]

        # Analytic Ridge Solution: W = (X^T X + alpha * I)^-1 X^T Y
        d = X_tr.shape[1]
        I_reg = np.eye(d, dtype=np.float32)
        W = np.linalg.solve(X_tr.T @ X_tr + best_alpha * I_reg, X_tr.T @ Y_tr)

        va_preds = X_va @ W
        # Clip negative predictions and renormalize
        va_preds = np.clip(va_preds, 0.0, None)
        row_sums = va_preds.sum(axis=1, keepdims=True)
        va_preds = np.divide(va_preds, row_sums, where=row_sums > 0)
        oof_ridge_probs[va_idx] = va_preds

        te_preds = test_features @ W
        te_preds = np.clip(te_preds, 0.0, None)
        te_sums = te_preds.sum(axis=1, keepdims=True)
        te_preds = np.divide(te_preds, te_sums, where=te_sums > 0)
        test_ridge_probs += te_preds / len(folds)

    map3_ridge = score_probs(y_true_str, oof_ridge_probs, classes)
    print(f"  --> Ridge Meta-Learner OOF MAP@3: {map3_ridge:.4f} (alpha={best_alpha})")

    # 4. Candidate B: In-Fold Logistic Regression Stacker
    print("\n[3/5] Training In-Fold Logistic Regression Meta-Learner...")
    oof_lr_probs = np.zeros((n_train, n_classes), dtype=np.float32)
    test_lr_probs = np.zeros((n_test, n_classes), dtype=np.float32)

    for tr_idx, va_idx in folds:
        X_tr = oof_features[tr_idx]
        y_tr = y_true_idx[tr_idx]
        X_va = oof_features[va_idx]

        lr = LogisticRegression(C=0.05, max_iter=200, solver="lbfgs", random_state=42)
        lr.fit(X_tr, y_tr)

        va_lr = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        va_lr_raw = lr.predict_proba(X_va)
        for i, c_idx in enumerate(lr.classes_):
            va_lr[:, c_idx] = va_lr_raw[:, i]
        oof_lr_probs[va_idx] = va_lr

        te_lr_fold = np.zeros((n_test, n_classes), dtype=np.float32)
        te_lr_raw = lr.predict_proba(test_features)
        for i, c_idx in enumerate(lr.classes_):
            te_lr_fold[:, c_idx] = te_lr_raw[:, i]
        test_lr_probs += te_lr_fold / len(folds)

    map3_lr = score_probs(y_true_str, oof_lr_probs, classes)
    print(f"  --> Logistic Regression Stacker OOF MAP@3: {map3_lr:.4f} (C=0.05)")

    # 5. Candidate C: Segmented Modality Weight Optimizer (Seen vs. Unseen Question Blends)
    print("\n[4/5] Tuning Segmented Modality Weights (Seen vs. Unseen Questions)...")
    # Identify seen vs unseen per validation fold
    oof_segmented = np.zeros((n_train, n_classes), dtype=np.float32)

    # For seen questions: prior is strong (e.g. 75% prior, 15% vision, 10% text)
    # For unseen questions: prior is coarse topic-grade (e.g. 40% prior, 45% vision, 15% text)
    is_seen_oof = np.zeros(n_train, dtype=bool)
    for tr_idx, va_idx in folds:
        tr_q = set(train_df.iloc[tr_idx]["question_id"])
        va_q = train_df.iloc[va_idx]["question_id"].values
        for local_i, q in enumerate(va_q):
            if q in tr_q:
                is_seen_oof[va_idx[local_i]] = True

    # Sweep grid for unseen questions
    best_unseen_w = (0.70, 0.20, 0.10)
    best_seg_score = -1.0
    for w_p in [0.30, 0.40, 0.50, 0.60, 0.70]:
        for w_v in [0.20, 0.30, 0.40, 0.50]:
            w_t = 1.0 - w_p - w_v
            if w_t < 0:
                continue
            cand_probs = np.zeros_like(oof_m10)
            # Seen rows get M10 weights (0.70, 0.20, 0.10)
            cand_probs[is_seen_oof] = oof_m10[is_seen_oof]
            # Unseen rows get test weights
            cand_probs[~is_seen_oof] = w_p * oof_prior[~is_seen_oof] + w_v * oof_dual_vis[~is_seen_oof] + w_t * oof_text[~is_seen_oof]
            score = score_probs(y_true_str, cand_probs, classes)
            if score > best_seg_score:
                best_seg_score = score
                best_unseen_w = (w_p, w_v, w_t)

    print(f"  --> Best Segmented Weights for Unseen: Prior={best_unseen_w[0]:.2f}, Vision={best_unseen_w[1]:.2f}, Text={best_unseen_w[2]:.2f}")
    oof_segmented[is_seen_oof] = oof_m10[is_seen_oof]
    oof_segmented[~is_seen_oof] = (
        best_unseen_w[0] * oof_prior[~is_seen_oof]
        + best_unseen_w[1] * oof_dual_vis[~is_seen_oof]
        + best_unseen_w[2] * oof_text[~is_seen_oof]
    )
    map3_segmented = score_probs(y_true_str, oof_segmented, classes)
    print(f"  --> Segmented Fusion OOF MAP@3: {map3_segmented:.4f}")

    # 6. Paired Bootstrap Significance Testing vs M10 SOTA
    print("\n=======================================================")
    print("[5/5] Paired Bootstrap Significance Testing (vs M10 SOTA)")
    print("=======================================================")

    eval_models = [
        ("Candidate A (Ridge Stacker)", oof_ridge_probs),
        ("Candidate B (Logistic Stacker)", oof_lr_probs),
        ("Candidate C (Segmented Fusion)", oof_segmented),
        ("Reference (Prior Alone mq=10)", oof_prior)
    ]

    bootstrap_results = {}
    for name, cand_probs in eval_models:
        boot = paired_bootstrap_test(
            y_true=y_true_str,
            probs_a=cand_probs,
            probs_b=oof_m10,
            classes=classes,
            n_resamples=10000,
            random_state=42
        )
        bootstrap_results[name] = boot
        print_bootstrap_report(boot, name, "M10 Dual-Vision")

    # Save complete study report
    report = {
        "m10_oof_map3": map3_m10,
        "prior_alone_oof_map3": prior_alone_map3,
        "ridge_oof_map3": map3_ridge,
        "lr_oof_map3": map3_lr,
        "segmented_oof_map3": map3_segmented,
        "best_unseen_weights": {
            "prior": best_unseen_w[0],
            "vision": best_unseen_w[1],
            "text": best_unseen_w[2]
        },
        "bootstrap_comparisons": {
            name: {
                "mean_a": b["mean_a"],
                "mean_b": b["mean_b"],
                "mean_delta": b["mean_delta"],
                "se_delta": b["se_delta"],
                "ci_95": [b["ci_lower"], b["ci_upper"]],
                "p_value": b["p_value"],
                "is_significant": b["is_significant"],
                "wins": b["wins"],
                "losses": b["losses"],
                "ties": b["ties"]
            }
            for name, b in bootstrap_results.items()
        }
    }

    report_path = PROJECT_ROOT / "artifacts" / "stacking_meta_learner_study.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Stacking Study] Saved full report to: {report_path}")

    return report


if __name__ == "__main__":
    run_stacking_study()
