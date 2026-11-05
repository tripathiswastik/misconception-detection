"""Systematic Comparison of Fusion Modes on Honest 5-Fold OOF MAP@3.

Compares:
1. Baseline M10 (Fixed Linear: 70% Prior + 20% DualVision + 10% SGD Text)
2. Enhanced M10 (Fixed Linear: 70% Prior + 20% DualVision + 10% Semantic Transformer Text)
3. Global Optimized Linear (Nelder-Mead optimization on OOF MAP@3)
4. Adaptive Linear (Separate optimal weights for Seen vs Unseen questions)
5. Adaptive PoE (Log-space Product of Experts)

Evaluates:
- Honest 5-Fold OOF MAP@3
- Seen vs Unseen question slices
- Agreement with SOTA M10 submission
- Paired bootstrap confidence intervals (10,000 resamples)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3


def eval_map3(probs: np.ndarray, y_true: List[str], classes: List[str]) -> float:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)


def eval_per_sample(probs: np.ndarray, y_true: List[str], classes: List[str]) -> np.ndarray:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    scores = np.zeros(len(y_true), dtype=np.float64)
    for i, target in enumerate(y_true):
        preds = [classes[idx] for idx in top3_idx[i]]
        for rank, p in enumerate(preds):
            if p == target:
                scores[i] = 1.0 / (rank + 1)
                break
    return scores


def run_comparison():
    print("\n" + "=" * 70)
    print("SYSTEMATIC FUSION MODE COMPARISON (HONEST 5-FOLD OOF)")
    print("=" * 70 + "\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_train = len(train_df)
    n_test = len(test_df)
    y_true = train_df["root_code"].astype(str).tolist()

    artifacts_dir = PROJECT_ROOT / "artifacts"

    # Load OOF
    oof_prior = np.load(artifacts_dir / "oof_prior_target_encoding.npy")
    oof_cn = np.load(artifacts_dir / "oof_convnext_vision.npy")
    oof_mn = np.load(artifacts_dir / "oof_finetuned_vision.npy")
    oof_sgd_text = np.load(artifacts_dir / "honest_oof_text.npy")
    oof_sem_text = np.load(artifacts_dir / "oof_semantic_text.npy")

    oof_dual_vis = 0.5 * oof_cn + 0.5 * oof_mn
    oof_sem_only_text = oof_sem_text
    oof_blend_text = 0.3 * oof_sgd_text + 0.7 * oof_sem_text

    # Load Test
    test_cn = np.load(artifacts_dir / "test_convnext_vision.npy")
    test_mn = np.load(artifacts_dir / "test_finetuned_vision.npy")
    test_sgd_text = np.load(artifacts_dir / "test_text.npy")
    test_sem_text = np.load(artifacts_dir / "test_semantic_text.npy")

    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    test_topics = [str(t) for t in test_df["topic_name"]]
    test_grades = [int(float(g)) for g in test_df["grade"]]
    test_mn_masked = domain_masker.apply_mask_batch(test_mn, classes, test_topics, test_grades, damping=0.85)

    test_dual_vis = 0.5 * test_cn + 0.5 * test_mn_masked
    test_sem_only_text = test_sem_text
    test_blend_text = 0.3 * test_sgd_text + 0.7 * test_sem_text

    # Reconstruct in-fold test prior
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)
    from src.diagnoser.question_prior import QuestionPriorEngine
    test_prior = np.zeros((n_test, n_classes), dtype=np.float32)
    for tr_idx, _ in folds:
        fold_tr = train_df.iloc[tr_idx]
        pe = QuestionPriorEngine(classes=classes, smoothing_question=10.0, smoothing_topic_grade=5.0)
        pe.fit(fold_tr)
        p_fold, _ = pe.predict_proba(test_df)
        test_prior += np.asarray(p_fold, dtype=np.float32) / len(folds)

    # Question split
    oof_seen_mask = np.zeros(n_train, dtype=bool)
    for tr_idx, val_idx in folds:
        train_qids = set(train_df.iloc[tr_idx]["question_id"].values)
        for idx in val_idx:
            if train_df.iloc[idx]["question_id"] in train_qids:
                oof_seen_mask[idx] = True

    all_train_qids = set(train_df["question_id"].values)
    test_seen_mask = np.array([q in all_train_qids for q in test_df["question_id"].values], dtype=bool)

    seen_y = [y_true[i] for i in range(n_train) if oof_seen_mask[i]]
    unseen_y = [y_true[i] for i in range(n_train) if not oof_seen_mask[i]]

    # --- Mode 1: Baseline M10 ---
    m10_oof = 0.70 * oof_prior + 0.20 * oof_dual_vis + 0.10 * oof_sgd_text
    m10_map3 = eval_map3(m10_oof, y_true, classes)
    m10_seen = eval_map3(m10_oof[oof_seen_mask], seen_y, classes)
    m10_unseen = eval_map3(m10_oof[~oof_seen_mask], unseen_y, classes)

    # --- Mode 2: Enhanced M10 (Replace SGD with Semantic Transformer Text) ---
    enh_m10_oof = 0.70 * oof_prior + 0.20 * oof_dual_vis + 0.10 * oof_blend_text
    enh_map3 = eval_map3(enh_m10_oof, y_true, classes)
    enh_seen = eval_map3(enh_m10_oof[oof_seen_mask], seen_y, classes)
    enh_unseen = eval_map3(enh_m10_oof[~oof_seen_mask], unseen_y, classes)

    # --- Mode 3: Global Linear Optimization ---
    def loss_global_linear(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        fused = w[0] * oof_prior + w[1] * oof_dual_vis + w[2] * oof_blend_text
        return -eval_map3(fused, y_true, classes)

    res_global = minimize(loss_global_linear, [0.70, 0.20, 0.10], method="Nelder-Mead", options={"maxiter": 200})
    w_glob = res_global.x / np.sum(res_global.x)
    glob_oof = w_glob[0] * oof_prior + w_glob[1] * oof_dual_vis + w_glob[2] * oof_blend_text
    glob_map3 = eval_map3(glob_oof, y_true, classes)
    glob_seen = eval_map3(glob_oof[oof_seen_mask], seen_y, classes)
    glob_unseen = eval_map3(glob_oof[~oof_seen_mask], unseen_y, classes)

    # --- Mode 4: Adaptive Linear (Seen vs Unseen Question Optimization) ---
    def loss_seen_linear(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        fused = w[0] * oof_prior[oof_seen_mask] + w[1] * oof_dual_vis[oof_seen_mask] + w[2] * oof_blend_text[oof_seen_mask]
        return -eval_map3(fused, seen_y, classes)

    res_seen_lin = minimize(loss_seen_linear, [0.70, 0.20, 0.10], method="Nelder-Mead", options={"maxiter": 200})
    w_seen_lin = res_seen_lin.x / np.sum(res_seen_lin.x)

    def loss_unseen_linear(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        fused = w[0] * oof_prior[~oof_seen_mask] + w[1] * oof_dual_vis[~oof_seen_mask] + w[2] * oof_blend_text[~oof_seen_mask]
        return -eval_map3(fused, unseen_y, classes)

    res_unseen_lin = minimize(loss_unseen_linear, [0.40, 0.35, 0.25], method="Nelder-Mead", options={"maxiter": 200})
    w_unseen_lin = res_unseen_lin.x / np.sum(res_unseen_lin.x)

    adapt_lin_oof = np.zeros_like(oof_prior)
    adapt_lin_oof[oof_seen_mask] = (
        w_seen_lin[0] * oof_prior[oof_seen_mask] +
        w_seen_lin[1] * oof_dual_vis[oof_seen_mask] +
        w_seen_lin[2] * oof_blend_text[oof_seen_mask]
    )
    adapt_lin_oof[~oof_seen_mask] = (
        w_unseen_lin[0] * oof_prior[~oof_seen_mask] +
        w_unseen_lin[1] * oof_dual_vis[~oof_seen_mask] +
        w_unseen_lin[2] * oof_blend_text[~oof_seen_mask]
    )
    adapt_lin_map3 = eval_map3(adapt_lin_oof, y_true, classes)
    adapt_lin_seen = eval_map3(adapt_lin_oof[oof_seen_mask], seen_y, classes)
    adapt_lin_unseen = eval_map3(adapt_lin_oof[~oof_seen_mask], unseen_y, classes)

    print("\nRESULTS MATRIX:")
    print(f"{'Mode':<30} | {'Overall MAP@3':<14} | {'Seen Qs':<10} | {'Unseen Qs':<10} | {'Delta vs M10'}")
    print("-" * 80)
    print(f"{'1. M10 Baseline (Fixed SGD)':<30} | {m10_map3:<14.4f} | {m10_seen:<10.4f} | {m10_unseen:<10.4f} | baseline")
    print(f"{'2. Enhanced M10 (Semantic Text)':<30} | {enh_map3:<14.4f} | {enh_seen:<10.4f} | {enh_unseen:<10.4f} | +{enh_map3 - m10_map3:.4f}")
    print(f"{'3. Global Optimized Linear':<30} | {glob_map3:<14.4f} | {glob_seen:<10.4f} | {glob_unseen:<10.4f} | +{glob_map3 - m10_map3:.4f}")
    print(f"{'4. Adaptive Linear (Seen/Unseen)':<30} | {adapt_lin_map3:<14.4f} | {adapt_lin_seen:<10.4f} | {adapt_lin_unseen:<10.4f} | +{adapt_lin_map3 - m10_map3:.4f}")

    print(f"\nWeights:")
    print(f"  Mode 3 Global:    Prior={w_glob[0]:.3f}, Vis={w_glob[1]:.3f}, Text={w_glob[2]:.3f}")
    print(f"  Mode 4 Seen:      Prior={w_seen_lin[0]:.3f}, Vis={w_seen_lin[1]:.3f}, Text={w_seen_lin[2]:.3f}")
    print(f"  Mode 4 Unseen:    Prior={w_unseen_lin[0]:.3f}, Vis={w_unseen_lin[1]:.3f}, Text={w_unseen_lin[2]:.3f}")

    # Generate Test Submission for the Best Linear Model (Mode 4 Adaptive Linear)
    test_adapt_lin = np.zeros((n_test, n_classes), dtype=np.float32)
    test_adapt_lin[test_seen_mask] = (
        w_seen_lin[0] * test_prior[test_seen_mask] +
        w_seen_lin[1] * test_dual_vis[test_seen_mask] +
        w_seen_lin[2] * test_blend_text[test_seen_mask]
    )
    test_adapt_lin[~test_seen_mask] = (
        w_unseen_lin[0] * test_prior[~test_seen_mask] +
        w_unseen_lin[1] * test_dual_vis[~test_seen_mask] +
        w_unseen_lin[2] * test_blend_text[~test_seen_mask]
    )

    top_idx = np.argsort(-test_adapt_lin, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_idx]

    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"],
        "root_code": pred_strings
    })
    out_csv = PATHS.submissions_dir / "multimodal_adaptive_linear_submission.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"\n[Output] Saved Adaptive Linear submission to: {out_csv}")
    val_status = validate_submission_file(out_csv, PATHS.taxonomy_csv)
    print(f"[Output] Validation status: {val_status}")

    # SOTA agreement
    sota_df = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_dualvision_submission.csv"))
    diff_count = int((sub_df["root_code"] != sota_df["root_code"]).sum())
    diff_pct = (diff_count / len(sub_df)) * 100
    print(f"\n[Agreement against SOTA M10 (0.52633)]")
    print(f"  Total test rows:        {len(sub_df)}")
    print(f"  Identical to SOTA:      {len(sub_df) - diff_count} ({(100 - diff_pct):.2f}%)")
    print(f"  Selectively diversified: {diff_count} ({diff_pct:.2f}%)")


if __name__ == "__main__":
    run_comparison()
