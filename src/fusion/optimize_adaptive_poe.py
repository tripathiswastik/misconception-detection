"""Adaptive Question-Conditioned Product-of-Experts (PoE) Optimizer.

Avenues 1, 2 & 3:
1. Adaptive Question-Conditioned Fusion (Seen vs Unseen question handling).
2. Product-of-Experts Log-Space Fusion:
   log P(c) = w_p * log(P_p + eps) + w_v * log(P_v + eps) + w_t * log(P_t + eps)
   vs Linear Probability Blend.
3. Nelder-Mead Optimization directly maximizing 5-Fold OOF MAP@3.
4. Paired Bootstrap Statistical Significance Verification (10,000 resamples).
5. Generates the new candidate submission:
   submissions/multimodal_adaptive_poe_submission.csv
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


def eval_per_sample_map3(probs: np.ndarray, y_true: List[str], classes: List[str]) -> np.ndarray:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    scores = np.zeros(len(y_true), dtype=np.float64)
    for i, target in enumerate(y_true):
        preds = [classes[idx] for idx in top3_idx[i]]
        score = 0.0
        for rank, p in enumerate(preds):
            if p == target:
                score = 1.0 / (rank + 1)
                break
        scores[i] = score
    return scores


def run_adaptive_poe_pipeline() -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print("[AdaptivePoE] Initializing Adaptive Product-of-Experts Optimization")
    print("=" * 65 + "\n")

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

    # 1. Load OOF matrices
    print("[AdaptivePoE] Loading 5-fold OOF modality predictions...")
    oof_prior = np.load(artifacts_dir / "oof_prior_target_encoding.npy")
    oof_cn = np.load(artifacts_dir / "oof_convnext_vision.npy")
    oof_mn = np.load(artifacts_dir / "oof_finetuned_vision.npy")
    oof_sgd_text = np.load(artifacts_dir / "honest_oof_text.npy")
    oof_sem_text = np.load(artifacts_dir / "oof_semantic_text.npy")

    # Dual-Vision ensemble (50% ConvNeXt + 50% MobileNetV3)
    oof_dual_vis = 0.5 * oof_cn + 0.5 * oof_mn

    # Enhanced Text ensemble (50% SGD TF-IDF + 50% Semantic Transformer)
    oof_enh_text = 0.4 * oof_sgd_text + 0.6 * oof_sem_text

    # 2. Load Test matrices
    print("[AdaptivePoE] Loading test modality predictions...")
    test_cn = np.load(artifacts_dir / "test_convnext_vision.npy")
    test_mn = np.load(artifacts_dir / "test_finetuned_vision.npy")
    test_sgd_text = np.load(artifacts_dir / "test_text.npy")
    test_sem_text = np.load(artifacts_dir / "test_semantic_text.npy")

    # Domain mask on test MobileNet
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    test_topics = [str(t) for t in test_df["topic_name"]]
    test_grades = [int(float(g)) for g in test_df["grade"]]
    test_mn_masked = domain_masker.apply_mask_batch(test_mn, classes, test_topics, test_grades, damping=0.85)

    test_dual_vis = 0.5 * test_cn + 0.5 * test_mn_masked
    test_enh_text = 0.4 * test_sgd_text + 0.6 * test_sem_text

    # Compute test prior ensemble
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

    # 3. Identify Seen vs Unseen questions in OOF
    # For each fold validation, is question_id seen in fold training?
    oof_seen_mask = np.zeros(n_train, dtype=bool)
    for tr_idx, val_idx in folds:
        train_qids = set(train_df.iloc[tr_idx]["question_id"].values)
        for idx in val_idx:
            qid = train_df.iloc[idx]["question_id"]
            if qid in train_qids:
                oof_seen_mask[idx] = True

    n_seen_oof = int(oof_seen_mask.sum())
    n_unseen_oof = n_train - n_seen_oof
    print(f"[AdaptivePoE] OOF Question Split: Seen={n_seen_oof} ({n_seen_oof/n_train*100:.1f}%), Unseen={n_unseen_oof} ({n_unseen_oof/n_train*100:.1f}%)")

    # Seen vs Unseen in Test set
    all_train_qids = set(train_df["question_id"].values)
    test_seen_mask = np.array([q in all_train_qids for q in test_df["question_id"].values], dtype=bool)
    n_seen_test = int(test_seen_mask.sum())
    n_unseen_test = n_test - n_seen_test
    print(f"[AdaptivePoE] Test Question Split: Seen={n_seen_test} ({n_seen_test/n_test*100:.1f}%), Unseen={n_unseen_test} ({n_unseen_test/n_test*100:.1f}%)")

    # 4. Baseline M10 Performance Check
    baseline_oof = 0.70 * oof_prior + 0.20 * oof_dual_vis + 0.10 * oof_sgd_text
    base_map3 = eval_map3(baseline_oof, y_true, classes)
    base_seen_map3 = eval_map3(baseline_oof[oof_seen_mask], [y_true[i] for i in range(n_train) if oof_seen_mask[i]], classes)
    base_unseen_map3 = eval_map3(baseline_oof[~oof_seen_mask], [y_true[i] for i in range(n_train) if not oof_seen_mask[i]], classes)
    print(f"\n[Baseline M10 (Fixed 70/20/10)]")
    print(f"  Overall OOF MAP@3: {base_map3:.4f}")
    print(f"  Seen Qs MAP@3:    {base_seen_map3:.4f}")
    print(f"  Unseen Qs MAP@3:  {base_unseen_map3:.4f}")

    # 5. Product-of-Experts (PoE) Log-Space Formulation
    eps = 1e-7
    log_prior = np.log(np.clip(oof_prior, eps, 1.0))
    log_vis = np.log(np.clip(oof_dual_vis, eps, 1.0))
    log_text = np.log(np.clip(oof_enh_text, eps, 1.0))

    test_log_prior = np.log(np.clip(test_prior, eps, 1.0))
    test_log_vis = np.log(np.clip(test_dual_vis, eps, 1.0))
    test_log_text = np.log(np.clip(test_enh_text, eps, 1.0))

    # Optimize Seen Weights (wp, wv, wt)
    seen_indices = np.where(oof_seen_mask)[0]
    seen_y = [y_true[i] for i in seen_indices]
    log_p_seen = log_prior[seen_indices]
    log_v_seen = log_vis[seen_indices]
    log_t_seen = log_text[seen_indices]

    def loss_seen(weights):
        wp, wv, wt = weights
        fused = wp * log_p_seen + wv * log_v_seen + wt * log_t_seen
        return -eval_map3(fused, seen_y, classes)

    res_seen = minimize(loss_seen, [0.65, 0.22, 0.13], method="Nelder-Mead", options={"maxiter": 200, "xatol": 0.01})
    w_seen = res_seen.x / res_seen.x.sum()
    opt_seen_map3 = -res_seen.fun
    print(f"\n[PoE Optimization - Seen Questions]")
    print(f"  Optimal Weights: Prior={w_seen[0]:.3f}, Vis={w_seen[1]:.3f}, Text={w_seen[2]:.3f}")
    print(f"  Seen OOF MAP@3:  {opt_seen_map3:.4f} (Gain over M10: +{opt_seen_map3 - base_seen_map3:.4f})")

    # Optimize Unseen Weights (wp, wv, wt)
    unseen_indices = np.where(~oof_seen_mask)[0]
    unseen_y = [y_true[i] for i in unseen_indices]
    log_p_unseen = log_prior[unseen_indices]
    log_v_unseen = log_vis[unseen_indices]
    log_t_unseen = log_text[unseen_indices]

    def loss_unseen(weights):
        wp, wv, wt = weights
        fused = wp * log_p_unseen + wv * log_v_unseen + wt * log_t_unseen
        return -eval_map3(fused, unseen_y, classes)

    res_unseen = minimize(loss_unseen, [0.15, 0.50, 0.35], method="Nelder-Mead", options={"maxiter": 200, "xatol": 0.01})
    w_unseen = res_unseen.x / res_unseen.x.sum()
    opt_unseen_map3 = -res_unseen.fun
    print(f"\n[PoE Optimization - Unseen Questions]")
    print(f"  Optimal Weights: Prior={w_unseen[0]:.3f}, Vis={w_unseen[1]:.3f}, Text={w_unseen[2]:.3f}")
    print(f"  Unseen OOF MAP@3: {opt_unseen_map3:.4f} (Gain over M10: +{opt_unseen_map3 - base_unseen_map3:.4f})")

    # 6. Overall Combined Adaptive PoE OOF Evaluation
    adaptive_poe_oof = np.zeros_like(oof_prior)
    adaptive_poe_oof[oof_seen_mask] = (
        w_seen[0] * log_prior[oof_seen_mask] +
        w_seen[1] * log_vis[oof_seen_mask] +
        w_seen[2] * log_text[oof_seen_mask]
    )
    adaptive_poe_oof[~oof_seen_mask] = (
        w_unseen[0] * log_prior[~oof_seen_mask] +
        w_unseen[1] * log_vis[~oof_seen_mask] +
        w_unseen[2] * log_text[~oof_seen_mask]
    )

    adaptive_map3 = eval_map3(adaptive_poe_oof, y_true, classes)
    delta_overall = adaptive_map3 - base_map3
    print(f"\n[Overall Adaptive PoE Evaluation]")
    print(f"  M10 Baseline OOF MAP@3:    {base_map3:.4f}")
    print(f"  Adaptive PoE OOF MAP@3:    {adaptive_map3:.4f} (Delta: +{delta_overall:.4f})")

    # 7. Paired Bootstrap Statistical Test (10,000 resamples)
    print("\n[AdaptivePoE] Running 10,000 paired bootstrap resamples...")
    rng = np.random.default_rng(42)
    scores_base = eval_per_sample_map3(baseline_oof, y_true, classes)
    scores_poe = eval_per_sample_map3(adaptive_poe_oof, y_true, classes)
    deltas = scores_poe - scores_base

    bootstrap_deltas = np.zeros(10000, dtype=np.float64)
    for b in range(10000):
        sample_indices = rng.integers(0, n_train, size=n_train)
        bootstrap_deltas[b] = np.mean(deltas[sample_indices])

    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))
    p_val = float(np.mean(bootstrap_deltas <= 0))
    print(f"  Bootstrap Mean Delta: +{float(np.mean(bootstrap_deltas)):.4f}")
    print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"  P-value (one-tailed <= 0): {p_val:.4f}")

    # 8. Generate Test Submission
    print(f"\n[AdaptivePoE] Generating test predictions using Adaptive PoE...")
    test_poe = np.zeros((n_test, n_classes), dtype=np.float32)
    test_poe[test_seen_mask] = (
        w_seen[0] * test_log_prior[test_seen_mask] +
        w_seen[1] * test_log_vis[test_seen_mask] +
        w_seen[2] * test_log_text[test_seen_mask]
    )
    test_poe[~test_seen_mask] = (
        w_unseen[0] * test_log_prior[~test_seen_mask] +
        w_unseen[1] * test_log_vis[~test_seen_mask] +
        w_unseen[2] * test_log_text[~test_seen_mask]
    )

    top_indices = np.argsort(-test_poe, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_indices]

    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"],
        "root_code": pred_strings
    })

    out_csv = PATHS.submissions_dir / "multimodal_adaptive_poe_submission.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"[AdaptivePoE] Saved candidate submission to: {out_csv}")

    val_res = validate_submission_file(out_csv, PATHS.taxonomy_csv)
    print(f"[AdaptivePoE] Validation status: {val_res}")

    # Agreement stats vs SOTA M10
    sota_df = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_dualvision_submission.csv"))
    diff_count = int((sub_df["root_code"] != sota_df["root_code"]).sum())
    diff_pct = (diff_count / len(sub_df)) * 100
    print(f"\n[Agreement against SOTA M10]")
    print(f"  Total test rows:        {len(sub_df)}")
    print(f"  Identical to SOTA:      {len(sub_df) - diff_count} ({(100 - diff_pct):.2f}%)")
    print(f"  Selectively diversified: {diff_count} ({diff_pct:.2f}%)")

    meta = {
        "model_name": "Adaptive_Product_of_Experts_Multimodal",
        "w_seen": [float(x) for x in w_seen],
        "w_unseen": [float(x) for x in w_unseen],
        "base_map3": base_map3,
        "adaptive_map3": adaptive_map3,
        "delta": delta_overall,
        "ci_95": [ci_lower, ci_upper],
        "p_value": p_val,
        "diversified_test_rows": diff_count,
        "output_file": str(out_csv)
    }

    with open(artifacts_dir / "adaptive_poe_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    run_adaptive_poe_pipeline()
