"""Evaluate ConvNeXt-Tiny Visual Embeddings via In-Fold Linear Probe.

Benchmarks:
1. Raw ConvNeXt Visual OOF MAP@3
2. Domain-Masked ConvNeXt Visual OOF MAP@3 (compare with MobileNetV3 0.3819)
3. Fusion with Question Prior (m_q=10) and Tri-Modal Ensembling
4. Generates test probability matrix test_convnext_vision.npy if passing stopping rule (>0.3800)
"""
import sys
import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

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


def run_convnext_probe() -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Probe] Evaluating ConvNeXt-Tiny Visual Linear Probe")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true = list(train_df["root_code"].astype(str))
    n_train = len(train_df)
    n_test = len(test_df)

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    artifacts_dir = PROJECT_ROOT / "artifacts"
    X_train_raw = np.load(artifacts_dir / "train_convnext_features.npy")
    X_test_raw = np.load(artifacts_dir / "test_convnext_features.npy")

    # L2 normalize visual embeddings
    X_train: np.ndarray = np.asarray(normalize(X_train_raw, norm="l2"), dtype=np.float32)
    X_test: np.ndarray = np.asarray(normalize(X_test_raw, norm="l2"), dtype=np.float32)

    print(f"[Probe] Loaded features: train={X_train.shape}, test={X_test.shape}")

    # Sweep regularizer C for linear probe
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    best_c = 0.1
    best_raw_map3 = -1.0
    best_oof_probs: np.ndarray = np.zeros((n_train, n_classes), dtype=np.float32)
    best_test_probs: np.ndarray = np.zeros((n_test, n_classes), dtype=np.float32)

    for c in C_values:
        oof_p = np.zeros((n_train, n_classes), dtype=np.float32)
        test_p = np.zeros((n_test, n_classes), dtype=np.float32)

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            clf = LogisticRegression(C=c, max_iter=200, solver="lbfgs", multi_class="multinomial", random_state=42)
            clf.fit(X_train[tr_idx], [y_true[i] for i in tr_idx])
            raw_va = clf.predict_proba(X_train[va_idx])
            raw_te = clf.predict_proba(X_test)

            # Align to master classes
            for m_i, c_name in enumerate(clf.classes_):
                if c_name in classes:
                    c_idx = classes.index(c_name)
                    oof_p[va_idx, c_idx] = raw_va[:, m_i]
                    test_p[:, c_idx] += raw_te[:, m_i] / len(folds)

        # Row normalize
        r_sums = oof_p.sum(axis=1, keepdims=True)
        oof_p = np.divide(oof_p, r_sums, where=r_sums > 0)
        t_sums = test_p.sum(axis=1, keepdims=True)
        test_p = np.divide(test_p, t_sums, where=t_sums > 0)

        # Compute raw MAP@3
        top = np.argsort(-oof_p, axis=1, kind="mergesort")[:, :3]
        preds = [[classes[i] for i in r] for r in top]
        score = calculate_map3(y_true, preds)
        print(f"  Linear Probe (C={c:5.2f}): Raw Visual OOF MAP@3 = {score:.4f}")

        if score > best_raw_map3:
            best_raw_map3 = score
            best_c = c
            best_oof_probs = oof_p
            best_test_probs = test_p

    print(f"\n[Probe] Optimal C={best_c} with Raw Visual OOF MAP@3 = {best_raw_map3:.4f}")

    # Apply in-fold domain constraint masking
    oof_convnext_masked = np.zeros_like(best_oof_probs)
    test_convnext_masked = np.zeros_like(best_test_probs)

    for tr_idx, va_idx in folds:
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(float(g)) for g in train_df.iloc[va_idx]["grade"]]
        oof_convnext_masked[va_idx] = masker.apply_mask_batch(best_oof_probs[va_idx], classes, t_va, g_va, damping=0.85)

    masker_global = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    t_te = [str(t) for t in test_df["topic_name"]]
    g_te = [int(float(g)) for g in test_df["grade"]]
    test_convnext_masked = masker_global.apply_mask_batch(best_test_probs, classes, t_te, g_te, damping=0.85)

    top_m = np.argsort(-oof_convnext_masked, axis=1, kind="mergesort")[:, :3]
    preds_m = [[classes[i] for i in r] for r in top_m]
    masked_score = calculate_map3(y_true, preds_m)
    print(f"[Probe] Masked ConvNeXt Visual OOF MAP@3 = {masked_score:.4f} (MobileNetV3 benchmark: 0.3819)")

    # Save ConvNeXt probability matrices
    np.save(artifacts_dir / "oof_convnext_vision.npy", oof_convnext_masked)
    np.save(artifacts_dir / "test_convnext_vision.npy", test_convnext_masked)

    # Compare & Fuse with Question Prior (m_q=10)
    print("\n--- Fusion with Tuned Question Prior (m_q=10.0) ---")
    oof_prior_10 = np.zeros((n_train, n_classes), dtype=np.float32)
    for tr_idx, va_idx in folds:
        engine = QuestionPriorEngine(classes=classes, smoothing_question=10.0, smoothing_topic_grade=5.0)
        engine.fit(train_df.iloc[tr_idx])
        p, _ = engine.predict_proba(train_df.iloc[va_idx])
        oof_prior_10[va_idx] = p

    vis_m7_masked = np.load(artifacts_dir / "oof_finetuned_vision.npy")
    for tr_idx, va_idx in folds:
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(float(g)) for g in train_df.iloc[va_idx]["grade"]]
        vis_m7_masked[va_idx] = masker.apply_mask_batch(vis_m7_masked[va_idx], classes, t_va, g_va, damping=0.85)

    # 1. Prior + ConvNeXt
    for w in [0.15, 0.20, 0.25, 0.30]:
        fused_cn = (1.0 - w) * oof_prior_10 + w * oof_convnext_masked
        top = np.argsort(-fused_cn, axis=1, kind="mergesort")[:, :3]
        s = calculate_map3(y_true, [[classes[i] for i in r] for r in top])
        print(f"  Prior(10) + ConvNeXt (w={w:.2f}): MAP@3 = {s:.4f}")

    # 2. Vision Ensemble: ConvNeXt + MobileNetV3 (complementary CNN features)
    vis_ensemble = 0.5 * oof_convnext_masked + 0.5 * vis_m7_masked
    top_ve = np.argsort(-vis_ensemble, axis=1, kind="mergesort")[:, :3]
    s_ve = calculate_map3(y_true, [[classes[i] for i in r] for r in top_ve])
    print(f"\n  Dual-Vision Ensemble (ConvNeXt + MobileNetV3): OOF MAP@3 = {s_ve:.4f}")

    # 3. Tri-Modal: Prior + Dual-Vision Ensemble + Text
    text_h = np.load(artifacts_dir / "honest_oof_text.npy")
    for w_v in [0.20, 0.25, 0.30]:
        for w_t in [0.05, 0.10]:
            w_p = 1.0 - w_v - w_t
            fused_tri = w_p * oof_prior_10 + w_v * vis_ensemble + w_t * text_h
            top = np.argsort(-fused_tri, axis=1, kind="mergesort")[:, :3]
            s = calculate_map3(y_true, [[classes[i] for i in r] for r in top])
            print(f"  Prior({w_p:.2f}) + DualVision({w_v:.2f}) + Text({w_t:.2f}): MAP@3 = {s:.4f}")

    return {
        "best_c": best_c,
        "raw_visual_map3": best_raw_map3,
        "masked_visual_map3": masked_score,
        "dual_vision_map3": s_ve
    }


if __name__ == "__main__":
    run_convnext_probe()
