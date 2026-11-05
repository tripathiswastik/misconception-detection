"""Evaluate Swin Transformer (swin_t) Visual Embeddings via In-Fold Linear Probe.

Benchmarks:
1. Raw Swin-T Visual OOF MAP@3 (across C regularizer sweep).
2. Domain-Masked Swin-T Visual OOF MAP@3 (compare with ConvNeXt-Tiny 0.3880 and MobileNetV3 0.3819).
3. Tri-Vision Ensemble: ConvNeXt-Tiny + Swin-T + MobileNetV3.
4. Generates:
   - artifacts/oof_swin_vision.npy
   - artifacts/test_swin_vision.npy
   - artifacts/oof_trivision.npy
   - artifacts/test_trivision.npy
   - artifacts/swin_probe_manifest.json
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from src.metrics.map3 import calculate_map3


def eval_map3(probs: np.ndarray, y_true: List[str], classes: List[str]) -> float:
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)


def run_swin_probe() -> Dict[str, Any]:
    print("\n" + "=" * 75)
    print("[Probe] Evaluating Swin-T Visual Linear Probe & Tri-Vision Ensembling")
    print("=" * 75 + "\n")

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
    X_train_raw = np.load(artifacts_dir / "train_swin_features.npy")
    X_test_raw = np.load(artifacts_dir / "test_swin_features.npy")

    # L2 normalize visual embeddings
    X_train = np.asarray(normalize(X_train_raw, norm="l2"), dtype=np.float32)
    X_test = np.asarray(normalize(X_test_raw, norm="l2"), dtype=np.float32)
    print(f"[Probe] Loaded Swin-T features: train={X_train.shape}, test={X_test.shape}")

    # Sweep regularizer C for linear probe
    C_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]
    best_c = 0.1
    best_raw_map3 = -1.0
    best_oof_probs: np.ndarray = np.zeros((n_train, n_classes), dtype=np.float32)
    best_test_probs: np.ndarray = np.zeros((n_test, n_classes), dtype=np.float32)

    print("\n--- Sweeping Linear Probe Regularizer C ---")
    for c in C_values:
        oof_p = np.zeros((n_train, n_classes), dtype=np.float32)
        test_p = np.zeros((n_test, n_classes), dtype=np.float32)

        for fold_idx, (tr_idx, va_idx) in enumerate(folds):
            clf = LogisticRegression(C=c, max_iter=250, solver="lbfgs", random_state=42)
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

        score = eval_map3(oof_p, y_true, classes)
        print(f"  Swin-T Probe (C={c:5.2f}): Raw Visual OOF MAP@3 = {score:.4f}")

        if score > best_raw_map3:
            best_raw_map3 = score
            best_c = c
            best_oof_probs = oof_p
            best_test_probs = test_p

    print(f"\n[Probe] Optimal C={best_c} with Raw Swin-T Visual OOF MAP@3 = {best_raw_map3:.4f}")

    # Apply in-fold domain constraint masking
    print("\n--- Applying In-Fold Domain Constraint Masking ---")
    oof_swin_masked = np.zeros_like(best_oof_probs)
    test_swin_masked = np.zeros_like(best_test_probs)

    for tr_idx, va_idx in folds:
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df.iloc[tr_idx])
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(float(g)) for g in train_df.iloc[va_idx]["grade"]]
        oof_swin_masked[va_idx] = masker.apply_mask_batch(best_oof_probs[va_idx], classes, t_va, g_va, damping=0.85)

    masker_global = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    t_te = [str(t) for t in test_df["topic_name"]]
    g_te = [int(float(g)) for g in test_df["grade"]]
    test_swin_masked = masker_global.apply_mask_batch(best_test_probs, classes, t_te, g_te, damping=0.85)

    swin_masked_score = eval_map3(oof_swin_masked, y_true, classes)
    print(f"[Probe] Masked Swin-T Visual OOF MAP@3: {swin_masked_score:.4f}")
    print(f"  ConvNeXt-Tiny Benchmark:  0.3880")
    print(f"  MobileNetV3 Benchmark:   0.3819")

    # Persist Swin-T matrices
    np.save(artifacts_dir / "oof_swin_vision.npy", oof_swin_masked)
    np.save(artifacts_dir / "test_swin_vision.npy", test_swin_masked)
    print(f"[Probe] Persisted Swin-T vision matrices to {artifacts_dir}")

    # Tri-Vision Ensemble Evaluation
    print("\n--- Tri-Vision Ensemble Benchmarks ---")
    oof_cn = np.load(artifacts_dir / "oof_convnext_vision.npy")
    oof_mn = np.load(artifacts_dir / "oof_finetuned_vision.npy")
    test_cn = np.load(artifacts_dir / "test_convnext_vision.npy")
    test_mn = np.load(artifacts_dir / "test_finetuned_vision.npy")
    test_mn_masked = masker_global.apply_mask_batch(test_mn, classes, t_te, g_te, damping=0.85)

    dual_vis_score = eval_map3(0.5 * oof_cn + 0.5 * oof_mn, y_true, classes)
    print(f"  Dual-Vision Baseline (50% ConvNeXt + 50% MobileNet): MAP@3 = {dual_vis_score:.4f}")

    # Test Tri-Vision blending topologies
    tri_configs = [
        (0.34, 0.33, 0.33, "Equal Tri-Vision (34% CN + 33% Swin + 33% MN)"),
        (0.40, 0.40, 0.20, "Transformer+ConvNeXt Heavy (40% CN + 40% Swin + 20% MN)"),
        (0.45, 0.35, 0.20, "ConvNeXt Anchored (45% CN + 35% Swin + 20% MN)"),
        (0.35, 0.45, 0.20, "Swin Anchored (35% CN + 45% Swin + 20% MN)")
    ]

    best_tri_score = -1.0
    best_tri_oof = np.zeros_like(oof_cn)
    best_tri_test = np.zeros_like(test_cn)
    best_tri_desc = ""

    for w_cn, w_sw, w_mn, desc in tri_configs:
        tri_oof = w_cn * oof_cn + w_sw * oof_swin_masked + w_mn * oof_mn
        score = eval_map3(tri_oof, y_true, classes)
        delta = score - dual_vis_score
        print(f"  {desc}: MAP@3 = {score:.4f} [Delta vs Dual-Vision: {delta:+.4f}]")

        if score > best_tri_score:
            best_tri_score = score
            best_tri_oof = tri_oof
            best_tri_test = w_cn * test_cn + w_sw * test_swin_masked + w_mn * test_mn_masked
            best_tri_desc = desc

    print(f"\n[Probe] Champion Tri-Vision: {best_tri_desc} with MAP@3 = {best_tri_score:.4f}")
    np.save(artifacts_dir / "oof_trivision.npy", best_tri_oof)
    np.save(artifacts_dir / "test_trivision.npy", best_tri_test)
    print(f"[Probe] Saved Tri-Vision matrices to {artifacts_dir / 'oof_trivision.npy'} and test_trivision.npy")

    manifest = {
        "best_c": best_c,
        "swin_raw_map3": best_raw_map3,
        "swin_masked_map3": swin_masked_score,
        "dual_vision_map3": dual_vis_score,
        "champion_trivision_desc": best_tri_desc,
        "champion_trivision_map3": best_tri_score,
        "trivision_gain_over_dualvision": best_tri_score - dual_vis_score
    }
    with open(artifacts_dir / "swin_probe_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


if __name__ == "__main__":
    run_swin_probe()
