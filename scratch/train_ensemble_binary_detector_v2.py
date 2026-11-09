"""Train Ensemble Binary Detector V2 and Evaluate on OOF & Test.

Outputs:
- artifacts/oof_compact_binary_prob_v2.npy
- artifacts/test_compact_binary_prob_v2.npy
"""
import json
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, precision_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

def compute_uncertainty_features(probs: np.ndarray, none_idx: int) -> np.ndarray:
    n_samples, n_classes = probs.shape
    p_none = probs[:, none_idx]

    probs_misc = probs.copy()
    probs_misc[:, none_idx] = -1.0
    p_top_misc = np.max(probs_misc, axis=1)

    margin = p_top_misc - p_none
    ratio = p_top_misc / np.maximum(p_none, 1e-6)

    eps = 1e-12
    p_safe = np.maximum(probs, eps)
    p_norm = p_safe / p_safe.sum(axis=1, keepdims=True)
    entropy = -np.sum(p_norm * np.log(p_norm), axis=1)
    gini = 1.0 - np.sum(p_norm ** 2, axis=1)

    part2 = np.partition(-probs_misc, kth=1, axis=1)[:, :2]
    p_second_misc = -part2[:, 1]
    misc_margin = p_top_misc - p_second_misc

    features = np.column_stack([
        p_none,
        p_top_misc,
        p_second_misc,
        margin,
        ratio,
        entropy,
        gini,
        misc_margin
    ])
    return features

def main():
    print("=" * 80, flush=True)
    print("ENSEMBLE BINARY DETECTOR V2: TRAINING & ARTIFACT GENERATION", flush=True)
    print("=" * 80, flush=True)

    classes = [line.strip().split(",")[0] for line in open(DATA_DIR / "taxonomy_v1.csv")][1:]
    if "NONE" not in classes:
        classes.append("NONE")
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    n_train = len(train_df)
    n_test = len(test_df)

    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_bin = np.array([0 if code == "NONE" else 1 for code in y_raw], dtype=np.int32)

    # 1. Multimodal Probabilities
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")

    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")

    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")

    oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    test_prior = np.load(ART_DIR / "test_adaptive_prior.npy")

    oof_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    test_sem = np.load(ART_DIR / "test_semantic_text.npy")

    p_none_tr = np.column_stack([
        oof_m15[:, none_idx],
        oof_v2[:, none_idx],
        oof_dual[:, none_idx],
        oof_comp[:, none_idx],
        oof_prior[:, none_idx],
        oof_sem[:, none_idx]
    ])
    p_none_te = np.column_stack([
        test_m15[:, none_idx],
        test_v2[:, none_idx],
        test_dual[:, none_idx],
        test_comp[:, none_idx],
        test_prior[:, none_idx],
        test_sem[:, none_idx]
    ])

    unc_tr = compute_uncertainty_features(oof_m15, none_idx)
    unc_te = compute_uncertainty_features(test_m15, none_idx)
    unc_v2_tr = compute_uncertainty_features(oof_v2, none_idx)
    unc_v2_te = compute_uncertainty_features(test_v2, none_idx)

    # SVD
    cn_tr = np.load(ART_DIR / "train_convnext_448_letterbox.npy")
    tr_tr = np.load(ART_DIR / "train_trocr_384_letterbox.npy")
    sw_tr = np.load(ART_DIR / "train_swin_features.npy")
    tx_tr = np.load(ART_DIR / "train_text_embeddings_384.npy")
    st_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")
    X_raw_tr = np.hstack([cn_tr, tr_tr, sw_tr, tx_tr, st_tr])

    cn_te = np.load(ART_DIR / "test_convnext_448_letterbox.npy")
    tr_te = np.load(ART_DIR / "test_trocr_384_letterbox.npy")
    sw_te = np.load(ART_DIR / "test_swin_features.npy")
    tx_te = np.load(ART_DIR / "test_text_embeddings_384.npy")
    st_te = np.load(ART_DIR / "test_perceptual_strokes.npy")
    X_raw_te = np.hstack([cn_te, tr_te, sw_te, tx_te, st_te])

    svd = TruncatedSVD(n_components=64, random_state=42)
    svd_tr = svd.fit_transform(X_raw_tr)
    svd_te = svd.transform(X_raw_te)

    X_meta_tr = np.hstack([p_none_tr, unc_tr, unc_v2_tr, svd_tr, st_tr])
    X_meta_te = np.hstack([p_none_te, unc_te, unc_v2_te, svd_te, st_te])

    oof_lr = np.zeros(n_train, dtype=np.float64)
    oof_hgb = np.zeros(n_train, dtype=np.float64)
    oof_rf = np.zeros(n_train, dtype=np.float64)

    test_lr = np.zeros(n_test, dtype=np.float64)
    test_hgb = np.zeros(n_test, dtype=np.float64)
    test_rf = np.zeros(n_test, dtype=np.float64)

    for fold in splits:
        tr_idx = fold["train_indices"]
        va_idx = fold["val_indices"]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_meta_tr[tr_idx])
        X_va = scaler.transform(X_meta_tr[va_idx])
        X_te = scaler.transform(X_meta_te)

        # 1. LR
        lr = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
        lr.fit(X_tr, y_bin[tr_idx])
        oof_lr[va_idx] = lr.predict_proba(X_va)[:, 1]
        test_lr += lr.predict_proba(X_te)[:, 1] / len(splits)

        # 2. HGB
        hgb = HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=2.0, learning_rate=0.05, random_state=42
        )
        hgb.fit(X_tr, y_bin[tr_idx])
        oof_hgb[va_idx] = hgb.predict_proba(X_va)[:, 1]
        test_hgb += hgb.predict_proba(X_te)[:, 1] / len(splits)

        # 3. RF
        rf = RandomForestClassifier(
            n_estimators=150, max_depth=6, min_samples_leaf=5, random_state=42, n_jobs=-1
        )
        rf.fit(X_tr, y_bin[tr_idx])
        oof_rf[va_idx] = rf.predict_proba(X_va)[:, 1]
        test_rf += rf.predict_proba(X_te)[:, 1] / len(splits)

    # Blend: 0.30 LR + 0.35 HGB + 0.35 RF
    oof_blend = 0.30 * oof_lr + 0.35 * oof_hgb + 0.35 * oof_rf
    test_blend = 0.30 * test_lr + 0.35 * test_hgb + 0.35 * test_rf

    auc_blend = roc_auc_score(y_bin, oof_blend)
    acc_blend = accuracy_score(y_bin, oof_blend >= 0.5)
    loss_blend = log_loss(y_bin, oof_blend)

    print(f"\n[Ensemble Binary Detector V2] OOF AUC: {auc_blend:.4f} | Acc: {acc_blend:.4f} | LogLoss: {loss_blend:.4f}", flush=True)

    # Save Artifacts
    np.save(ART_DIR / "oof_compact_binary_prob_v2.npy", oof_blend)
    np.save(ART_DIR / "test_compact_binary_prob_v2.npy", test_blend)
    print(f"[Saved Artifact] {ART_DIR / 'oof_compact_binary_prob_v2.npy'}", flush=True)
    print(f"[Saved Artifact] {ART_DIR / 'test_compact_binary_prob_v2.npy'}", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

