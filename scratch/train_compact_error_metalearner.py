"""Track 1: Fast Compact Meta-Learner for P(is_error) with SVD Dimension Compression.

Features:
1. 48-d TruncatedSVD from 2,328-d raw multimodal features (ConvNeXt-448, Swin-T, TrOCR-384, MiniLM-384, Strokes).
2. Model uncertainty signals: P(NONE), max P(misc), margin (max P(misc) - P(NONE)), ratio, entropy, Gini.
3. Fold-local question empirical error rates (Laplace smoothed).
4. Perceptual stroke summary metrics.

Target:
y = 1 if root_code != 'NONE' else 0.
Strictly evaluated with 5-fold CV (splits.json).
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
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
    print("TRACK 1: FAST COMPACT META-LEARNER FOR P(is_error)", flush=True)
    print("=" * 80, flush=True)

    classes = [line.strip().split(",")[0] for line in open(DATA_DIR / "taxonomy_v1.csv")][1:]
    if "NONE" not in classes:
        classes.append("NONE")
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    n_train = len(train_df)
    n_test = len(test_df)

    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_bin = np.array([0 if code == "NONE" else 1 for code in y_raw], dtype=np.int32)
    print(f"Dataset: {n_train} samples | Clean (0): {(y_bin == 0).sum()} ({(y_bin == 0).mean():.1%}), Misconception (1): {(y_bin == 1).sum()} ({(y_bin == 1).mean():.1%})", flush=True)

    print("Loading multimodal features...", flush=True)
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

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    p_test_champ = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    unc_tr = compute_uncertainty_features(p_oof_champ, none_idx)
    unc_te = compute_uncertainty_features(p_test_champ, none_idx)

    # Precompute fold features ONCE
    print("Precomputing SVD & meta-features for each of 5 folds...", flush=True)
    fold_data = []
    for f_idx, fold in enumerate(splits):
        tr_idx = fold["train_indices"]
        va_idx = fold["val_indices"]

        tr_df = train_df.iloc[tr_idx]
        global_err_rate = float(y_bin[tr_idx].mean())
        q_err: Dict[str, float] = {}
        for qid, grp in tr_df.groupby("question_id"):
            err_cnt = float((grp["root_code"].fillna("NONE") != "NONE").sum())
            tot_cnt = float(len(grp))
            q_err[qid] = (err_cnt + 1.0) / (tot_cnt + 2.0)

        q_tr_col = np.array([q_err.get(qid, global_err_rate) for qid in tr_df["question_id"]]).reshape(-1, 1)
        q_va_col = np.array([q_err.get(qid, global_err_rate) for qid in train_df.iloc[va_idx]["question_id"]]).reshape(-1, 1)
        q_te_col = np.array([q_err.get(qid, global_err_rate) for qid in test_df["question_id"]]).reshape(-1, 1)

        print(f"  Fitting SVD on Fold {f_idx}...", flush=True)
        svd = TruncatedSVD(n_components=48, random_state=42)
        X_svd_tr = svd.fit_transform(X_raw_tr[tr_idx])
        X_svd_va = svd.transform(X_raw_tr[va_idx])
        X_svd_te = svd.transform(X_raw_te)

        X_meta_tr = np.hstack([X_svd_tr, unc_tr[tr_idx], q_tr_col, st_tr[tr_idx, :4]])
        X_meta_va = np.hstack([X_svd_va, unc_tr[va_idx], q_va_col, st_tr[va_idx, :4]])
        X_meta_te = np.hstack([X_svd_te, unc_te, q_te_col, st_te[:, :4]])

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_meta_tr)
        X_va_sc = scaler.transform(X_meta_va)
        X_te_sc = scaler.transform(X_meta_te)

        fold_data.append({
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "X_tr": X_tr_sc,
            "X_va": X_va_sc,
            "X_te": X_te_sc,
            "y_tr": y_bin[tr_idx],
            "y_va": y_bin[va_idx]
        })

    print("All fold features precomputed successfully!", flush=True)

    # Grid search C
    c_vals = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
    best_c = 0.1
    best_auc = 0.0
    best_oof = np.zeros(n_train, dtype=np.float64)

    print("\nSweeping C parameter across precomputed folds:", flush=True)
    print("-" * 80, flush=True)
    for c_val in c_vals:
        fold_aucs = []
        fold_losses = []
        current_oof = np.zeros(n_train, dtype=np.float64)

        for fd in fold_data:
            clf = LogisticRegression(C=c_val, penalty="l2", max_iter=500, random_state=42)
            clf.fit(fd["X_tr"], fd["y_tr"])
            val_prob = clf.predict_proba(fd["X_va"])[:, 1]
            current_oof[fd["va_idx"]] = val_prob

            fold_aucs.append(roc_auc_score(fd["y_va"], val_prob))
            fold_losses.append(log_loss(fd["y_va"], val_prob))

        mean_auc = float(np.mean(fold_aucs))
        mean_loss = float(np.mean(fold_losses))
        print(f"C = {c_val:7.3f} | 5-Fold AUC: {mean_auc:.4f} | LogLoss: {mean_loss:.4f} | Per-fold AUCs: {[round(a, 4) for a in fold_aucs]}", flush=True)

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_c = c_val
            best_oof = current_oof.copy()

    print("=" * 80, flush=True)
    print(f"OPTIMAL C = {best_c} (Mean 5-Fold AUC: {best_auc:.4f})", flush=True)

    # Generate Test Probabilities
    test_preds_folds = np.zeros((len(splits), n_test), dtype=np.float64)
    for f_idx, fd in enumerate(fold_data):
        clf = LogisticRegression(C=best_c, penalty="l2", max_iter=500, random_state=42)
        clf.fit(fd["X_tr"], fd["y_tr"])
        test_preds_folds[f_idx] = clf.predict_proba(fd["X_te"])[:, 1]

    test_prob = test_preds_folds.mean(axis=0)

    # Save artifacts
    np.save(ART_DIR / "oof_compact_binary_prob.npy", best_oof)
    np.save(ART_DIR / "test_compact_binary_prob.npy", test_prob)
    print(f"\n[Saved Artifact] {ART_DIR / 'oof_compact_binary_prob.npy'}", flush=True)
    print(f"[Saved Artifact] {ART_DIR / 'test_compact_binary_prob.npy'}", flush=True)
    print(f"Test Probability Distribution: min={test_prob.min():.4f}, mean={test_prob.mean():.4f}, max={test_prob.max():.4f}", flush=True)


if __name__ == "__main__":
    main()
