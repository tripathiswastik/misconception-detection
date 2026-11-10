"""Evaluate Rare-Class Pilot against Current Champion Pipeline with In-Fold Rare Class Definitions.

Methodological Upgrades:
1. Rare class threshold (<= 10 samples) defined STRICTLY per fold using trn_df["root_code"] ONLY (zero full-dataset leakage).
2. Uses the verified SOTA 550 Champion foundation (M15 + ArcFace V1 + DualHead + Composite).
3. Pseudo candidates are filtered strictly in-fold:
   - Must belong to questions present in that fold's training partition.
   - Predicted class must belong to that fold's identified rare classes.
   - Metric centroid similarity margin must exceed threshold.
4. Evaluates whether in-fold pseudo-labeling improves the champion pipeline.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("RARE-CLASS PILOT vs CHAMPION PIPELINE (IN-FOLD RARE CLASS DEFINITIONS)")
    print("Zero leakage: rare classes defined strictly using fold training labels")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    # 1. Load Champion Features & Foundation
    print("Loading multimodal features...")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    st_feat = np.load(ART_DIR / "train_perceptual_strokes.npy")

    X_train_2328 = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(st_feat)
    ])

    # Pseudo pool
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_raw = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    X_pseudo_norm = np.hstack([
        normalize(pseudo_raw[:, :768]),
        normalize(pseudo_raw[:, 768:1536]),
        normalize(pseudo_raw[:, 1536:1920]),
        normalize(pseudo_raw[:, 1920:2304]),
        normalize(pseudo_raw[:, 2304:2328])
    ])

    # 2. Champion Baseline (Zero Pseudo Added)
    # Reconstruct Champion Foundation OOF
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_champ_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    champ_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_champ_base, axis=1)]
    champ_base_map3 = calculate_map3(y_raw, champ_top3)
    print(f"\nChampion Base OOF MAP@3 (Baseline): {champ_base_map3:.5f}")

    # 3. Test Pilot with In-Fold Rare Definitions across multiple margins
    for min_margin in [0.005, 0.010, 0.015]:
        print(f"\n--- Testing In-Fold Rare Pilot with min_margin = {min_margin:.3f} ---")
        oof_blend = np.copy(p_champ_base)
        total_in_fold_added = 0
        fold_rare_counts_list = []

        for f_idx, f in enumerate(splits, start=1):
            trn_idx = f["train_indices"]
            val_idx = f["val_indices"]
            trn_df = train_df.iloc[trn_idx]
            trn_questions = set(trn_df["question_id"])

            # INVARIANT: Define rare classes STRICTLY from this fold's training labels!
            fold_counts = Counter(trn_df["root_code"].dropna())
            fold_rare_classes = set(c for c, cnt in fold_counts.items() if cnt <= 10)
            fold_rare_counts_list.append(len(fold_rare_classes))

            # Compute in-fold centroids strictly from training fold
            fold_y = y_indices[trn_idx]
            fold_X = X_train_2328[trn_idx]
            fold_centroids = np.zeros((n_classes, fold_X.shape[1]), dtype=np.float32)
            for c_i in range(n_classes):
                c_mask = (fold_y == c_i)
                if c_mask.sum() > 0:
                    c_m = fold_X[c_mask].mean(axis=0)
                    fold_centroids[c_i] = c_m / (np.linalg.norm(c_m) + 1e-12)

            # In-fold pseudo candidates
            sims = np.dot(X_pseudo_norm, fold_centroids.T)
            top1_idx = np.argmax(sims, axis=1)
            sorted_sims = np.sort(sims, axis=1)[:, ::-1]
            margins = sorted_sims[:, 0] - sorted_sims[:, 1]

            in_fold_pseudo_X = []
            in_fold_pseudo_y = []

            for p_i in range(len(pseudo_df)):
                p_qid = pseudo_df.iloc[p_i]["question_id"]
                # Must be in this fold's training questions
                if p_qid in trn_questions:
                    p_code = classes[top1_idx[p_i]]
                    p_margin = margins[p_i]
                    # Must be in THIS FOLD'S identified rare classes
                    if p_code in fold_rare_classes and p_margin >= min_margin:
                        in_fold_pseudo_X.append(X_pseudo_norm[p_i])
                        in_fold_pseudo_y.append(top1_idx[p_i])

            total_in_fold_added += len(in_fold_pseudo_y)

            # Train linear probe on augmented in-fold data
            if len(in_fold_pseudo_y) > 0:
                X_aug = np.vstack([fold_X, np.array(in_fold_pseudo_X)])
                y_aug = np.concatenate([fold_y, np.array(in_fold_pseudo_y)])
            else:
                X_aug = fold_X
                y_aug = fold_y

            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
            clf.fit(X_aug, y_aug)

            val_preds_raw = clf.predict_proba(X_train_2328[val_idx])
            val_probe = np.zeros((len(val_idx), n_classes), dtype=np.float64)
            for col_i, cls_i in enumerate(clf.classes_):
                val_probe[:, cls_i] = val_preds_raw[:, col_i]

            # In-fold ensemble: blend probe predictions with champion foundation
            # Test blend weight w_probe in [0.02, 0.05]
            w_probe = 0.05
            oof_blend[val_idx] = (1.0 - w_probe) * p_champ_base[val_idx] + w_probe * val_probe

        print(f"  Rare classes defined per fold: mean = {np.mean(fold_rare_counts_list):.1f} classes")
        print(f"  Total pseudo traces added across 5 folds: {total_in_fold_added}")

        blend_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_blend, axis=1)]
        new_map3 = calculate_map3(y_raw, blend_top3)
        delta = new_map3 - champ_base_map3
        print(f"  Champion + In-Fold Rare Pilot MAP@3: {new_map3:.5f} ({delta:+.5f})")

    print("=" * 80)

if __name__ == "__main__":
    main()

