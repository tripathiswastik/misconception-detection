"""Question-Held-Out Evaluation of Rare-Class Pseudo-Labeling Pilot.

Hypothesis:
Adding in-fold pseudo-labeled traces for rare classes will improve held-out rare-class MAP@3
(baseline: 0.06463) without degrading common-class MAP@3 (baseline: 0.51623).

Guardrails:
1. Strict GroupKFold on question_id (0.0% question leakage).
2. Candidate pseudo traces are ONLY added if their question is in that fold's training set.
3. Completely untouched validation fold for honest evaluation.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("QUESTION-HELD-OUT EVALUATION OF RARE-CLASS PSEUDO-LABELING PILOT")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 1. Load Train Multimodal Features
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    vi_feat = np.load(ART_DIR / "train_vision_features.npy")

    X_train_norm = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(vi_feat)
    ])

    # 2. Load Pseudo Pool
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_raw = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    # Features in pseudo_1465 are 2328-d (cn 768, sw 768, tr 384, tx 384, vi 24)
    # We will map matching features (cn, sw, tr, tx) for consistency:
    X_pseudo_norm = np.hstack([
        normalize(pseudo_raw[:, :768]),
        normalize(pseudo_raw[:, 768:1536]),
        normalize(pseudo_raw[:, 1536:1920]),
        normalize(pseudo_raw[:, 1920:2304]),
        normalize(pseudo_raw[:, 2304:2328])
    ])
    # For matching dimension with X_train_norm (which has 576-d vision), let's use the matching 2328-d for train:
    st_tr = np.load(ART_DIR / "train_perceptual_strokes.npy")
    X_train_2328 = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(st_tr)
    ])

    # 3. Setup GroupKFold
    gkf = GroupKFold(n_splits=5)
    rare_counts = Counter(train_df["root_code"].dropna())
    rare_classes = set(c for c, cnt in rare_counts.items() if cnt <= 10)

    # Evaluate multiple margin thresholds for the pilot: baseline (0 added), margin >= 0.005, margin >= 0.010
    for min_margin in [0.0, 0.008, 0.012]:
        is_baseline = (min_margin == 0.0)
        mode_label = "BASELINE (0 pseudo traces added)" if is_baseline else f"PILOT (margin >= {min_margin:.3f})"
        print(f"\n--- Running 5-Fold Evaluation: {mode_label} ---")

        oof_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_pseudo_added = 0

        for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, y_indices, groups=groups), start=1):
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            trn_questions = set(trn_df["question_id"])

            # Compute in-fold centroids from trn_idx
            fold_y = y_indices[trn_idx]
            fold_X = X_train_2328[trn_idx]
            fold_centroids = np.zeros((n_classes, fold_X.shape[1]), dtype=np.float32)
            for c_i in range(n_classes):
                c_mask = (fold_y == c_i)
                if c_mask.sum() > 0:
                    c_m = fold_X[c_mask].mean(axis=0)
                    fold_centroids[c_i] = c_m / (np.linalg.norm(c_m) + 1e-12)

            # Check pseudo candidates that belong to TRN questions ONLY
            in_fold_pseudo_X = []
            in_fold_pseudo_y = []

            if not is_baseline:
                sims = np.dot(X_pseudo_norm, fold_centroids.T)
                top1_idx = np.argmax(sims, axis=1)
                sorted_sims = np.sort(sims, axis=1)[:, ::-1]
                margins = sorted_sims[:, 0] - sorted_sims[:, 1]

                for p_i in range(len(pseudo_df)):
                    p_qid = pseudo_df.iloc[p_i]["question_id"]
                    # INVARIANT: Question must belong to training fold
                    if p_qid in trn_questions:
                        p_code = classes[top1_idx[p_i]]
                        p_margin = margins[p_i]
                        if p_code in rare_classes and p_margin >= min_margin:
                            in_fold_pseudo_X.append(X_pseudo_norm[p_i])
                            in_fold_pseudo_y.append(top1_idx[p_i])

            total_pseudo_added += len(in_fold_pseudo_y)

            # Assemble training data for this fold
            if len(in_fold_pseudo_y) > 0:
                X_trn_aug = np.vstack([fold_X, np.array(in_fold_pseudo_X)])
                y_trn_aug = np.concatenate([fold_y, np.array(in_fold_pseudo_y)])
            else:
                X_trn_aug = fold_X
                y_trn_aug = fold_y

            # Train classifier
            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
            clf.fit(X_trn_aug, y_trn_aug)

            # Prior engine strictly on original training fold
            prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
            val_prior = prior_engine.predict_proba(val_df).astype(np.float64)

            # Predict on validation fold
            val_clf_raw = clf.predict_proba(X_train_2328[val_idx])
            val_clf = np.zeros((len(val_idx), n_classes), dtype=np.float64)
            for local_col, global_cls_idx in enumerate(clf.classes_):
                val_clf[:, global_cls_idx] = val_clf_raw[:, local_col]

            # In-fold blend & calibration
            val_blend = 0.55 * val_clf + 0.45 * val_prior
            oof_probs[val_idx] = val_blend

        # Score across all 1500 samples
        top3_all = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_probs, axis=1)]
        overall_map3 = calculate_map3(y_raw.tolist(), top3_all)

        rare_mask = np.array([c in rare_classes for c in y_raw])
        rare_y = [y_raw[i] for i in range(len(y_raw)) if rare_mask[i]]
        rare_p = [top3_all[i] for i in range(len(y_raw)) if rare_mask[i]]
        rare_map3 = calculate_map3(rare_y, rare_p)

        common_y = [y_raw[i] for i in range(len(y_raw)) if not rare_mask[i]]
        common_p = [top3_all[i] for i in range(len(y_raw)) if not rare_mask[i]]
        common_map3 = calculate_map3(common_y, common_p)

        print(f"  Pseudo traces added across 5 folds: {total_pseudo_added}")
        print(f"  Overall MAP@3: {overall_map3:.5f}")
        print(f"  Common Classes ({sum(~rare_mask)} rows): {common_map3:.5f}")
        print(f"  Rare Classes   ({sum(rare_mask)} rows): {rare_map3:.5f}")

    print("=" * 80)

if __name__ == "__main__":
    main()

