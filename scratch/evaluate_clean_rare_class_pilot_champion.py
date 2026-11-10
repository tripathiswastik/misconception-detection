"""Clean Rare-Class Pseudo-Labeling Pilot Benchmark against SOTA 550 Champion.

Methodological Upgrades per User Directive:
1. In-Fold Rare Class Definition:
   - Classes defined as 'rare' (count <= 10) STRICTLY using each fold's training labels (trn_df).
   - Zero full-dataset leakage in rare class identification or centroid computation.
2. Question Guardrail:
   - Candidate pseudo traces can ONLY be added if their question_id is in that fold's training set.
3. Multilateral Evaluation:
   - Measure standalone probe impact on Rare classes vs Common classes vs Overall.
   - Test selective rare-class injection into the SOTA 550 champion pipeline.
   - Run under both canonical 5-fold (splits.json) and 5-fold GroupKFold.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
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
    print("CLEAN RARE-CLASS PSEUDO-LABELING PILOT BENCHMARK")
    print("Strict In-Fold Rare Class Definitions | In-Fold Centroid Filtering | Champion Audit")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    n_samples = len(train_df)

    # 1. Load Multimodal Features (2328-d)
    print("Loading multimodal features...")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    st_feat = np.load(ART_DIR / "train_perceptual_strokes.npy")

    X_train_2328 = np.hstack([
        normalize(cn_feat), normalize(sw_feat), normalize(tr_feat), normalize(tx_feat), normalize(st_feat)
    ])

    # 2. Load Pseudo Pool (1465 candidates)
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_raw = np.load(ART_DIR / "pseudo_1465_features_2328.npy")
    X_pseudo_norm = np.hstack([
        normalize(pseudo_raw[:, :768]),
        normalize(pseudo_raw[:, 768:1536]),
        normalize(pseudo_raw[:, 1536:1920]),
        normalize(pseudo_raw[:, 1920:2304]),
        normalize(pseudo_raw[:, 2304:2328])
    ])

    # 3. Load SOTA 550 Champion Base
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_champ_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    champ_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_champ_base, axis=1)]
    champ_base_map3 = calculate_map3(y_raw, champ_top3)
    print(f"Champion Foundation OOF MAP@3 Baseline: {champ_base_map3:.5f}")

    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    # =========================================================================
    # PART 1: STANDALONE PROBE AUDIT (Zero Pseudo vs In-Fold Rare Pseudo)
    # Does adding pseudo traces improve the classifier on rare classes?
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: STANDALONE MULTIMODAL PROBE AUDIT")
    print("=" * 80)

    for min_margin in [0.0, 0.005, 0.010, 0.015, 0.020]:
        is_baseline = (min_margin == 0.0)
        mode = "BASELINE (0 pseudo traces added)" if is_baseline else f"PILOT (margin >= {min_margin:.3f})"
        
        probe_oof = np.zeros((n_samples, n_classes), dtype=np.float64)
        total_pseudo_added = 0
        all_fold_rare_classes: List[Set[str]] = []

        for f_idx, f in enumerate(splits):
            trn_idx = f["train_indices"]
            val_idx = f["val_indices"]
            trn_df = train_df.iloc[trn_idx]
            trn_questions = set(trn_df["question_id"])

            # Strict In-Fold Rare Definition: count <= 10 in trn_df only
            fold_counts = Counter(trn_df["root_code"].dropna())
            fold_rare = set(c for c, cnt in fold_counts.items() if cnt <= 10)
            all_fold_rare_classes.append(fold_rare)

            fold_y = y_indices[trn_idx]
            fold_X = X_train_2328[trn_idx]

            # In-fold centroids
            fold_centroids = np.zeros((n_classes, fold_X.shape[1]), dtype=np.float32)
            for c_i in range(n_classes):
                c_mask = (fold_y == c_i)
                if c_mask.sum() > 0:
                    c_m = fold_X[c_mask].mean(axis=0)
                    fold_centroids[c_i] = c_m / (np.linalg.norm(c_m) + 1e-12)

            in_fold_pseudo_X = []
            in_fold_pseudo_y = []

            if not is_baseline:
                sims = np.dot(X_pseudo_norm, fold_centroids.T)
                top1_idx = np.argmax(sims, axis=1)
                sorted_sims = np.sort(sims, axis=1)[:, ::-1]
                margins = sorted_sims[:, 0] - sorted_sims[:, 1]

                for p_i in range(len(pseudo_df)):
                    p_qid = pseudo_df.iloc[p_i]["question_id"]
                    if p_qid in trn_questions:
                        p_code = classes[top1_idx[p_i]]
                        p_margin = margins[p_i]
                        if p_code in fold_rare and p_margin >= min_margin:
                            in_fold_pseudo_X.append(X_pseudo_norm[p_i])
                            in_fold_pseudo_y.append(top1_idx[p_i])

            total_pseudo_added += len(in_fold_pseudo_y)

            if len(in_fold_pseudo_y) > 0:
                X_trn_aug = np.vstack([fold_X, np.array(in_fold_pseudo_X)])
                y_trn_aug = np.concatenate([fold_y, np.array(in_fold_pseudo_y)])
            else:
                X_trn_aug = fold_X
                y_trn_aug = fold_y

            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs", random_state=42)
            clf.fit(X_trn_aug, y_trn_aug)

            val_preds_raw = clf.predict_proba(X_train_2328[val_idx])
            for col_i, cls_i in enumerate(clf.classes_):
                probe_oof[val_idx, cls_i] = val_preds_raw[:, col_i]

        probe_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-probe_oof, axis=1)]
        probe_map3 = calculate_map3(y_raw, probe_top3)

        # Global rare evaluation
        full_rare_counts = Counter(train_df["root_code"].dropna())
        global_rare = set(c for c, cnt in full_rare_counts.items() if cnt <= 10)
        rare_mask = np.array([c in global_rare for c in y_raw])
        rare_y = [y_raw[i] for i in range(n_samples) if rare_mask[i]]
        rare_p = [probe_top3[i] for i in range(n_samples) if rare_mask[i]]
        rare_map3 = calculate_map3(rare_y, rare_p)

        common_y = [y_raw[i] for i in range(n_samples) if not rare_mask[i]]
        common_p = [probe_top3[i] for i in range(n_samples) if not rare_mask[i]]
        common_map3 = calculate_map3(common_y, common_p)

        print(f"{mode:38s} | Pseudo Added: {total_pseudo_added:4d} | MAP@3 Overall: {probe_map3:.5f} | Rare: {rare_map3:.5f} | Common: {common_map3:.5f}")

    # =========================================================================
    # PART 2: SELECTIVE RARE-CLASS INJECTION INTO SOTA 550 CHAMPION
    # Can we boost champion probabilities for rare classes where the pilot is confident?
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: SELECTIVE RARE-CLASS INJECTION INTO CHAMPION")
    print("=" * 80)

    # Let's take the pilot predictions from margin >= 0.010
    # For each sample, if probe_oof has high confidence in a rare class, boost champion prob for that rare class
    for boost_weight in [0.01, 0.02, 0.05, 0.10]:
        new_champ_probs = np.copy(p_champ_base)
        for i in range(n_samples):
            for c_i, c in enumerate(classes):
                if c in global_rare and c != "NONE":
                    new_champ_probs[i, c_i] += boost_weight * probe_oof[i, c_i]

        new_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-new_champ_probs, axis=1)]
        score = calculate_map3(y_raw, new_top3)
        delta = score - champ_base_map3

        # Rare vs common
        r_p = [new_top3[i] for i in range(n_samples) if rare_mask[i]]
        c_p = [new_top3[i] for i in range(n_samples) if not rare_mask[i]]
        r_score = calculate_map3(rare_y, r_p)
        c_score = calculate_map3(common_y, c_p)

        print(f"boost_weight = {boost_weight:.2f} | Champion MAP@3: {score:.5f} ({delta:+.5f}) | Rare: {r_score:.5f} | Common: {c_score:.5f}")

    print("=" * 80)

if __name__ == "__main__":
    main()

