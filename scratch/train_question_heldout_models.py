"""Train and Generate Honest Question-Held-Out Validation Predictions.

Protocol:
1. 5-Fold GroupKFold strictly on question_id.
   0.0% question overlap between training and validation in every fold.
2. For each fold:
   - Exclude all validation questions from model training.
   - Fit in-fold AdaptiveQuestionPriorEngine (falls back to topic-grade prior on unseen questions).
   - Fit Multimodal Linear Probes / Logistic Classifiers on concatenated normalized features:
     * ConvNeXt (768-d)
     * Swin (768-d)
     * TrOCR (384-d)
     * Text (384-d)
     * Vision (576-d)
   - Fit In-Fold Frequency Calibration mult strictly on training rows.
   - Predict probability distributions on the held-out validation questions.
3. Save full honest question-held-out validation prediction matrix:
   artifacts/oof_question_heldout_honest_baseline.npy (1500, 76).
4. Evaluate:
   - Overall Question-Held-Out MAP@3
   - Question-Held-Out MAP@3 on Rare-Class Rows (<= 10 samples)
   - Question-Held-Out MAP@3 on Common-Class Rows
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
    print("GENERATING HONEST QUESTION-HELD-OUT VALIDATION PREDICTIONS")
    print("All validation questions 100% excluded from model training per fold")
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

    # 1. Load Multimodal Embeddings
    print("Loading multimodal features for 1,500 training samples...")
    cn_feat = np.load(ART_DIR / "train_convnext_features.npy")
    sw_feat = np.load(ART_DIR / "train_swin_features.npy")
    tr_feat = np.load(ART_DIR / "train_trocr_features.npy")
    tx_feat = np.load(ART_DIR / "train_text_embeddings_384.npy")
    vi_feat = np.load(ART_DIR / "train_vision_features.npy")

    # L2-normalize each feature block before concatenation
    cn_norm = normalize(cn_feat, norm="l2")
    sw_norm = normalize(sw_feat, norm="l2")
    tr_norm = normalize(tr_feat, norm="l2")
    tx_norm = normalize(tx_feat, norm="l2")
    vi_norm = normalize(vi_feat, norm="l2")

    X_all = np.hstack([cn_norm, sw_norm, tr_norm, tx_norm, vi_norm])
    print(f"Concatenated feature matrix: {X_all.shape} ({X_all.dtype})")

    # 2. Setup 5-Fold GroupKFold
    gkf = GroupKFold(n_splits=5)
    oof_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_prior = np.zeros((n_samples, n_classes), dtype=np.float64)
    oof_clf = np.zeros((n_samples, n_classes), dtype=np.float64)

    fold_scores = []
    rare_counts = Counter(train_df["root_code"].dropna())
    rare_classes = set(c for c, cnt in rare_counts.items() if cnt <= 10)

    for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, y_indices, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]

        # Verify zero question leakage
        val_q = set(val_df["question_id"])
        trn_q = set(trn_df["question_id"])
        assert len(val_q & trn_q) == 0, f"Leakage detected in fold {fold}!"

        # In-fold class presence
        present_classes = np.unique(y_indices[trn_idx])

        # A. Fit AdaptiveQuestionPriorEngine strictly on training fold
        prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        val_prior = prior_engine.predict_proba(val_df).astype(np.float64)
        oof_prior[val_idx] = val_prior

        # B. Train Multimodal Classifier strictly on training fold
        clf = LogisticRegression(
            C=1.0,
            max_iter=300,
            solver="lbfgs",
            random_state=42,
            multi_class="multinomial"
        )
        clf.fit(X_all[trn_idx], y_indices[trn_idx])

        val_clf_raw = clf.predict_proba(X_all[val_idx])
        # Map classifier probabilities to full 76-class matrix
        val_clf = np.zeros((len(val_idx), n_classes), dtype=np.float64)
        for local_col, global_cls_idx in enumerate(clf.classes_):
            val_clf[:, global_cls_idx] = val_clf_raw[:, local_col]
        oof_clf[val_idx] = val_clf

        # C. In-fold Ensemble & Calibration
        # Blend multimodal classifier (which generalizes visually) with topic-grade prior
        val_blend = 0.55 * val_clf + 0.45 * val_prior

        # In-fold frequency calibration
        trn_pred_indices = np.argmax(clf.predict_proba(X_all[trn_idx]), axis=1)
        trn_pred_classes = [clf.classes_[i] for i in trn_pred_indices]
        pred_counts = Counter(trn_pred_classes)
        true_counts = Counter(y_indices[trn_idx])

        in_fold_mult = np.ones(n_classes, dtype=np.float64)
        for c_idx in range(n_classes):
            if c_idx != none_idx:
                pc = max(pred_counts.get(c_idx, 1), 1)
                tc = max(true_counts.get(c_idx, 1), 1)
                in_fold_mult[c_idx] = (tc / pc) ** 0.05
        in_fold_mult[none_idx] = 0.9850

        val_calibrated = val_blend * in_fold_mult
        oof_probs[val_idx] = val_calibrated

        # Score fold
        val_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-val_calibrated, axis=1)]
        fold_map3 = calculate_map3(y_raw[val_idx].tolist(), val_top3)
        fold_scores.append(fold_map3)
        print(f"Fold {fold}: {len(trn_idx)} trn, {len(val_idx)} val | {len(val_q)} unseen questions | Val MAP@3: {fold_map3:.5f}")

    # 3. Overall Evaluation
    print("-" * 80)
    top3_all = [[classes[idx] for idx in row[:3]] for row in np.argsort(-oof_probs, axis=1)]
    overall_map3 = calculate_map3(y_raw.tolist(), top3_all)
    print(f"Overall Honest Question-Held-Out MAP@3: {overall_map3:.5f} (Mean fold: {np.mean(fold_scores):.5f} +/- {np.std(fold_scores):.5f})")

    # 4. Stratified Subgroup Analysis
    rare_mask = np.array([c in rare_classes for c in y_raw])
    rare_y = [y_raw[i] for i in range(len(y_raw)) if rare_mask[i]]
    rare_p = [top3_all[i] for i in range(len(y_raw)) if rare_mask[i]]
    rare_map3 = calculate_map3(rare_y, rare_p)

    common_y = [y_raw[i] for i in range(len(y_raw)) if not rare_mask[i]]
    common_p = [top3_all[i] for i in range(len(y_raw)) if not rare_mask[i]]
    common_map3 = calculate_map3(common_y, common_p)

    print(f"\nSubgroup Performance Breakdown:")
    print(f"  Common Classes ({sum(~rare_mask)} rows): MAP@3 = {common_map3:.5f}")
    print(f"  Rare Classes   ({sum(rare_mask)} rows, <=10 samples): MAP@3 = {rare_map3:.5f}")

    # What percentage of rare-class rows were predicted as NONE at Rank 1?
    rare_r1 = [p[0] for p in rare_p]
    print(f"  Rare-class rows predicted as NONE at Rank 1: {sum(1 for x in rare_r1 if x == 'NONE')} / {len(rare_r1)} ({sum(1 for x in rare_r1 if x == 'NONE')/len(rare_r1):.1%})")

    # 5. Save Honest Question-Held-Out Validation Matrix
    out_npy = ART_DIR / "oof_question_heldout_honest_baseline.npy"
    np.save(out_npy, oof_probs)
    print(f"\n[Saved] {out_npy} ({oof_probs.shape})")
    print("=" * 80)

if __name__ == "__main__":
    main()

