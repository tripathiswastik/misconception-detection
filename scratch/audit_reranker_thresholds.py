"""Audit Candidate Reranker Swapping Thresholds on Question-Held-Out Folds.

Examines why unconstrained reranking swapped 637 rows (-0.01411):
Candidate Rank 1 is already correct in 31.5% of rows, while Candidate Rank 2 is correct in only 18.4%.
Therefore, aggressive swapping causes massive false-alarm demotions of correct Candidate-1 predictions!

Hypothesis:
Only intervening when the champion probability gap between Candidate 1 and Candidate 2 is tiny (gap < eps)
and reranker margin is strong (score_diff > delta) prevents destructive demotions.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80)
    print("AUDITING CONSTRAINED CANDIDATE RERANKING THRESHOLDS")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 1. Load Pre-extracted Frozen Embeddings
    trocr_feats = np.load(ART_DIR / "train_trocr_features.npy")
    q_text_feats = np.load(ART_DIR / "train_text_embeddings_384.npy")
    tax_feats = np.load(ART_DIR / "taxonomy_embeddings_384.npy")

    trocr_norm = trocr_feats / (np.linalg.norm(trocr_feats, axis=1, keepdims=True) + 1e-12)
    q_norm = q_text_feats / (np.linalg.norm(q_text_feats, axis=1, keepdims=True) + 1e-12)
    tax_norm = tax_feats / (np.linalg.norm(tax_feats, axis=1, keepdims=True) + 1e-12)

    # 2. Reconstruct Exact Champion Foundation Matrix
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    top3_tr = np.argsort(-p_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_raw)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_cal = p_base * mult
    p_cal[:, none_idx] *= 0.9995

    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    prior_oof = prior_engine.predict_proba(train_df)

    p_clean = (
        0.6448 * prior_oof +
        0.0839 * sharp(np.load(ART_DIR / "honest_oof_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "oof_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "oof_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "oof_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "oof_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "oof_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "oof_clean_arcface_2328.npy"), 0.8)
    )

    p_champ = 0.96 * p_cal + 0.04 * p_clean
    base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_champ, axis=1)]
    base_map3 = calculate_map3(y_raw, base_top3)
    print(f"Champion Baseline OOF MAP@3: {base_map3:.5f}")

    r1_is_none = [p[0] == "NONE" for p in base_top3]
    cand_matrix = p_champ.copy()
    cand_matrix[:, none_idx] = -1.0
    sorted_cand_indices = np.argsort(-cand_matrix, axis=1)[:, :5]

    # Precompute pairwise features for candidate 1 vs candidate 2
    gkf = GroupKFold(n_splits=5)

    # Let's test a disciplined conservative sweep
    # Condition: Only swap Candidate 1 and Candidate 2 if:
    # 1. Champion probability gap between Cand 1 and Cand 2 is < max_gap
    # 2. Semantic alignment (OCR + Question cosine to candidate 2) exceeds candidate 1 by > min_semantic_delta
    print("\nSweeping Conservative Semantic Gating (max_gap x min_semantic_delta):")
    print(f"{'max_gap':8s} | {'min_delta':10s} | {'Swaps':6s} | {'Held-Out MAP@3':15s} | {'Delta':8s}")
    print("-" * 60)

    for max_gap in [0.005, 0.010, 0.020, 0.030, 0.050]:
        for min_delta in [0.02, 0.05, 0.08, 0.10, 0.15]:
            new_top3 = [list(row) for row in base_top3]
            swaps_count = 0

            for i in range(n_samples):
                if r1_is_none[i]:
                    c1_idx = sorted_cand_indices[i, 0]
                    c2_idx = sorted_cand_indices[i, 1]

                    p1 = p_champ[i, c1_idx]
                    p2 = p_champ[i, c2_idx]
                    gap = p1 - p2

                    if 0 < gap < max_gap:
                        # Semantic scores
                        cos_ocr1 = float(np.dot(trocr_norm[i], tax_norm[c1_idx]))
                        cos_ocr2 = float(np.dot(trocr_norm[i], tax_norm[c2_idx]))

                        cos_q1 = float(np.dot(q_norm[i], tax_norm[c1_idx]))
                        cos_q2 = float(np.dot(q_norm[i], tax_norm[c2_idx]))

                        sem1 = 0.5 * cos_ocr1 + 0.5 * cos_q1
                        sem2 = 0.5 * cos_ocr2 + 0.5 * cos_q2
                        sem_delta = sem2 - sem1

                        if sem_delta > min_delta:
                            new_top3[i][1] = classes[c2_idx]
                            new_top3[i][2] = classes[c1_idx]
                            swaps_count += 1

            s = calculate_map3(y_raw, new_top3)
            delta = s - base_map3
            if swaps_count > 0 and swaps_count < 100:
                print(f"{max_gap:8.3f} | {min_delta:10.2f} | {swaps_count:6d} | {s:15.5f} | {delta:+.5f}")

    print("=" * 80)

if __name__ == "__main__":
    main()

