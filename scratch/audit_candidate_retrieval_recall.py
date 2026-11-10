"""Audit Candidate Retrieval Recall@K on SOTA 550 Champion Pipeline.

Prerequisite for any candidate reranker:
A reranker cannot promote a class that is missing from its candidate set.
Therefore, on rows where the student has a misconception (y_true != 'NONE'):
1. Does the champion's candidate list contain the true misconception?
2. What is Recall@1, Recall@2, Recall@3, Recall@5, Recall@10?
3. Where is the true misconception located when Rank 1 is 'NONE'?
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80)
    print("AUDITING CANDIDATE RETRIEVAL RECALL@K FOR SOTA 550 CHAMPION")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # 1. Reconstruct exact SOTA 550 Continuous Foundation on 1,500 rows
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    # Mult calibration
    top3_tr = np.argsort(-p_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_raw)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_cal = p_base * mult
    p_cal[:, none_idx] *= 0.9995

    # Clean multimodal
    from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
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

    p_final = 0.96 * p_cal + 0.04 * p_clean

    # Candidate lists excluding NONE
    cand_matrix = p_final.copy()
    cand_matrix[:, none_idx] = -1.0  # mask out NONE to get pure misconception candidate ranking
    sorted_cand_indices = np.argsort(-cand_matrix, axis=1)

    # Filter rows where student actually has a misconception
    misc_rows = [i for i in range(n_samples) if y_raw[i] != "NONE"]
    print(f"Total labeled student traces: {n_samples}")
    print(f"Traces where student has a misconception (y_true != 'NONE'): {len(misc_rows)}")

    # Check Recall@K
    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in misc_rows:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_cand_indices[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        recall = hits / len(misc_rows)
        print(f"  Misconception Candidate Recall@{k:2d}: {hits:4d} / {len(misc_rows)} ({recall:.1%})")

    # Critical subgroup: Rows where the model predicted NONE at Rank 1, but student had a misconception!
    r1_is_none_and_true_misc = [i for i in misc_rows if np.argmax(p_final[i]) == none_idx]
    print(f"\nSubgroup: NONE at Rank 1 while true label is a misconception ({len(r1_is_none_and_true_misc)} rows):")
    for k in [1, 2, 3, 5, 8, 10]:
        hits = 0
        for i in r1_is_none_and_true_misc:
            true_label = y_raw[i]
            top_k_codes = [classes[idx] for idx in sorted_cand_indices[i, :k]]
            if true_label in top_k_codes:
                hits += 1
        recall = hits / len(r1_is_none_and_true_misc)
        print(f"  Candidate Recall@{k:2d} on 'NONE-at-Rank-1' rows: {hits:4d} / {len(r1_is_none_and_true_misc)} ({recall:.1%})")

    # Rank distribution of true label among candidate misconceptions
    ranks = []
    for i in r1_is_none_and_true_misc:
        true_label = y_raw[i]
        cand_list = [classes[idx] for idx in sorted_cand_indices[i]]
        if true_label in cand_list:
            r = cand_list.index(true_label) + 1
            ranks.append(r)
        else:
            ranks.append(999)

    print("\nExact rank distribution of true label among non-NONE candidates:")
    rank_counts = Counter(ranks)
    for r in range(1, 11):
        print(f"  Rank {r:2d}: {rank_counts.get(r, 0):3d} traces ({rank_counts.get(r, 0)/len(r1_is_none_and_true_misc):.1%})")
    print(f"  Rank >10: {sum(cnt for r, cnt in rank_counts.items() if r > 10):3d} traces ({sum(cnt for r, cnt in rank_counts.items() if r > 10)/len(r1_is_none_and_true_misc):.1%})")

    print("=" * 80)

if __name__ == "__main__":
    main()

