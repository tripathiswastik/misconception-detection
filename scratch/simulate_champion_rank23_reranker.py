"""Simulate Rank-2/3 Candidate Reranker on Champion Test Submission.

Invariants:
1. Rank 1 is 100% strictly fixed (all 11 golden flips and all Rank 1 predictions untouched).
2. For test rows where Rank 1 is NONE, candidate misconceptions at Rank 2/3 are re-scored using
   S(c) = P_champ(c)^0.90 * P_topic_grade(c)^0.10.
3. Swaps are audited: how many rows swap Rank 2 and Rank 3?
"""
from collections import Counter
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

def main():
    print("=" * 80)
    print("SIMULATING RANK-2/3 CANDIDATE RERANKER ON SOTA 550 CHAMPION")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))

    n_test = len(test_df)
    s550_preds = s550["root_code"].tolist()

    # Fit topic-grade prior on entire train set
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_tg_prior = prior_engine.predict_proba(test_df)

    # Reconstruct champion continuous probability matrix on test
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_v1 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")

    p_test_champ = 0.95 * test_m15 + 0.02 * test_dual + 0.03 * test_comp

    alpha = 0.90
    reranked_preds = list(s550_preds)
    swaps = []

    for i in range(n_test):
        tokens = s550_preds[i].split()
        r1 = tokens[0]
        r2 = tokens[1]
        r3 = tokens[2]

        if r1 == "NONE":
            c2_idx = class_to_idx[r2]
            c3_idx = class_to_idx[r3]

            p2 = max(p_test_champ[i, c2_idx], 1e-12)
            p3 = max(p_test_champ[i, c3_idx], 1e-12)

            tg2 = max(test_tg_prior[i, c2_idx], 1e-12)
            tg3 = max(test_tg_prior[i, c3_idx], 1e-12)

            score2 = (p2 ** alpha) * (tg2 ** (1.0 - alpha))
            score3 = (p3 ** alpha) * (tg3 ** (1.0 - alpha))

            if score3 > score2:
                new_pred = f"{r1} {r3} {r2}"
                reranked_preds[i] = new_pred
                swaps.append((i, r2, r3, p2, p3, tg2, tg3))

    print(f"Total test rows: {n_test}")
    print(f"Rows with NONE at Rank 1 in SOTA 550: {sum(1 for p in s550_preds if p.startswith('NONE'))} ({sum(1 for p in s550_preds if p.startswith('NONE'))/n_test:.1%})")
    print(f"Total Rank-2/Rank-3 swaps under alpha=0.90: {len(swaps)}")

    print(f"\nSample candidate swaps (Rank 1 strictly unchanged):")
    print(f"{'Row':5s} | {'Before (R2 R3)':20s} -> {'After (R3 R2)':20s} | {'P_champ(R2/R3)':16s} | {'P_tg(R2/R3)':16s}")
    print("-" * 80)
    for row_idx, r2, r3, p2, p3, tg2, tg3 in swaps[:15]:
        print(f"{row_idx:5d} | {r2:9s} {r3:9s} -> {r3:9s} {r2:9s} | {p2:.4f} / {p3:.4f}   | {tg2:.4f} / {tg3:.4f}")

    print("=" * 80)

if __name__ == "__main__":
    main()

