"""Audit Topic-Prefix Filtering on Rank 2 and Rank 3 in 5-Fold Cross Validation.

Method:
1. Estimate allowed misconception prefixes per topic_name strictly inside each fold.
2. In out-of-fold validation, if a predicted misconception at Rank 2 or Rank 3 has a prefix
   that NEVER appeared for that topic in the training fold, demote its probability so the next
   highest in-domain candidate misconception takes its place.
3. Keep Rank 1 strictly untouched.
4. Keep NONE strictly allowed in all topics.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def score_triplets(y_true: List[str], triplets: List[List[str]]) -> float:
    scores = []
    for true_label, row in zip(y_true, triplets):
        score = 0.0
        for r, pred in enumerate(row[:3]):
            if pred == true_label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_oof_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Baseline predictions
    row_idx = np.arange(n_samples)
    r1_base = np.argmax(p_oof_champ, axis=1)
    cand_base = p_oof_champ.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]

    base_triplets = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, base_triplets)
    print(f"Baseline Honest OOF MAP@3: {base_score:.5f}")

    # Now apply fold-local topic-prefix domain filtering on Rank 2 and Rank 3
    filtered_triplets = [list(t) for t in base_triplets]
    filtered_r2_count = 0
    filtered_r3_count = 0

    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        tr_df = train_df.iloc[tr]

        # Learn allowed prefixes per topic from train fold
        topic_prefixes: Dict[str, Set[str]] = {}
        for topic, grp in tr_df.groupby("topic_name"):
            codes = grp["root_code"].dropna().unique()
            prefixes = {c.split("-")[0].upper() for c in codes if c != "NONE"}
            topic_prefixes[topic] = prefixes

        # Filter val fold
        for i in va:
            topic = train_df.iloc[i]["topic_name"]
            if topic in topic_prefixes:
                allowed = topic_prefixes[topic]
                row_cand = cand_base[i].copy()

                # Zero out candidates with disallowed prefix (never zero out NONE!)
                for ci, c in enumerate(classes):
                    if c != "NONE":
                        pref = c.split("-")[0].upper()
                        if pref not in allowed:
                            row_cand[ci] = -1.0

                new_top2 = np.argsort(-row_cand)[:2]
                new_r2 = classes[new_top2[0]]
                new_r3 = classes[new_top2[1]]

                if new_r2 != base_triplets[i][1]:
                    filtered_r2_count += 1
                if new_r3 != base_triplets[i][2]:
                    filtered_r3_count += 1

                filtered_triplets[i] = [base_triplets[i][0], new_r2, new_r3]

    new_score = score_triplets(y_true, filtered_triplets)
    print(f"Filtered OOF MAP@3: {new_score:.5f} (Delta: {new_score - base_score:+.5f})")
    print(f"Filtered Rank 2 slots: {filtered_r2_count}")
    print(f"Filtered Rank 3 slots: {filtered_r3_count}")


if __name__ == "__main__":
    main()

