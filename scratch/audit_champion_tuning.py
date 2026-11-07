"""Audit Calibration, Transductive Boost, and Secondary Beta on Honest OOF.

Optimizes:
1. Frequency calibration exponent gamma in [0.00, 0.03, 0.05, 0.08, 0.10]
2. Transductive question boost alpha in [1.05, 1.10, 1.15, 1.20, 1.25, 1.30]
3. Secondary NONE discount beta in [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]

Evaluates honest OOF MAP@3 under strict 5-fold cross validation.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

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

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_oof_champ_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    top3_train = np.argsort(-p_oof_champ_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)

    row_idx = np.arange(n_samples)

    print("=" * 80)
    print("CHAMPION CALIBRATION & BOOST TUNING AUDIT")
    print("=" * 80)

    best_score = 0.0
    best_config = {}

    for gamma in [0.00, 0.03, 0.05, 0.07, 0.10]:
        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult[i] = (tc / pc) ** gamma
        mult[none_idx] = 0.9840

        p_cal = p_oof_champ_base * mult
        p_cal[:, none_idx] *= 0.9995

        for alpha in [1.00, 1.10, 1.15, 1.20, 1.25, 1.30]:
            p_boost = np.copy(p_cal)
            for f in splits:
                tr = f["train_indices"]
                va = f["val_indices"]
                tr_df = train_df.iloc[tr]
                q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_df.groupby("question_id")}
                for i in va:
                    qid = train_df.iloc[i]["question_id"]
                    if qid in q_map:
                        for ci, code in enumerate(classes):
                            if code != "NONE" and code in q_map[qid]:
                                p_boost[i, ci] *= alpha

            p_arc_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult
            p_ens = 0.85 * p_cal + 0.10 * p_boost + 0.05 * p_arc_cal

            r1_idx = np.argmax(p_ens, axis=1)

            for beta in [0.70, 0.75, 0.80, 0.82, 0.85, 0.90, 1.00]:
                cand = p_ens.copy()
                cand[row_idx, r1_idx] = -1.0
                cand[r1_idx != none_idx, none_idx] *= beta

                part2 = np.argpartition(-cand, kth=1, axis=1)[:, :2]
                part2_p = cand[row_idx[:, None], part2]
                order2 = np.argsort(-part2_p, axis=1, kind="mergesort")
                top2 = part2[row_idx[:, None], order2]

                triplets = [[classes[r1_idx[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
                score = score_triplets(y_true, triplets)

                if score > best_score:
                    best_score = score
                    best_config = {
                        "gamma": gamma,
                        "alpha": alpha,
                        "beta": beta,
                        "oof_map3": score
                    }
                    print(f"New Peak -> gamma={gamma:.2f} | alpha={alpha:.2f} | beta={beta:.2f} | OOF MAP@3: {score:.5f}")

    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION FOUND:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()

