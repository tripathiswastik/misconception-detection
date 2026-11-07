"""Test Empirical Question Prior Gate in 5-Fold Cross Validation.

Investigates whether questions with empirical training evidence of 0% NONE rate
can reliably have NONE demoted from Rank 1 without leakage.
"""
import json
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def main():
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    y_true = train_df["root_code"].fillna("NONE").values

    def eval_preds(preds, y):
        s = 0.0
        for p, t in zip(preds, y):
            for r, c in enumerate(p[:3]):
                if c == t:
                    s += 1.0 / (r + 1)
                    break
        return s / len(y)

    base_preds = [[classes[idx] for idx in np.argsort(-oof_m15[i])[:3]] for i in range(len(train_df))]
    base_score = eval_preds(base_preds, y_true)
    print(f"Baseline OOF MAP@3: {base_score:.5f}")

    # Test varying min_count thresholds: 2, 3, 4, 5
    for min_count in [1, 2, 3, 4, 5]:
        gated_preds = [list(p) for p in base_preds]
        flips = 0

        for f in splits:
            tr = f["train_indices"]
            va = f["val_indices"]

            tr_df = train_df.iloc[tr]
            q_stats = tr_df.groupby("question_id")["root_code"].agg(
                count="count",
                none_rate=lambda s: (s == "NONE").mean()
            )

            zero_none_qs = set(q_stats[(q_stats["count"] >= min_count) & (q_stats["none_rate"] == 0.0)].index)

            for i in va:
                qid = train_df.iloc[i]["question_id"]
                if qid in zero_none_qs:
                    row_p = oof_m15[i].copy()
                    if np.argmax(row_p) == none_idx:
                        # Top candidate misconception
                        row_p_nom = row_p.copy()
                        row_p_nom[none_idx] = -1.0
                        best_cand = int(np.argmax(row_p_nom))

                        row_p[best_cand] = -1.0
                        row_p[none_idx] *= 0.82
                        rem_top2 = np.argsort(-row_p)[:2]
                        gated_preds[i] = [classes[best_cand], classes[rem_top2[0]], classes[rem_top2[1]]]
                        flips += 1

        new_score = eval_preds(gated_preds, y_true)
        print(f"Min Count = {min_count} -> Flips: {flips:3d} | Gated OOF MAP@3: {new_score:.5f} | Delta: {new_score - base_score:+.5f}")


if __name__ == "__main__":
    main()

