"""Fold-local audit of transductive priors for MAP@3.

Unlike the original sweep, each fold's frequency calibration and question/topic
maps use only that fold's training indices. This is an evaluation utility; it
does not make a submission or claim an unbiased score after selecting the best
grid cell. Use a separate validation set for final hyperparameter selection.

The saved OOF probability artifacts were produced before this outer-fold
calibration. If their model-training folds overlap an outer validation fold,
they can still carry indirect fold information. Fully nested model OOFs are
needed for a strict outer-CV estimate.
"""

from collections import Counter
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ART_DIR = ROOT / "artifacts"
DATA_DIR = ROOT / "data"


def map3(y_true, probs, classes):
    top3 = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
    rank_score = {0: 1.0, 1: 0.5, 2: 1.0 / 3.0}
    scores = []
    for label, row in zip(y_true, top3):
        found = [classes[idx] for idx in row]
        scores.append(next((rank_score[i] for i, pred in enumerate(found)
                            if pred == label), 0.0))
    return float(np.mean(scores))


def fold_calibrator(train_indices, train_df, y, p_base, classes):
    """Fit the repository's mild top-3 frequency correction on fold-train only."""
    none_idx = classes.index("NONE")
    p_train = p_base[train_indices]
    top3 = np.argsort(-p_train.astype(np.float64), axis=1)[:, :3]
    predicted = Counter(classes[idx] for row in top3 for idx in row)
    actual = Counter(y[train_indices])
    mult = np.ones(len(classes), dtype=np.float64)
    for ci, label in enumerate(classes):
        if ci == none_idx:
            continue
        pc = max(predicted.get(label, 0), 1)
        tc = max(actual.get(label, 0), 1)
        mult[ci] = (tc / pc) ** 0.05
    # Do not impose the old globally selected 0.984 NONE multiplier.
    return mult


def masks_for_rows(train_idx, rows_idx, train_df, classes):
    """Question/topic label support derived only from the current fold's train rows."""
    fold_train = train_df.iloc[train_idx]
    q_map = {key: set(group["root_code"].dropna())
             for key, group in fold_train.groupby("question_id")}
    topic_map = {key: set(group["root_code"].dropna())
                 for key, group in fold_train.groupby("topic_name")}
    none_idx = classes.index("NONE")
    q_mask = np.zeros((len(rows_idx), len(classes)), dtype=bool)
    topic_mask = np.zeros_like(q_mask)
    for local_i, source_i in enumerate(rows_idx):
        row = train_df.iloc[source_i]
        for ci, label in enumerate(classes):
            if ci == none_idx:
                continue
            q_mask[local_i, ci] = label in q_map.get(row["question_id"], set())
            topic_mask[local_i, ci] = label in topic_map.get(row["topic_name"], set())
    return q_mask, topic_mask


def main():
    with (ART_DIR / "taxonomy_classes.json").open(encoding="utf-8") as handle:
        classes = json.load(handle)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y = train_df["root_code"].fillna("NONE").astype(str).to_numpy()
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    if p_base.shape != (len(train_df), len(classes)):
        raise ValueError(f"Expected OOF shape {(len(train_df), len(classes))}, got {p_base.shape}")

    q_grid = np.linspace(0.0, 0.8, 9)
    topic_damp_grid = np.linspace(0.0, 0.6, 7)
    results = {(float(q), float(d)): [] for q in q_grid for d in topic_damp_grid}
    baseline = []

    for fold in splits:
        train_idx = np.asarray(fold["train_indices"], dtype=int)
        val_idx = np.asarray(fold["val_indices"], dtype=int)
        mult = fold_calibrator(train_idx, train_df, y, p_base, classes)
        p_val = p_base[val_idx] * mult
        q_mask, topic_mask = masks_for_rows(train_idx, val_idx, train_df, classes)
        baseline.append(map3(y[val_idx], p_val, classes))
        non_none = np.ones(len(classes), dtype=bool)
        non_none[classes.index("NONE")] = False

        for q_boost in q_grid:
            for topic_damp in topic_damp_grid:
                adjusted = p_val.copy()
                if q_boost:
                    adjusted[q_mask] *= (1.0 + q_boost)
                if topic_damp:
                    invalid = (~topic_mask) & non_none[None, :]
                    adjusted[invalid] *= (1.0 - topic_damp)
                results[(float(q_boost), float(topic_damp))].append(
                    map3(y[val_idx], adjusted, classes)
                )

    print("Fold-local baseline MAP@3:", ", ".join(f"{x:.5f}" for x in baseline))
    print(f"Mean baseline: {np.mean(baseline):.5f} +/- {np.std(baseline, ddof=1):.5f}")
    print("\nq_boost topic_damp fold_mean fold_sd delta_vs_baseline")
    for (q, d), scores in sorted(results.items()):
        mean = float(np.mean(scores))
        sd = float(np.std(scores, ddof=1))
        print(f"{q:.2f} {d:.2f} {mean:.5f} {sd:.5f} {mean-np.mean(baseline):+.5f}")
    print("\nGrid results are exploratory; choosing the best row on these same folds is optimistic.")


if __name__ == "__main__":
    main()
