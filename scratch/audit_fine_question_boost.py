"""Fine-grained sweep of effective question boost alpha_q on full champion OOF matrix.

Compare against baseline 0.54011.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
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

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    from collections import Counter
    top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_oof_cal = p_oof_base * mult
    p_oof_cal[:, none_idx] *= 0.9995

    row_idx = np.arange(n_samples)
    r1_base = np.argmax(p_oof_cal, axis=1)
    cand_base = p_oof_cal.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]
    base_triplets = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, base_triplets)
    print(f"Baseline OOF MAP@3: {base_score:.5f}")

    mask_q_boost = np.zeros((n_samples, n_classes), dtype=bool)
    for fold in splits:
        tr = fold["train_indices"]
        va = fold["val_indices"]
        tr_df = train_df.iloc[tr]
        q_allowed: Dict[str, Set[str]] = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allowed[q] = set(grp["root_code"].dropna().unique())
        for i in va:
            qid = train_df.iloc[i]["question_id"]
            allowed_q = q_allowed.get(qid, set())
            for ci in range(n_classes):
                if ci != none_idx and classes[ci] in allowed_q:
                    mask_q_boost[i, ci] = True

    # Grid search over alpha_q with 0.01 step size from 1.00 to 1.15
    print("\nFine-grained sweep of alpha_q:")
    best_score = base_score
    best_aq = 1.00

    for aq in np.arange(1.00, 1.16, 0.01):
        mult_matrix = np.ones((n_samples, n_classes), dtype=np.float64)
        mult_matrix[mask_q_boost] = aq
        p_boosted = p_oof_cal * mult_matrix

        r1 = np.argmax(p_boosted, axis=1)
        cand = p_boosted.copy()
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]

        triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        score = score_triplets(y_true, triplets)
        flips = sum(1 for a, b in zip(r1, r1_base) if a != b)

        flag = " *" if score > best_score else ""
        if score > best_score:
            best_score = score
            best_aq = aq
        print(f"alpha_q = {aq:.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R1 flips: {flips:2d}){flag}")

    print("=" * 80)
    print(f"Optimal alpha_q: {best_aq:.2f} (Peak MAP@3: {best_score:.5f}, Delta: {best_score - base_score:+.5f})")


if __name__ == "__main__":
    main()

