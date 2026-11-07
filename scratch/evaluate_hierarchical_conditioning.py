"""Fast Vectorized Evaluation of Hierarchical Bayesian Topic & Question Conditioning on 5-Fold OOF.

Hierarchy:
1. Exact Question Match: If misconception c was observed for question_id in train fold -> boost by alpha_q
2. Out-of-Topic Prefix: If prefix(c) was NEVER observed for topic_name in train fold AND not in question -> dampen by delta_topic
3. NONE is NEVER dampened or boosted (retains calibrated probability).
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
    prefixes = [c.split("-")[0].upper() if c != "NONE" else "NONE" for c in classes]

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Frequency calibration multiplier
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

    # Baseline score with beta = 0.82
    row_idx = np.arange(n_samples)
    r1_base = np.argmax(p_oof_cal, axis=1)
    cand_base = p_oof_cal.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]
    base_triplets = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, base_triplets)
    print(f"Honest Baseline OOF MAP@3: {base_score:.5f}")

    # PRECOMPUTE BOOLEAN MASKS ACROSS THE ENTIRE DATASET
    # mask_q_boost[i, c] == True if c was seen in question for row i
    # mask_topic_damp[i, c] == True if c is NOT in question AND prefix(c) NOT seen in topic for row i
    mask_q_boost = np.zeros((n_samples, n_classes), dtype=bool)
    mask_topic_damp = np.zeros((n_samples, n_classes), dtype=bool)

    for fold in splits:
        tr = fold["train_indices"]
        va = fold["val_indices"]
        tr_df = train_df.iloc[tr]

        q_allowed: Dict[str, Set[str]] = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allowed[q] = set(grp["root_code"].dropna().unique())

        topic_prefixes: Dict[str, Set[str]] = {}
        for t, grp in tr_df.groupby("topic_name"):
            codes = grp["root_code"].dropna().unique()
            topic_prefixes[t] = {c.split("-")[0].upper() for c in codes if c != "NONE"}

        for i in va:
            qid = train_df.iloc[i]["question_id"]
            t = train_df.iloc[i]["topic_name"]
            allowed_q = q_allowed.get(qid, set())
            allowed_pref = topic_prefixes.get(t, set())

            for ci in range(n_classes):
                if ci == none_idx:
                    continue
                c = classes[ci]
                pref = prefixes[ci]
                if c in allowed_q:
                    mask_q_boost[i, ci] = True
                elif pref not in allowed_pref:
                    mask_topic_damp[i, ci] = True

    print(f"Precomputed masks:")
    print(f"  Total Question-boosted slots: {int(mask_q_boost.sum())} / {n_samples * (n_classes - 1)}")
    print(f"  Total Out-of-topic dampened slots: {int(mask_topic_damp.sum())} / {n_samples * (n_classes - 1)}")

    # FAST VECTORIZED GRID SEARCH
    alpha_q_vals = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40]
    delta_topic_vals = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]

    best_score = base_score
    best_params = (1.00, 1.00)

    for aq in alpha_q_vals:
        for dt in delta_topic_vals:
            # Vectorized multiplier
            mult_matrix = np.ones((n_samples, n_classes), dtype=np.float64)
            if aq != 1.0:
                mult_matrix[mask_q_boost] = aq
            if dt != 1.0:
                mult_matrix[mask_topic_damp] = dt

            p_oof_cond = p_oof_cal * mult_matrix

            # Rank with beta = 0.82
            r1_cond = np.argmax(p_oof_cond, axis=1)
            cand_cond = p_oof_cond.copy()
            cand_cond[row_idx, r1_cond] = -1.0
            cand_cond[r1_cond != none_idx, none_idx] *= 0.82
            top2_cond = np.argsort(-cand_cond, axis=1)[:, :2]
            cond_triplets = [[classes[r1_cond[i]], classes[top2_cond[i, 0]], classes[top2_cond[i, 1]]] for i in range(n_samples)]
            score = score_triplets(y_true, cond_triplets)

            r1_flips = sum(1 for a, b in zip(r1_cond, r1_base) if a != b)
            if score > best_score:
                best_score = score
                best_params = (aq, dt)
                print(f"--> [NEW BEST] alpha_q={aq:.2f}, delta_topic={dt:.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R1 flips: {r1_flips})")
            else:
                print(f"    alpha_q={aq:.2f}, delta_topic={dt:.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R1 flips: {r1_flips})")

    print("=" * 80)
    print(f"OPTIMAL HIERARCHICAL CONDITIONING: alpha_q={best_params[0]:.2f}, delta_topic={best_params[1]:.2f} | Peak MAP@3: {best_score:.5f} (Delta: {best_score - base_score:+.5f})")


if __name__ == "__main__":
    main()
