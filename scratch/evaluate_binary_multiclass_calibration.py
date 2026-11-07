"""Evaluate Binary Meta-Learner Calibration with Multiclass Ensemble on 5-Fold OOF.

Mathematical Fusion:
P*(NONE) = (1 - w_bin) * p_none + w_bin * (1 - q_error)
P*(c) = p_c * [ (1 - w_bin) + w_bin * q_error / (sum_{k != NONE} p_k + 1e-12) ] for c != NONE.

Combined with:
- Question Empirical Boost alpha_q in [1.00, 1.05, 1.10]
- Secondary NONE recalibration beta = 0.82

Metric: MAP@3 across 1,500 train rows.
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
    print("=" * 80)
    print("EVALUATING BINARY META-LEARNER MULTICLASS CALIBRATION")
    print("=" * 80)

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

    # Baseline MAP@3
    row_idx = np.arange(n_samples)
    r1_base = np.argmax(p_oof_cal, axis=1)
    cand_base = p_oof_cal.copy()
    cand_base[row_idx, r1_base] = -1.0
    cand_base[r1_base != none_idx, none_idx] *= 0.82
    top2_base = np.argsort(-cand_base, axis=1)[:, :2]
    base_triplets = [[classes[r1_base[i]], classes[top2_base[i, 0]], classes[top2_base[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, base_triplets)
    print(f"Honest Baseline OOF MAP@3: {base_score:.5f}")

    # Load Compact Binary Error Probabilities
    q_oof = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    print(f"Loaded Compact Binary Error Probabilities: min={q_oof.min():.4f}, mean={q_oof.mean():.4f}, max={q_oof.max():.4f}")

    # Precompute Question Mask
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

    # Sweep w_bin and alpha_q
    w_bin_vals = [0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20]
    alpha_q_vals = [1.00, 1.05, 1.10]

    best_score = base_score
    best_config = (0.0, 1.00)

    print("\nGrid Search over Binary Weight w_bin and Question Boost alpha_q:")
    print("-" * 80)

    for w_bin in w_bin_vals:
        # Fuse multiclass with binary probability
        p_fused = p_oof_cal.copy()
        if w_bin > 0:
            p_none_orig = p_fused[:, none_idx]
            p_misc_sum = 1.0 - p_none_orig
            p_fused[:, none_idx] = (1.0 - w_bin) * p_none_orig + w_bin * (1.0 - q_oof)

            # Rescale misconceptions
            misc_scale = (1.0 - w_bin) + w_bin * (q_oof / np.maximum(p_misc_sum, 1e-6))
            for ci in range(n_classes):
                if ci != none_idx:
                    p_fused[:, ci] *= misc_scale

        for aq in alpha_q_vals:
            p_cand = p_fused.copy()
            if aq != 1.0:
                p_cand[mask_q_boost] *= aq

            r1 = np.argmax(p_cand, axis=1)
            cand = p_cand.copy()
            cand[row_idx, r1] = -1.0
            cand[r1 != none_idx, none_idx] *= 0.82
            top2 = np.argsort(-cand, axis=1)[:, :2]

            triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
            score = score_triplets(y_true, triplets)
            r1_flips = sum(1 for a, b in zip(r1, r1_base) if a != b)

            flag = " *" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_config = (w_bin, aq)
            print(f"w_bin = {w_bin:.2f}, alpha_q = {aq:.2f} | MAP@3: {score:.5f} (Delta: {score - base_score:+.5f}, R1 flips: {r1_flips:2d}){flag}")

    print("=" * 80)
    print(f"OPTIMAL CONFIGURATION: w_bin = {best_config[0]:.2f}, alpha_q = {best_config[1]:.2f} | Peak MAP@3: {best_score:.5f} (Delta: {best_score - base_score:+.5f})")


if __name__ == "__main__":
    main()

