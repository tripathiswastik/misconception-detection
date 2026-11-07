"""Evaluate Rank-2 and Rank-3 ranking strategies specifically on NONE rows in 5-fold CV.

For rows where Champion predicts NONE at Rank 1:
- If ground truth is NONE, MAP@3 is 1.0 regardless of R2/R3.
- If ground truth is a Misconception (false-NONE row), MAP@3 is:
    - 0.50 if true misconception is at Rank 2
    - 0.33 if true misconception is at Rank 3
    - 0.00 otherwise.

We test:
1. Baseline Champion ranking
2. Clean Multimodal ranking (w = 1.0)
3. Blended ranking for various weights w in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # 1. Load Champion OOF
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Frequency calibration multiplier
    from collections import Counter
    top3_train = np.argsort(-p_oof_champ.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_oof_champ_cal = p_oof_champ * mult
    p_oof_champ_cal[:, none_idx] *= 0.9995

    # 2. Load Clean Multimodal OOF
    oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    oof_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_mn = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_th = np.load(ART_DIR / "oof_text.npy")
    oof_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_arc = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    def sharp(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return s / s.sum(axis=1, keepdims=True)

    p_oof_clean = (
        0.6448 * oof_prior +
        0.0839 * sharp(oof_th, 0.6) +
        0.0401 * sharp(oof_ts, 0.6) +
        0.0458 * oof_cn +
        0.0454 * oof_swin +
        0.0483 * oof_mn +
        0.0455 * oof_trocr +
        0.0462 * sharp(oof_arc, 0.8)
    )

    # 3. Identify rows where Champion predicts NONE at Rank 1
    r1_champ = np.argmax(p_oof_champ_cal, axis=1)
    none_rows = (r1_champ == none_idx)
    n_none_rows = int(np.sum(none_rows))
    print(f"Total rows: {n_samples}")
    print(f"Champion Rank-1 NONE rows: {n_none_rows} ({n_none_rows / n_samples:.2%})")

    # How many of these NONE rows actually have a misconception ground truth?
    y_true_arr = np.array(y_true)
    none_rows_gt_misc = (none_rows & (y_true_arr != "NONE"))
    n_false_none = int(np.sum(none_rows_gt_misc))
    print(f"False-NONE rows (where R2/R3 matters): {n_false_none} / {n_none_rows} ({n_false_none / n_none_rows:.2%})")

    # Evaluate R2/R3 MAP@3 specifically on these rows
    # For a row i in none_rows:
    # If y_true[i] == 'NONE', MAP@3 is 1.00 for all strategies.
    # If y_true[i] != 'NONE', MAP@3 is: 0.5 if r2 == y_true, 0.3333 if r3 == y_true, else 0.0

    weights = [0.00, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50, 0.70, 1.00]
    print("\nEvaluating Blend Weights w for p_eval = (1 - w) * p_champ + w * p_clean on NONE rows:")
    print("-" * 80)

    best_w = 0.0
    best_score = 0.0

    for w in weights:
        p_blend = (1.0 - w) * p_oof_champ_cal + w * p_oof_clean
        # Zero out NONE
        cand = p_blend.copy()
        cand[:, none_idx] = -1.0
        # Get top 2
        top2 = np.argsort(-cand, axis=1)[:, :2]

        # Compute MAP@3 across all n_samples
        scores = []
        for i in range(n_samples):
            yt = y_true[i]
            if r1_champ[i] == none_idx:
                # Rank 1 is NONE
                if yt == "NONE":
                    scores.append(1.0)
                else:
                    r2 = classes[top2[i, 0]]
                    r3 = classes[top2[i, 1]]
                    if r2 == yt:
                        scores.append(0.5)
                    elif r3 == yt:
                        scores.append(1.0 / 3.0)
                    else:
                        scores.append(0.0)
            else:
                # Misconception row (keep champion baseline behavior with beta=0.82)
                r1_c = classes[r1_champ[i]]
                if yt == r1_c:
                    scores.append(1.0)
                else:
                    # R2 is NONE
                    if yt == "NONE":
                        scores.append(0.5)
                    else:
                        # R3 is top secondary misconception
                        cand_i = cand[i].copy()
                        cand_i[r1_champ[i]] = -1.0
                        r3_c = classes[int(np.argmax(cand_i))]
                        if r3_c == yt:
                            scores.append(1.0 / 3.0)
                        else:
                            scores.append(0.0)

        map3 = float(np.mean(scores))
        delta = map3 - scores[0]  # will compare with baseline w=0
        print(f"Weight w = {w:.2f} | Overall OOF MAP@3: {map3:.5f}")
        if map3 > best_score:
            best_score = map3
            best_w = w

    print("-" * 80)
    print(f"Optimal w on NONE rows: {best_w:.2f} (Peak MAP@3: {best_score:.5f})")


if __name__ == "__main__":
    main()
