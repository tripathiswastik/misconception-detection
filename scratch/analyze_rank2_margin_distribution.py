"""Analyze Probability Gap and Binary Confidence on the 164 Rank-2 Misconceptions.

For all rows where:
- Model predicts NONE at Rank 1
- True label is sitting at Rank 2
We inspect:
1. Probability gap: P(NONE) - P(Rank 2)
2. Ensemble Binary Detector V2 confidence P(is_error)
3. Precision vs Flip Yield curves
"""
from collections import Counter
import json
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

from src.data.taxonomy import TaxonomyManager

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80, flush=True)
    print("ANALYZING MARGIN & BINARY CONFIDENCE ON RANK-2 MISCONCEPTIONS", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_base_v1 = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    oof_clean = (
        0.6448 * np.load(ART_DIR / "best_adaptive_prior_oof.npy") +
        0.0839 * sharp(np.load(ART_DIR / "honest_oof_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "oof_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "oof_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "oof_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "oof_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "oof_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "oof_clean_arcface_2328.npy"), 0.8)
    )

    oof_bin_v1 = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    oof_bin_v2 = np.load(ART_DIR / "oof_compact_binary_prob_v2.npy")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    oof_final_probs = np.zeros((n_samples, n_classes), dtype=np.float64)

    for fold in splits:
        tr_idx = np.array(fold["train_indices"], dtype=np.int32)
        va_idx = np.array(fold["val_indices"], dtype=np.int32)
        n_eval = len(va_idx)

        top3_tr = np.argsort(-p_base_v1[tr_idx].astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
        true_labels_tr = train_df.iloc[tr_idx]["root_code"].fillna("NONE").astype(str).tolist()
        true_counts = Counter(true_labels_tr)

        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult[i] = (tc / pc) ** 0.05
        mult[none_idx] = 0.9840

        p_eval_cal = p_base_v1[va_idx] * mult
        p_eval_cal[:, none_idx] *= 0.9995

        tr_sub_df = train_df.iloc[tr_idx]
        q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}

        p_eval_boosted = np.copy(p_eval_cal)
        eval_df = train_df.iloc[va_idx].reset_index(drop=True)
        for i in range(n_eval):
            qid = eval_df.iloc[i]["question_id"]
            if qid in q_map:
                allowed = q_map[qid]
                for ci, code in enumerate(classes):
                    if code != "NONE" and code in allowed:
                        p_eval_boosted[i, ci] *= 1.20

        p_eval_arc_cal = (0.95 * oof_m15[va_idx] + 0.05 * oof_v1[va_idx]) * mult
        p_eval_champ = 0.70 * p_eval_cal + 0.25 * p_eval_boosted + 0.05 * p_eval_arc_cal
        oof_final_probs[va_idx] = 0.96 * p_eval_champ + 0.04 * oof_clean[va_idx]

    # Decision Engine Base Ranking
    cand = oof_final_probs.copy()
    row_idx = np.arange(n_samples)
    r1 = np.argmax(cand, axis=1)
    cand[row_idx, r1] = -1.0
    cand[r1 != none_idx, none_idx] *= 0.82
    top2 = np.argsort(-cand, axis=1)[:, :2]

    # Inspect all rows where R1 == NONE
    records = []
    for i in range(n_samples):
        if r1[i] == none_idx:
            gold = y_true[i]
            r2_idx = top2[i, 0]
            r2_code = classes[r2_idx]
            p_none = oof_final_probs[i, none_idx]
            p_r2 = oof_final_probs[i, r2_idx]
            gap = p_none - p_r2

            is_gold_clean = (gold == "NONE")
            is_gold_r2 = (gold == r2_code)
            is_gold_other = (gold != "NONE" and gold != r2_code)

            records.append({
                "idx": i,
                "gold": gold,
                "r2_code": r2_code,
                "gap": gap,
                "p_bin_v1": oof_bin_v1[i],
                "p_bin_v2": oof_bin_v2[i],
                "is_gold_clean": is_gold_clean,
                "is_gold_r2": is_gold_r2,
                "is_gold_other": is_gold_other
            })

    df = pd.DataFrame(records)
    print(f"Total Rows with R1 == NONE: {len(df)}")
    print(f"  True Clean (NONE)         : {df['is_gold_clean'].sum()} ({df['is_gold_clean'].mean():.1%})")
    print(f"  True Misconception is R2  : {df['is_gold_r2'].sum()} ({df['is_gold_r2'].mean():.1%})")
    print(f"  True Misconception is Other: {df['is_gold_other'].sum()} ({df['is_gold_other'].mean():.1%})")

    # Audit Precision and Net Points if we flip R1 (NONE) to R2 when p_bin_v2 >= tau and gap < eps
    print("\n" + "=" * 80)
    print("PRECISION FLIP MATRIX: Net MAP@3 Point Gain on 1500 OOF Rows")
    print("=" * 80)
    print(f"{'tau':6s} | {'eps':6s} | {'Flips':6s} | {'True Misc (R2)':15s} | {'True Clean (Loss)':17s} | {'Net Point Delta':15s}")
    print("-" * 80)

    for tau in [0.70, 0.75, 0.78, 0.80, 0.82, 0.85]:
        for eps in [0.020, 0.030, 0.040, 0.050, 0.060]:
            sub = df[(df["p_bin_v2"] >= tau) & (df["gap"] < eps)]
            n_flips = len(sub)
            if n_flips == 0:
                continue
            n_r2_hit = sub["is_gold_r2"].sum()       # Was R2 (0.50 pts), now R1 (1.00 pt) -> +0.50 each
            n_clean_fa = sub["is_gold_clean"].sum()  # Was R1 (1.00 pt), now R2 (0.50 pts) -> -0.50 each
            n_other = sub["is_gold_other"].sum()     # Was R1 (0.00 pts), now R2 (still 0.00) -> 0.00 delta
            
            # Note: when we flip [NONE, R2, R3] to [R2, NONE, R3]:
            # If gold == R2: score goes from 0.50 to 1.00 -> +0.50 points!
            # If gold == NONE: score goes from 1.00 to 0.50 -> -0.50 points!
            # If gold == other: score remains 0.00 -> 0.00 delta!
            net_pts = n_r2_hit * 0.50 - n_clean_fa * 0.50
            prec = (n_r2_hit / n_flips) * 100.0

            print(f"{tau:6.2f} | {eps:6.3f} | {n_flips:6d} | {n_r2_hit:4d} ({prec:5.1f}%)   | {n_clean_fa:4d}             | {net_pts:+12.2f} pts")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

