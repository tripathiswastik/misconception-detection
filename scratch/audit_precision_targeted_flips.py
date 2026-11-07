"""Audit Precision-Targeted Rank-1 Flips on 5-Fold OOF.

Hypothesis:
Instead of flipping 100-200 rows with a blunt threshold, find the ultra-narrow, high-precision
subset of rows where:
1. Baseline Champion predicts NONE at Rank 1
2. Compact Meta-Learner P(is_error) >= tau (high confidence of error)
3. Top candidate misconception c_top was observed for question_id in train fold
4. Probability gap P(NONE) - P(c_top) < epsilon (close race)

We test:
tau in [0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
epsilon in [0.01, 0.02, 0.03, 0.05, 0.08]

Metric: Exact MAP@3 on 1,500 train rows.
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

    # 1. Load SOTA 0.54949 Probability Matrix
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

    # Transductive Question Boost Matrix (1.20)
    p_oof_boosted_120 = p_oof_cal.copy()
    q_allowed_folds = []
    for fold in splits:
        tr = fold["train_indices"]
        va = fold["val_indices"]
        tr_df = train_df.iloc[tr]
        q_allowed: Dict[str, Set[str]] = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allowed[q] = set(grp["root_code"].dropna().unique())
        q_allowed_folds.append(q_allowed)
        for i in va:
            qid = train_df.iloc[i]["question_id"]
            allowed_q = q_allowed.get(qid, set())
            for ci in range(n_classes):
                if ci != none_idx and classes[ci] in allowed_q:
                    p_oof_boosted_120[i, ci] *= 1.20

    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    p_oof_arc3500_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult

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

    p_champ_549 = 0.70 * p_oof_cal + 0.25 * p_oof_boosted_120 + 0.05 * p_oof_arc3500_cal
    p_final_549 = 0.96 * p_champ_549 + 0.04 * p_oof_clean

    row_idx = np.arange(n_samples)
    r1_549 = np.argmax(p_final_549, axis=1)
    cand_549 = p_final_549.copy()
    cand_549[row_idx, r1_549] = -1.0
    cand_549[r1_549 != none_idx, none_idx] *= 0.82
    top2_549 = np.argsort(-cand_549, axis=1)[:, :2]
    triplets_549 = [[classes[r1_549[i]], classes[top2_549[i, 0]], classes[top2_549[i, 1]]] for i in range(n_samples)]
    score_baseline = score_triplets(y_true, triplets_549)
    print(f"Active 0.54949 SOTA Peak OOF Baseline: {score_baseline:.5f}")

    # Load Compact Binary Meta-Learner (AUC 0.6289)
    p_bin_error = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    # Evaluate Precision-Targeted Flips
    print("\nAuditing Precision-Targeted Flips across (tau, epsilon):")
    print("-" * 80)
    best_score = score_baseline
    best_params = (0.0, 0.0)

    tau_vals = [0.65, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80]
    eps_vals = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.12]

    for tau in tau_vals:
        for eps in eps_vals:
            triplets_mod = [list(t) for t in triplets_549]
            flips = 0

            for f_idx, fold in enumerate(splits):
                va = fold["val_indices"]
                q_allowed = q_allowed_folds[f_idx]

                for i in va:
                    # Condition 1: Baseline predicts NONE at Rank 1
                    if r1_549[i] == none_idx:
                        # Condition 2: Meta-learner has high error confidence
                        if p_bin_error[i] >= tau:
                            # Find top misconception
                            row_p = p_final_549[i].copy()
                            p_none_val = row_p[none_idx]
                            row_p[none_idx] = -1.0
                            top_misc_idx = int(np.argmax(row_p))
                            top_misc_code = classes[top_misc_idx]
                            p_misc_val = row_p[top_misc_idx]

                            # Condition 3: Close race (gap < eps)
                            gap = p_none_val - p_misc_val
                            if 0 < gap < eps:
                                # Condition 4: Question allowed
                                qid = train_df.iloc[i]["question_id"]
                                if top_misc_code in q_allowed.get(qid, set()):
                                    # Target flip: Elevate top misconception to Rank 1, NONE to Rank 2, Rank 3 is second misc
                                    row_p[top_misc_idx] = -1.0
                                    second_misc_idx = int(np.argmax(row_p))
                                    second_misc_code = classes[second_misc_idx]

                                    triplets_mod[i] = [top_misc_code, "NONE", second_misc_code]
                                    flips += 1

            score = score_triplets(y_true, triplets_mod)
            flag = " *" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_params = (tau, eps)
            print(f"tau={tau:.2f}, eps={eps:.2f} | MAP@3: {score:.5f} (Delta: {score - score_baseline:+.5f}, Flips: {flips:2d}){flag}")

    print("=" * 80)
    print(f"Optimal Targeted Gating: tau={best_params[0]:.2f}, eps={best_params[1]:.2f} | Peak MAP@3: {best_score:.5f} (Delta: {best_score - score_baseline:+.5f})")


if __name__ == "__main__":
    main()

