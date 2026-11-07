"""Audit Fine-Grained Gating Thresholds around the 0.55049 Winning Formula.

Winning formula:
- Active baseline: 0.54949 peak
- tau = 0.78, eps = 0.04 -> 11 test flips -> 0.55049 Public Leaderboard!

Let's test:
- tau in [0.765, 0.770, 0.775, 0.780]
- eps in [0.030, 0.035, 0.040, 0.045, 0.050]

Report:
1. Exact 5-fold OOF MAP@3 and number of train flips.
2. Exact number of test flips vs 0.55049.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"


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
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)
    n_test = len(test_df)

    # Reconstruct 0.54949 Baseline
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

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

    p_oof_boosted = p_oof_cal.copy()
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
                    p_oof_boosted[i, ci] *= 1.20

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

    p_champ_549 = 0.70 * p_oof_cal + 0.25 * p_oof_boosted + 0.05 * p_oof_arc3500_cal
    p_final_549 = 0.96 * p_champ_549 + 0.04 * p_oof_clean

    row_idx = np.arange(n_samples)
    r1_549 = np.argmax(p_final_549, axis=1)
    cand_549 = p_final_549.copy()
    cand_549[row_idx, r1_549] = -1.0
    cand_549[r1_549 != none_idx, none_idx] *= 0.82
    top2_549 = np.argsort(-cand_549, axis=1)[:, :2]
    triplets_549 = [[classes[r1_549[i]], classes[top2_549[i, 0]], classes[top2_549[i, 1]]] for i in range(n_samples)]
    base_score = score_triplets(y_true, triplets_549)

    # Test Matrix Reconstitution
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")
    p_test_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448
    p_test_cal = p_test_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_allowed[qid]:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    p_test_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal

    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)
    p_test_clean = (
        0.6448 * test_prior_probs +
        0.0839 * sharp(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )
    p_test_final = 0.96 * p_test_champ + 0.04 * p_test_clean
    test_r1_549 = np.argmax(p_test_final, axis=1)

    # Binary error probabilities
    p_bin_oof = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    p_bin_test = np.load(ART_DIR / "test_compact_binary_prob.npy")

    print(f"Active Baseline 5-Fold OOF MAP@3: {base_score:.5f}")
    print("=" * 80)
    print(f"{'Tau':>6s} | {'Epsilon':>8s} | {'OOF MAP@3':>10s} | {'Delta vs Peak':>13s} | {'Train Flips':>11s} | {'Test Flips':>10s}")
    print("=" * 80)

    for tau in [0.765, 0.770, 0.775, 0.780]:
        for eps in [0.030, 0.035, 0.040, 0.045, 0.050]:
            # CV evaluation
            trips_mod = [list(t) for t in triplets_549]
            train_flips = 0

            for f_idx, fold in enumerate(splits):
                va = fold["val_indices"]
                q_allow = q_allowed_folds[f_idx]

                for i in va:
                    if r1_549[i] == none_idx and p_bin_oof[i] >= tau:
                        row_p = p_final_549[i].copy()
                        p_none_val = row_p[none_idx]
                        row_p[none_idx] = -1.0
                        top_idx = int(np.argmax(row_p))
                        top_code = classes[top_idx]
                        p_misc_val = row_p[top_idx]
                        gap = p_none_val - p_misc_val

                        if 0 < gap < eps:
                            qid = train_df.iloc[i]["question_id"]
                            if top_code in q_allow.get(qid, set()):
                                row_p[top_idx] = -1.0
                                second_idx = int(np.argmax(row_p))
                                trips_mod[i] = [top_code, "NONE", classes[second_idx]]
                                train_flips += 1

            sc = score_triplets(y_true, trips_mod)

            # Test evaluation
            test_flips = 0
            for i in range(n_test):
                if test_r1_549[i] == none_idx and p_bin_test[i] >= tau:
                    row_p = p_test_final[i].copy()
                    p_none_val = row_p[none_idx]
                    row_p[none_idx] = -1.0
                    top_idx = int(np.argmax(row_p))
                    top_code = classes[top_idx]
                    p_misc_val = row_p[top_idx]
                    gap = p_none_val - p_misc_val

                    if 0 < gap < eps:
                        qid = test_df.iloc[i]["question_id"]
                        if top_code in q_allowed.get(qid, set()):
                            test_flips += 1

            flag = " *" if sc > base_score else ""
            print(f"{tau:6.3f} | {eps:8.3f} | {sc:10.5f} | {sc - base_score:+13.5f} | {train_flips:11d} | {test_flips:10d}{flag}")


if __name__ == "__main__":
    main()

