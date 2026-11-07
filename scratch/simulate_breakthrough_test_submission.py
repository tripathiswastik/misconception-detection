"""Simulate and Audit Breakthrough Gated Decision Engine on Test Set.

Compares:
1. Active SOTA Peak: blend_optimal_multimodal_theoretical_submission.csv (LB: 0.54899)
2. Gated Theoretical Decision Rule with Lambda = 1.04 - 1.05 and Binary Error Gating
3. Examines Rank-1 flips, Top-1 NONE count, and candidate confidence.
"""
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.fusion.validator import validate_submission_file

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    if abs(t - 1.0) < 1e-5:
        return probs.copy()
    sharp = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    row_sums = sharp.sum(axis=1, keepdims=True)
    return np.divide(sharp, row_sums, where=row_sums > 0)


def simulate():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)

    # 1. Load active peak champion (0.54899)
    peak_path = SUB_DIR / "blend_optimal_multimodal_theoretical_submission.csv"
    peak_df = cast(pd.DataFrame, pd.read_csv(peak_path))
    peak_preds = peak_df["root_code"].tolist()
    peak_r1 = [p.split()[0] for p in peak_preds]
    peak_none_count = sum(1 for r in peak_r1 if r == "NONE")
    print(f"Active SOTA Peak (0.54899) Top-1 NONE Count: {peak_none_count} / {n_test} ({peak_none_count / n_test:.2%})")

    # 2. Build Champion Test Probability Foundation
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_test_champ_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    # Frequency calibration multiplier
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = (
        0.95 * oof_m15 +
        0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") +
        0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    )
    from collections import Counter
    top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_test_54616 = p_test_champ_base * mult
    p_test_54616[:, none_idx] *= 0.9995

    # Transductive question conditioning
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_54616)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    p_test_champ_ensemble = 0.85 * p_test_54616 + 0.10 * p_test_boosted + 0.05 * p_test_arc3500_cal

    # Clean Multimodal Test Probability Matrix
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    test_cn = np.load(ART_DIR / "test_convnext_vision.npy")
    test_swin = np.load(ART_DIR / "test_swin_vision.npy")
    test_mn = np.load(ART_DIR / "test_finetuned_vision.npy")
    test_th = np.load(ART_DIR / "test_text.npy")
    test_ts = np.load(ART_DIR / "test_semantic_text.npy")
    test_trocr = np.load(ART_DIR / "test_trocr_vision.npy")
    test_arc_clean = np.load(ART_DIR / "test_clean_arcface_2328.npy")

    test_th_sharp = apply_temperature(test_th, 0.6)
    test_ts_sharp = apply_temperature(test_ts, 0.6)
    test_arc_sharp = apply_temperature(test_arc_clean, 0.8)

    p_test_clean = (
        0.6448 * test_prior_probs +
        0.0839 * test_th_sharp +
        0.0401 * test_ts_sharp +
        0.0458 * test_cn +
        0.0454 * test_swin +
        0.0483 * test_mn +
        0.0455 * test_trocr +
        0.0462 * test_arc_sharp
    )

    p_final = 0.92 * p_test_champ_ensemble + 0.08 * p_test_clean

    # Load Binary Error Detector Test Probabilities
    test_p_error = np.load(ART_DIR / "test_binary_error_prob.npy")

    # 3. Simulate Reranking with Lambdas
    secondary_beta = 0.82
    print("\n" + "=" * 80)
    print("SIMULATING GATED DECISION ENGINE AT VARIOUS THRESHOLDS")
    print("=" * 80)

    for lam in [1.20, 1.15, 1.10, 1.05, 1.04, 1.00, 0.95]:
        new_preds = []
        for i in range(n_test):
            row_p = p_final[i].copy()
            p_none = row_p[none_idx]

            row_p_nom = row_p.copy()
            row_p_nom[none_idx] = -1.0
            top_cand_idx = int(np.argmax(row_p_nom))
            p_cand = row_p_nom[top_cand_idx]

            sum_cand = np.sum(row_p[row_p > 0]) - p_none
            cond_cand = p_cand / max(sum_cand, 1e-12)

            expected_gain = test_p_error[i] * cond_cand - lam * p_none

            # Condition: If expected_gain > 0 and model has high error confidence
            if expected_gain > 0 and top_cand_idx != none_idx:
                r1_idx = top_cand_idx
                row_p[r1_idx] = -1.0
                row_p[none_idx] *= secondary_beta
                rem_top2 = np.argsort(-row_p)[:2]
                r2_idx, r3_idx = rem_top2[0], rem_top2[1]
            else:
                top3_idx = np.argsort(-row_p)[:3]
                r1_idx, r2_idx, r3_idx = top3_idx[0], top3_idx[1], top3_idx[2]

            new_preds.append(f"{classes[r1_idx]} {classes[r2_idx]} {classes[r3_idx]}")

        r1_new = [p.split()[0] for p in new_preds]
        none_cnt = sum(1 for r in r1_new if r == "NONE")
        r1_diff = sum(1 for a, b in zip(r1_new, peak_r1) if a != b)
        tot_diff = sum(1 for a, b in zip(new_preds, peak_preds) if a != b)

        print(f"Lambda={lam:.2f} -> NONE R1: {none_cnt} ({none_cnt/n_test:.2%}) | R1 Diffs vs 0.54899: {r1_diff:3d} | Total Triplet Diffs: {tot_diff:3d}")


if __name__ == "__main__":
    simulate()

