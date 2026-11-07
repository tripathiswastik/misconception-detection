"""Audit Flip Stability and Risk vs 0.54949 Peak.

Examine:
How many test rows flip Rank 1 from NONE to Misconception across:
- alpha in [0.20, 0.40, 0.60, 0.80]
- none_mult in [0.975, 0.980, 0.984, 0.990]

Find the sweet spot that maximizes OOF MAP@3 while keeping Rank-1 flips gentle (<= 50-100 flips),
avoiding the catastrophic over-flipping trap.
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

    # Active Peak 0.54949
    peak_path = SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv"
    peak_df = cast(pd.DataFrame, pd.read_csv(peak_path))
    peak_r1 = [p.split()[0] for p in peak_df["root_code"].tolist()]

    test_m18 = np.load(ART_DIR / "test_m18_tuned_convex.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    oof_m18 = np.load(ART_DIR / "oof_m18_tuned_convex.npy")
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

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

    true_counts = Counter(y_true)
    q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in train_df.groupby("question_id")}

    print("=" * 85)
    print(f"{'Alpha':>6s} | {'None_Mult':>9s} | {'Beta':>5s} | {'OOF MAP@3':>10s} | {'Delta vs Peak':>13s} | {'Test R1 Flips':>13s} | {'Test Top1 NONE':>14s}")
    print("=" * 85)

    for alpha in [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80]:
        test_base_core = alpha * test_m18 + (1.0 - alpha) * test_m15
        oof_base_core = alpha * oof_m18 + (1.0 - alpha) * oof_m15

        test_base = 0.95 * test_base_core + 0.02 * test_dual448 + 0.03 * test_comp448
        oof_base = 0.95 * oof_base_core + 0.02 * oof_dual448 + 0.03 * oof_comp448

        top3_tr = np.argsort(-oof_base.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
        mult_base = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult_base[i] = (tc / pc) ** 0.05

        for none_mult in [0.975, 0.980, 0.984]:
            mult = mult_base.copy()
            mult[none_idx] = none_mult

            p_oof_cal = oof_base * mult
            p_oof_cal[:, none_idx] *= 0.9995

            p_oof_boosted = p_oof_cal.copy()
            for fold in splits:
                tr = fold["train_indices"]
                va = fold["val_indices"]
                tr_df = train_df.iloc[tr]
                q_allow_f: Dict[str, Set[str]] = {}
                for q, grp in tr_df.groupby("question_id"):
                    q_allow_f[q] = set(grp["root_code"].dropna().unique())
                for i in va:
                    qid = train_df.iloc[i]["question_id"]
                    allowed_q = q_allow_f.get(qid, set())
                    for ci in range(n_classes):
                        if ci != none_idx and classes[ci] in allowed_q:
                            p_oof_boosted[i, ci] *= 1.20

            p_oof_arc3500_cal = (0.95 * oof_base_core + 0.05 * oof_arc3500) * mult
            p_oof_champ = 0.70 * p_oof_cal + 0.25 * p_oof_boosted + 0.05 * p_oof_arc3500_cal
            p_oof_final = 0.96 * p_oof_champ + 0.04 * p_oof_clean

            # OOF Score
            r1_oof = np.argmax(p_oof_final, axis=1)
            cand_oof = p_oof_final.copy()
            cand_oof[np.arange(n_samples), r1_oof] = -1.0
            cand_oof[r1_oof != none_idx, none_idx] *= 0.82
            top2_oof = np.argsort(-cand_oof, axis=1)[:, :2]
            triplets_oof = [[classes[r1_oof[i]], classes[top2_oof[i, 0]], classes[top2_oof[i, 1]]] for i in range(n_samples)]
            oof_sc = score_triplets(y_true, triplets_oof)

            # Test Matrix
            p_test_cal = test_base * mult
            p_test_cal[:, none_idx] *= 0.9995
            p_test_boosted = np.copy(p_test_cal)
            for i in range(n_test):
                qid = test_df.iloc[i]["question_id"]
                if qid in q_allowed:
                    for ci, c in enumerate(classes):
                        if c != "NONE" and c in q_allowed[qid]:
                            p_test_boosted[i, ci] *= 1.20
            p_test_arc3500_cal = (0.95 * test_base_core + 0.05 * test_arc3500) * mult
            p_test_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal
            p_test_final = 0.96 * p_test_champ + 0.04 * p_test_clean

            r1_test = np.argmax(p_test_final, axis=1)
            r1_test_codes = [classes[idx] for idx in r1_test]
            test_flips = sum(1 for a, b in zip(r1_test_codes, peak_r1) if a != b)
            test_nones = sum(1 for c in r1_test_codes if c == "NONE")

            print(f"{alpha:6.2f} | {none_mult:9.4f} |   0.82 | {oof_sc:10.5f} | {oof_sc - 0.54122:+13.5f} | {test_flips:13d} | {test_nones:6d} ({test_nones/n_test:.1%})")


if __name__ == "__main__":
    main()

