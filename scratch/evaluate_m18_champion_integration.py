"""Evaluate Integrating Model M18 (OOF 0.54822) into the Champion Architecture.

Compare:
1. Baseline Champion with M15 (Current OOF: 0.54122)
2. Champion replacing M15 with M18
3. Convex blend of M15 and M18
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
    oof_m18 = np.load(ART_DIR / "oof_m18_tuned_convex.npy")
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

    true_counts = Counter(y_true)

    print("=" * 80)
    print("M18 vs M15 INTEGRATION BENCHMARK ACROSS CONVEX MIXTURES")
    print("=" * 80)

    # Test alpha * M18 + (1 - alpha) * M15
    for alpha in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        base_core = alpha * oof_m18 + (1.0 - alpha) * oof_m15
        p_oof_base = 0.95 * base_core + 0.02 * oof_dual448 + 0.03 * oof_comp448

        top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
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
                        p_oof_boosted[i, ci] *= 1.20

        p_oof_arc3500_cal = (0.95 * base_core + 0.05 * oof_arc3500) * mult
        p_champ = 0.70 * p_oof_cal + 0.25 * p_oof_boosted + 0.05 * p_oof_arc3500_cal
        p_final = 0.96 * p_champ + 0.04 * p_oof_clean

        row_idx = np.arange(n_samples)
        r1 = np.argmax(p_final, axis=1)
        cand = p_final.copy()
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]
        triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        sc = score_triplets(y_true, triplets)
        r1_none = sum(1 for r in r1 if r == none_idx)
        print(f"Alpha M18 = {alpha:4.2f} (M15 = {1.0-alpha:4.2f}) | OOF MAP@3: {sc:.5f} | R1 NONE: {r1_none:4d}")


if __name__ == "__main__":
    main()

