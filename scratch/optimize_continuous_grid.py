"""Optimize Continuous Multimodal Hyperparameters on OOF MAP@3.

Sweep grid around the 0.54899 peak:
1. blend_weight_clean in [0.03, 0.05, 0.08, 0.10, 0.12]
2. secondary_beta in [0.75, 0.78, 0.82, 0.85, 0.88]
3. transductive_boost in [1.10, 1.15, 1.20, 1.25]
4. Evaluates OOF MAP@3 and counts Rank-1 changes vs Champion foundation.
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
from src.evaluation.cross_validator import ModelCrossValidator
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    if abs(t - 1.0) < 1e-5:
        return probs.copy()
    sharp = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    row_sums = sharp.sum(axis=1, keepdims=True)
    return np.divide(sharp, row_sums, where=row_sums > 0)


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

    # 1. Champion Base OOF Probabilities
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_oof_champ_base = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Frequency calibration
    from collections import Counter
    top3_train = np.argsort(-p_oof_champ_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_oof_54616 = p_oof_champ_base * mult
    p_oof_54616[:, none_idx] *= 0.9995

    # Fold-local transductive boosting
    p_oof_boosted = np.copy(p_oof_54616)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        tr_df = train_df.iloc[tr]
        q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_df.groupby("question_id")}
        for i in va:
            qid = train_df.iloc[i]["question_id"]
            if qid in q_map:
                allowed = q_map[qid]
                for ci, code in enumerate(classes):
                    if code != "NONE" and code in allowed:
                        p_oof_boosted[i, ci] *= 1.20

    p_oof_arc3500_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult
    p_oof_champ_ensemble = 0.85 * p_oof_54616 + 0.10 * p_oof_boosted + 0.05 * p_oof_arc3500_cal

    # Clean Multimodal OOF
    oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")

    oof_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_mn = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_th = np.load(ART_DIR / "honest_oof_text.npy")
    oof_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_arc_clean = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    oof_th_sharp = apply_temperature(oof_th, 0.6)
    oof_ts_sharp = apply_temperature(oof_ts, 0.6)
    oof_arc_sharp = apply_temperature(oof_arc_clean, 0.8)

    p_oof_clean = (
        0.6448 * oof_prior +
        0.0839 * oof_th_sharp +
        0.0401 * oof_ts_sharp +
        0.0458 * oof_cn +
        0.0454 * oof_swin +
        0.0483 * oof_mn +
        0.0455 * oof_trocr +
        0.0462 * oof_arc_sharp
    )

    print("=" * 80)
    print("CONTINUOUS MULTIMODAL GRID SEARCH AROUND 0.54899 PEAK")
    print("=" * 80)

    best_score = 0.0
    best_params = {}

    row_idx = np.arange(n_samples)

    for w_clean in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        p_fused = (1.0 - w_clean) * p_oof_champ_ensemble + w_clean * p_oof_clean
        r1_indices = np.argmax(p_fused, axis=1)

        for beta in [0.75, 0.80, 0.82, 0.85, 0.90, 1.00]:
            cand = p_fused.copy()
            cand[row_idx, r1_indices] = -1.0
            is_non_none = (r1_indices != none_idx)
            cand[is_non_none, none_idx] *= beta

            part2 = np.argpartition(-cand, kth=1, axis=1)[:, :2]
            part2_p = cand[row_idx[:, None], part2]
            order2 = np.argsort(-part2_p, axis=1, kind="mergesort")
            top2 = part2[row_idx[:, None], order2]

            triplets = [[classes[r1_indices[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
            score = score_triplets(y_true, triplets)

            r1_none_cnt = sum(1 for r in r1_indices if r == none_idx)

            if score > best_score:
                best_score = score
                best_params = {
                    "w_clean": w_clean,
                    "beta": beta,
                    "r1_none_count": r1_none_cnt,
                    "oof_map3": score
                }
                print(f"New Best -> w_clean: {w_clean:.2f} | beta: {beta:.2f} | NONE R1: {r1_none_cnt} | OOF MAP@3: {score:.5f}")

    print("\n" + "=" * 80)
    print("OPTIMAL CONTINUOUS CONFIGURATION:")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    print("=" * 80)


if __name__ == "__main__":
    main()

