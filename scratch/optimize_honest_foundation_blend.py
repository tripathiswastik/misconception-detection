"""Optimize Foundation Blend Weights on 100% Leak-Free Honest Pipeline.

Components available:
1. best_adaptive_prior_oof (0.51022)
2. oof_arcface_pseudo_3500 (0.46233)
3. oof_semantic_text (0.44956)
4. oof_arcface_composite_448 (0.42467)
5. oof_arcface_dualhead_448_soft (0.41678)
6. honest_oof_text (0.39044)
7. oof_swin_vision (0.38889)
8. oof_convnext_vision (0.38800)
9. oof_trocr_vision (0.36789)

Objective:
Find the convex combination w = [w_prior, w_arc3500, w_sem, w_comp, w_dual, w_txt, w_swin, w_cn, w_trocr]
that maximizes Honest OOF MAP@3 across the 5 canonical folds with zero leakage.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

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

    # 1. Load All 9 Honest Components
    c_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    c_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    c_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    c_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    c_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    c_text = np.load(ART_DIR / "honest_oof_text.npy")
    c_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    c_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    c_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    component_list = [c_prior, c_arc3500, c_sem, c_comp, c_dual, c_text, c_swin, c_cn, c_trocr]
    comp_names = ["Prior", "Arc3500", "SemText", "ArcComp", "ArcDual", "HonestText", "Swin", "ConvNeXt", "TrOCR"]

    # In-fold question allowed lists and fold indices precomputation
    fold_info = []
    for f in splits:
        tr_idx = f["train_indices"]
        va_idx = f["val_indices"]
        tr_df = train_df.iloc[tr_idx]
        q_allowed = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allowed[q] = set(grp["root_code"].dropna().unique())
        fold_info.append({
            "tr_idx": tr_idx,
            "va_idx": va_idx,
            "q_allowed": q_allowed,
            "y_tr": [y_true[i] for i in tr_idx],
            "y_va": [y_true[i] for i in va_idx]
        })

    def evaluate_weights(weights: np.ndarray, beta: float = 0.82, boost_factor: float = 1.20) -> float:
        w = np.maximum(weights, 0.0)
        w_sum = w.sum()
        if w_sum <= 0:
            return 0.0
        w = w / w_sum

        fused_oof = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, c_mat in enumerate(component_list):
            if w[i] > 0:
                fused_oof += w[i] * c_mat

        triplets = [[] for _ in range(n_samples)]

        for f in fold_info:
            tr_idx = f["tr_idx"]
            va_idx = f["va_idx"]
            q_allow = f["q_allowed"]
            y_tr = f["y_tr"]

            p_base_tr = fused_oof[tr_idx]
            p_base_va = fused_oof[va_idx]

            # In-fold mult
            top3_tr = np.argsort(-p_base_tr, axis=1)[:, :3]
            pred_counts_tr = Counter(classes[idx] for row in top3_tr for idx in row)
            true_counts_tr = Counter(y_tr)
            mult = np.ones(n_classes, dtype=np.float64)
            for ci, c in enumerate(classes):
                if c != "NONE":
                    pc = max(pred_counts_tr.get(c, 1), 1)
                    tc = max(true_counts_tr.get(c, 1), 1)
                    mult[ci] = (tc / pc) ** 0.05
            mult[none_idx] = 0.9840

            p_va_cal = p_base_va * mult
            p_va_cal[:, none_idx] *= 0.9995

            # In-fold question boost
            p_va_boosted = p_va_cal.copy()
            for loc_i, glob_i in enumerate(va_idx):
                qid = train_df.iloc[glob_i]["question_id"]
                allow_q = q_allow.get(qid, set())
                for ci in range(n_classes):
                    if ci != none_idx and classes[ci] in allow_q:
                        p_va_boosted[loc_i, ci] *= boost_factor

            p_final_va = 0.75 * p_va_cal + 0.25 * p_va_boosted

            loc_rows = np.arange(len(va_idx))
            r1_va = np.argmax(p_final_va, axis=1)
            cand_va = p_final_va.copy()
            cand_va[loc_rows, r1_va] = -1.0
            cand_va[r1_va != none_idx, none_idx] *= beta
            top2_va = np.argsort(-cand_va, axis=1)[:, :2]

            for loc_i, glob_i in enumerate(va_idx):
                triplets[glob_i] = [classes[r1_va[loc_i]], classes[top2_va[loc_i, 0]], classes[top2_va[loc_i, 1]]]

        return score_triplets(y_true, triplets)

    # 1. Baseline Weights (Approximating M15 in honest space)
    # [Prior: 0.45, Arc3500: 0.05, SemText: 0.18, ArcComp: 0.03, ArcDual: 0.02, HonestText: 0.18, Swin: 0.03, ConvNeXt: 0.03, TrOCR: 0.03]
    w_baseline = np.array([0.45, 0.05, 0.18, 0.03, 0.02, 0.18, 0.03, 0.03, 0.03])
    base_score = evaluate_weights(w_baseline)
    print("=" * 80)
    print(f"HONEST M15-LIKE BASELINE OOF MAP@3: {base_score:.5f}")
    print("=" * 80)

    # 2. Hypothesis Testing: What if we increase the weight of strong honest models?
    # ArcFace Pseudo 3500 (honest 0.46233) from 0.05 -> 0.15
    # ArcFace Composite 448 (honest 0.42467) from 0.03 -> 0.10
    # Text SGD reduced from 0.18 -> 0.08
    print("\nSweeping weight configurations:")
    print("-" * 80)

    configs = [
        ("Base M15-Like", [0.45, 0.05, 0.18, 0.03, 0.02, 0.18, 0.03, 0.03, 0.03]),
        ("Higher ArcFace Pseudo (10%)", [0.42, 0.10, 0.18, 0.05, 0.03, 0.12, 0.03, 0.04, 0.03]),
        ("Higher ArcFace Pseudo (15%)", [0.40, 0.15, 0.16, 0.06, 0.03, 0.10, 0.03, 0.04, 0.03]),
        ("Heavy Metric Learning (20% Arc3500, 10% ArcComp)", [0.35, 0.20, 0.15, 0.10, 0.04, 0.06, 0.03, 0.04, 0.03]),
        ("Prior + ArcFace Heavy", [0.45, 0.20, 0.12, 0.08, 0.03, 0.04, 0.03, 0.03, 0.02]),
        ("Prior + SemText + Arc3500 Balanced", [0.45, 0.15, 0.18, 0.08, 0.02, 0.04, 0.03, 0.03, 0.02]),
        ("Zero Noisy Text SGD (Prior + Sem + Vision + ArcFace)", [0.48, 0.18, 0.18, 0.08, 0.02, 0.00, 0.02, 0.02, 0.02]),
    ]

    best_score = base_score
    best_name = "Base M15-Like"
    best_w = w_baseline

    for name, w_cand in configs:
        sc = evaluate_weights(np.array(w_cand))
        flag = " *" if sc > best_score else ""
        if sc > best_score:
            best_score = sc
            best_name = name
            best_w = np.array(w_cand)
        print(f"{name:55s} | MAP@3: {sc:.5f} (Delta: {sc - base_score:+.5f}){flag}")

    print("=" * 80)
    print(f"WINNING CONFIGURATION: {best_name} | MAP@3: {best_score:.5f} (Delta: {best_score - base_score:+.5f})")
    print(f"Weights: {list(zip(comp_names, best_w))}")
    print("=" * 80)


if __name__ == "__main__":
    main()

