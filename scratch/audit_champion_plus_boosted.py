"""Evaluate Full 0.54899 Architecture with w_boosted = 0.30 + w_clean = 0.08 in 5-Fold CV.

Compare:
1. Active SOTA Peak formulation (w_boost = 0.10, w_clean = 0.08, beta = 0.82)
2. Upgraded formulation (w_boost = 0.30, w_clean = 0.08, beta = 0.82)
3. Grid of w_clean in [0.04, 0.06, 0.08, 0.10]
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

    # 1. Base Champion OOF
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

    # 2. Transductive Question Boost Matrix (1.20)
    p_oof_boosted_120 = p_oof_cal.copy()
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
                    p_oof_boosted_120[i, ci] *= 1.20

    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    p_oof_arc3500_cal = (0.95 * oof_m15 + 0.05 * oof_arc3500) * mult

    # 3. Clean Multimodal OOF
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

    row_idx = np.arange(n_samples)

    # 4. Active Peak Formulation (0.54899 architecture: w_boost=0.10, w_clean=0.08)
    p_champ_548 = 0.85 * p_oof_cal + 0.10 * p_oof_boosted_120 + 0.05 * p_oof_arc3500_cal
    p_final_548 = 0.92 * p_champ_548 + 0.08 * p_oof_clean
    r1_548 = np.argmax(p_final_548, axis=1)
    cand_548 = p_final_548.copy()
    cand_548[row_idx, r1_548] = -1.0
    cand_548[r1_548 != none_idx, none_idx] *= 0.82
    top2_548 = np.argsort(-cand_548, axis=1)[:, :2]
    triplets_548 = [[classes[r1_548[i]], classes[top2_548[i, 0]], classes[top2_548[i, 1]]] for i in range(n_samples)]
    score_548 = score_triplets(y_true, triplets_548)
    print(f"Active 0.54899 OOF MAP@3 Baseline: {score_548:.5f}")

    # 5. Grid search around w_boosted in [0.20, 0.30, 0.40] and w_clean in [0.04, 0.06, 0.08, 0.10]
    print("\nSweeping w_boosted and w_clean:")
    print("-" * 80)
    best_score = score_548
    best_params = (0.10, 0.08)

    for wb in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40]:
        w_base = 0.95 - wb
        p_champ = w_base * p_oof_cal + wb * p_oof_boosted_120 + 0.05 * p_oof_arc3500_cal

        for wc in [0.04, 0.06, 0.08, 0.10]:
            p_final = (1.0 - wc) * p_champ + wc * p_oof_clean

            r1 = np.argmax(p_final, axis=1)
            cand = p_final.copy()
            cand[row_idx, r1] = -1.0
            cand[r1 != none_idx, none_idx] *= 0.82
            top2 = np.argsort(-cand, axis=1)[:, :2]

            triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
            score = score_triplets(y_true, triplets)
            r1_diffs = sum(1 for a, b in zip(r1, r1_548) if a != b)

            flag = " *" if score > best_score else ""
            if score > best_score:
                best_score = score
                best_params = (wb, wc)
            print(f"w_boosted={wb:.2f}, w_clean={wc:.2f} | MAP@3: {score:.5f} (Delta vs 0.54899: {score - score_548:+.5f}, R1 flips: {r1_diffs:2d}){flag}")

    print("=" * 80)
    print(f"OPTIMAL COMBINATION: w_boosted={best_params[0]:.2f}, w_clean={best_params[1]:.2f} | Peak MAP@3: {best_score:.5f} (Delta: {best_score - score_548:+.5f})")


if __name__ == "__main__":
    main()

