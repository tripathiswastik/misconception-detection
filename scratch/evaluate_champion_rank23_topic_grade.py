"""Evaluate Topic-Grade Prior Re-weighting on the Actual SOTA 550 Champion Pipeline.

Invariants:
1. Pipeline Alignment: Uses the actual SOTA 550 Champion predictions.
2. Rank 1 Locked: Rank 1 prediction is 100% strictly anchored and immutable.
3. Zero Leakage: Evaluated with strictly in-fold topic-grade distributions.
4. Gate: Promote only if it improves held-out MAP@3 without hurting baseline.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80)
    print("EVALUATING TOPIC-GRADE PRIOR RE-WEIGHTING ON SOTA 550 CHAMPION (RANK 1 LOCKED)")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values
    n_samples = len(train_df)

    # 1. Load Champion Components
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    honest_text = np.load(ART_DIR / "honest_oof_text.npy")
    oof_sem_text = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_convnext = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_finetuned = np.load(ART_DIR / "oof_finetuned_vision.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    oof_clean_arcface = np.load(ART_DIR / "oof_clean_arcface_2328.npy")
    oof_bin_err = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    p_base_all = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # 2. Build Champion Baseline Across 5-Fold GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    champion_probs = np.zeros((n_samples, n_classes), dtype=np.float64)
    champion_top3: List[List[str]] = [[] for _ in range(n_samples)]
    champion_r1_is_none = np.zeros(n_samples, dtype=bool)

    for fold_num, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
        trn_df = train_df.iloc[trn_idx]
        val_df = train_df.iloc[val_idx]
        n_val = len(val_idx)

        prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
        val_prior = prior_eng.predict_proba(val_df).astype(np.float64)

        p_base_trn = p_base_all[trn_idx]
        top3_trn = np.argsort(-p_base_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df["root_code"].fillna("NONE").values)
        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[none_idx] = 0.9840

        p_val_cal = p_base_all[val_idx] * mult
        p_val_cal[:, none_idx] *= 0.9995

        q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in trn_df.groupby("question_id")}
        p_val_boosted = np.copy(p_val_cal)
        for i in range(n_val):
            qid = val_df.iloc[i]["question_id"]
            if qid in q_allowed:
                for ci, c in enumerate(classes):
                    if c != "NONE" and c in q_allowed[qid]:
                        p_val_boosted[i, ci] *= 1.20

        p_val_arc3500_cal = (0.95 * oof_m15[val_idx] + 0.05 * oof_arc3500[val_idx]) * mult
        p_champ_val = 0.70 * p_val_cal + 0.25 * p_val_boosted + 0.05 * p_val_arc3500_cal

        p_clean_val = (
            0.6448 * val_prior +
            0.0839 * sharp(honest_text[val_idx], 0.6) +
            0.0401 * sharp(oof_sem_text[val_idx], 0.6) +
            0.0458 * oof_convnext[val_idx] +
            0.0454 * oof_swin[val_idx] +
            0.0483 * oof_finetuned[val_idx] +
            0.0455 * oof_trocr[val_idx] +
            0.0462 * sharp(oof_clean_arcface[val_idx], 0.8)
        )

        p_final_val = 0.96 * p_champ_val + 0.04 * p_clean_val
        champion_probs[val_idx] = p_final_val

        val_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_final_val, axis=1)]

        p_bin_err_val = oof_bin_err[val_idx]
        for i in range(n_val):
            if val_top3[i][0] == "NONE" and p_bin_err_val[i] >= 0.78:
                row_p = p_final_val[i].copy()
                p_none_val = row_p[none_idx]
                row_p[none_idx] = -1.0
                top_misc_idx = int(np.argmax(row_p))
                top_misc_code = classes[top_misc_idx]
                if 0 < (p_none_val - row_p[top_misc_idx]) < 0.04:
                    qid = val_df.iloc[i]["question_id"]
                    if top_misc_code in q_allowed.get(qid, set()):
                        row_p[top_misc_idx] = -1.0
                        second_misc = classes[int(np.argmax(row_p))]
                        val_top3[i] = [top_misc_code, "NONE", second_misc]

        for loc_i, glob_i in enumerate(val_idx):
            champion_top3[glob_i] = val_top3[loc_i]
            champion_r1_is_none[glob_i] = (val_top3[loc_i][0] == "NONE")

    base_map3 = calculate_map3(y_raw.tolist(), champion_top3)
    r1_none_count = int(champion_r1_is_none.sum())
    print(f"SOTA 550 Champion Baseline MAP@3: {base_map3:.5f}")
    print(f"Total Rows with Rank 1 == NONE   : {r1_none_count} / {n_samples} ({r1_none_count/n_samples:.1%})")

    # 3. Test In-Fold Topic-Grade Prior Re-weighting for Rank 2 vs Rank 3
    # Candidate score: S(c) = P_champ(c)^alpha * P_topic_grade(c)^(1 - alpha)
    # Applied ONLY to Rank 2 & Rank 3 (Rank 1 strictly locked!)
    print("\n--- Evaluating Topic-Grade Prior Re-weighting across alphas ---")
    for alpha in [0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50]:
        new_top3 = [list(r) for r in champion_top3]
        total_swaps = 0
        helped = 0
        hurt = 0
        neutral = 0

        for fold_num, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
            val_prior = prior_eng.predict_proba(val_df)

            for loc_i, glob_i in enumerate(val_idx):
                if champion_r1_is_none[glob_i]:
                    r2_code = champion_top3[glob_i][1]
                    r3_code = champion_top3[glob_i][2]
                    c2_idx = class_to_idx[r2_code]
                    c3_idx = class_to_idx[r3_code]

                    p_c2 = champion_probs[glob_i, c2_idx]
                    p_c3 = champion_probs[glob_i, c3_idx]

                    p_tg2 = max(val_prior[loc_i, c2_idx], 1e-6)
                    p_tg3 = max(val_prior[loc_i, c3_idx], 1e-6)

                    s2 = (p_c2 ** alpha) * (p_tg2 ** (1.0 - alpha))
                    s3 = (p_c3 ** alpha) * (p_tg3 ** (1.0 - alpha))

                    if s3 > s2:
                        new_top3[glob_i][1] = r3_code
                        new_top3[glob_i][2] = r2_code
                        total_swaps += 1

                        true_y = y_raw[glob_i]
                        if true_y == r3_code:
                            helped += 1
                        elif true_y == r2_code:
                            hurt += 1
                        else:
                            neutral += 1

        score = calculate_map3(y_raw.tolist(), new_top3)
        delta = score - base_map3
        prec = (helped / max(1, helped + hurt)) if (helped + hurt) > 0 else 0.0
        print(f"alpha = {alpha:.2f} | Swaps: {total_swaps:3d} (Helped: {helped:2d}, Hurt: {hurt:2d}, Neutral: {neutral:3d}, Prec: {prec:4.1%}) | MAP@3: {score:.5f} ({delta:+.5f})")

    # 4. Test Topic-Grade Prior Re-weighting with Confidence Gap Threshold (delta_gap)
    # Only swap if gap = p_c2 - p_c3 < max_gap
    print("\n--- Evaluating Topic-Grade Prior Re-weighting (alpha=0.90) with Max Gap Gating ---")
    alpha = 0.90
    for max_gap in [0.010, 0.015, 0.020, 0.030, 0.050]:
        new_top3 = [list(r) for r in champion_top3]
        total_swaps = 0
        helped = 0
        hurt = 0
        neutral = 0

        for fold_num, (trn_idx, val_idx) in enumerate(outer_gkf.split(train_df, y_indices, groups=groups), start=1):
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            prior_eng = AdaptiveQuestionPriorEngine(classes=classes).fit(trn_df)
            val_prior = prior_eng.predict_proba(val_df)

            for loc_i, glob_i in enumerate(val_idx):
                if champion_r1_is_none[glob_i]:
                    r2_code = champion_top3[glob_i][1]
                    r3_code = champion_top3[glob_i][2]
                    c2_idx = class_to_idx[r2_code]
                    c3_idx = class_to_idx[r3_code]

                    p_c2 = champion_probs[glob_i, c2_idx]
                    p_c3 = champion_probs[glob_i, c3_idx]
                    gap = p_c2 - p_c3

                    if 0 < gap < max_gap:
                        p_tg2 = max(val_prior[loc_i, c2_idx], 1e-6)
                        p_tg3 = max(val_prior[loc_i, c3_idx], 1e-6)

                        s2 = (p_c2 ** alpha) * (p_tg2 ** (1.0 - alpha))
                        s3 = (p_c3 ** alpha) * (p_tg3 ** (1.0 - alpha))

                        if s3 > s2:
                            new_top3[glob_i][1] = r3_code
                            new_top3[glob_i][2] = r2_code
                            total_swaps += 1

                            true_y = y_raw[glob_i]
                            if true_y == r3_code:
                                helped += 1
                            elif true_y == r2_code:
                                hurt += 1
                            else:
                                neutral += 1

        score = calculate_map3(y_raw.tolist(), new_top3)
        delta = score - base_map3
        prec = (helped / max(1, helped + hurt)) if (helped + hurt) > 0 else 0.0

        # Calculate per-fold deltas
        fold_deltas = []
        for trn_idx, val_idx in outer_gkf.split(train_df, y_indices, groups=groups):
            v_y = y_raw[val_idx].tolist()
            b_p = [champion_top3[i] for i in val_idx]
            n_p = [new_top3[i] for i in val_idx]
            fold_deltas.append(calculate_map3(v_y, n_p) - calculate_map3(v_y, b_p))

        print(f"max_gap = {max_gap:.3f} | Swaps: {total_swaps:3d} (Helped: {helped:2d}, Hurt: {hurt:2d}, Neutral: {neutral:3d}, Prec: {prec:4.1%}) | MAP@3: {score:.5f} ({delta:+.5f}) | Folds: {[round(d, 5) for d in fold_deltas]}")

    print("=" * 80)

if __name__ == "__main__":
    main()

