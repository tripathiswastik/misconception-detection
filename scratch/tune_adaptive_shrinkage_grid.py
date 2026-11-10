"""Fine-grained grid search for adaptive shrinkage parameters (m_q, power) on test-mix splits."""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple, cast
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
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).values
    y_indices = np.array([class_to_idx[c] for c in y_raw], dtype=np.int64)
    groups = train_df["question_id"].values

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

    # Splits
    q_counts = train_df["question_id"].value_counts()
    multi_qids = set(q_counts[q_counts >= 2].index)
    multi_mask = train_df["question_id"].isin(multi_qids).values
    multi_indices = np.where(multi_mask)[0]

    seen_folds: List[Tuple[np.ndarray, np.ndarray]] = []
    np.random.seed(42)
    fold_assignments = np.zeros(len(multi_indices), dtype=int)
    for qid in multi_qids:
        q_locs = np.where(train_df.iloc[multi_indices]["question_id"] == qid)[0]
        perm = np.random.permutation(len(q_locs))
        for rank, p_idx in enumerate(perm):
            fold_assignments[q_locs[p_idx]] = rank % 5

    for f in range(5):
        val_loc = np.where(fold_assignments == f)[0]
        trn_loc = np.where(fold_assignments != f)[0]
        val_glob = multi_indices[val_loc]
        single_indices = np.where(~multi_mask)[0]
        trn_glob = np.concatenate([multi_indices[trn_loc], single_indices])
        seen_folds.append((trn_glob, val_glob))

    outer_gkf = GroupKFold(n_splits=5)
    unseen_folds = list(outer_gkf.split(train_df, y_indices, groups=groups))

    def eval_seen_unseen(m_q: float, power: float) -> Tuple[float, float, float]:
        # Seen
        seen_y = []
        seen_preds = []
        for trn_idx, val_idx in seen_folds:
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            n_val = len(val_idx)

            engine = AdaptiveQuestionPriorEngine(classes=classes, m_q_base=m_q, m_tg=5.0, adaptive_nq=True, adaptive_power=power).fit(trn_df)
            val_prior = engine.predict_proba(val_df).astype(np.float64)

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

            seen_y.extend(y_raw[val_idx].tolist())
            seen_preds.extend(val_top3)

        seen_score = calculate_map3(seen_y, seen_preds)

        # Unseen
        unseen_y = []
        unseen_preds = []
        for trn_idx, val_idx in unseen_folds:
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]

            engine = AdaptiveQuestionPriorEngine(classes=classes, m_q_base=m_q, m_tg=5.0, adaptive_nq=True, adaptive_power=power).fit(trn_df)
            val_prior = engine.predict_proba(val_df).astype(np.float64)

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

            p_val_boosted = np.copy(p_val_cal)
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
            val_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_final_val, axis=1)]

            unseen_y.extend(y_raw[val_idx].tolist())
            unseen_preds.extend(val_top3)

        unseen_score = calculate_map3(unseen_y, unseen_preds)
        composite_score = 0.957 * seen_score + 0.043 * unseen_score
        return seen_score, unseen_score, composite_score

    # Reference Baseline
    base_seen = 0.54311
    base_comp = 0.54281

    print(f"{'m_q':<6} | {'power':<6} | {'Seen':<8} | {'Unseen':<8} | {'Composite':<10} | {'Delta':<10}")
    print("-" * 65)

    results = []
    for mq in [6.0, 7.0, 8.0, 9.0, 10.0]:
        for p in [0.5, 0.6, 0.7, 0.8]:
            sn, un, cp = eval_seen_unseen(mq, p)
            delta = cp - base_comp
            print(f"{mq:<6.1f} | {p:<6.1f} | {sn:<8.5f} | {un:<8.5f} | {cp:<10.5f} | {delta:<+10.5f}")
            results.append({"mq": mq, "power": p, "seen": sn, "unseen": un, "composite": cp, "delta": delta})

    best = max(results, key=lambda x: x["composite"])
    print(f"\nBest Config: m_q={best['mq']}, power={best['power']} -> Composite={best['composite']:.5f} (Delta={best['delta']:+.5f})")

if __name__ == "__main__":
    main()

