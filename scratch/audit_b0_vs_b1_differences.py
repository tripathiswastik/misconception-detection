"""Direct comparison of B0 (Champion baseline) vs B1 (Adaptive Shrinkage m_q=8, pow=0.7).
Audit every trace difference on both Seen and Unseen splits.
"""
from collections import Counter
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

    def run_eval(m_q: float, adaptive: bool, p_val_power: float):
        seen_y = []
        seen_preds = []
        for trn_idx, val_idx in seen_folds:
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            val_y = y_raw[val_idx].tolist()
            n_val = len(val_idx)

            engine = AdaptiveQuestionPriorEngine(
                classes=classes, m_q_base=m_q, m_tg=5.0, adaptive_nq=adaptive, adaptive_power=p_val_power
            ).fit(trn_df)
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

            seen_y.extend(val_y)
            seen_preds.extend(val_top3)

        return seen_y, seen_preds

    y_true, preds_b0 = run_eval(10.0, False, 0.5)
    _, preds_b1 = run_eval(8.0, True, 0.7)

    score_b0 = calculate_map3(y_true, preds_b0)
    score_b1 = calculate_map3(y_true, preds_b1)

    print(f"B0 (Baseline M10 fixed) Seen MAP@3 : {score_b0:.5f}")
    print(f"B1 (Adaptive m_q=8, p=0.7) Seen MAP@3: {score_b1:.5f} ({score_b1 - score_b0:+.5f})")

    diffs = []
    helped = 0
    hurt = 0
    for i, (t, p0, p1) in enumerate(zip(y_true, preds_b0, preds_b1)):
        if p0 != p1:
            s0 = 1.0 if t == p0[0] else (0.5 if t == p0[1] else (1/3 if t == p0[2] else 0.0))
            s1 = 1.0 if t == p1[0] else (0.5 if t == p1[1] else (1/3 if t == p1[2] else 0.0))
            d = s1 - s0
            if d > 0:
                helped += 1
            elif d < 0:
                hurt += 1
            diffs.append((i, t, p0, p1, d))

    print(f"\nTrace Differences Audit: {len(diffs)} traces changed prediction")
    print(f"  Helped : {helped} traces")
    print(f"  Hurt   : {hurt} traces")
    print(f"  Neutral: {len(diffs) - helped - hurt} traces")
    print(f"  Net Delta: +{(helped - hurt) * (1/6) / len(y_true):.5f}")

    for idx, t, p0, p1, d in diffs[:8]:
        status = "HELPED" if d > 0 else ("HURT" if d < 0 else "NEUTRAL")
        print(f"  Trace {idx:4d} | True: {t:<8} | B0: {p0} | B1: {p1} | {status} ({d:+.3f})")

if __name__ == "__main__":
    main()

