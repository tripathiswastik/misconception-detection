"""Diagnose Exact Lost Points Breakdown on 100% Leak-Free OOF.

Categorizes all lost points into:
1. False Alarm: True = NONE, Pred Rank 1 = Misconception
2. Missed Misconception: True = Misconception, Pred Rank 1 = NONE
3. Misconception Confusion: True = Misc A, Pred Rank 1 = Misc B
4. Rank-2/3 degradation: True was in top 3 but not rank 1
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80, flush=True)
    print("OOF ERROR TAXONOMY & LOST POINTS DECOMPOSITION", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")

    p_base_v1 = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    oof_clean = (
        0.6448 * np.load(ART_DIR / "best_adaptive_prior_oof.npy") +
        0.0839 * sharp(np.load(ART_DIR / "honest_oof_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "oof_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "oof_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "oof_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "oof_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "oof_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "oof_clean_arcface_2328.npy"), 0.8)
    )

    oof_bin = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]

    outer_preds = [[] for _ in range(n_samples)]

    for fold in splits:
        tr_idx = np.array(fold["train_indices"], dtype=np.int32)
        va_idx = np.array(fold["val_indices"], dtype=np.int32)
        n_eval = len(va_idx)

        top3_tr = np.argsort(-p_base_v1[tr_idx].astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
        true_labels_tr = train_df.iloc[tr_idx]["root_code"].fillna("NONE").astype(str).tolist()
        true_counts = Counter(true_labels_tr)

        mult = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult[i] = (tc / pc) ** 0.05
        mult[none_idx] = 0.9840

        p_eval_cal = p_base_v1[va_idx] * mult
        p_eval_cal[:, none_idx] *= 0.9995

        tr_sub_df = train_df.iloc[tr_idx]
        q_map = {q: set(grp["root_code"].dropna()) for q, grp in tr_sub_df.groupby("question_id")}

        p_eval_boosted = np.copy(p_eval_cal)
        eval_df = train_df.iloc[va_idx].reset_index(drop=True)
        for i in range(n_eval):
            qid = eval_df.iloc[i]["question_id"]
            if qid in q_map:
                allowed = q_map[qid]
                for ci, code in enumerate(classes):
                    if code != "NONE" and code in allowed:
                        p_eval_boosted[i, ci] *= 1.20

        p_eval_arc_cal = (0.95 * oof_m15[va_idx] + 0.05 * oof_v1[va_idx]) * mult
        p_eval_champ = 0.70 * p_eval_cal + 0.25 * p_eval_boosted + 0.05 * p_eval_arc_cal
        p_eval_final = 0.96 * p_eval_champ + 0.04 * oof_clean[va_idx]

        cand = p_eval_final.copy()
        row_idx = np.arange(n_eval)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_eval)]

        for i in range(n_eval):
            if preds[i][0] == "NONE" and oof_bin[va_idx[i]] >= 0.780:
                row_p = p_eval_final[i].copy()
                p_none_val = row_p[none_idx]
                row_p[none_idx] = -1.0
                top_misc_idx = int(np.argmax(row_p))
                top_misc_code = classes[top_misc_idx]
                gap = p_none_val - row_p[top_misc_idx]

                if 0 < gap < 0.040:
                    qid = eval_df.iloc[i]["question_id"]
                    if top_misc_code in q_map.get(qid, set()):
                        row_p[top_misc_idx] = -1.0
                        second_misc_idx = int(np.argmax(row_p))
                        second_misc_code = classes[second_misc_idx]
                        preds[i] = [top_misc_code, "NONE", second_misc_code]

        for local_i, global_i in enumerate(va_idx):
            outer_preds[global_i] = preds[local_i]

    # Analyze Error Categories
    scores = []
    cat_loss = {"false_alarm_none": 0.0, "missed_none": 0.0, "misc_confusion": 0.0, "rank2_hit": 0.0, "rank3_hit": 0.0, "rank1_hit": 0.0}
    cat_counts = Counter()

    for i in range(n_samples):
        gold = y_true[i]
        pred = outer_preds[i]

        if gold == pred[0]:
            point = 1.0
            cat_counts["rank1_hit"] += 1
        elif gold == pred[1]:
            point = 0.5
            cat_counts["rank2_hit"] += 1
            cat_loss["rank2_hit"] += 0.5  # lost 0.5 points
        elif gold == pred[2]:
            point = 1.0 / 3.0
            cat_counts["rank3_hit"] += 1
            cat_loss["rank3_hit"] += (2.0 / 3.0)  # lost 0.67 points
        else:
            point = 0.0
            # Total miss (lost 1.0 point)
            if gold == "NONE" and pred[0] != "NONE":
                cat_counts["false_alarm_none"] += 1
                cat_loss["false_alarm_none"] += 1.0
            elif gold != "NONE" and pred[0] == "NONE":
                cat_counts["missed_none"] += 1
                cat_loss["missed_none"] += 1.0
            else:
                cat_counts["misc_confusion"] += 1
                cat_loss["misc_confusion"] += 1.0

        scores.append(point)

    total_map3 = np.mean(scores)
    total_lost_points = (1.0 - total_map3) * n_samples

    print(f"Overall Leak-Free OOF MAP@3 : {total_map3:.5f}")
    print(f"Total Lost Points across 1500: {total_lost_points:.2f} points\n")

    print(f"{'Category':30s} | {'Sample Count':12s} | {'Lost Points':12s} | {'% of Total Lost Points':22s}")
    print("-" * 82)
    for cat in ["misc_confusion", "false_alarm_none", "missed_none", "rank2_hit", "rank3_hit"]:
        lp = cat_loss[cat]
        cnt = cat_counts[cat]
        pct = (lp / total_lost_points) * 100.0
        print(f"{cat:30s} | {cnt:12d} | {lp:12.2f} | {pct:21.1f}%")

    # Analyze NONE-At-Rank-1 cases specifically
    none_r1_cases = []
    for i in range(n_samples):
        gold = y_true[i]
        pred = outer_preds[i]
        if pred[0] == "NONE":
            none_r1_cases.append({
                "idx": i,
                "gold": gold,
                "pred": pred,
                "gold_is_clean": (gold == "NONE"),
                "gold_in_rank2": (gold == pred[1]),
                "gold_in_rank3": (gold == pred[2]),
                "gold_total_miss": (gold not in pred)
            })

    df_none = pd.DataFrame(none_r1_cases)
    print("\n" + "=" * 80)
    print(f"DEEP DIVE: All Rows where Model Predicts NONE at Rank 1 ({len(df_none)} rows):")
    print("=" * 80)
    print(f"  True is Clean (NONE) [CORRECT CALL]          : {sum(df_none['gold_is_clean'])} / {len(df_none)} ({sum(df_none['gold_is_clean'])/len(df_none):.1%})")
    print(f"  True is Misconception, sitting at Rank 2      : {sum(df_none['gold_in_rank2'])} / {len(df_none)} ({sum(df_none['gold_in_rank2'])/len(df_none):.1%}) [Lost 0.50 pts each]")
    print(f"  True is Misconception, sitting at Rank 3      : {sum(df_none['gold_in_rank3'])} / {len(df_none)} ({sum(df_none['gold_in_rank3'])/len(df_none):.1%}) [Lost 0.67 pts each]")
    print(f"  True is Misconception, not in Top 3 (Total Miss): {sum(df_none['gold_total_miss'])} / {len(df_none)} ({sum(df_none['gold_total_miss'])/len(df_none):.1%}) [Lost 1.00 pt each]")

    # Total lost points solely inside the NONE-at-Rank-1 bucket:
    lost_in_none = sum(df_none['gold_in_rank2']) * 0.50 + sum(df_none['gold_in_rank3']) * (2.0/3.0) + sum(df_none['gold_total_miss']) * 1.00
    print(f"\n  Total Lost Points when Model calls NONE at Rank 1: {lost_in_none:.2f} points ({lost_in_none/total_lost_points:.1%} of ALL lost points!)")

if __name__ == "__main__":
    main()

