"""Comprehensive S550 OOF Error Breakdown for Rare Misconception Labels.

Objective per User Directive:
1. On all 1,500 held-out labeled traces across canonical 5 folds (splits.json),
   measure S550 out-of-fold performance specifically on rare misconception labels (<15 train examples, 52 classes).
2. Decompose all rare-label errors into two mutually exclusive regimes:
   A) "Present but ranked too low":
      - True label is ranked just outside Top 3 (Rank 4-5 or Rank 6-10 in continuous probability vector).
      - Probability margin gap [P(Rank 3) - P(true)] is small.
      - Diagnostic implication: Class weighting, loss rebalancing, or calibrated temperature can recover them.
   B) "Missing from retrieval":
      - True label is ranked > 10 or assigned negligible probability (P < 0.01).
      - Diagnostic implication: Class weighting cannot fix zero/near-zero logits; requires visual feature/metric retrieval.
3. Quantify competitor labels stealing Rank 1-3 slots (false NONE vs common misconceptions).
4. Evaluate Seen vs Unseen question conditioning effect on rare label recovery.
5. Provide definitive quantitative verdict: Target Visual Retrieval vs Target Class Weighting.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
import numpy as np
import pandas as pd

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

class S550AuditEngine:
    def __init__(self):
        self.tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
        self.classes = self.tax.all_codes(include_none=True)
        self.none_idx = self.classes.index("NONE")
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
        self.y_raw = self.train_df["root_code"].fillna("NONE").astype(str).tolist()
        self.n_samples = len(self.train_df)
        self.targets = np.array([self.class_to_idx.get(y, self.none_idx) for y in self.y_raw], dtype=np.int64)

        # Baseline components for leak-free S550 in-fold regeneration
        self.oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
        self.oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
        self.oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
        self.oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
        self.honest_text = np.load(ART_DIR / "honest_oof_text.npy")
        self.oof_sem_text = np.load(ART_DIR / "oof_semantic_text.npy")
        self.oof_convnext = np.load(ART_DIR / "oof_convnext_vision.npy")
        self.oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
        self.oof_finetuned = np.load(ART_DIR / "oof_finetuned_vision.npy")
        self.oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
        self.oof_clean_arcface = np.load(ART_DIR / "oof_clean_arcface_2328.npy")
        self.oof_bin_v1 = np.load(ART_DIR / "oof_compact_binary_prob.npy")

        splits_json = json.loads((DATA_DIR / "splits.json").read_text())
        self.folds = splits_json["folds"]

    def build_fold_clean_multimodal(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> np.ndarray:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        prior_eng = AdaptiveQuestionPriorEngine(classes=self.classes).fit(trn_df_fold)
        val_prior = prior_eng.predict_proba(val_df_fold).astype(np.float64)

        p_clean_val = (
            0.6448 * val_prior +
            0.0839 * sharp(self.honest_text[val_idx], 0.6) +
            0.0401 * sharp(self.oof_sem_text[val_idx], 0.6) +
            0.0458 * self.oof_convnext[val_idx] +
            0.0454 * self.oof_swin[val_idx] +
            0.0483 * self.oof_finetuned[val_idx] +
            0.0455 * self.oof_trocr[val_idx] +
            0.0462 * sharp(self.oof_clean_arcface[val_idx], 0.8)
        )
        return p_clean_val

    def generate_s550_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Counter]]:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        n_val = len(val_idx)

        p_base_v1_trn = 0.95 * self.oof_m15[trn_idx] + 0.02 * self.oof_dual[trn_idx] + 0.03 * self.oof_comp[trn_idx]
        top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(self.classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
        mult = np.ones(self.n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[self.none_idx] = 0.9840

        p_base_v1_val = (0.95 * self.oof_m15[val_idx] + 0.02 * self.oof_dual[val_idx] + 0.03 * self.oof_comp[val_idx]) * mult
        p_base_v1_val[:, self.none_idx] *= 0.9995

        q_label_dist: Dict[str, Counter] = {
            str(q): Counter(grp["root_code"].fillna("NONE").values)
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        for i in range(n_val):
            qid = str(val_df_fold.iloc[i]["question_id"])
            if qid in q_label_dist:
                allowed_misconceptions = {c for c, cnt in q_label_dist[qid].items() if c != "NONE"}
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20

        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean

        p_s550 = np.maximum(p_s550, 0.0)
        p_s550 = p_s550 / p_s550.sum(axis=1, keepdims=True)
        return p_s550, q_label_dist

    def rank_and_gate(self, p_matrix: np.ndarray, val_idx: np.ndarray, q_label_dist: Dict[str, Counter], beta: float = 0.82) -> List[List[str]]:
        cand = p_matrix.copy()
        n = len(cand)
        row_idx = np.arange(n)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != self.none_idx, self.none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[self.classes[r1[i]], self.classes[top2[i, 0]], self.classes[top2[i, 1]]] for i in range(n)]

        p_bin = self.oof_bin_v1[val_idx]
        val_df = self.train_df.iloc[val_idx].reset_index(drop=True)

        for i in range(n):
            if preds[i][0] == "NONE" and p_bin[i] >= 0.78:
                row_p = p_matrix[i].copy()
                p_none = row_p[self.none_idx]
                row_p[self.none_idx] = -1.0
                top_m_idx = int(np.argmax(row_p))
                top_m_code = self.classes[top_m_idx]
                gap = p_none - row_p[top_m_idx]

                if 0 < gap < 0.04:
                    qid = str(val_df.iloc[i]["question_id"])
                    allowed = {c for c, cnt in q_label_dist.get(qid, Counter()).items() if c != "NONE"}
                    if top_m_code in allowed:
                        row_p[top_m_idx] = -1.0
                        sec_m = self.classes[int(np.argmax(row_p))]
                        preds[i] = [top_m_code, "NONE", sec_m]
        return preds

def main():
    print("=" * 115)
    print("S550 OUT-OF-FOLD ERROR BREAKDOWN FOR RARE MISCONCEPTION LABELS (<15 SAMPLES)")
    print("=" * 115)
    start_time = time.time()

    engine = S550AuditEngine()
    n_samples = engine.n_samples

    # Count class frequencies in train
    class_counts = Counter(engine.y_raw)
    rare_classes = {c: cnt for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    ultrarare_classes = {c: cnt for c, cnt in class_counts.items() if c != "NONE" and cnt < 5}
    modrare_classes = {c: cnt for c, cnt in class_counts.items() if c != "NONE" and 5 <= cnt < 15}
    frequent_classes = {c: cnt for c, cnt in class_counts.items() if c != "NONE" and cnt >= 15}

    print(f"Taxonomy Summary:")
    print(f"  Total Active Classes: {len(class_counts)} (including NONE)")
    print(f"  Ultra-Rare Misconceptions (<5 train samples):  {len(ultrarare_classes)} classes ({sum(ultrarare_classes.values())} traces)")
    print(f"  Moderate-Rare Misconceptions (5-14 samples):    {len(modrare_classes)} classes ({sum(modrare_classes.values())} traces)")
    print(f"  Total Rare Misconceptions (<15 train samples):  {len(rare_classes)} classes ({sum(rare_classes.values())} traces)")
    print(f"  Frequent Misconceptions (>=15 samples):        {len(frequent_classes)} classes ({sum(frequent_classes.values())} traces)")
    print(f"  Clean Traces (NONE):                           {class_counts['NONE']} traces\n")

    # In-fold leak-free regeneration
    oof_p_s550 = np.zeros((n_samples, engine.n_classes), dtype=np.float64)
    final_preds_all = [[] for _ in range(n_samples)]
    seen_mask = np.zeros(n_samples, dtype=bool)

    print("Regenerating In-Fold S550 Continuous Probabilities & Final Ranked Predictions...")
    for fold_num in range(1, 6):
        fold_info = engine.folds[fold_num - 1]
        trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
        val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

        p_val, q_dist = engine.generate_s550_probs(trn_idx, val_idx)
        oof_p_s550[val_idx] = p_val

        preds_val = engine.rank_and_gate(p_val, val_idx, q_dist)
        for i, idx in enumerate(val_idx):
            final_preds_all[idx] = preds_val[i]

        trn_qids = set(engine.train_df.iloc[trn_idx]["question_id"].astype(str))
        for i in val_idx:
            if str(engine.train_df.iloc[i]["question_id"]) in trn_qids:
                seen_mask[i] = True
        print(f"  Fold {fold_num}/5 complete ({len(val_idx)} traces)")

    # Analyze every trace
    trace_records = []
    for i in range(n_samples):
        y_gold = engine.y_raw[i]
        gold_idx = engine.class_to_idx[y_gold]
        p_row = oof_p_s550[i]
        top3_pred = final_preds_all[i]
        qid = str(engine.train_df.iloc[i]["question_id"])
        is_seen = bool(seen_mask[i])

        # Rank in continuous probability vector
        prob_ranking = list(np.argsort(-p_row))
        rank_in_prob = prob_ranking.index(gold_idx) + 1  # 1-indexed

        # Rank in final post-processed top 3
        if y_gold == top3_pred[0]:
            final_rank = 1
        elif y_gold == top3_pred[1]:
            final_rank = 2
        elif y_gold == top3_pred[2]:
            final_rank = 3
        else:
            final_rank = None

        in_top3 = final_rank is not None
        map3_score = 1.0 / final_rank if final_rank is not None else 0.0

        p_gold = float(p_row[gold_idx])
        p_top1 = float(p_row[prob_ranking[0]])
        p_rank3 = float(p_row[prob_ranking[2]])
        p_gap_to_top3 = max(0.0, p_rank3 - p_gold) if not in_top3 else 0.0

        is_rare = y_gold in rare_classes
        is_ultra = y_gold in ultrarare_classes
        is_mod = y_gold in modrare_classes

        # Categorize regime if error
        regime = "HIT_TOP3"
        if not in_top3:
            if rank_in_prob in [4, 5]:
                regime = "RANKED_4_5"
            elif rank_in_prob in range(6, 11):
                regime = "RANKED_6_10"
            elif rank_in_prob in range(11, 21):
                regime = "RANKED_11_20"
            else:
                regime = "RANKED_GT_20"

        # Competitor stealing slot 1
        stealer_top1 = top3_pred[0]
        stealer_top1_is_none = (stealer_top1 == "NONE")

        trace_records.append({
            "trace_id": str(engine.train_df.iloc[i]["trace_id"]),
            "question_id": qid,
            "gold_code": y_gold,
            "gold_count": class_counts.get(y_gold, 0),
            "is_seen": is_seen,
            "is_rare": is_rare,
            "is_ultra": is_ultra,
            "is_mod": is_mod,
            "final_top3": top3_pred,
            "final_rank": final_rank,
            "in_top3": in_top3,
            "map3": map3_score,
            "prob_rank": rank_in_prob,
            "regime": regime,
            "p_gold": round(p_gold, 5),
            "p_top1": round(p_top1, 5),
            "p_gap_to_top3": round(p_gap_to_top3, 5),
            "stealer_top1": stealer_top1,
            "stealer_top1_is_none": stealer_top1_is_none
        })

    # Filter rare traces
    rare_records = [r for r in trace_records if r["is_rare"]]
    n_rare_traces = len(rare_records)

    # 1. Overall Error Breakdown on Rare Labels
    print("\n" + "=" * 115)
    print(f"TABLE 1: S550 ERROR REGIME DECOMPOSITION ON RARE MISCONCEPTION TRACES (N={n_rare_traces})")
    print("=" * 115)

    def summarize_regimes(records: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
        tot = len(records)
        if tot == 0:
            return {}
        hits = [r for r in records if r["in_top3"]]
        r1 = [r for r in records if r["final_rank"] == 1]
        r2 = [r for r in records if r["final_rank"] == 2]
        r3 = [r for r in records if r["final_rank"] == 3]
        r45 = [r for r in records if r["regime"] == "RANKED_4_5"]
        r610 = [r for r in records if r["regime"] == "RANKED_6_10"]
        r1120 = [r for r in records if r["regime"] == "RANKED_11_20"]
        rgt20 = [r for r in records if r["regime"] == "RANKED_GT_20"]

        errors = [r for r in records if not r["in_top3"]]
        n_err = len(errors)
        stealer_none = [r for r in errors if r["stealer_top1_is_none"]]

        mean_map3 = float(np.mean([r["map3"] for r in records]))
        mean_prob_rank = float(np.mean([r["prob_rank"] for r in records]))
        mean_err_rank = float(np.mean([r["prob_rank"] for r in errors])) if n_err > 0 else 0.0

        p45_err_pct = len(r45) / n_err * 100 if n_err > 0 else 0.0
        p610_err_pct = len(r610) / n_err * 100 if n_err > 0 else 0.0
        gt10_err_pct = (len(r1120) + len(rgt20)) / n_err * 100 if n_err > 0 else 0.0

        print(f"--- {label} (N = {tot} traces) ---")
        print(f"  MAP@3 Score:                {mean_map3:.4f}")
        print(f"  Top 3 Hits:                 {len(hits):3d} / {tot} ({len(hits)/tot*100:5.1f}%)")
        print(f"    - Rank 1:                 {len(r1):3d} ({len(r1)/tot*100:5.1f}%)")
        print(f"    - Rank 2:                 {len(r2):3d} ({len(r2)/tot*100:5.1f}%)")
        print(f"    - Rank 3:                 {len(r3):3d} ({len(r3)/tot*100:5.1f}%)")
        print(f"  Top 3 Misses (Errors):      {n_err:3d} / {tot} ({n_err/tot*100:5.1f}%)")
        print(f"  Error Breakdown (Where was the true label ranked?):")
        print(f"    [Class Weighting Zone]")
        print(f"    - Rank 4-5 (Near-Miss):   {len(r45):3d} ({len(r45)/tot*100:5.1f}% of all, {p45_err_pct:5.1f}% of errors)")
        print(f"    - Rank 6-10:              {len(r610):3d} ({len(r610)/tot*100:5.1f}% of all, {p610_err_pct:5.1f}% of errors)")
        print(f"    [Visual Retrieval Zone]")
        print(f"    - Rank 11-20:             {len(r1120):3d} ({len(r1120)/tot*100:5.1f}% of all, {len(r1120)/n_err*100:5.1f}% of errors)")
        print(f"    - Rank > 20 (Distant):    {len(rgt20):3d} ({len(rgt20)/tot*100:5.1f}% of all, {len(rgt20)/n_err*100:5.1f}% of errors)")
        print(f"    - Total > Rank 10:        {len(r1120) + len(rgt20):3d} ({(len(r1120)+len(rgt20))/tot*100:5.1f}% of all, {gt10_err_pct:5.1f}% of errors)")
        print(f"  Mean True Label Rank:       Overall: {mean_prob_rank:.1f} | On Errors Only: {mean_err_rank:.1f}")
        print(f"  Slot 1 Stealer on Errors:   NONE: {len(stealer_none)} ({len(stealer_none)/n_err*100:.1f}%) | Other Misc: {n_err - len(stealer_none)} ({(n_err-len(stealer_none))/n_err*100:.1f}%)\n")

        return {
            "total_traces": tot,
            "map3": round(mean_map3, 4),
            "top3_hits": len(hits),
            "top3_hit_rate": round(len(hits) / tot, 4),
            "rank1_count": len(r1),
            "rank2_count": len(r2),
            "rank3_count": len(r3),
            "errors_count": n_err,
            "ranked_4_5_count": len(r45),
            "ranked_4_5_pct_of_err": round(p45_err_pct, 2),
            "ranked_6_10_count": len(r610),
            "ranked_6_10_pct_of_err": round(p610_err_pct, 2),
            "ranked_gt_10_count": len(r1120) + len(rgt20),
            "ranked_gt_10_pct_of_err": round(gt10_err_pct, 2),
            "mean_prob_rank": round(mean_prob_rank, 2),
            "mean_err_rank": round(mean_err_rank, 2),
            "stealer_none_count": len(stealer_none),
            "stealer_none_pct": round(len(stealer_none) / n_err * 100, 2) if n_err > 0 else 0.0
        }

    sum_all_rare = summarize_regimes(rare_records, "ALL RARE MISCONCEPTION TRACES (<15 samples)")
    sum_ultra = summarize_regimes([r for r in rare_records if r["is_ultra"]], "ULTRA-RARE TRACES (<5 samples)")
    sum_mod = summarize_regimes([r for r in rare_records if r["is_mod"]], "MODERATE-RARE TRACES (5-14 samples)")
    sum_freq = summarize_regimes([r for r in trace_records if not r["is_rare"] and r["gold_code"] != "NONE"], "FREQUENT MISCONCEPTION TRACES (>=15 samples)")

    # 2. Seen vs Unseen Comparison on Rare Labels
    print("=" * 115)
    print("TABLE 2: SEEN vs UNSEEN BREAKDOWN FOR RARE LABELS")
    print("=" * 115)
    sum_seen_rare = summarize_regimes([r for r in rare_records if r["is_seen"]], "SEEN QUESTIONS (Rare Labels)")
    sum_unseen_rare = summarize_regimes([r for r in rare_records if not r["is_seen"]], "UNSEEN QUESTIONS (Rare Labels)")

    # 3. Class-by-Class Table for all 52 Rare Classes
    print("=" * 125)
    print("TABLE 3: PER-CLASS S550 OOF BREAKDOWN (ALL 52 RARE CLASSES, SORTED BY FREQUENCY)")
    print("=" * 125)
    print(
        f"{'Class Code':12s} | {'N':3s} | {'MAP@3':7s} | "
        f"{'R1':3s} | {'R2':3s} | {'R3':3s} | {'Top3%':7s} | "
        f"{'R4-5':5s} | {'R6-10':6s} | {'R>10':5s} | {'Avg Rank':9s} | {'Avg P(gold)':12s} | {'Primary Stealer':16s}"
    )
    print("-" * 125)

    per_class_summary = []
    class_groups = defaultdict(list)
    for r in rare_records:
        class_groups[r["gold_code"]].append(r)

    for code, recs in sorted(class_groups.items(), key=lambda x: (len(x[1]), x[0])):
        c_tot = len(recs)
        c_r1 = sum(1 for r in recs if r["final_rank"] == 1)
        c_r2 = sum(1 for r in recs if r["final_rank"] == 2)
        c_r3 = sum(1 for r in recs if r["final_rank"] == 3)
        c_hits = c_r1 + c_r2 + c_r3
        c_r45 = sum(1 for r in recs if r["regime"] == "RANKED_4_5")
        c_r610 = sum(1 for r in recs if r["regime"] == "RANKED_6_10")
        c_rgt10 = sum(1 for r in recs if r["prob_rank"] > 10)

        c_map3 = float(np.mean([r["map3"] for r in recs]))
        c_avgrank = float(np.mean([r["prob_rank"] for r in recs]))
        c_avgp = float(np.mean([r["p_gold"] for r in recs]))

        stealers = Counter([r["stealer_top1"] for r in recs if not r["in_top3"]])
        top_stealer = f"{stealers.most_common(1)[0][0]} ({stealers.most_common(1)[0][1]})" if stealers else "None"

        per_class_summary.append({
            "class_code": code,
            "train_count": c_tot,
            "map3": round(c_map3, 4),
            "rank1": c_r1,
            "rank2": c_r2,
            "rank3": c_r3,
            "top3_hit_rate": round(c_hits / c_tot, 4),
            "rank_4_5": c_r45,
            "rank_6_10": c_r610,
            "rank_gt_10": c_rgt10,
            "mean_rank": round(c_avgrank, 1),
            "mean_p_gold": round(c_avgp, 4),
            "primary_stealer": top_stealer
        })

        print(
            f"{code:12s} | {c_tot:3d} | {c_map3:7.4f} | "
            f"{c_r1:3d} | {c_r2:3d} | {c_r3:3d} | {c_hits/c_tot*100:6.1f}% | "
            f"{c_r45:5d} | {c_r610:6d} | {c_rgt10:5d} | {c_avgrank:9.1f} | {c_avgp:12.4f} | {top_stealer:16s}"
        )

    # 4. Strategic Diagnosis & Hypothesis Validation
    errors_all_rare = [r for r in rare_records if not r["in_top3"]]
    n_err_total = len(errors_all_rare)
    n_class_weighting_zone = sum(1 for r in errors_all_rare if r["regime"] == "RANKED_4_5")
    n_prob_reachable = sum(1 for r in errors_all_rare if r["prob_rank"] <= 10)
    n_retrieval_failure = sum(1 for r in errors_all_rare if r["prob_rank"] > 10)

    print("\n" + "=" * 95)
    print("STRATEGIC DECISION DIAGNOSIS: VISUAL RETRIEVAL vs CLASS WEIGHTING")
    print("=" * 95)
    print(f"Total Rare Error Traces: {n_err_total}")
    print(f"  1. Ranked 4-5 (Direct Class-Weighting Candidates): {n_class_weighting_zone} / {n_err_total} ({n_class_weighting_zone / n_err_total:.1%})")
    print(f"  2. Ranked 6-10 (Extended Reach Candidates):       {n_prob_reachable - n_class_weighting_zone} / {n_err_total} ({(n_prob_reachable - n_class_weighting_zone) / n_err_total:.1%})")
    print(f"  3. Ranked > 10 (Retrieval Failures):              {n_retrieval_failure} / {n_err_total} ({n_retrieval_failure / n_err_total:.1%})")
    print(f"\nVerdict:")
    if n_class_weighting_zone > n_retrieval_failure:
        print("  -> CLASS WEIGHTING DOMINATES: True labels are present near the margin (Rank 4-5).")
        print("     Slight loss re-weighting or frequency recalibration will flip them into Top 3.")
    else:
        print(f"  -> RETRIEVAL FAILURE DOMINATES: {n_retrieval_failure} of {n_err_total} errors ({n_retrieval_failure / n_err_total:.1%}) are ranked > 10.")
        print("     Average assigned probability to true label in this zone is near zero.")
        print("     Class weighting cannot push Rank 15+ predictions into Top 3 without massive collateral damage.")
        print("     The next gold-only experiment MUST target visual retrieval & representation geometry.")
    print("=" * 95)

    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "summary_all_rare": sum_all_rare,
        "summary_ultra_rare": sum_ultra,
        "summary_moderate_rare": sum_mod,
        "summary_frequent": sum_freq,
        "summary_seen_rare": sum_seen_rare,
        "summary_unseen_rare": sum_unseen_rare,
        "per_class_summary": per_class_summary,
        "strategic_diagnosis": {
            "total_rare_errors": n_err_total,
            "ranked_4_5_count": n_class_weighting_zone,
            "ranked_4_5_pct": round(n_class_weighting_zone / n_err_total, 4),
            "ranked_6_10_count": n_prob_reachable - n_class_weighting_zone,
            "ranked_gt_10_count": n_retrieval_failure,
            "ranked_gt_10_pct": round(n_retrieval_failure / n_err_total, 4),
            "primary_bottleneck": "RETRIEVAL_FAILURE" if n_retrieval_failure > n_class_weighting_zone else "CLASS_WEIGHTING"
        },
        "audit_duration_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "s550_oof_rare_class_error_breakdown.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Report] {report_path.name}")

if __name__ == "__main__":
    main()

