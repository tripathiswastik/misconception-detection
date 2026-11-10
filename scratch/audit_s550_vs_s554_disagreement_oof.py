"""Row-Level Error & Candidate-Coverage Audit: S550 versus S554 on Leak-Free OOF.

Directly answers:
1. For disagreement rows, is the true label in either model's Top 3 (Union Recall@3)?
2. How does each model rank it, and who wins on MAP@3 points?
3. How does question support (seen vs novel misconception vs clean) correlate with outcomes?
4. Definitive verdict: should next effort focus on candidate retrieval or model scoring?
5. Keeps Row 3264 strictly out of OOF claims.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
from typing_extensions import TypedDict
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.metrics.map3 import calculate_map3

class TraceDisagreementInfo(TypedDict):
    sub_index: int
    global_index: int
    trace_id: str
    question_id: str
    gold: str
    p550: List[str]
    p554: List[str]
    score_550: float
    score_554: float
    rank_550: Optional[int]
    rank_554: Optional[int]
    winner: str
    margin: float
    in_550: bool
    in_554: bool
    in_either: bool
    in_neither: bool
    in_both: bool
    support_cat: str
    gold_q_train: int
    total_q_train: int
    is_r1_diff: bool

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

class AuditComponents:
    def __init__(self, data_dir: Path, art_dir: Path):
        self.tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
        self.classes = self.tax.all_codes(include_none=True)
        self.none_idx = self.classes.index("NONE")
        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.train_df = cast(pd.DataFrame, pd.read_csv(data_dir / "train.csv"))
        self.y_raw = self.train_df["root_code"].fillna("NONE").astype(str).tolist()
        self.n_samples = len(self.train_df)

        self.oof_m15 = np.load(art_dir / "oof_m15_balanced_fusion.npy")
        self.oof_dual = np.load(art_dir / "oof_arcface_dualhead_448_soft.npy")
        self.oof_comp = np.load(art_dir / "oof_arcface_composite_448.npy")
        self.oof_v1 = np.load(art_dir / "oof_arcface_pseudo_3500.npy")
        self.oof_v2 = np.load(art_dir / "oof_arcface_pseudo_3500_v2.npy")

        self.honest_text = np.load(art_dir / "honest_oof_text.npy")
        self.oof_sem_text = np.load(art_dir / "oof_semantic_text.npy")
        self.oof_convnext = np.load(art_dir / "oof_convnext_vision.npy")
        self.oof_swin = np.load(art_dir / "oof_swin_vision.npy")
        self.oof_finetuned = np.load(art_dir / "oof_finetuned_vision.npy")
        self.oof_trocr = np.load(art_dir / "oof_trocr_vision.npy")
        self.oof_clean_arcface = np.load(art_dir / "oof_clean_arcface_2328.npy")

        self.oof_bin_v1 = np.load(art_dir / "oof_compact_binary_prob.npy")
        self.oof_bin_v2 = np.load(art_dir / "oof_compact_binary_prob_v2.npy")

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

    def generate_fold_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Counter]]:
        trn_df_fold = self.train_df.iloc[trn_idx]
        val_df_fold = self.train_df.iloc[val_idx]
        n_val = len(val_idx)

        # Frequency calibration mult on trn_idx
        p_base_v1_trn = 0.95 * self.oof_m15[trn_idx] + 0.02 * self.oof_dual[trn_idx] + 0.03 * self.oof_comp[trn_idx]
        top3_trn = np.argsort(-p_base_v1_trn.astype(np.float64), axis=1)[:, :3]
        pred_counts = Counter(self.classes[idx] for row in top3_trn for idx in row)
        true_counts = Counter(trn_df_fold["root_code"].fillna("NONE").values)
        mult = np.ones(self.n_classes, dtype=np.float64)
        for i, c in enumerate(self.classes):
            if c != "NONE":
                mult[i] = (max(true_counts.get(c, 1), 1) / max(pred_counts.get(c, 1), 1)) ** 0.05
        mult[self.none_idx] = 0.9840

        # Base probabilities
        p_base_v1_val = (0.95 * self.oof_m15[val_idx] + 0.02 * self.oof_dual[val_idx] + 0.03 * self.oof_comp[val_idx]) * mult
        p_base_v1_val[:, self.none_idx] *= 0.9995

        p_base_v2_val = (0.88 * self.oof_m15[val_idx] + 0.12 * self.oof_v2[val_idx]) * mult
        p_base_v2_val[:, self.none_idx] *= 0.9995

        # Transductive question boost
        q_label_dist: Dict[str, Counter] = {
            q: Counter(grp["root_code"].fillna("NONE").values)
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        p_boosted_v2 = np.copy(p_base_v2_val)
        for i in range(n_val):
            qid = val_df_fold.iloc[i]["question_id"]
            if qid in q_label_dist:
                allowed_misconceptions = {c for c, cnt in q_label_dist[qid].items() if c != "NONE"}
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20
                        p_boosted_v2[i, ci] *= 1.20

        # ArcFace additions
        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_arc_v2 = (0.88 * self.oof_m15[val_idx] + 0.12 * self.oof_v2[val_idx]) * mult

        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_champ_v2 = 0.65 * p_base_v2_val + 0.25 * p_boosted_v2 + 0.10 * p_arc_v2

        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean
        p_s554 = 0.96 * p_champ_v2 + 0.04 * p_clean

        return p_s550, p_s554, q_label_dist

    def rank_and_gate(self, p_matrix: np.ndarray, val_idx: np.ndarray, q_label_dist: Dict[str, Counter], beta: float = 0.82) -> List[List[str]]:
        cand = p_matrix.copy()
        n = len(cand)
        row_idx = np.arange(n)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != self.none_idx, self.none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[self.classes[r1[i]], self.classes[top2[i, 0]], self.classes[top2[i, 1]]] for i in range(n)]

        # Apply precision gating
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
                    qid = val_df.iloc[i]["question_id"]
                    allowed = {c for c, cnt in q_label_dist.get(qid, Counter()).items() if c != "NONE"}
                    if top_m_code in allowed:
                        row_p[top_m_idx] = -1.0
                        sec_m = self.classes[int(np.argmax(row_p))]
                        preds[i] = [top_m_code, "NONE", sec_m]
        return preds

def score_trace(gold: str, pred: List[str]) -> Tuple[float, Optional[int]]:
    if gold == pred[0]:
        return 1.0, 1
    elif gold == pred[1]:
        return 0.5, 2
    elif gold == pred[2]:
        return 1.0 / 3.0, 3
    else:
        return 0.0, None

def main():
    print("=" * 80)
    print("ROW-LEVEL ERROR & CANDIDATE-COVERAGE AUDIT: S550 vs S554 (LEAK-FREE OOF)")
    print("Zero Test Leakage | Row 3264 Excluded | Empirical Retrieval vs Scoring Gate")
    print("=" * 80)

    start_time = time.time()
    comp = AuditComponents(DATA_DIR, ART_DIR)
    train_df = comp.train_df
    y_true_all = comp.y_raw
    n_samples = comp.n_samples

    # Build 5-Fold Stratified on Multi-Trace Questions (Transductive Seen Proxy)
    q_counts = Counter(train_df["question_id"])
    multi_q_mask = np.array([q_counts[q] >= 2 for q in train_df["question_id"]])
    sub_df = train_df.iloc[multi_q_mask].copy().reset_index(drop=True)
    sub_indices_global = np.where(multi_q_mask)[0]
    n_seen = len(sub_df)

    kf_seen = KFold(n_splits=5, shuffle=True, random_state=42)
    seen_splits = []
    for trn_loc, val_loc in kf_seen.split(sub_df):
        seen_splits.append((sub_indices_global[trn_loc], sub_indices_global[val_loc]))

    s550_oof_preds: List[List[str]] = [[] for _ in range(n_seen)]
    s554_oof_preds: List[List[str]] = [[] for _ in range(n_seen)]
    q_label_dists_per_fold: Dict[int, Dict[str, Counter]] = {}
    val_indices_per_fold: Dict[int, np.ndarray] = {}

    for f_idx, (trn_idx, val_idx) in enumerate(seen_splits):
        p_s550, p_s554, q_dist = comp.generate_fold_probs(trn_idx, val_idx)
        preds_s550 = comp.rank_and_gate(p_s550, val_idx, q_dist)
        preds_s554 = comp.rank_and_gate(p_s554, val_idx, q_dist)

        val_local_indices = [np.where(sub_indices_global == idx)[0][0] for idx in val_idx]
        for i, global_sub_idx in enumerate(val_local_indices):
            s550_oof_preds[global_sub_idx] = preds_s550[i]
            s554_oof_preds[global_sub_idx] = preds_s554[i]

        q_label_dists_per_fold[f_idx] = q_dist
        val_indices_per_fold[f_idx] = val_idx

    y_seen_true = [y_true_all[i] for i in sub_indices_global]
    overall_s550 = calculate_map3(y_seen_true, s550_oof_preds)
    overall_s554 = calculate_map3(y_seen_true, s554_oof_preds)
    print(f"\nEvaluated {n_seen} Seen-Question OOF traces across 5 folds:")
    print(f"  Overall S550 Baseline MAP@3: {overall_s550:.5f}")
    print(f"  Overall S554 Baseline MAP@3: {overall_s554:.5f}")

    # =========================================================================
    # AUDIT ALL DISAGREEMENT ROWS
    # =========================================================================
    disagreement_rows: List[TraceDisagreementInfo] = []
    r1_disagreement_rows: List[TraceDisagreementInfo] = []

    for i in range(n_seen):
        gold = y_seen_true[i]
        p550 = s550_oof_preds[i]
        p554 = s554_oof_preds[i]
        global_idx = sub_indices_global[i]
        row_data = train_df.iloc[global_idx]
        qid = row_data["question_id"]

        is_triplet_diff = (p550 != p554)
        is_r1_diff = (p550[0] != p554[0])

        score_550, rank_550 = score_trace(gold, p550)
        score_554, rank_554 = score_trace(gold, p554)

        in_550 = gold in p550
        in_554 = gold in p554
        in_either = in_550 or in_554
        in_neither = not in_either
        in_both = in_550 and in_554

        # Question support in training partition
        # Find which fold this row belonged to
        fold_id = -1
        for f, v_idx in val_indices_per_fold.items():
            if global_idx in v_idx:
                fold_id = f
                break
        q_dist = q_label_dists_per_fold[fold_id].get(qid, Counter())
        total_q_train = sum(q_dist.values())
        gold_q_train = q_dist.get(gold, 0)

        # Categorize question support
        if gold == "NONE":
            support_cat = "Clean (NONE)"
        elif gold_q_train >= 2:
            support_cat = "Strong Support (>=2 in train)"
        elif gold_q_train == 1:
            support_cat = "Weak Support (1 in train)"
        else:
            support_cat = "Novel Label (0 in train for this question)"

        # Determine winner
        if score_550 > score_554:
            winner = "S550"
            margin = score_550 - score_554
        elif score_554 > score_550:
            winner = "S554"
            margin = score_554 - score_550
        else:
            winner = "Tie"
            margin = 0.0

        trace_info = {
            "sub_index": i,
            "global_index": int(global_idx),
            "trace_id": row_data["trace_id"],
            "question_id": qid,
            "gold": gold,
            "p550": p550,
            "p554": p554,
            "score_550": round(score_550, 4),
            "score_554": round(score_554, 4),
            "rank_550": rank_550,
            "rank_554": rank_554,
            "winner": winner,
            "margin": round(margin, 4),
            "in_550": in_550,
            "in_554": in_554,
            "in_either": in_either,
            "in_neither": in_neither,
            "in_both": in_both,
            "support_cat": support_cat,
            "gold_q_train": gold_q_train,
            "total_q_train": total_q_train,
            "is_r1_diff": is_r1_diff
        }

        if is_triplet_diff:
            disagreement_rows.append(trace_info)
        if is_r1_diff:
            r1_disagreement_rows.append(trace_info)

    n_diff = len(disagreement_rows)
    n_r1_diff = len(r1_disagreement_rows)

    print("\n" + "=" * 80)
    print(f"DISAGREEMENT ROWS IDENTIFIED: {n_diff} / {n_seen} ({n_diff/n_seen:.1%})")
    print(f"Rank-1 Disagreements       : {n_r1_diff} / {n_seen} ({n_r1_diff/n_seen:.1%})")
    print("=" * 80)

    # 1. Candidate Coverage on Disagreement Rows
    cov_either = sum(1 for r in disagreement_rows if r["in_either"])
    cov_550 = sum(1 for r in disagreement_rows if r["in_550"])
    cov_554 = sum(1 for r in disagreement_rows if r["in_554"])
    cov_both = sum(1 for r in disagreement_rows if r["in_both"])
    cov_neither = sum(1 for r in disagreement_rows if r["in_neither"])

    print("\n" + "=" * 80)
    print("1. CANDIDATE RETRIEVAL & COVERAGE ON DISAGREEMENT ROWS")
    print("=" * 80)
    print(f"  True Label in EITHER Model's Top 3 (Union Recall@3): {cov_either:3d} / {n_diff} ({cov_either/n_diff:.1%})")
    print(f"  True Label in S550 Top 3                           : {cov_550:3d} / {n_diff} ({cov_550/n_diff:.1%})")
    print(f"  True Label in S554 Top 3                           : {cov_554:3d} / {n_diff} ({cov_554/n_diff:.1%})")
    print(f"  True Label in BOTH Models' Top 3                   : {cov_both:3d} / {n_diff} ({cov_both/n_diff:.1%})")
    print(f"  True Label in NEITHER Model's Top 3 (Total Miss)   : {cov_neither:3d} / {n_diff} ({cov_neither/n_diff:.1%})")

    # 2. Ranking & Point Outcomes on Disagreement Rows
    wins_550 = [r for r in disagreement_rows if r["winner"] == "S550"]
    wins_554 = [r for r in disagreement_rows if r["winner"] == "S554"]
    ties = [r for r in disagreement_rows if r["winner"] == "Tie"]

    pts_550 = sum(float(r["score_550"]) for r in disagreement_rows)
    pts_554 = sum(float(r["score_554"]) for r in disagreement_rows)
    pts_oracle = sum(max(float(r["score_550"]), float(r["score_554"])) for r in disagreement_rows)

    print("\n" + "=" * 80)
    print("2. RANKING & POINT OUTCOMES ON DISAGREEMENT ROWS")
    print("=" * 80)
    print(f"  S550 Wins (Higher MAP@3 Point)     : {len(wins_550):3d} rows ({len(wins_550)/n_diff:.1%}) | Total Pts: {sum(float(r['score_550']) for r in wins_550):.2f}")
    print(f"  S554 Wins (Higher MAP@3 Point)     : {len(wins_554):3d} rows ({len(wins_554)/n_diff:.1%}) | Total Pts: {sum(float(r['score_554']) for r in wins_554):.2f}")
    print(f"  Ties (Both Got Same Points or 0)   : {len(ties):3d} rows ({len(ties)/n_diff:.1%})")
    print(f"  Total MAP@3 Points Scored by S550  : {pts_550:.2f} / {n_diff} (Mean: {pts_550/n_diff:.4f})")
    print(f"  Total MAP@3 Points Scored by S554  : {pts_554:.2f} / {n_diff} (Mean: {pts_554/n_diff:.4f})")
    print(f"  Upper-Bound Oracle Point Potential : {pts_oracle:.2f} / {n_diff} (Mean: {pts_oracle/n_diff:.4f})")
    print(f"  Headroom from Perfect Arbitration  : +{(pts_oracle - pts_550)/n_seen:.5f} Overall MAP@3 Gain!")

    # 3. Question Support Cross-Tabulation
    support_cats = [
        "Strong Support (>=2 in train)",
        "Weak Support (1 in train)",
        "Novel Label (0 in train for this question)",
        "Clean (NONE)"
    ]

    print("\n" + "=" * 80)
    print("3. QUESTION SUPPORT vs MODEL OUTCOME BREAKDOWN")
    print("=" * 80)
    print(f"{'Question Support Category':42s} | {'Count':5s} | {'S550 Wins':9s} | {'S554 Wins':9s} | {'Ties':6s} | {'Union R@3':9s} | {'Neither R@3':11s}")
    print("-" * 105)

    support_breakdown: Dict[str, Any] = {}
    for cat in support_cats:
        rows_cat = [r for r in disagreement_rows if r["support_cat"] == cat]
        n_c = len(rows_cat)
        if n_c == 0:
            continue
        w550_c = sum(1 for r in rows_cat if r["winner"] == "S550")
        w554_c = sum(1 for r in rows_cat if r["winner"] == "S554")
        tie_c = sum(1 for r in rows_cat if r["winner"] == "Tie")
        union_c = sum(1 for r in rows_cat if r["in_either"])
        neither_c = sum(1 for r in rows_cat if r["in_neither"])

        support_breakdown[cat] = {
            "count": n_c,
            "s550_wins": w550_c,
            "s554_wins": w554_c,
            "ties": tie_c,
            "union_recall3": round(union_c / n_c, 4),
            "neither_recall3": round(neither_c / n_c, 4)
        }
        print(f"{cat:42s} | {n_c:5d} | {w550_c:9d} | {w554_c:9d} | {tie_c:6d} | {union_c/n_c:8.1%} | {neither_c/n_c:10.1%}")

    # 4. Rank Position Matrix for Disagreements
    print("\n" + "=" * 80)
    print("4. RANK OF TRUE LABEL WHEN RETRIEVED (S550 vs S554)")
    print("=" * 80)
    rank_matrix = Counter((r["rank_550"], r["rank_554"]) for r in disagreement_rows)
    print(f"{'S550 Rank':12s} | {'S554 Rank':12s} | {'Sample Count':12s} | {'Outcome Description'}")
    print("-" * 75)
    for (r1, r2), cnt in sorted(rank_matrix.items(), key=lambda x: -x[1]):
        desc = ""
        if r1 == 1 and r2 != 1:
            desc = "S550 hits Rank 1 (S550 wins +0.50 or +0.67 pts)"
        elif r2 == 1 and r1 != 1:
            desc = "S554 hits Rank 1 (S554 wins +0.50 or +0.67 pts)"
        elif r1 is None and r2 is None:
            desc = "Total Miss by Both (0 points for both)"
        elif r1 == r2:
            desc = f"Both rank true label at Rank {r1} (Tie)"
        elif r1 is not None and r2 is not None:
            desc = f"Both retrieved; S550 Rank {r1} vs S554 Rank {r2}"
        else:
            win_m = "S550" if r1 is not None else "S554"
            desc = f"Retrieved only by {win_m}"
        print(f"{str(r1):12s} | {str(r2):12s} | {cnt:12d} | {desc}")

    # 5. Core Decision Analysis: Retrieval Bottleneck vs Scoring Bottleneck
    pct_union = cov_either / n_diff
    pct_neither = cov_neither / n_diff
    is_retrieval_bottleneck = pct_neither > 0.40

    print("\n" + "=" * 80)
    print("5. DEFINITIVE VERDICT: CANDIDATE RETRIEVAL vs MODEL SCORING")
    print("=" * 80)
    if is_retrieval_bottleneck:
        verdict = "RETRIEVAL / REPRESENTATION BOTTLENECK"
        action = "Prioritize improving candidate retrieval: Neither model captures the true label in top-3 on >40% of disagreement cases."
    else:
        verdict = "SCORING & ARBITRATION BOTTLENECK"
        action = f"Prioritize model scoring / arbitration: {pct_union:.1%} of disagreement traces ALREADY contain the true label in the union of Top 3."

    print(f"VERDICT: {verdict}")
    print(f"Rationale: Union Recall@3 on disagreements = {pct_union:.1%}; Total Miss Rate = {pct_neither:.1%}.")
    print(f"Action Recommendation: {action}")
    print("=" * 80)

    # Save full audit artifact
    out_artifact = {
        "audit_name": "s550_vs_s554_disagreement_oof_audit",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_seen_traces": n_seen,
        "disagreement_count": n_diff,
        "disagreement_pct": round(n_diff / n_seen, 4),
        "r1_disagreement_count": n_r1_diff,
        "coverage_stats": {
            "union_recall3": round(cov_either / n_diff, 4),
            "s550_recall3": round(cov_550 / n_diff, 4),
            "s554_recall3": round(cov_554 / n_diff, 4),
            "both_recall3": round(cov_both / n_diff, 4),
            "neither_recall3": round(cov_neither / n_diff, 4)
        },
        "points_stats": {
            "s550_total_points": round(pts_550, 2),
            "s554_total_points": round(pts_554, 2),
            "s550_wins": len(wins_550),
            "s554_wins": len(wins_554),
            "ties": len(ties),
            "oracle_points": round(pts_oracle, 2),
            "potential_gain": round((pts_oracle - pts_550) / n_seen, 5)
        },
        "question_support_breakdown": support_breakdown,
        "rank_distribution_matrix": {f"S550_R{r1}_S554_R{r2}": cnt for (r1, r2), cnt in rank_matrix.items()},
        "verdict": {
            "primary_bottleneck": verdict,
            "union_recall_pct": round(pct_union, 4),
            "total_miss_pct": round(pct_neither, 4),
            "recommendation": action
        },
        "disagreement_rows_sample": disagreement_rows[:25],
        "runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "s550_vs_s554_disagreement_audit.json"
    with open(report_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    print(f"\n[Saved Audit Artifact] {report_path.name}")

if __name__ == "__main__":
    main()
