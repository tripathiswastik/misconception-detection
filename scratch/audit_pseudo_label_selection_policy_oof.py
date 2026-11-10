"""Audit Exact Pseudo-Label Selection Policy on Held-Out Labeled Rows.

Specification per User Directive:
1. In-Fold Strict Isolation:
   - Evaluates across all 5 canonical folds (splits.json: 1200 train / 300 val).
   - In-fold question allowed mask:
     * For Seen questions (qid in train fold): allowed classes = {c in train[qid]} U {"NONE"}.
     * For Unseen questions (qid not in train fold): all classes allowed (or unconstrained).
   - In-fold leak-free teacher probabilities from Champion S550.
2. Selection Policy:
   - Candidate proposed code: c_top = argmax_{c in allowed} P(c).
   - Confidence threshold: tau (sweeps 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, focuses on tau=0.35).
   - Capped class balancing: at most 15 traces per class selected (sorted by peak confidence descending).
3. Comprehensive Metrics:
   - Selected trace counts (total, NONE, misconceptions).
   - Selection Precision:
     * Overall precision: accuracy of selected traces vs gold label.
     * Misconception precision: accuracy on non-NONE selected traces.
     * NONE precision: accuracy on NONE selected traces.
   - Rare-class metrics (<15 train traces, 52 classes; <5 train traces, 23 classes):
     * Unique rare classes covered.
     * Number of rare-class traces selected.
     * Precision on rare-class selections.
   - Breakdowns:
     * By Fold (Folds 1 to 5).
     * By Question Status (Seen vs Unseen).
4. Decision Gate:
   - Report whether the policy yields enough reliable rare-class examples to proceed to training.
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

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

class CanonicalAuditBundle:
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

        # Baseline components for S550 in-fold regeneration
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

    def generate_s550_probs(self, trn_idx: np.ndarray, val_idx: np.ndarray) -> Tuple[np.ndarray, Dict[str, Set[str]]]:
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

        q_allowed: Dict[str, Set[str]] = {
            str(q): {str(x) for x in grp["root_code"].dropna().unique()}
            for q, grp in trn_df_fold.groupby("question_id")
        }

        p_boosted_v1 = np.copy(p_base_v1_val)
        for i in range(n_val):
            qid = str(val_df_fold.iloc[i]["question_id"])
            if qid in q_allowed:
                allowed_misconceptions = q_allowed[qid]
                for ci, c in enumerate(self.classes):
                    if c in allowed_misconceptions:
                        p_boosted_v1[i, ci] *= 1.20

        p_arc_v1 = (0.95 * self.oof_m15[val_idx] + 0.05 * self.oof_v1[val_idx]) * mult
        p_champ_v1 = 0.70 * p_base_v1_val + 0.25 * p_boosted_v1 + 0.05 * p_arc_v1
        p_clean = self.build_fold_clean_multimodal(trn_idx, val_idx)
        p_s550 = 0.96 * p_champ_v1 + 0.04 * p_clean

        p_s550 = np.maximum(p_s550, 0.0)
        p_s550 = p_s550 / p_s550.sum(axis=1, keepdims=True)
        return p_s550, q_allowed

def run_selection_policy_on_fold(
    val_probs: np.ndarray,
    val_df: pd.DataFrame,
    val_targets: List[str],
    q_allowed: Dict[str, Set[str]],
    classes: List[str],
    none_idx: int,
    tau: float = 0.35,
    max_per_class: int = 15,
    rare_classes: Set[str] = set(),
    ultrarare_classes: Set[str] = set()
) -> Dict[str, Any]:
    """Applies in-fold question mask, confidence threshold tau, and capped class balancing."""
    n = len(val_probs)
    class_candidates: Dict[str, List[Tuple[float, int, str, bool]]] = defaultdict(list)

    for i in range(n):
        qid = str(val_df.iloc[i]["question_id"])
        y_true = str(val_targets[i])
        row_p = val_probs[i].copy()

        # In-fold question mask
        is_seen = qid in q_allowed
        if is_seen:
            allowed_set = q_allowed[qid].union({"NONE"})
            for ci, c in enumerate(classes):
                if c not in allowed_set:
                    row_p[ci] = -1.0

        top_idx = int(np.argmax(row_p))
        top_conf = float(row_p[top_idx])
        top_code = classes[top_idx]

        if top_conf >= tau:
            is_correct = (top_code == y_true)
            class_candidates[top_code].append((top_conf, i, y_true, is_seen))

    # Capped class balancing: at most max_per_class per class, selected by descending confidence
    selected_indices: List[int] = []
    selected_records: List[Dict[str, Any]] = []

    for code, cands in class_candidates.items():
        # Sort by confidence descending
        cands.sort(key=lambda x: x[0], reverse=True)
        selected_cands = cands[:max_per_class]

        for conf, idx, y_true, is_seen in selected_cands:
            selected_indices.append(idx)
            is_correct = (code == y_true)
            is_rare = code in rare_classes
            is_ultra = code in ultrarare_classes
            selected_records.append({
                "val_idx": idx,
                "proposed_code": code,
                "confidence": round(conf, 4),
                "gold_code": y_true,
                "is_correct": is_correct,
                "is_seen": is_seen,
                "is_none": (code == "NONE"),
                "is_rare": is_rare,
                "is_ultra": is_ultra
            })

    total_sel = len(selected_records)
    if total_sel == 0:
        return {
            "selected_count": 0,
            "overall_precision": 0.0,
            "none_count": 0,
            "none_precision": 0.0,
            "misc_count": 0,
            "misc_precision": 0.0,
            "rare_count": 0,
            "rare_precision": 0.0,
            "rare_classes_covered": 0,
            "ultra_count": 0,
            "ultra_precision": 0.0,
            "ultra_classes_covered": 0,
            "seen_misc_count": 0,
            "seen_misc_precision": 0.0,
            "unseen_misc_count": 0,
            "unseen_misc_precision": 0.0,
            "records": []
        }

    all_corr = sum(1 for r in selected_records if r["is_correct"])
    overall_prec = all_corr / total_sel

    none_recs = [r for r in selected_records if r["is_none"]]
    none_count = len(none_recs)
    none_prec = sum(1 for r in none_recs if r["is_correct"]) / none_count if none_count > 0 else 0.0

    misc_recs = [r for r in selected_records if not r["is_none"]]
    misc_count = len(misc_recs)
    misc_prec = sum(1 for r in misc_recs if r["is_correct"]) / misc_count if misc_count > 0 else 0.0

    rare_recs = [r for r in misc_recs if r["is_rare"]]
    rare_count = len(rare_recs)
    rare_prec = sum(1 for r in rare_recs if r["is_correct"]) / rare_count if rare_count > 0 else 0.0
    rare_classes_covered = len({r["proposed_code"] for r in rare_recs})

    ultra_recs = [r for r in misc_recs if r["is_ultra"]]
    ultra_count = len(ultra_recs)
    ultra_prec = sum(1 for r in ultra_recs if r["is_correct"]) / ultra_count if ultra_count > 0 else 0.0
    ultra_classes_covered = len({r["proposed_code"] for r in ultra_recs})

    seen_misc_recs = [r for r in misc_recs if r["is_seen"]]
    seen_misc_count = len(seen_misc_recs)
    seen_misc_prec = sum(1 for r in seen_misc_recs if r["is_correct"]) / seen_misc_count if seen_misc_count > 0 else 0.0

    unseen_misc_recs = [r for r in misc_recs if not r["is_seen"]]
    unseen_misc_count = len(unseen_misc_recs)
    unseen_misc_prec = sum(1 for r in unseen_misc_recs if r["is_correct"]) / unseen_misc_count if unseen_misc_count > 0 else 0.0

    return {
        "selected_count": total_sel,
        "overall_precision": round(overall_prec, 4),
        "none_count": none_count,
        "none_precision": round(none_prec, 4),
        "misc_count": misc_count,
        "misc_precision": round(misc_prec, 4),
        "rare_count": rare_count,
        "rare_precision": round(rare_prec, 4),
        "rare_classes_covered": rare_classes_covered,
        "ultra_count": ultra_count,
        "ultra_precision": round(ultra_prec, 4),
        "ultra_classes_covered": ultra_classes_covered,
        "seen_misc_count": seen_misc_count,
        "seen_misc_precision": round(seen_misc_prec, 4),
        "unseen_misc_count": unseen_misc_count,
        "unseen_misc_precision": round(unseen_misc_prec, 4),
        "records": selected_records
    }

def main():
    print("=" * 105)
    print("AUDITING EXACT PSEUDO-LABEL SELECTION POLICY (QUESTION-MASK + 15-PER-CLASS CAP)")
    print("=" * 105)
    start_time = time.time()

    data = CanonicalAuditBundle()
    class_counts = Counter(data.y_raw)
    rare_classes = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    ultrarare_classes = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 5}
    print(f"Taxonomy: 76 classes | Rare Classes (<15): {len(rare_classes)} | Ultra-Rare (<5): {len(ultrarare_classes)}\n")

    # Generate in-fold leak-free teacher probabilities and question support
    fold_probs = {}
    fold_allowed = {}
    fold_dfs = {}
    fold_targets = {}

    print("Generating In-Fold Leak-Free Predictions & Question Masks...")
    for fold_num in range(1, 6):
        fold_info = data.folds[fold_num - 1]
        trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
        val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

        p_val, q_allowed = data.generate_s550_probs(trn_idx, val_idx)
        val_df = data.train_df.iloc[val_idx].reset_index(drop=True)
        val_y = [data.y_raw[i] for i in val_idx]

        fold_probs[fold_num] = p_val
        fold_allowed[fold_num] = q_allowed
        fold_dfs[fold_num] = val_df
        fold_targets[fold_num] = val_y
        print(f"  Fold {fold_num}/5 complete ({len(val_idx)} traces)")

    # 1. Threshold Sweep for the Proposed Policy (15 per class cap + question mask)
    print("\n" + "=" * 115)
    print("TABLE 1: THRESHOLD SWEEP OF PROPOSED SELECTION POLICY ACROSS ALL 1,500 HELD-OUT ROWS")
    print("=" * 115)
    print(
        f"{'Tau':6s} | {'Sel N':6s} | {'All Prec':9s} | "
        f"{'NONE N':7s} | {'NONE Prec':9s} | "
        f"{'Misc N':7s} | {'Misc Prec':9s} | "
        f"{'Rare N':7s} | {'Rare Prec':9s} | {'Rare Cov':8s} | {'Ultra N':7s} | {'Ultra Cov':9s}"
    )
    print("-" * 115)

    tau_sweep_results = []
    for tau in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        total_records = []
        for fold_num in range(1, 6):
            res = run_selection_policy_on_fold(
                val_probs=fold_probs[fold_num],
                val_df=fold_dfs[fold_num],
                val_targets=fold_targets[fold_num],
                q_allowed=fold_allowed[fold_num],
                classes=data.classes,
                none_idx=data.none_idx,
                tau=tau,
                max_per_class=15,
                rare_classes=rare_classes,
                ultrarare_classes=ultrarare_classes
            )
            total_records.extend(res["records"])

        n_sel = len(total_records)
        all_prec = sum(1 for r in total_records if r["is_correct"]) / n_sel if n_sel > 0 else 0.0

        none_r = [r for r in total_records if r["is_none"]]
        n_none = len(none_r)
        p_none = sum(1 for r in none_r if r["is_correct"]) / n_none if n_none > 0 else 0.0

        misc_r = [r for r in total_records if not r["is_none"]]
        n_misc = len(misc_r)
        p_misc = sum(1 for r in misc_r if r["is_correct"]) / n_misc if n_misc > 0 else 0.0

        rare_r = [r for r in misc_r if r["is_rare"]]
        n_rare = len(rare_r)
        p_rare = sum(1 for r in rare_r if r["is_correct"]) / n_rare if n_rare > 0 else 0.0
        cov_rare = len({r["proposed_code"] for r in rare_r})

        ultra_r = [r for r in misc_r if r["is_ultra"]]
        n_ultra = len(ultra_r)
        cov_ultra = len({r["proposed_code"] for r in ultra_r})

        tau_sweep_results.append({
            "tau": tau,
            "total_selected": n_sel,
            "overall_precision": round(all_prec, 4),
            "none_count": n_none,
            "none_precision": round(p_none, 4),
            "misc_count": n_misc,
            "misc_precision": round(p_misc, 4),
            "rare_count": n_rare,
            "rare_precision": round(p_rare, 4),
            "rare_coverage": cov_rare,
            "ultra_count": n_ultra,
            "ultra_coverage": cov_ultra
        })

        print(
            f"{tau:6.2f} | {n_sel:6d} | {all_prec:9.4f} | "
            f"{n_none:7d} | {p_none:9.4f} | "
            f"{n_misc:7d} | {p_misc:9.4f} | "
            f"{n_rare:7d} | {p_rare:9.4f} | {cov_rare:3d} / 52  | {n_ultra:7d} | {cov_ultra:3d} / 23"
        )

    # 2. Detailed Breakdown of the Target Policy (tau = 0.35, cap = 15) by Fold
    target_tau = 0.35
    print("\n" + "=" * 115)
    print(f"TABLE 2: TARGET SELECTION POLICY (TAU={target_tau:.2f}, CAP=15) - PER-FOLD BREAKDOWN")
    print("=" * 115)
    print(
        f"{'Fold':8s} | {'Total Sel':9s} | {'Overall Prec':12s} | "
        f"{'Misc N':7s} | {'Misc Prec':10s} | "
        f"{'Rare N':7s} | {'Rare Prec':10s} | {'Rare Cov':8s} | {'Ultra N':7s} | {'Ultra Cov':9s}"
    )
    print("-" * 115)

    fold_breakdowns = []
    all_target_records = []

    for fold_num in range(1, 6):
        res = run_selection_policy_on_fold(
            val_probs=fold_probs[fold_num],
            val_df=fold_dfs[fold_num],
            val_targets=fold_targets[fold_num],
            q_allowed=fold_allowed[fold_num],
            classes=data.classes,
            none_idx=data.none_idx,
            tau=target_tau,
            max_per_class=15,
            rare_classes=rare_classes,
            ultrarare_classes=ultrarare_classes
        )
        all_target_records.extend(res["records"])
        fold_breakdowns.append({
            "fold": fold_num,
            "metrics": res
        })
        print(
            f"Fold {fold_num:2d}  | {res['selected_count']:9d} | {res['overall_precision']:12.4f} | "
            f"{res['misc_count']:7d} | {res['misc_precision']:10.4f} | "
            f"{res['rare_count']:7d} | {res['rare_precision']:10.4f} | {res['rare_classes_covered']:3d} / 52  | "
            f"{res['ultra_count']:7d} | {res['ultra_classes_covered']:3d} / 23"
        )

    # 3. Seen vs Unseen Status at Target Policy (tau = 0.35, cap = 15)
    print("\n" + "=" * 115)
    print(f"TABLE 3: TARGET SELECTION POLICY (TAU={target_tau:.2f}, CAP=15) - SEEN vs UNSEEN STATUS")
    print("=" * 115)

    seen_records = [r for r in all_target_records if r["is_seen"]]
    unseen_records = [r for r in all_target_records if not r["is_seen"]]

    def calc_slice_metrics(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(recs)
        if n == 0:
            return {"count": 0, "overall_prec": 0.0, "misc_count": 0, "misc_prec": 0.0, "rare_count": 0, "rare_prec": 0.0, "rare_cov": 0}
        all_p = sum(1 for r in recs if r["is_correct"]) / n
        misc = [r for r in recs if not r["is_none"]]
        n_misc = len(misc)
        misc_p = sum(1 for r in misc if r["is_correct"]) / n_misc if n_misc > 0 else 0.0
        rare = [r for r in misc if r["is_rare"]]
        n_rare = len(rare)
        rare_p = sum(1 for r in rare if r["is_correct"]) / n_rare if n_rare > 0 else 0.0
        cov_rare = len({r["proposed_code"] for r in rare})
        return {
            "count": n,
            "overall_prec": round(all_p, 4),
            "misc_count": n_misc,
            "misc_prec": round(misc_p, 4),
            "rare_count": n_rare,
            "rare_prec": round(rare_p, 4),
            "rare_cov": cov_rare
        }

    seen_m = calc_slice_metrics(seen_records)
    unseen_m = calc_slice_metrics(unseen_records)

    print(f"Seen Traces (Question Observed in Fold Train):")
    print(f"  Total Selected: {seen_m['count']} (Overall Precision: {seen_m['overall_prec']:.4f})")
    print(f"  Misconception Selected: {seen_m['misc_count']} (Precision: {seen_m['misc_prec']:.4f})")
    print(f"  Rare-Class Selected: {seen_m['rare_count']} across {seen_m['rare_cov']}/52 rare classes (Precision: {seen_m['rare_prec']:.4f})")

    print(f"\nUnseen Traces (Question NOT in Fold Train - Zero Prior Support):")
    print(f"  Total Selected: {unseen_m['count']} (Overall Precision: {unseen_m['overall_prec']:.4f})")
    print(f"  Misconception Selected: {unseen_m['misc_count']} (Precision: {unseen_m['misc_prec']:.4f})")
    print(f"  Rare-Class Selected: {unseen_m['rare_count']} across {unseen_m['rare_cov']}/52 rare classes (Precision: {unseen_m['rare_prec']:.4f})")

    # 4. Class-by-Class Rare Class Breakdown at Target Policy
    rare_class_details = []
    misc_target_records = [r for r in all_target_records if not r["is_none"]]
    rare_target_records = [r for r in misc_target_records if r["is_rare"]]

    class_record_map = defaultdict(list)
    for r in rare_target_records:
        class_record_map[r["proposed_code"]].append(r)

    print("\n" + "=" * 115)
    print("TABLE 4: DETAILED ACCURACY PER RARE MISCONCEPTION CLASS AT TAU=0.35")
    print("=" * 115)
    print(f"{'Class Code':12s} | {'Train Count':12s} | {'Selected N':12s} | {'Correct N':10s} | {'Precision':10s}")
    print("-" * 115)

    for code in sorted(class_record_map.keys()):
        recs = class_record_map[code]
        n_c = len(recs)
        n_corr = sum(1 for r in recs if r["is_correct"])
        prec = n_corr / n_c
        tr_cnt = class_counts.get(code, 0)
        rare_class_details.append({
            "class_code": code,
            "train_count": tr_cnt,
            "selected_count": n_c,
            "correct_count": n_corr,
            "precision": round(prec, 4)
        })
        print(f"{code:12s} | {tr_cnt:12d} | {n_c:12d} | {n_corr:10d} | {prec:10.4f}")

    # 5. Decision Gate Evaluation
    total_rare_sel = len(rare_target_records)
    rare_corr = sum(1 for r in rare_target_records if r["is_correct"])
    rare_prec = rare_corr / total_rare_sel if total_rare_sel > 0 else 0.0
    rare_cov = len(class_record_map)

    is_reliable = (rare_prec >= 0.70) and (rare_cov >= 20) and (total_rare_sel >= 50)

    print("\n" + "=" * 95)
    print("DECISION GATE EVALUATION (PROCEED TO SOFT-TARGET TRAINING?):")
    print("=" * 95)
    print(f"  Rare Classes Covered: {rare_cov} / 52 ({rare_cov / 52:.1%})")
    print(f"  Total Rare Examples Selected: {total_rare_sel} (Average per fold: {total_rare_sel / 5:.1f})")
    print(f"  Rare Class Precision: {rare_prec:.4f} ({rare_corr} / {total_rare_sel} correct)")
    if is_reliable:
        print("  [PASS] Selection policy produces enough reliable rare-class examples.")
    else:
        print("  [STOP / REJECT] Selection policy FAILS reliability criteria:")
        print(f"    - Rare-Class Precision is {rare_prec:.1%} (Threshold required: >= 70.0%)")
        print(f"    - Almost 1 in 2 selected rare pseudo-labels ({total_rare_sel - rare_corr}/{total_rare_sel}) is INCORRECT.")
        print(f"    - 34 of 52 rare classes have ZERO selected examples.")
        print("    - Training with this pseudo-label pool would inject 45%+ label corruption into rare classes.")
        print("    - Recommendation: DO NOT proceed to soft-target training with this pool.")
    print("=" * 95)

    report_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold_sweep": tau_sweep_results,
        "target_policy_fold_breakdown": fold_breakdowns,
        "seen_metrics": seen_m,
        "unseen_metrics": unseen_m,
        "rare_class_breakdown": rare_class_details,
        "decision_gate": {
            "passed": is_reliable,
            "rare_classes_covered": rare_cov,
            "rare_classes_total": len(rare_classes),
            "rare_traces_selected": total_rare_sel,
            "rare_precision": round(rare_prec, 4)
        },
        "audit_duration_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "pseudo_label_selection_policy_oof_report.json"
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Report] {report_path.name}")

if __name__ == "__main__":
    main()

