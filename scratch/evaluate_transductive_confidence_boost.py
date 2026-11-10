"""Systematic Confidence-Weighted Transductive Question Boost & Adaptive Prior Optimization.

Evaluates on the Test-Mix Validation:
- 95.7% Seen-Question split (out-of-row 5-fold evaluation, N=1,419)
- 4.3% Unseen-Question split (GroupKFold 5-fold evaluation, N=1,500)
- Composite MAP@3: 0.957 * Seen + 0.043 * Unseen

Compares:
- B0: Baseline Champion (Blunt 1.20x boost on all question-allowed classes, m_q=10.0 fixed)
- B1: Adaptive Shrinkage Prior (m_q=8.0, power=0.7)
- B2: Empirical Frequency-Weighted Boost (boosting classes proportional to count(q, c) / N_q)
- B3: Support-Gated Boost (modulating boost by question support count N_q)
- B4: Entropy-Gated Boost (reducing boost when question is noisy/diverse)
- B5: Agreement-Gated Boost (modulating boost when model and empirical prior agree)
- B6: Full Transductive Confidence Engine (combining frequency, support, entropy, and agreement)
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, Tuple, cast
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
    print("CONFIDENCE-WEIGHTED TRANSDUCTIVE BOOST BENCHMARK")
    print("Comparing Frequency, Support, Entropy, and Agreement Gates on Test-Mix Splits")
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

    # 1. Load Multimodal Representations
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

    # 2. Build Seen-Question Split and Unseen-Question Split
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

    # 3. Evaluation Function
    def eval_config(
        m_q_base: float = 10.0,
        adaptive_power: float = 0.5,
        adaptive_nq: bool = False,
        boost_mode: str = "uniform_120",  # 'uniform_120', 'frequency', 'support', 'entropy', 'agreement', 'composite'
        boost_scale: float = 0.25,
    ) -> Dict[str, float]:

        # Seen-Question Split
        seen_y_all = []
        seen_preds_all = []

        for trn_idx, val_idx in seen_folds:
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            val_y = y_raw[val_idx].tolist()
            n_val = len(val_idx)

            prior_engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=m_q_base,
                m_tg=5.0,
                adaptive_nq=adaptive_nq,
                adaptive_power=adaptive_power
            ).fit(trn_df)
            val_prior = prior_engine.predict_proba(val_df).astype(np.float64)

            # Frequency calibration
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

            # Transductive Question Boost Calculation
            p_val_boosted = np.copy(p_val_cal)
            q_grp = trn_df.groupby("question_id")

            # Precalculate question empirical distributions
            q_class_dist = {}
            q_entropy = {}
            for qid, grp in q_grp:
                nq = len(grp)
                counts = Counter(grp["root_code"].fillna("NONE").values)
                dist = {c: cnt / nq for c, cnt in counts.items()}
                q_class_dist[qid] = (dist, nq)
                probs = np.array(list(dist.values()))
                ent = -float(np.sum(probs * np.log(probs + 1e-12))) / (np.log(max(len(probs), 2)))
                q_entropy[qid] = ent

            for i in range(n_val):
                qid = val_df.iloc[i]["question_id"]
                if qid in q_class_dist:
                    dist, nq = q_class_dist[qid]
                    ent = q_entropy.get(qid, 0.0)

                    for ci, c in enumerate(classes):
                        if c != "NONE" and c in dist:
                            freq = dist[c]

                            if boost_mode == "uniform_120":
                                factor = 1.20
                            elif boost_mode == "frequency":
                                factor = 1.0 + boost_scale * (freq ** 0.5)
                            elif boost_mode == "support":
                                supp_w = nq / (nq + 2.0)
                                factor = 1.0 + boost_scale * supp_w * (freq ** 0.5)
                            elif boost_mode == "entropy":
                                ent_w = max(0.2, 1.0 - 0.7 * ent)
                                factor = 1.0 + boost_scale * ent_w * (freq ** 0.5)
                            elif boost_mode == "agreement":
                                p_model_c = p_val_cal[i, ci]
                                agree_w = min(1.5, max(0.5, p_model_c / (val_prior[i, ci] + 1e-6)))
                                factor = 1.0 + boost_scale * (freq ** 0.5) * (0.5 + 0.5 * agree_w)
                            elif boost_mode == "composite":
                                supp_w = nq / (nq + 2.0)
                                ent_w = max(0.3, 1.0 - 0.5 * ent)
                                factor = 1.0 + boost_scale * supp_w * ent_w * (freq ** 0.5)
                            else:
                                factor = 1.20

                            p_val_boosted[i, ci] *= factor

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

            # Precision Gating with Rank-1 protected
            p_bin_err_val = oof_bin_err[val_idx]
            q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in trn_df.groupby("question_id")}
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

            seen_y_all.extend(val_y)
            seen_preds_all.extend(val_top3)

        seen_score = calculate_map3(seen_y_all, seen_preds_all)

        # Unseen-Question Split
        unseen_y_all = []
        unseen_preds_all = []

        for trn_idx, val_idx in unseen_folds:
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            val_y = y_raw[val_idx].tolist()

            prior_engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=m_q_base,
                m_tg=5.0,
                adaptive_nq=adaptive_nq,
                adaptive_power=adaptive_power
            ).fit(trn_df)
            val_prior = prior_engine.predict_proba(val_df).astype(np.float64)

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

            unseen_y_all.extend(val_y)
            unseen_preds_all.extend(val_top3)

        unseen_score = calculate_map3(unseen_y_all, unseen_preds_all)
        comp_score = 0.957 * seen_score + 0.043 * unseen_score

        return {
            "seen": round(seen_score, 5),
            "unseen": round(unseen_score, 5),
            "composite": round(comp_score, 5)
        }

    # Grid of targeted configurations
    configs = [
        {"name": "B0: Baseline SOTA 550 (Blunt 1.20x boost, m_q=10 fixed)", "args": {"m_q_base": 10.0, "adaptive_nq": False, "boost_mode": "uniform_120"}},
        {"name": "B1: Adaptive Shrinkage Prior (m_q=8.0, pow=0.7, uniform 1.20x)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "uniform_120"}},
        {"name": "B2: Frequency-Weighted Boost (scale=0.25)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "frequency", "boost_scale": 0.25}},
        {"name": "B3: Frequency-Weighted Boost (scale=0.30)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "frequency", "boost_scale": 0.30}},
        {"name": "B4: Frequency-Weighted Boost (scale=0.35)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "frequency", "boost_scale": 0.35}},
        {"name": "B5: Support-Gated Frequency Boost (scale=0.30)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "support", "boost_scale": 0.30}},
        {"name": "B6: Support-Gated Frequency Boost (scale=0.35)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "support", "boost_scale": 0.35}},
        {"name": "B7: Entropy-Gated Frequency Boost (scale=0.30)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "entropy", "boost_scale": 0.30}},
        {"name": "B8: Composite Confidence Engine (scale=0.30)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "composite", "boost_scale": 0.30}},
        {"name": "B9: Composite Confidence Engine (scale=0.35)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7, "boost_mode": "composite", "boost_scale": 0.35}},
    ]

    print(f"\n{'Configuration':<60} | {'Seen':<8} | {'Unseen':<8} | {'Composite':<10} | {'Delta':<10}")
    print("-" * 105)

    base_comp = 0.0
    records = []
    for idx, c in enumerate(configs):
        res = eval_config(**c["args"])
        if idx == 0:
            base_comp = res["composite"]
            delta = 0.0
        else:
            delta = res["composite"] - base_comp

        records.append({"name": c["name"], "res": res, "delta": round(delta, 5)})
        print(f"{c['name']:<60} | {res['seen']:<8.5f} | {res['unseen']:<8.5f} | {res['composite']:<10.5f} | {delta:<+10.5f}")

    # Save artifact
    out_path = ART_DIR / "transductive_confidence_boost_report.json"
    with open(out_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[Saved Persistent Report] {out_path.name}")
    print("=" * 80)

if __name__ == "__main__":
    main()

