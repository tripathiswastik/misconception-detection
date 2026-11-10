"""Test-Mix Validation & Transductive Prior-Confidence Experiment.

1. Test-Mix Validation Architecture:
   - Seen-Question Evaluation: 5-fold out-of-row split for all questions with >= 2 traces (1,419 traces).
     For every validation trace, other traces from the same question are strictly in training.
     Evaluates transductive performance (95.7% of Kaggle test set).
   - Unseen-Question Evaluation: 5-fold GroupKFold strictly on question_id (100% question held-out).
     Evaluates inductive performance (4.3% of Kaggle test set).
   - Composite Metric: 0.957 * MAP@3_seen + 0.043 * MAP@3_unseen.

2. Prior-Confidence Experiments on SOTA 550 Champion Pipeline:
   - Baseline: Standard SOTA 550 pipeline with fixed prior smoothing (m_q=10.0).
   - Exp 1 (Adaptive Shrinkage): m_q(N_q) = m_0 / (N_q^0.7), allowing high-count questions to express counts.
   - Exp 2 (Support Count Weighting): Prior weight scaled by N_q / (N_q + k).
   - Exp 3 (Entropy & Diversity): Downweights prior when question label entropy H_q is high (diverse errors).
   - Exp 4 (Model Agreement): Modulates prior based on Bhattacharyya agreement between multimodal model & prior.
   - Exp 5 (Composite Confidence Engine): Combines support, purity/entropy, and model agreement.

Strictly protects Rank 1 during all evaluations.
Outputs: artifacts/testmix_prior_confidence_report.json
"""
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, cast
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import normalize

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

def bhattacharyya_coeff(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Compute Bhattacharyya agreement coefficient between probability distributions."""
    return np.sum(np.sqrt(np.maximum(p1, 1e-12) * np.maximum(p2, 1e-12)), axis=1)

def compute_question_stats(train_df: pd.DataFrame, target_col: str = "root_code") -> Dict[str, Dict[str, float]]:
    """Compute per-question support count, normalized Shannon entropy, and top margin."""
    stats = {}
    for qid, grp in train_df.groupby("question_id"):
        nq = len(grp)
        labels = grp[target_col].fillna("NONE").values
        _, counts = np.unique(labels, return_counts=True)
        probs = counts / nq
        # Normalized entropy: 0 when pure, 1 when maximally uniform
        entropy = -float(np.sum(probs * np.log(probs + 1e-12))) / (np.log(max(len(probs), 2)))
        top_counts = np.sort(counts)[::-1]
        top_margin = float(top_counts[0] - (top_counts[1] if len(top_counts) > 1 else 0)) / nq
        stats[str(qid)] = {
            "nq": float(nq),
            "entropy": float(entropy),
            "top_margin": float(top_margin),
            "top_ratio": float(top_counts[0] / nq)
        }
    return stats

def main():
    print("=" * 80)
    print("TEST-MIX VALIDATION & PRIOR-CONFIDENCE OPTIMIZATION")
    print("Mirroring Competition Test Set: 95.7% Seen Questions | 4.3% Unseen Questions")
    print("=" * 80)

    start_time = time.time()

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

    # 1. Load Multimodal Component Representations (Leak-Free Backbones)
    print("Loading multimodal representations...")
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

    # Base Model (multimodal ensemble without prior)
    p_base_all = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # -------------------------------------------------------------------------
    # 2. Partition Validation into Seen-Question Folds and Unseen-Question Folds
    # -------------------------------------------------------------------------
    # Seen questions: questions with >= 2 traces
    q_counts = train_df["question_id"].value_counts()
    multi_qids = set(q_counts[q_counts >= 2].index)
    multi_mask = train_df["question_id"].isin(multi_qids).values
    multi_indices = np.where(multi_mask)[0]

    print(f"Total samples: {n_samples}")
    print(f"Seen-Question pool: {len(multi_indices)} traces across {len(multi_qids)} questions (100% seen in train)")
    print(f"Unseen-Question pool: 5-Fold GroupKFold across all {len(np.unique(groups))} questions")

    # Construct balanced 5 folds for Seen-Question pool (every trace's question is present in its training fold)
    seen_folds: List[Tuple[np.ndarray, np.ndarray]] = []
    np.random.seed(42)
    # Stratify by question_id so traces of each question are distributed across folds
    fold_assignments = np.zeros(len(multi_indices), dtype=int)
    for qid in multi_qids:
        q_locs = np.where(train_df.iloc[multi_indices]["question_id"] == qid)[0]
        # Distribute randomly across the 5 folds
        perm = np.random.permutation(len(q_locs))
        for rank, p_idx in enumerate(perm):
            fold_assignments[q_locs[p_idx]] = rank % 5

    for f in range(5):
        val_loc = np.where(fold_assignments == f)[0]
        trn_loc = np.where(fold_assignments != f)[0]
        val_glob = multi_indices[val_loc]
        # Trn includes remaining multi-trace rows PLUS the single-trace questions
        single_indices = np.where(~multi_mask)[0]
        trn_glob = np.concatenate([multi_indices[trn_loc], single_indices])
        seen_folds.append((trn_glob, val_glob))

    # Verify that in seen_folds, 100% of validation questions are present in training!
    for f, (tr, va) in enumerate(seen_folds, 1):
        tr_q = set(train_df.iloc[tr]["question_id"])
        va_q = set(train_df.iloc[va]["question_id"])
        assert va_q.issubset(tr_q), f"Seen fold {f} contains unseen questions!"
    print("Verified: 100% of validation questions in seen_folds have support in their training fold.")

    # Unseen folds: GroupKFold
    outer_gkf = GroupKFold(n_splits=5)
    unseen_folds = list(outer_gkf.split(train_df, y_indices, groups=groups))

    # -------------------------------------------------------------------------
    # 3. Pipeline Evaluation Function
    # -------------------------------------------------------------------------
    def evaluate_pipeline(
        m_q_base: float = 10.0,
        adaptive_nq: bool = False,
        adaptive_power: float = 0.5,
        use_confidence_modulation: bool = False,
        conf_mode: str = "none",  # 'support', 'entropy', 'margin', 'agreement', 'composite'
        conf_power: float = 1.0,
    ) -> Dict[str, Any]:
        """Evaluates pipeline on both Seen-Question split and Unseen-Question split."""

        # A. Evaluate on Seen-Question Split
        seen_y_all = []
        seen_preds_all = []
        seen_fold_scores = []

        for f_idx, (trn_idx, val_idx) in enumerate(seen_folds, 1):
            trn_df = train_df.iloc[trn_idx]
            val_df = train_df.iloc[val_idx]
            val_y = y_raw[val_idx].tolist()
            n_val = len(val_idx)

            # Fit prior engine on trn_df
            prior_engine = AdaptiveQuestionPriorEngine(
                classes=classes,
                m_q_base=m_q_base,
                m_tg=5.0,
                adaptive_nq=adaptive_nq,
                adaptive_power=adaptive_power
            ).fit(trn_df)

            val_prior = prior_engine.predict_proba(val_df).astype(np.float64)

            # In-fold frequency calibration
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

            # Question-allowed boost
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

            # Confidence Modulation on prior
            if use_confidence_modulation:
                q_stats = compute_question_stats(trn_df)
                conf_weights = np.ones(n_val, dtype=np.float64)

                for i in range(n_val):
                    qid = str(val_df.iloc[i]["question_id"])
                    st = q_stats.get(qid, {"nq": 1.0, "entropy": 1.0, "top_margin": 0.0, "top_ratio": 1.0})
                    nq = st["nq"]
                    ent = st["entropy"]
                    mrg = st["top_margin"]

                    if conf_mode == "support":
                        # Weight proportional to question sample support
                        conf = float(nq / (nq + 3.0))
                    elif conf_mode == "entropy":
                        # Weight inverse to label entropy (pure questions = 1.0, high entropy = 0.3)
                        conf = float(1.0 - 0.7 * ent)
                    elif conf_mode == "margin":
                        # Weight proportional to dominant class margin
                        conf = float(0.5 + 0.5 * mrg)
                    elif conf_mode == "agreement":
                        # Agreement between multimodal base model and empirical prior
                        agree = float(np.sum(np.sqrt(np.maximum(p_champ_val[i], 1e-12) * np.maximum(val_prior[i], 1e-12))))
                        conf = float(np.clip(agree * 1.2, 0.4, 1.5))
                    elif conf_mode == "composite":
                        # Combined: Support * (1 - entropy) * Agreement
                        supp_factor = float(nq / (nq + 2.0))
                        ent_factor = float(1.0 - 0.5 * ent)
                        agree = float(np.sum(np.sqrt(np.maximum(p_champ_val[i], 1e-12) * np.maximum(val_prior[i], 1e-12))))
                        conf = float(np.clip(supp_factor * ent_factor * (0.5 + agree), 0.3, 1.6))
                    else:
                        conf = 1.0

                    conf_weights[i] = conf ** conf_power

                # Modulate prior contribution
                val_prior_mod = val_prior * conf_weights[:, None]
                val_prior_mod = val_prior_mod / np.maximum(val_prior_mod.sum(axis=1, keepdims=True), 1e-12)
            else:
                val_prior_mod = val_prior

            p_clean_val = (
                0.6448 * val_prior_mod +
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

            score_f = calculate_map3(val_y, val_top3)
            seen_fold_scores.append(score_f)
            seen_y_all.extend(val_y)
            seen_preds_all.extend(val_top3)

        seen_map3 = calculate_map3(seen_y_all, seen_preds_all)

        # B. Evaluate on Unseen-Question Split (5-Fold GroupKFold)
        unseen_y_all = []
        unseen_preds_all = []
        unseen_fold_scores = []

        for f_idx, (trn_idx, val_idx) in enumerate(unseen_folds, 1):
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

            # No question-allowed boost for unseen questions
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

            score_f = calculate_map3(val_y, val_top3)
            unseen_fold_scores.append(score_f)
            unseen_y_all.extend(val_y)
            unseen_preds_all.extend(val_top3)

        unseen_map3 = calculate_map3(unseen_y_all, unseen_preds_all)

        # Composite Test-Mix MAP@3 (95.7% Seen + 4.3% Unseen)
        composite_map3 = 0.957 * seen_map3 + 0.043 * unseen_map3

        return {
            "seen_map3": round(seen_map3, 5),
            "unseen_map3": round(unseen_map3, 5),
            "composite_map3": round(composite_map3, 5),
            "seen_folds": [round(s, 5) for s in seen_fold_scores],
            "unseen_folds": [round(s, 5) for s in unseen_fold_scores]
        }

    # -------------------------------------------------------------------------
    # 4. Run Systematic Benchmark Matrix
    # -------------------------------------------------------------------------
    experiments = [
        {"name": "M0: Baseline SOTA 550 (m_q=10.0 fixed)", "args": {"m_q_base": 10.0, "adaptive_nq": False}},
        {"name": "M1: Adaptive Shrinkage (m_q=10, pow=0.5)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.5}},
        {"name": "M2: Adaptive Shrinkage (m_q=10, pow=0.7)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7}},
        {"name": "M3: Adaptive Shrinkage (m_q=8, pow=0.7)", "args": {"m_q_base": 8.0, "adaptive_nq": True, "adaptive_power": 0.7}},
        {"name": "M4: Support Count Modulation (pow=0.5)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7, "use_confidence_modulation": True, "conf_mode": "support", "conf_power": 0.5}},
        {"name": "M5: Entropy Diversity Modulation (pow=0.5)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7, "use_confidence_modulation": True, "conf_mode": "entropy", "conf_power": 0.5}},
        {"name": "M6: Margin Modulation (pow=0.5)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7, "use_confidence_modulation": True, "conf_mode": "margin", "conf_power": 0.5}},
        {"name": "M7: Model Agreement Modulation", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7, "use_confidence_modulation": True, "conf_mode": "agreement", "conf_power": 0.5}},
        {"name": "M8: Composite Confidence Engine (Support + Entropy + Agreement)", "args": {"m_q_base": 10.0, "adaptive_nq": True, "adaptive_power": 0.7, "use_confidence_modulation": True, "conf_mode": "composite", "conf_power": 0.5}},
    ]

    print("\nRunning systematic prior-confidence benchmark across Test-Mix validation splits...")
    print(f"{'Experiment':<52} | {'Seen MAP@3':<10} | {'Unseen MAP@3':<12} | {'Composite':<10} | {'Delta vs M0':<12}")
    print("-" * 105)

    base_composite = 0.0
    benchmark_results = []

    for idx, exp in enumerate(experiments):
        res = evaluate_pipeline(**exp["args"])
        if idx == 0:
            base_composite = res["composite_map3"]
            delta = 0.0
        else:
            delta = res["composite_map3"] - base_composite

        benchmark_results.append({
            "experiment": exp["name"],
            "config": exp["args"],
            "metrics": res,
            "delta_composite": round(delta, 5)
        })

        print(f"{exp['name']:<52} | {res['seen_map3']:<10.5f} | {res['unseen_map3']:<12.5f} | {res['composite_map3']:<10.5f} | {delta:<+12.5f}")

    # Save Persistent Report Artifact
    out_artifact = {
        "benchmark_name": "testmix_prior_confidence_optimization",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_proportions": {"seen_questions": 0.957, "unseen_questions": 0.043},
        "experiments": benchmark_results,
        "total_runtime_seconds": round(time.time() - start_time, 2)
    }

    report_path = ART_DIR / "testmix_prior_confidence_report.json"
    with open(report_path, "w") as f:
        json.dump(out_artifact, f, indent=2)
    print(f"\n[Saved Persistent Report Artifact] {report_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

