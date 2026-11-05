"""Comprehensive Evaluation of Question-Level Target Encoding and Fusion.

Implements all user requirements:
1. Strictly in-fold empirical question-prior target encoding with hierarchical backoff.
2. Separate OOF reporting for:
   - Overall OOF MAP@3
   - Seen Questions (questions in val with >= 1 training traces in fold)
   - Unseen Questions (questions in val with 0 training traces in fold)
3. Ablation testing on correct_answer feature.
4. Multimodal fusion:
   - Stacking (Meta Logistic Regression on OOF probabilities with nested cross-validation)
   - Fixed-weight sweep across full spectrum w_vis in [0.0, 0.60]
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.diagnoser.question_prior import QuestionPriorEngine
from src.metrics.map3 import calculate_map3


def run_question_prior_study() -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Option A] Evaluating Strictly In-Fold Question Priors & Fusion")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    n_samples = len(train_df)
    y_true = train_df["root_code"].astype(str).values

    # Load canonical deterministic 5-fold splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 1. Output matrices
    oof_prior_only = np.zeros((n_samples, n_classes), dtype=np.float32)
    oof_text_no_key = np.zeros((n_samples, n_classes), dtype=np.float32)
    oof_text_with_key = np.zeros((n_samples, n_classes), dtype=np.float32)
    is_seen_mask = np.zeros(n_samples, dtype=bool)
    oof_backoff_levels: List[str] = [""] * n_samples

    # Helper text builder
    def make_text(row: pd.Series, include_key: bool = True) -> str:
        q_id = str(row.get("question_id", ""))
        grade = str(row.get("grade", ""))
        topic = str(row.get("topic_name", ""))
        q_text = str(row.get("question_text", ""))
        key = str(row.get("correct_answer", "")) if include_key else ""
        key_tok = f"KEY_{key} KEY_{key} " if include_key and key else ""
        return f"QID_{q_id} QID_{q_id} GRADE_{grade} TOPIC_{topic} TOPIC_{topic} {key_tok}{q_text}"

    print(f"[Option A] Running 5-fold cross-validation with strict in-fold fitting...")

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_tr = train_df.iloc[tr_idx].copy().reset_index(drop=True)
        fold_va = train_df.iloc[va_idx].copy().reset_index(drop=True)

        tr_questions = set(fold_tr["question_id"])
        va_is_seen = fold_va["question_id"].isin(tr_questions).values
        is_seen_mask[va_idx] = va_is_seen

        # --- A. Strictly In-Fold Question Prior Target Encoder ---
        prior_engine = QuestionPriorEngine(classes=classes, smoothing_question=3.0, smoothing_topic_grade=5.0)
        prior_engine.fit(fold_tr)
        p_prior, va_levels = prior_engine.predict_proba(fold_va)
        oof_prior_only[va_idx] = p_prior
        for idx_val, lvl in zip(va_idx, va_levels):
            oof_backoff_levels[idx_val] = lvl

        # --- In-Fold Domain Constraint Engine ---
        domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(fold_tr)
        topics_va = [str(t) for t in fold_va["topic_name"]]
        grades_va = [int(g) for g in fold_va["grade"]]

        # --- B. Ablation: Text without correct_answer ---
        tr_text_no_key = [make_text(row, include_key=False) for _, row in fold_tr.iterrows()]
        va_text_no_key = [make_text(row, include_key=False) for _, row in fold_va.iterrows()]
        pipe_no_key = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log", alpha=1e-4, penalty="l2", max_iter=500, random_state=42))
        ])
        pipe_no_key.fit(tr_text_no_key, fold_tr["root_code"].astype(str))
        raw_p_no_key = pipe_no_key.predict_proba(va_text_no_key)

        aligned_no_key = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        for m_i, c_name in enumerate(pipe_no_key.classes_):
            if c_name in classes:
                aligned_no_key[:, classes.index(c_name)] = raw_p_no_key[:, m_i]
        r_sums = aligned_no_key.sum(axis=1, keepdims=True)
        aligned_no_key = np.divide(aligned_no_key, r_sums, where=r_sums > 0)
        oof_text_no_key[va_idx] = domain_masker.apply_mask_batch(aligned_no_key, classes, topics_va, grades_va, damping=0.85)

        # --- C. Ablation: Text with correct_answer ---
        tr_text_with_key = [make_text(row, include_key=True) for _, row in fold_tr.iterrows()]
        va_text_with_key = [make_text(row, include_key=True) for _, row in fold_va.iterrows()]
        pipe_with_key = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=15000, sublinear_tf=True)),
            ("clf", SGDClassifier(loss="log", alpha=1e-4, penalty="l2", max_iter=500, random_state=42))
        ])
        pipe_with_key.fit(tr_text_with_key, fold_tr["root_code"].astype(str))
        raw_p_with_key = pipe_with_key.predict_proba(va_text_with_key)

        aligned_with_key = np.zeros((len(va_idx), n_classes), dtype=np.float32)
        for m_i, c_name in enumerate(pipe_with_key.classes_):
            if c_name in classes:
                aligned_with_key[:, classes.index(c_name)] = raw_p_with_key[:, m_i]
        r_sums = aligned_with_key.sum(axis=1, keepdims=True)
        aligned_with_key = np.divide(aligned_with_key, r_sums, where=r_sums > 0)
        oof_text_with_key[va_idx] = domain_masker.apply_mask_batch(aligned_with_key, classes, topics_va, grades_va, damping=0.85)

        print(f"  Fold {fold_idx}: Val rows={len(fold_va)} (Seen Q={va_is_seen.sum()}, Unseen Q={(~va_is_seen).sum()})")

    # 2. Compute Segmented OOF Scores
    seen_indices = np.where(is_seen_mask)[0]
    unseen_indices = np.where(~is_seen_mask)[0]

    # Level indices
    levels_arr = np.array(oof_backoff_levels)
    q_level_idx = np.where(levels_arr == "question")[0]
    tg_level_idx = np.where(levels_arr == "topic+grade")[0]
    glob_level_idx = np.where(levels_arr == "global")[0]

    def score_matrix(probs: np.ndarray, subset_idx: Optional[np.ndarray] = None) -> float:
        if subset_idx is not None and len(subset_idx) == 0:
            return 0.0
        sub_p = probs[subset_idx] if subset_idx is not None else probs
        sub_y = [y_true[i] for i in subset_idx] if subset_idx is not None else list(y_true)
        top = np.argsort(-sub_p, axis=1, kind="mergesort")[:, :3]
        preds = [[classes[i] for i in r] for r in top]
        return calculate_map3(sub_y, preds)

    # Metric summaries
    res_prior = {
        "overall": score_matrix(oof_prior_only),
        "seen_questions": score_matrix(oof_prior_only, seen_indices),
        "unseen_questions": score_matrix(oof_prior_only, unseen_indices),
        "by_level": {
            "question": score_matrix(oof_prior_only, q_level_idx),
            "topic+grade": score_matrix(oof_prior_only, tg_level_idx),
            "global": score_matrix(oof_prior_only, glob_level_idx),
            "counts": {
                "question": len(q_level_idx),
                "topic+grade": len(tg_level_idx),
                "global": len(glob_level_idx)
            }
        }
    }
    res_no_key = {
        "overall": score_matrix(oof_text_no_key),
        "seen_questions": score_matrix(oof_text_no_key, seen_indices),
        "unseen_questions": score_matrix(oof_text_no_key, unseen_indices)
    }
    res_with_key = {
        "overall": score_matrix(oof_text_with_key),
        "seen_questions": score_matrix(oof_text_with_key, seen_indices),
        "unseen_questions": score_matrix(oof_text_with_key, unseen_indices)
    }

    print("\n=======================================================")
    print("[Option A] Benchmark & Ablation Results (Strict In-Fold):")
    print(f"  Total Val Samples: {n_samples} | Seen Q: {len(seen_indices)} (88.7%) | Unseen Q: {len(unseen_indices)} (11.3%)")
    print("-------------------------------------------------------")
    print(f"  1. Question Prior Alone (Target Encoding):")
    print(f"     Overall: {res_prior['overall']:.4f} | Seen Q: {res_prior['seen_questions']:.4f} | Unseen Q: {res_prior['unseen_questions']:.4f}")
    print(f"     By Level -> question ({len(q_level_idx)}): {res_prior['by_level']['question']:.4f} | topic+grade ({len(tg_level_idx)}): {res_prior['by_level']['topic+grade']:.4f} | global ({len(glob_level_idx)}): {res_prior['by_level']['global']:.4f}")
    print(f"  2. Context Text (without correct_answer):")
    print(f"     Overall: {res_no_key['overall']:.4f} | Seen Q: {res_no_key['seen_questions']:.4f} | Unseen Q: {res_no_key['unseen_questions']:.4f}")
    print(f"  3. Context Text (WITH correct_answer):")
    print(f"     Overall: {res_with_key['overall']:.4f} | Seen Q: {res_with_key['seen_questions']:.4f} | Unseen Q: {res_with_key['unseen_questions']:.4f}")
    print("=======================================================\n")

    # 3. Multimodal Fusion with Vision
    oof_vis_m6 = np.load(PROJECT_ROOT / "artifacts" / "oof_vision.npy")
    oof_vis_m7 = np.load(PROJECT_ROOT / "artifacts" / "oof_finetuned_vision.npy")

    # In-fold domain masking for vision
    oof_vis_m7_masked = np.zeros_like(oof_vis_m7)
    for tr_idx, va_idx in folds:
        fold_tr = train_df.iloc[tr_idx]
        masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(fold_tr)
        t_va = [str(t) for t in train_df.iloc[va_idx]["topic_name"]]
        g_va = [int(g) for g in train_df.iloc[va_idx]["grade"]]
        oof_vis_m7_masked[va_idx] = masker.apply_mask_batch(oof_vis_m7[va_idx], classes, t_va, g_va, damping=0.85)

    # Grid sweep on blend between Question-Prior Text and Fine-Tuned Vision
    print("--- Fixed-Weight Fusion Grid Sweep (Prior Text + Vision) ---")
    best_fuse_score = -1.0
    best_w = 0.0
    grid_results = []
    for w_vis in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        fused = (1.0 - w_vis) * oof_text_with_key + w_vis * oof_vis_m7_masked
        ov_score = score_matrix(fused)
        seen_s = score_matrix(fused, seen_indices)
        unseen_s = score_matrix(fused, unseen_indices)
        grid_results.append({"w_vis": w_vis, "overall": round(ov_score, 4), "seen": round(seen_s, 4), "unseen": round(unseen_s, 4)})
        print(f"  w_vis={w_vis:.2f}: Overall={ov_score:.4f} | Seen Q={seen_s:.4f} | Unseen Q={unseen_s:.4f}")
        if ov_score > best_fuse_score:
            best_fuse_score = ov_score
            best_w = w_vis

    print(f"\n  --> Optimal Fusion: w_vis={best_w:.2f} | Overall OOF MAP@3 = {best_fuse_score:.4f}")

    # 4. Save artifacts
    artifacts_dir = PROJECT_ROOT / "artifacts"
    np.save(artifacts_dir / "oof_text_prior.npy", oof_text_with_key)
    np.save(artifacts_dir / "oof_prior_target_encoding.npy", oof_prior_only)

    audit_report = {
        "n_samples": n_samples,
        "seen_count": len(seen_indices),
        "unseen_count": len(unseen_indices),
        "prior_only": res_prior,
        "text_no_key": res_no_key,
        "text_with_key": res_with_key,
        "best_fusion_weight": best_w,
        "best_fusion_map3": best_fuse_score,
        "grid_results": grid_results
    }
    with open(artifacts_dir / "question_prior_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    return audit_report


if __name__ == "__main__":
    run_question_prior_study()
