"""Transductive Test-Time Bayesian Prior Refinement & Consensus Blending.

Exploits test set graph structure (~7.1 traces per question across 492 test questions):
1. Computes soft empirical question posteriors mu_q from SOTA M10 test predictions.
2. For unseen questions (37 questions, 150 traces): Replaces generic (topic, grade) backoff
   with Bayesian smoothed test-consensus prior.
3. For seen questions (455 questions, 3,348 traces): Refines the prior with conservative (alpha=0.15)
   empirical test evidence.
4. Synthesizes individual visual stroke evidence with refined transductive priors.

Outputs:
- submissions/multimodal_transductive_submission.csv (3,498 rows, valid schema)
- artifacts/transductive_metadata.json
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.splits import FoldSplitter
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.diagnoser.question_prior import QuestionPriorEngine
from src.fusion.validator import validate_submission_file


def run_transductive_refinement(alpha_seen: float = 0.15, m_unseen: float = 5.0) -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Transductive] Running Bayesian Test-Time Prior Refinement")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_test = len(test_df)

    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 1. Base Question Prior on test (m_q=10.0)
    print(f"[Transductive] Computing in-fold base prior (m_q=10.0)...")
    test_prior_probs = np.zeros((n_test, n_classes), dtype=np.float32)
    for fold_idx, (tr_idx, _) in enumerate(folds, start=1):
        fold_tr = train_df.iloc[tr_idx]
        engine = QuestionPriorEngine(classes=classes, smoothing_question=10.0, smoothing_topic_grade=5.0)
        engine.fit(fold_tr)
        p_fold, _ = engine.predict_proba(test_df)
        test_prior_probs += p_fold / len(folds)

    # 2. Dual-Vision Modality
    test_vis_cn = np.load(PROJECT_ROOT / "artifacts" / "test_convnext_vision.npy")
    test_vis_mn = np.load(PROJECT_ROOT / "artifacts" / "test_finetuned_vision.npy")
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    t_te = [str(t) for t in test_df["topic_name"]]
    g_te = [int(float(g)) for g in test_df["grade"]]
    test_vis_mn_masked = domain_masker.apply_mask_batch(test_vis_mn, classes, t_te, g_te, damping=0.85)
    test_dual_vis = 0.5 * test_vis_cn + 0.5 * test_vis_mn_masked

    # 3. Context Text Modality
    test_text = np.load(PROJECT_ROOT / "artifacts" / "test_text.npy")

    # 4. M10 Base Posterior (LB 0.52633)
    p_m10 = 0.70 * test_prior_probs + 0.20 * test_dual_vis + 0.10 * test_text

    # 5. Transductive Test-Time E-Step: Compute soft consensus mu_q per test question
    print(f"[Transductive] Computing empirical test question consensus across {test_df['question_id'].nunique()} questions...")
    q_groups = test_df.groupby("question_id").indices
    tr_q_set = set(train_df["question_id"])

    refined_prior_probs = np.zeros_like(test_prior_probs)
    unseen_count = 0
    seen_count = 0

    for q_id, q_indices in q_groups.items():
        n_q = len(q_indices)
        mu_q = p_m10[q_indices].mean(axis=0)

        if q_id not in tr_q_set:
            # Unseen Question in train:
            # Base prior was generic topic-grade backoff.
            # Replace with Bayesian shrinkage between test consensus and topic-grade prior:
            base_tg_prior = test_prior_probs[q_indices[0]]  # was backed off to tg
            refined_q = (n_q * mu_q + m_unseen * base_tg_prior) / (n_q + m_unseen)
            for idx in q_indices:
                refined_prior_probs[idx] = refined_q
            unseen_count += n_q
        else:
            # Seen Question in train:
            # Blend training prior with test empirical evidence
            for idx in q_indices:
                refined_prior_probs[idx] = (1.0 - alpha_seen) * test_prior_probs[idx] + alpha_seen * mu_q
            seen_count += n_q

    # Row normalize refined prior
    r_sums = refined_prior_probs.sum(axis=1, keepdims=True)
    refined_prior_probs = np.divide(refined_prior_probs, r_sums, where=r_sums > 0)
    print(f"[Transductive] Updated {seen_count} seen traces and {unseen_count} unseen traces.")

    # 6. Final Transductive Blend:
    # 70% Refined Prior + 20% Dual-Vision + 10% Text
    fused_transductive = 0.70 * refined_prior_probs + 0.20 * test_dual_vis + 0.10 * test_text

    # 7. Extract Top-3 predictions
    top_indices = np.argsort(-fused_transductive, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_indices]

    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_transductive_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"[Transductive] Saved submission to: {out_path} ({len(submission_df)} rows)")

    # 8. Strict validation
    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Transductive] Validation status: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # Save metadata
    meta = {
        "model_name": "Transductive_Bayesian_Prior_Refinement",
        "alpha_seen": alpha_seen,
        "m_unseen": m_unseen,
        "seen_traces_updated": seen_count,
        "unseen_traces_updated": unseen_count,
        "total_test_questions": test_df["question_id"].nunique(),
        "unseen_questions_in_test": len([q for q in q_groups if q not in tr_q_set]),
        "output_file": str(out_path)
    }
    with open(PROJECT_ROOT / "artifacts" / "transductive_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    run_transductive_refinement()
