"""Generate SOTA + Transductive Bayesian Ensemble Submission.

Constructs the optimal diversity candidate for final competition selection:
1. Base SOTA Model (M10, Public LB: 0.52633):
   70% Prior (m_q=10.0) + 20% Dual-Vision (ConvNeXt-Tiny + MobileNetV3) + 10% Text
2. Transductive Bayesian Refinement (Public LB: 0.52583):
   Bayesian test consensus smoothing across 492 test questions (37 unseen in train)
3. Conservative Ensemble Blend:
   P_Ensemble = 0.60 * P_SOTA + 0.40 * P_Transductive
4. Output:
   submissions/multimodal_sota_transductive_blend_submission.csv
   (3,498 rows, 100% valid schema)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, cast
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


def generate_sota_transductive_blend(
    w_sota: float = 0.60,
    w_transductive: float = 0.40,
    alpha_seen: float = 0.15,
    m_unseen: float = 5.0
) -> Dict[str, Any]:
    print("\n" + "=" * 65)
    print("GENERATING SOTA + TRANSDUCTIVE BAYESIAN ENSEMBLE SUBMISSION")
    print("=" * 65 + "\n")

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
    print(f"[Ensemble] Computing in-fold base prior (m_q=10.0)...")
    test_prior_probs = np.zeros((n_test, n_classes), dtype=np.float32)
    for fold_idx, (tr_idx, _) in enumerate(folds, start=1):
        fold_tr = train_df.iloc[tr_idx]
        engine = QuestionPriorEngine(classes=classes, smoothing_question=10.0, smoothing_topic_grade=5.0)
        engine.fit(fold_tr)
        p_fold, _ = engine.predict_proba(test_df)
        test_prior_probs += p_fold / len(folds)

    # 2. Dual-Vision Modality (ConvNeXt-Tiny + MobileNetV3)
    print(f"[Ensemble] Loading Dual-Vision (ConvNeXt + MobileNetV3) test predictions...")
    test_vis_cn = np.load(PROJECT_ROOT / "artifacts" / "test_convnext_vision.npy")
    test_vis_mn = np.load(PROJECT_ROOT / "artifacts" / "test_finetuned_vision.npy")
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    t_te = [str(t) for t in test_df["topic_name"]]
    g_te = [int(float(g)) for g in test_df["grade"]]
    test_vis_mn_masked = domain_masker.apply_mask_batch(test_vis_mn, classes, t_te, g_te, damping=0.85)
    test_dual_vis = 0.5 * test_vis_cn + 0.5 * test_vis_mn_masked

    # 3. Context Text Modality
    print(f"[Ensemble] Loading Context Text SGD test predictions...")
    test_text = np.load(PROJECT_ROOT / "artifacts" / "test_text.npy")

    # 4. SOTA M10 Posterior (Public LB: 0.52633 🏆)
    p_m10 = 0.70 * test_prior_probs + 0.20 * test_dual_vis + 0.10 * test_text

    # 5. Transductive Bayesian Refinement
    print(f"[Ensemble] Computing Transductive Bayesian test consensus (alpha={alpha_seen}, m_unseen={m_unseen})...")
    q_groups = test_df.groupby("question_id").indices
    tr_q_set = set(train_df["question_id"])

    refined_prior_probs = np.zeros_like(test_prior_probs)
    unseen_count = 0
    seen_count = 0

    for q_id, q_indices in q_groups.items():
        n_q = len(q_indices)
        mu_q = p_m10[q_indices].mean(axis=0)

        if q_id not in tr_q_set:
            base_tg_prior = test_prior_probs[q_indices[0]]
            refined_q = (n_q * mu_q + m_unseen * base_tg_prior) / (n_q + m_unseen)
            for idx in q_indices:
                refined_prior_probs[idx] = refined_q
            unseen_count += n_q
        else:
            for idx in q_indices:
                refined_prior_probs[idx] = (1.0 - alpha_seen) * test_prior_probs[idx] + alpha_seen * mu_q
            seen_count += n_q

    r_sums = refined_prior_probs.sum(axis=1, keepdims=True)
    refined_prior_probs = np.divide(refined_prior_probs, r_sums, where=r_sums > 0)

    p_transductive = 0.70 * refined_prior_probs + 0.20 * test_dual_vis + 0.10 * test_text

    # 6. Ensemble Blend: 60% SOTA M10 + 40% Transductive
    print(f"[Ensemble] Blending {w_sota:.0%} SOTA M10 + {w_transductive:.0%} Transductive...")
    p_blend = w_sota * p_m10 + w_transductive * p_transductive

    # 7. Extract Top-3 Rank Predictions
    top_indices = np.argsort(-p_blend, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_indices]

    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_sota_transductive_blend_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"[Ensemble] Saved blended submission to: {out_path} ({len(submission_df)} rows)")

    # 8. Schema & Sanity Validation
    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Ensemble] Submission validation status: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # 9. Agreement Statistics against SOTA M10
    sota_df = cast(pd.DataFrame, pd.read_csv(PATHS.submissions_dir / "multimodal_dualvision_submission.csv"))
    diff_count = int((submission_df["root_code"] != sota_df["root_code"]).sum())
    diff_pct = (diff_count / len(submission_df)) * 100

    print(f"\n[Ensemble Analysis]")
    print(f"  Total test rows:        {len(submission_df)}")
    print(f"  Rows identical to SOTA: {len(submission_df) - diff_count} ({(100 - diff_pct):.2f}%)")
    print(f"  Rows diversified:       {diff_count} ({diff_pct:.2f}%)")

    meta = {
        "model_name": "SOTA_Transductive_Bayesian_Blend",
        "w_sota": w_sota,
        "w_transductive": w_transductive,
        "alpha_seen": alpha_seen,
        "m_unseen": m_unseen,
        "total_test_rows": len(submission_df),
        "rows_diversified_from_sota": diff_count,
        "diversification_pct": diff_pct,
        "output_file": str(out_path)
    }

    out_meta = PROJECT_ROOT / "artifacts" / "sota_transductive_blend_metadata.json"
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[Ensemble] Metadata written to: {out_meta}")

    return meta


if __name__ == "__main__":
    generate_sota_transductive_blend()
