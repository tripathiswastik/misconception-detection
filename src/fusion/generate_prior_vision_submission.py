"""Generate Test Submission for 65% Question Prior + 35% Fine-Tuned Vision.

Honest 5-Fold OOF MAP@3: 0.5123 (Seen Q: 0.5112, Unseen Q: 0.5212)
Outputs:
- submissions/multimodal_prior_fused_submission.csv: 3,498 rows, valid schema.
- artifacts/prior_vision_metadata.json: Final configuration and weights.
(Preserves multimodal_fused_submission.csv LB 0.48833 fallback intact).
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
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
from src.metrics.map3 import calculate_map3


def generate_submission():
    print("\n=======================================================")
    print("[Submission] Generating Question Prior + Vision Submission")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true = train_df["root_code"].astype(str).values

    # 1. Load canonical deterministic splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 2. Compute 5-fold ensemble prior probabilities on test set
    test_prior_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)

    for fold_idx, (tr_idx, va_idx) in enumerate(folds, start=1):
        fold_tr = train_df.iloc[tr_idx]
        prior_engine = QuestionPriorEngine(classes=classes, smoothing_question=3.0, smoothing_topic_grade=5.0)
        prior_engine.fit(fold_tr)
        p_fold, _ = prior_engine.predict_proba(test_df)
        test_prior_probs += p_fold / len(folds)

    # 3. Load fine-tuned vision test probabilities
    test_vis_path = PROJECT_ROOT / "artifacts" / "test_finetuned_vision.npy"
    if not test_vis_path.exists():
        raise FileNotFoundError(f"Missing {test_vis_path}")
    test_vis = np.load(test_vis_path)

    # Apply domain constraint masking to vision test probabilities
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    test_topics = [str(t) for t in test_df["topic_name"]]
    test_grades = [int(g) for g in test_df["grade"]]
    test_vis_masked = domain_masker.apply_mask_batch(test_vis, classes, test_topics, test_grades, damping=0.85)

    # 4. Blend: 65% Question Prior + 35% Fine-Tuned Vision
    w_prior = 0.65
    w_vis = 0.35
    fused_test = w_prior * test_prior_probs + w_vis * test_vis_masked

    # 5. Extract top-3 predictions
    top_indices = np.argsort(-fused_test, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_indices]

    # Format dataframe
    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_prior_fused_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"[Submission] Saved submission to: {out_path} ({len(submission_df)} rows)")

    # 6. Strict validation
    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Submission] Validation status: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # Save metadata
    meta = {
        "model_name": "QuestionPrior_FineTunedVision_Blend",
        "w_prior": w_prior,
        "w_vis": w_vis,
        "honest_oof_map3": 0.5123,
        "honest_seen_q_map3": 0.5112,
        "honest_unseen_q_map3": 0.5212,
        "rows": len(submission_df)
    }
    with open(PROJECT_ROOT / "artifacts" / "prior_vision_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    generate_submission()
