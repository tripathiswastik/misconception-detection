"""Generate Test Submission for Tri-Modal Fusion (Prior + Vision + Text).

Recipe:
- 65% In-Fold Empirical Question Prior (Hierarchical target encoder)
- 25% Fine-Tuned MobileNetV3 Vision (Top-2 blocks unfrozen on cuda:0)
- 10% Honest Context Text (SGD Classifier on problem stem)

Honest 5-Fold OOF Metrics:
- Overall MAP@3: 0.5143 (+0.0036 over bi-modal)
- Seen Questions MAP@3: 0.5120
- Unseen Questions MAP@3: 0.5333 (+0.0121 over bi-modal)

Outputs:
- submissions/multimodal_trimodal_submission.csv (3,498 rows, valid schema)
- artifacts/trimodal_metadata.json
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


def generate_trimodal_submission() -> Dict[str, Any]:
    print("\n=======================================================")
    print("[Submission] Generating Tri-Modal Prior + Vision + Text Submission")
    print("=======================================================\n")

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)
    n_classes = len(classes)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    n_test = len(test_df)

    # 1. Load canonical deterministic splits
    splitter = FoldSplitter(n_splits=CONFIG.n_splits, random_state=CONFIG.random_state)
    folds = splitter.load_splits(PATHS.splits_cache)

    # 2. Compute 5-fold ensemble question prior probabilities on test set
    print(f"[Submission] Computing in-fold Question Prior ensemble on {n_test} test rows...")
    test_prior_probs = np.zeros((n_test, n_classes), dtype=np.float32)
    for fold_idx, (tr_idx, _) in enumerate(folds, start=1):
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
    test_grades = [int(float(g)) for g in test_df["grade"]]
    test_vis_masked = domain_masker.apply_mask_batch(test_vis, classes, test_topics, test_grades, damping=0.85)

    # 4. Load honest context text test probabilities
    test_text_path = PROJECT_ROOT / "artifacts" / "test_text.npy"
    if not test_text_path.exists():
        raise FileNotFoundError(f"Missing {test_text_path}")
    test_text = np.load(test_text_path)

    # 5. Tri-Modal Blend: 65% Prior + 25% Vision + 10% Text
    w_prior = 0.65
    w_vis = 0.25
    w_text = 0.10

    print(f"[Submission] Blending modalities: Prior={w_prior:.2f}, Vision={w_vis:.2f}, Text={w_text:.2f}...")
    fused_test = w_prior * test_prior_probs + w_vis * test_vis_masked + w_text * test_text

    # 6. Extract Top-3 predictions per row
    top_indices = np.argsort(-fused_test, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in top_indices]

    # Format dataframe
    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_trimodal_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"[Submission] Saved submission to: {out_path} ({len(submission_df)} rows)")

    # 7. Strict validation against competition schema
    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Submission] Validation status: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # Save metadata
    meta = {
        "model_name": "TriModal_Prior_Vision_Text_Fusion",
        "w_prior": w_prior,
        "w_vis": w_vis,
        "w_text": w_text,
        "honest_oof_map3": 0.5143,
        "honest_seen_q_map3": 0.5120,
        "honest_unseen_q_map3": 0.5333,
        "rows": len(submission_df),
        "output_file": str(out_path)
    }
    with open(PROJECT_ROOT / "artifacts" / "trimodal_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    generate_trimodal_submission()
