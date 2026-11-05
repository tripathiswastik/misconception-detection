"""Tune Late Fusion Weights on Out-of-Fold Predictions and Generate Final Submission.

Saves:
- artifacts/optimal_weights.json: Optimal fusion parameters and OOF scores.
- submissions/multimodal_fused_submission.csv: Final verified competition submission.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file
from src.models.fusion import MultimodalLateFusion


def run_fusion_and_submission():
    artifacts_dir = PROJECT_ROOT / "artifacts"
    oof_text_path = artifacts_dir / "oof_text.npy"
    oof_vision_path = artifacts_dir / "oof_vision.npy"
    test_text_path = artifacts_dir / "test_text.npy"
    test_vision_path = artifacts_dir / "test_vision.npy"

    for path in [oof_text_path, oof_vision_path, test_text_path, test_vision_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required artifact: {path}")

    oof_text = np.load(oof_text_path)
    oof_vision = np.load(oof_vision_path)
    test_text = np.load(test_text_path)
    test_vision = np.load(test_vision_path)

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    all_classes = tax.all_codes(include_none=True)

    data_mgr = MathMisconceptionData(PATHS.data_dir)
    train_df = data_mgr.load_train()
    test_df = data_mgr.load_test()
    y_true = train_df["root_code"].astype(str).values

    # Apply domain constraints to vision branch to eliminate cross-domain hallucination
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)
    train_topics = [str(t) for t in train_df["topic_name"].values]
    train_grades = [int(g) for g in train_df["grade"].values]
    test_topics = [str(t) for t in test_df["topic_name"].values]
    test_grades = [int(g) for g in test_df["grade"].values]

    oof_vision = domain_masker.apply_mask_batch(
        oof_vision, all_classes, train_topics, train_grades, damping=CONFIG.domain_mask_damping
    )
    test_vision = domain_masker.apply_mask_batch(
        test_vision, all_classes, test_topics, test_grades, damping=CONFIG.domain_mask_damping
    )

    print("\n[Fusion] Grid searching optimal vision weight on OOF predictions...")
    candidate_weights = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    results = MultimodalLateFusion.tune_fusion_weight(
        y_true=y_true,
        text_scores=oof_text,
        vision_scores=oof_vision,
        classes=all_classes,
        candidate_weights=candidate_weights
    )

    print(f"  Baseline Text OOF MAP@3: {results['baseline_text_map3']:.4f}")
    print(f"  Best Fused OOF MAP@3:    {results['best_oof_map3']:.4f}")
    print(f"  Optimal Vision Weight:   {results['optimal_vision_weight']:.4f}")

    opt_weight = results["optimal_vision_weight"]

    # Fuse test predictions
    fused_test = MultimodalLateFusion.apply_fusion(
        text_scores=test_text,
        vision_scores=test_vision,
        vision_weight=opt_weight
    )

    # Generate Top-3 predictions
    top_indices = np.argsort(-fused_test, axis=1, kind="mergesort")[:, :3]
    pred_strings = []
    for row in top_indices:
        pred_codes = [all_classes[idx] for idx in row]
        pred_strings.append(" ".join(pred_codes))

    # Format Kaggle submission
    submission_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings
    })

    PATHS.submissions_dir.mkdir(parents=True, exist_ok=True)
    out_path = PATHS.submissions_dir / "multimodal_fused_submission.csv"
    submission_df.to_csv(out_path, index=False)
    print(f"\n[Fusion] Saved submission to: {out_path} ({len(submission_df)} rows)")

    # Validate submission file format
    validation = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"[Fusion] Validation result: {validation}")
    if not validation["valid"]:
        raise ValueError(f"Submission validation failed: {validation['errors']}")

    # Save metadata
    with open(artifacts_dir / "optimal_weights.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results, out_path


if __name__ == "__main__":
    run_fusion_and_submission()
