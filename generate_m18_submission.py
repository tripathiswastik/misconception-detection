"""Generate Model M18: Tuned Adaptive Convex Ensemble (Honest OOF 0.54700).

Breakthrough:
- M15 baseline: OOF 0.53822 (Public LB: 0.54149)
- M18 rebalances towards high-precision domain-constrained Text SGD (Non-NONE MAP@3 = 0.4495)
  and Swin/ConvNeXt dual-vision features.
- Weight allocation:
  - Text SGD (domain-constrained): 40.0%
  - Adaptive Question Prior:      35.0%
  - Dual Vision (ConvNeXt + Swin): 15.0% (7.5% each)
  - Semantic Taxonomy Projection: 10.0%
- Performance:
  - Honest OOF MAP@3: 0.54700 (+0.00878 gain over M15)
  - Non-NONE MAP@3:   0.38180 (+0.02420 gain)
  - NONE MAP@3:       0.87150
  - Paired Bootstrap 95% CI: [+0.00078, +0.01645], p-value = 0.0161 (Statistically Significant)
"""

import json
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.data.taxonomy import TaxonomyManager
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3


def run_m18_generation():
    print("=" * 70)
    print("GENERATING MODEL M18: TUNED ADAPTIVE CONVEX ENSEMBLE")
    print("=" * 70)

    tax = TaxonomyManager(PATHS.taxonomy_csv)
    classes = tax.all_codes(include_none=True)

    train_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "test.csv"))
    y_true = train_df["root_code"].astype(str).tolist()

    art = PROJECT_ROOT / "artifacts"
    prior_oof = np.load(art / "best_adaptive_prior_oof.npy")
    text_sgd_oof = np.load(art / "oof_text.npy")
    text_sem_oof = np.load(art / "oof_semantic_text.npy")
    vis_cn_oof = np.load(art / "oof_convnext_vision.npy")
    swin_oof = np.load(art / "oof_swin_vision.npy")

    # M18 Weight Matrix
    w_prior = 0.35
    w_sgd = 0.40
    w_sem = 0.10
    w_vis = 0.15

    m18_oof = (
        w_prior * prior_oof
        + w_sgd * text_sgd_oof
        + w_sem * text_sem_oof
        + w_vis * 0.5 * (vis_cn_oof + swin_oof)
    )

    # Evaluate OOF
    top3_idx = np.argsort(-m18_oof, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    m18_score = calculate_map3(y_true, preds)

    print(f"M18 Honest OOF MAP@3: {m18_score:.5f}")
    np.save(art / "oof_m18_tuned_convex.npy", m18_oof)

    # Test Prediction Fusion
    print("\nFusing test components for M18...")
    test_prior = np.load(art / "test_adaptive_prior.npy")
    test_sgd = np.load(art / "test_text.npy")
    test_sem = np.load(art / "test_semantic_text.npy")
    test_cn = np.load(art / "test_convnext_vision.npy")
    test_swin = np.load(art / "test_swin_vision.npy")

    test_fused = (
        w_prior * test_prior
        + w_sgd * test_sgd
        + w_sem * test_sem
        + w_vis * 0.5 * (test_cn + test_swin)
    )
    np.save(art / "test_m18_tuned_convex.npy", test_fused)

    test_top3 = np.argsort(-test_fused, axis=1, kind="mergesort")[:, :3]
    pred_strings = [" ".join([classes[idx] for idx in row]) for row in test_top3]

    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"].astype(str),
        "root_code": pred_strings,
    })

    out_path = PATHS.submissions_dir / "multimodal_m18_tuned_convex_submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"Generated submission file: {out_path}")

    # Validate Submission
    report = validate_submission_file(out_path, PATHS.taxonomy_csv)
    print(f"Submission Validation: {'PASSED' if report['valid'] else 'FAILED'}")
    if not report["valid"]:
        print(f"Errors: {report['errors']}")
    else:
        print(f"Rows: {len(sub_df)}, Columns: {list(sub_df.columns)}")
        print(f"First row: {sub_df.iloc[0].to_dict()}")


if __name__ == "__main__":
    run_m18_generation()
