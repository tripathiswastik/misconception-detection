"""Kaggle Submission Generator and Submission Format Validator.

Generates and verifies submission CSVs against competition sample_submission.csv format.
"""
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, cast
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MathMisconceptionData
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.pipeline import TwoStageDiagnosticPipeline


def validate_submission_file(
    submission_path: Path,
    sample_submission_path: Path,
    taxonomy_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate generated submission against sample_submission.csv schema and constraints."""
    if not submission_path.exists():
        raise FileNotFoundError(f"Submission file does not exist: {submission_path}")
    if not sample_submission_path.exists():
        raise FileNotFoundError(f"Sample submission file does not exist: {sample_submission_path}")

    sub_df = cast(pd.DataFrame, pd.read_csv(submission_path))
    sample_df = cast(pd.DataFrame, pd.read_csv(sample_submission_path))

    results: Dict[str, Any] = {
        "is_valid": True,
        "submission_path": str(submission_path),
        "total_rows": len(sub_df),
        "expected_rows": len(sample_df),
        "columns_match": list(sub_df.columns) == list(sample_df.columns),
        "trace_ids_match_exactly": True,
        "nan_count": int(sub_df.isna().sum().sum()),
        "invalid_codes_found": [],
        "errors": []
    }

    # 1. Row count check
    if len(sub_df) != len(sample_df):
        results["is_valid"] = False
        results["errors"].append(
            f"Row count mismatch: got {len(sub_df)}, expected {len(sample_df)}"
        )

    # 2. Columns check
    expected_cols = ["trace_id", "root_code"]
    if list(sub_df.columns) != expected_cols:
        results["is_valid"] = False
        results["errors"].append(
            f"Columns mismatch: got {list(sub_df.columns)}, expected {expected_cols}"
        )

    # 3. Trace ID alignment check
    if not (sub_df["trace_id"].values == sample_df["trace_id"].values).all():
        results["is_valid"] = False
        results["trace_ids_match_exactly"] = False
        results["errors"].append("Order of trace_id does not match sample_submission.csv exactly.")

    # 4. Null value check
    if results["nan_count"] > 0:
        results["is_valid"] = False
        results["errors"].append(f"Found {results['nan_count']} NaN / null values in submission.")

    # 5. Taxonomy code validation
    if taxonomy_path and taxonomy_path.exists():
        tax = TaxonomyManager(taxonomy_path)
        valid_codes = set(tax.all_codes(include_none=True))
        
        invalid_set = set()
        for val in sub_df["root_code"].astype(str):
            codes = val.strip().split()
            if len(codes) == 0 or len(codes) > 3:
                results["errors"].append(f"Row has invalid number of predictions ({len(codes)}): {val}")
                results["is_valid"] = False
            for c in codes:
                if c not in valid_codes:
                    invalid_set.add(c)

        if invalid_set:
            results["is_valid"] = False
            results["invalid_codes_found"] = list(invalid_set)
            results["errors"].append(f"Found {len(invalid_set)} unknown taxonomy codes: {list(invalid_set)[:5]}")

    return results


def build_and_validate_competition_submission(
    data_dir: Path,
    output_path: Path,
    use_gpu: bool = True,
    alpha: float = 5e-5
) -> Dict[str, Any]:
    """Train pipeline, generate test predictions, write submission.csv, and validate."""
    data = MathMisconceptionData(data_dir)
    train_df = data.load_train()
    test_traces = data.get_test_traces()
    sample_sub_path = data_dir / "sample_submission.csv"
    tax_path = data_dir / "taxonomy_v1.csv"

    print(f"[Submission] Building submission for {len(test_traces)} test traces (alpha={alpha})...")
    
    # Initialize and train pipeline on full 1,500 training traces with tuned alpha
    tax = TaxonomyManager(tax_path)
    diagnoser = TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=alpha)
    pipeline = TwoStageDiagnosticPipeline(diagnoser=diagnoser, taxonomy=tax, use_gpu=use_gpu)
    pipeline.fit_diagnoser(train_df)

    # Batch diagnose test traces
    results = pipeline.diagnose_batch(test_traces, batch_size=200, show_progress=True)

    # Construct submission DataFrame
    sub_df = pd.DataFrame({
        "trace_id": [r.trace_id for r in results],
        "root_code": [r.to_submission_string() for r in results]
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"[Submission] Saved submission to: {output_path}")

    # Validate generated submission
    val_report = validate_submission_file(
        output_path,
        sample_sub_path,
        taxonomy_path=tax_path
    )
    print(f"[Submission] Validation Status: {'PASSED (READY FOR LEADERBOARD)' if val_report['is_valid'] else 'FAILED'}")
    return val_report


if __name__ == "__main__":
    data_path = PROJECT_ROOT / "data"
    sub_path = PROJECT_ROOT / "submissions" / "math_enhanced_domain_masked_submission.csv"
    report = build_and_validate_competition_submission(data_path, sub_path, use_gpu=True, alpha=1e-4)
    import json
    print(json.dumps(report, indent=2))
