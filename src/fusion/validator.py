"""Submission CSV validation engine for Kaggle competition compliance."""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.data.taxonomy import TaxonomyManager


def validate_submission_file(
    csv_path: Union[str, Path],
    taxonomy_csv: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Validate submission CSV file for formatting, dimensions, and taxonomy compliance."""
    path = Path(csv_path)
    if not path.exists():
        return {"valid": False, "errors": [f"File not found: {path}"], "row_count": 0}

    try:
        raw_df = pd.read_csv(path)
    except Exception as e:
        return {"valid": False, "errors": [f"Failed to read CSV: {e}"], "row_count": 0}

    if not isinstance(raw_df, pd.DataFrame):
        return {"valid": False, "errors": ["File parsed as TextFileReader instead of DataFrame"], "row_count": 0}
    df: pd.DataFrame = raw_df

    errors: List[str] = []
    expected_cols = ["trace_id", "root_code"]
    if list(df.columns) != expected_cols:
        errors.append(f"Invalid columns: {list(df.columns)}, expected {expected_cols}")

    if len(df) != 3498:
        errors.append(f"Expected 3498 rows, got {len(df)}")

    if bool(df.isna().any().any()):
        errors.append("Null or NaN values detected in submission")

    tax_path: Optional[Path] = Path(taxonomy_csv) if taxonomy_csv is not None else None
    tax = TaxonomyManager(csv_path=tax_path)
    valid_codes = set(tax.all_codes(include_none=True))

    for idx, val in enumerate(df["root_code"]):
        parts = str(val).split()
        if len(parts) != 3:
            errors.append(f"Row {idx}: expected 3 codes, got {len(parts)} ('{val}')")
            break
        if len(set(parts)) != 3:
            errors.append(f"Row {idx}: duplicate codes found in '{val}'")
            break
        invalid = [p for p in parts if p not in valid_codes]
        if invalid:
            errors.append(f"Row {idx}: invalid codes {invalid} in '{val}'")
            break

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "row_count": len(df)
    }
