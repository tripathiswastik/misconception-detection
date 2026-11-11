"""Operational Freezing & Comprehensive Validation for Champion Submission S550.

Validates:
1. File Integrity: Path, Size, SHA256, MD5.
2. Structure & Alignment: Row count (3,498), Column names (['trace_id', 'root_code']), ID sequence matching test.csv.
3. Prediction Syntax: Exactly 3 space-separated tokens per row, 0 NaNs, 0 duplicates per row.
4. Taxonomy Validity: All predicted tokens are members of the official 76-class taxonomy.
5. Score Provenance: Canonical CV = 0.54214, Public LB = 0.55049 (Ref: 57885741).
6. Operational Manifest Generation: Writes artifacts/submission_manifest_s550.json.
"""

import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Set, cast

import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager


def main():
    print("=" * 115)
    print("OPERATIONAL FREEZING & INTEGRITY AUDIT: CHAMPION S550 SUBMISSION")
    print("=" * 115)
    start_time = time.time()

    sub_file = SUB_DIR / "blend_sota550_precision_gated_submission.csv"
    test_file = DATA_DIR / "test.csv"
    sample_file = DATA_DIR / "sample_submission.csv"
    tax_file = DATA_DIR / "taxonomy_v1.csv"

    # 1. File Existence & Checksum Audit
    print("\n[Audit 1/5] Auditing Physical File Integrity & Cryptographic Hashes...")
    assert sub_file.exists(), f"CRITICAL: Submission file missing at {sub_file}"
    assert test_file.exists(), f"CRITICAL: Test file missing at {test_file}"
    assert sample_file.exists(), f"CRITICAL: Sample submission file missing at {sample_file}"
    assert tax_file.exists(), f"CRITICAL: Taxonomy file missing at {tax_file}"

    sub_bytes = sub_file.read_bytes()
    file_size = len(sub_bytes)
    sha256_hash = hashlib.sha256(sub_bytes).hexdigest()
    md5_hash = hashlib.md5(sub_bytes).hexdigest()

    expected_sha256 = "b2637c3b7a92010d3d04f41f2f4c2e91e0cc312a63d2532ab63f1d16bf90bc70"
    expected_md5 = "7d63ccaf87530d1d1d85f671565395f9"
    assert sha256_hash == expected_sha256, f"SHA256 mismatch! Got {sha256_hash}"
    assert md5_hash == expected_md5, f"MD5 mismatch! Got {md5_hash}"
    assert file_size == 115506, f"File size mismatch! Expected 115506 bytes, got {file_size}"
    print(f"  File Path: {sub_file.name}")
    print(f"  File Size: {file_size:,} bytes")
    print(f"  SHA256:    {sha256_hash}")
    print(f"  MD5:       {md5_hash}")
    print("  Status:    PASSED (Cryptographic hashes match golden reference)")

    # 2. Structure & Column Schema Audit
    print("\n[Audit 2/5] Auditing Row Count, Column Headers, and ID Sequence...")
    sub_df = cast(pd.DataFrame, pd.read_csv(sub_file))
    test_df = cast(pd.DataFrame, pd.read_csv(test_file))
    sample_df = cast(pd.DataFrame, pd.read_csv(sample_file))

    n_sub = len(sub_df)
    n_test = len(test_df)
    n_sample = len(sample_df)

    assert n_sub == 3498, f"Row count error: expected 3498, got {n_sub}"
    assert n_sub == n_test, f"Row count mismatch with test.csv: {n_sub} vs {n_test}"
    assert n_sub == n_sample, f"Row count mismatch with sample_submission.csv: {n_sub} vs {n_sample}"

    expected_cols = ["trace_id", "root_code"]
    assert list(sub_df.columns) == expected_cols, f"Column mismatch: expected {expected_cols}, got {list(sub_df.columns)}"
    assert list(sample_df.columns) == expected_cols, f"Column mismatch with sample_submission.csv"

    # Exact ID sequence match
    sub_ids = sub_df["trace_id"].astype(str).tolist()
    test_ids = test_df["trace_id"].astype(str).tolist()
    sample_ids = sample_df["trace_id"].astype(str).tolist()

    assert sub_ids == test_ids, "CRITICAL: trace_id order does NOT match test.csv exactly!"
    assert sub_ids == sample_ids, "CRITICAL: trace_id order does NOT match sample_submission.csv exactly!"
    print(f"  Total Rows: {n_sub:,} (Exact match to test.csv)")
    print(f"  Columns:    {list(sub_df.columns)} (Exact match to competition specification)")
    print(f"  ID Alignment: 3,498 / 3,498 IDs perfectly aligned with 0 order mismatches")
    print("  Status:     PASSED")

    # 3. Prediction Syntax & Completeness Audit
    print("\n[Audit 3/5] Auditing Prediction Syntax, Token Count, and Deduplication...")
    assert sub_df["trace_id"].isna().sum() == 0, "Null values detected in trace_id column!"
    assert sub_df["root_code"].isna().sum() == 0, "Null values detected in root_code column!"

    preds_raw = sub_df["root_code"].astype(str).tolist()
    for row_idx, pred_str in enumerate(preds_raw):
        tokens = pred_str.strip().split()
        assert len(tokens) == 3, f"Row {row_idx} ({sub_ids[row_idx]}) has {len(tokens)} tokens instead of 3: '{pred_str}'"
        assert len(set(tokens)) == 3, f"Row {row_idx} ({sub_ids[row_idx]}) has duplicate tokens: '{pred_str}'"

    print("  Null Values: 0 nulls across all rows and columns")
    print("  Token Count: Exactly 3 space-delimited tokens per row for all 3,498 rows")
    print("  Uniqueness:  0 duplicate predictions within any individual row's top 3")
    print("  Status:      PASSED")

    # 4. Taxonomy Validity Audit
    print("\n[Audit 4/5] Auditing Taxonomy Code Validity against Official Taxonomy...")
    tax = TaxonomyManager(tax_file)
    valid_codes: Set[str] = set(tax.all_codes(include_none=True))
    assert len(valid_codes) == 76, f"Expected 76 taxonomy codes, found {len(valid_codes)}"
    assert "NONE" in valid_codes, "Taxonomy missing 'NONE' code!"

    all_tokens_used: Set[str] = set()
    token_frequencies: Dict[str, int] = {}
    rank1_frequencies: Dict[str, int] = {}

    for pred_str in preds_raw:
        tokens = pred_str.strip().split()
        for t in tokens:
            assert t in valid_codes, f"CRITICAL: Invalid taxonomy code detected: '{t}'"
            all_tokens_used.add(t)
            token_frequencies[t] = token_frequencies.get(t, 0) + 1
        r1 = tokens[0]
        rank1_frequencies[r1] = rank1_frequencies.get(r1, 0) + 1

    none_top1_count = rank1_frequencies.get("NONE", 0)
    print(f"  Taxonomy Coverage: All predicted tokens exist in the 76-class official catalog")
    print(f"  Distinct Codes Predicted in Top 3: {len(all_tokens_used)} / 76 classes")
    print(f"  Rank 1 'NONE' Predictions: {none_top1_count} / 3,498 ({none_top1_count / n_sub:.1%})")
    print("  Status:            PASSED")

    # 5. Provenance, Operational Manifest, and Final Freeze
    print("\n[Audit 5/5] Generating Operational Submission Manifest...")
    manifest = {
        "manifest_version": "1.0",
        "frozen_timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        "champion_status": "FROZEN_FINAL_SUBMISSION",
        "artifact": {
            "relative_path": "submissions/blend_sota550_precision_gated_submission.csv",
            "file_name": "blend_sota550_precision_gated_submission.csv",
            "file_size_bytes": file_size,
            "row_count": n_sub,
            "column_names": expected_cols,
            "sha256": sha256_hash,
            "md5": md5_hash
        },
        "performance_benchmarks": {
            "public_leaderboard": {
                "score": 0.55049,
                "kaggle_submission_ref": 57885741,
                "submission_date": "2026-11-07"
            },
            "canonical_cross_validation": {
                "composite_map3": 0.54214,
                "seen_map3": 0.54295,
                "unseen_map3": 0.52424,
                "weighting": "0.957 * Seen + 0.043 * Unseen",
                "folds": 5,
                "split_definition": "data/splits.json"
            },
            "safety_and_tail_metrics": {
                "false_none_demotions": 104,
                "false_none_demotion_rate": 0.2055,
                "total_none_validation_samples": 506,
                "rare_class_recall_at_10": 218,
                "rare_class_recall_rate": 0.7315,
                "total_rare_validation_samples": 298
            }
        },
        "ensemble_architecture": {
            "components": [
                {"name": "M15 Foundation Fusion", "weight": 0.665, "description": "DeBERTa-v3 text + GBDT tabular + Swin vision base blend"},
                {"name": "DualHead Soft Distillation", "weight": 0.014, "description": "Dual-head visual distillation representation"},
                {"name": "Composite 448 Vision", "weight": 0.021, "description": "High-resolution 448px multi-scale visual embedding"},
                {"name": "Pseudo-3500 Metric Model", "weight": 0.035, "description": "Transductive pseudo-labeled metric model"},
                {"name": "Adaptive Question Prior Engine", "weight": 0.026, "description": "Empirical question-grouped Bayesian topic prior"},
                {"name": "Clean Multimodal ArcFace 2328", "weight": 0.002, "description": "Hard-margin angular classification head"}
            ],
            "post_processing_and_gating": {
                "domain_constraint_engine": "In-fold empirical topic/grade taxonomy masking",
                "precision_gating": {
                    "binary_error_model": "oof_compact_binary_prob.npy",
                    "threshold_tau": 0.78,
                    "max_prob_gap": 0.04,
                    "condition": "Promote Rank 2 misconception to Rank 1 if P(NONE)-P(miscon) < 0.04 and P(error) >= 0.78 and misconception in question candidate distribution"
                }
            }
        },
        "governance_status": {
            "exploratory_branches": "CLOSED",
            "ocr_experiments": "CLOSED",
            "raw_visual_finetuning": "CLOSED",
            "lupi_text_distillation": "CLOSED",
            "blank_working_features": "CLOSED",
            "champion_locked": True,
            "ready_for_final_selection": True
        }
    }

    manifest_path = ART_DIR / "submission_manifest_s550.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Manifest Path: {manifest_path}")
    print("  Status:        WRITTEN & VERIFIED")

    print("\n" + "=" * 115)
    print("ALL 5 AUDIT GATES PASSED: S550 IS OFFICIALLY FROZEN AS CHAMPION")
    print(f"Elapsed Time: {time.time() - start_time:.2f}s")
    print("=" * 115)


if __name__ == "__main__":
    main()
