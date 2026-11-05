"""Model M20: Transductive Question-Consensus Champion (Target: 0.550+).

Robust file resolution: Auto-locates test.csv, test_kaggle_sota.npy,
taxonomy_classes.json, and test_m15 across /kaggle/input and /kaggle/working.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
import numpy as np
import pandas as pd


def find_file(filename: str) -> Optional[Path]:
    """Fast search across Kaggle and local directories without traversing large trees."""
    direct_candidates = [
        # Kaggle paths
        Path("/kaggle/working/data") / filename,
        Path("/kaggle/working/artifacts") / filename,
        Path("/kaggle/working") / filename,
        Path("/kaggle/input/misconception-detection-in-k-12-student-math-workings") / filename,
        # Local paths
        Path("d:/ag-ai/misconception-detection/data") / filename,
        Path("d:/ag-ai/misconception-detection/artifacts") / filename,
        Path("d:/ag-ai/misconception-detection/submissions") / filename,
        Path("data") / filename,
        Path("artifacts") / filename,
        Path("submissions") / filename,
        Path(filename),
    ]
    for p in direct_candidates:
        if p.exists():
            return p

    # Targeted search in /kaggle/input (top-level only, depth 2)
    if os.path.exists("/kaggle/input"):
        for root, dirs, files in os.walk("/kaggle/input"):
            if filename in files:
                return Path(root) / filename
    return None


def run_m20_pipeline(consensus_weight: float = 0.12):
    print("=" * 75)
    print(f"[Model M20] Transductive Question-Consensus Champion Pipeline (lambda={consensus_weight})")
    print("=" * 75)

    # 1. Locate test.csv
    test_csv_path = find_file("test.csv")
    if test_csv_path is None:
        raise FileNotFoundError("Could not find test.csv in /kaggle/working, /kaggle/input, or local directory!")
    print(f"Located test.csv:               {test_csv_path}")
    test_df = cast(pd.DataFrame, pd.read_csv(test_csv_path))

    # 2. Locate taxonomy_classes.json
    tax_path = find_file("taxonomy_classes.json")
    if tax_path is None:
        raise FileNotFoundError("Could not find taxonomy_classes.json!")
    print(f"Located taxonomy_classes.json:  {tax_path}")
    with open(tax_path) as f:
        tax_classes: List[str] = json.load(f)

    # 3. Locate M15 Baseline Test Probabilities
    m15_path = find_file("test_m15_balanced_fusion.npy")
    if m15_path is None:
        raise FileNotFoundError("Could not find test_m15_balanced_fusion.npy!")
    print(f"Located test_m15:               {m15_path}")
    test_m15 = np.load(m15_path)

    # 4. Locate ArcFace Test Probabilities (0.54399 Champion component)
    out_dir = Path("/kaggle/working") if os.path.exists("/kaggle") else Path("submissions")
    out_dir.mkdir(parents=True, exist_ok=True)

    arcface_path = find_file("test_kaggle_sota.npy")
    if arcface_path is not None:
        print(f"Located test_arcface:           {arcface_path}")
        test_arcface = np.load(arcface_path)
        # Active Champion 90/10 blend formula (Public LB: 0.54399 🏆)
        p_champ = 0.90 * test_m15 + 0.10 * test_arcface
        print("  Successfully loaded Active Champion base (0.90 M15 + 0.10 ArcFace = 0.54399)")
    else:
        print("  Note: test_kaggle_sota.npy not found, using pure M15 base.")
        p_champ = test_m15

    # 5. Transductive Question-Consensus Smoothing
    q_count = test_df["question_id"].nunique()
    print(f"\nComputing Question Consensus across {q_count} test questions...")
    q_groups = test_df.groupby("question_id").indices
    q_consensus = np.zeros_like(p_champ)

    for q_id, idxs in q_groups.items():
        mu_q = np.mean(p_champ[idxs], axis=0)
        for idx in idxs:
            q_consensus[idx] = mu_q

    # Blend with consensus weight lambda
    p_m20 = (1.0 - consensus_weight) * p_champ + consensus_weight * q_consensus

    # 6. Analyze Differences against Champion
    top3_m20 = np.argsort(-p_m20.astype(np.float64), axis=1)[:, :3]
    top3_champ = np.argsort(-p_champ.astype(np.float64), axis=1)[:, :3]

    diff_r1 = int(np.sum(top3_m20[:, 0] != top3_champ[:, 0]))
    diff_r2 = int(np.sum(top3_m20[:, 1] != top3_champ[:, 1]))
    diff_r3 = int(np.sum(top3_m20[:, 2] != top3_champ[:, 2]))

    print("\n[M20 Consensus Adjustments vs 0.54399 Champion]")
    print(f"  Total test samples:   {len(test_df)}")
    print(f"  Rank 1 refined:       {diff_r1} ({diff_r1/len(test_df)*100:.2f}%)")
    print(f"  Rank 2 refined:       {diff_r2} ({diff_r2/len(test_df)*100:.2f}%)")
    print(f"  Rank 3 refined:       {diff_r3} ({diff_r3/len(test_df)*100:.2f}%)")

    # 7. Generate Validated Submission File
    pred_strings = []
    for row in top3_m20:
        pred_strings.append(" ".join(tax_classes[idx] for idx in row))

    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"],
        "root_code": pred_strings,
    })

    out_csv = out_dir / "transductive_champion_m20_submission.csv"
    sub_df.to_csv(out_csv, index=False)
    print(f"\nSaved M20 Submission: {out_csv} ({len(sub_df)} rows)")
    print(f"  Sample row 0: {sub_df.iloc[0].to_dict()}")

    # Schema Validation
    assert list(sub_df.columns) == ["trace_id", "root_code"], "Invalid columns!"
    assert len(sub_df) == 3498, "Invalid row count!"
    assert not sub_df.isna().any().any(), "NaN values found!"
    print("\n[SUCCESS] Model M20 Schema Validation: 100% PASSED!")


if __name__ == "__main__":
    run_m20_pipeline(consensus_weight=0.12)
