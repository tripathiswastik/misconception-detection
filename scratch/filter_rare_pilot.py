"""Filter High-Precision Rare-Class Pseudo-Labeling Pilot Candidates.

Selection Criteria:
1. Candidate must belong to questions associated with the 44 rare classes (<= 10 train samples).
2. Must have high prediction confidence (>= 0.70) and margin (>= 0.20) from the teacher model.
3. Must NOT be 'NONE'.
4. Batch size capped to a small pilot (~100-200 traces) to test the mechanism without teacher error propagation.
"""
from collections import Counter
from pathlib import Path
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager

def main():
    print("=" * 80)
    print("FILTERING HIGH-PRECISION RARE-CLASS PSEUDO-LABELING PILOT CANDIDATES")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    counts = Counter(train_df["root_code"].dropna())
    rare_10 = set(c for c, cnt in counts.items() if cnt <= 10)
    print(f"Targeting {len(rare_10)} rare classes (<= 10 training samples)")

    # Load existing 1465 pseudo trace pool
    pseudo_df = cast(pd.DataFrame, pd.read_csv(ART_DIR / "pseudo_1465_metadata.csv"))
    pseudo_soft = np.load(ART_DIR / "pseudo_1465_soft_targets_76.npy")

    print(f"Total candidate pseudo traces in pool: {len(pseudo_df)}")
    
    top1_idx = np.argmax(pseudo_soft, axis=1)
    top1_probs = np.max(pseudo_soft, axis=1)
    top1_codes = [classes[i] for i in top1_idx]

    # Compute margin to second class
    sorted_probs = np.sort(pseudo_soft, axis=1)[:, ::-1]
    margins = sorted_probs[:, 0] - sorted_probs[:, 1]

    # Filter for high-precision rare-class traces
    qualified_indices = []
    for i in range(len(pseudo_df)):
        code = top1_codes[i]
        conf = top1_probs[i]
        margin = margins[i]

        if code != "NONE" and code in rare_10 and conf >= 0.50 and margin >= 0.15:
            qualified_indices.append(i)

    print(f"\nQualified High-Confidence Rare-Class Candidates: {len(qualified_indices)}")
    
    if len(qualified_indices) > 0:
        pilot_df = pseudo_df.iloc[qualified_indices].copy()
        pilot_df["pseudo_code"] = [top1_codes[i] for i in qualified_indices]
        pilot_df["confidence"] = [top1_probs[i] for i in qualified_indices]
        pilot_df["margin"] = [margins[i] for i in qualified_indices]

        print("\nCandidate breakdown by predicted rare class:")
        for c, count in Counter(pilot_df["pseudo_code"]).most_common():
            print(f"  {c:10s}: {count:3d} traces (Mean conf: {pilot_df[pilot_df['pseudo_code']==c]['confidence'].mean():.3f})")

        out_path = ART_DIR / "pilot_rare_candidates.csv"
        pilot_df.to_csv(out_path, index=False)
        print(f"\n[Saved] Pilot candidates ({len(pilot_df)} traces) to {out_path.name}")
    else:
        print("No candidates met the high-precision filter from this pool.")

    print("=" * 80)

if __name__ == "__main__":
    main()

