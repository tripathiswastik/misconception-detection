"""Evaluate In-Fold Rank-2 / Rank-3 Question Admissibility Reordering on Canonical Folds.

Rigorous Invariants:
1. Rank 1 is 100% FIXED (never altered).
2. Uses canonical 5-fold split (splits.json).
3. Question allowed misconception set q_allowed is computed strictly from training partition of each fold.
4. Gate: Promote only if in-fold MAP@3 strictly improves without hurting baseline.
"""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("EVALUATING IN-FOLD QUESTION ADMISSIBILITY RANK-2/3 REORDERING (RANK 1 FIXED)")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Load SOTA 550 Champion In-Fold Baseline Matrix (M15 + ArcFace)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp

    base_top3 = [[classes[idx] for idx in row[:3]] for row in np.argsort(-p_base, axis=1)]
    base_map3 = calculate_map3(y_raw, base_top3)
    print(f"Base Champion OOF MAP@3: {base_map3:.5f}")

    # Check rows where Rank 1 is NONE
    r1_is_none = [p[0] == "NONE" for p in base_top3]
    print(f"Total rows where Rank 1 is NONE: {sum(r1_is_none)} / {len(r1_is_none)}")

    # -------------------------------------------------------------------------
    # RULE A: In-Fold Question Allowed Set Priority for Rank 2 vs Rank 3
    # If R1 is NONE, and R3 is in the fold-train question's allowed set,
    # but R2 is NOT in the question's allowed set:
    # Swap R2 and R3!
    # -------------------------------------------------------------------------
    print("\n--- Evaluating Rule A: In-Fold Question Admissibility Swap (R2 vs R3) ---")
    
    # We can also test a confidence gap threshold (only swap if gap < delta)
    for max_gap in [0.01, 0.02, 0.03, 0.05, 0.10, 1.00]:
        new_top3 = [list(row) for row in base_top3]
        swaps_count = 0
        correct_swaps = 0
        harmful_swaps = 0

        for f in splits:
            trn_idx = f["train_indices"]
            val_idx = f["val_indices"]
            trn_df = train_df.iloc[trn_idx]

            # Build in-fold q_allowed strictly from training fold
            q_allowed = {}
            for _, r in trn_df.iterrows():
                code = str(r["root_code"])
                if code != "NONE" and pd.notna(r["root_code"]):
                    q_allowed.setdefault(r["question_id"], set()).add(code)

            for i in val_idx:
                if r1_is_none[i]:
                    qid = train_df.iloc[i]["question_id"]
                    allowed = q_allowed.get(qid, set())

                    r2 = base_top3[i][1]
                    r3 = base_top3[i][2]
                    c2_idx = class_to_idx[r2]
                    c3_idx = class_to_idx[r3]

                    p2 = p_base[i, c2_idx]
                    p3 = p_base[i, c3_idx]
                    gap = p2 - p3

                    # Condition: R3 is allowed, R2 is NOT allowed, and gap < max_gap
                    if r3 in allowed and r2 not in allowed and gap < max_gap:
                        new_top3[i][1] = r3
                        new_top3[i][2] = r2
                        swaps_count += 1
                        true_y = y_raw[i]
                        if true_y == r3:
                            correct_swaps += 1
                        elif true_y == r2:
                            harmful_swaps += 1

        score = calculate_map3(y_raw, new_top3)
        delta = score - base_map3
        precision = (correct_swaps / max(1, correct_swaps + harmful_swaps)) if (correct_swaps + harmful_swaps) > 0 else 0.0
        print(f"  max_gap = {max_gap:.2f} | Swaps: {swaps_count:3d} (Win={correct_swaps:2d}, Loss={harmful_swaps:2d}, Prec={precision:.1%}) | MAP@3: {score:.5f} ({delta:+.5f})")

    print("=" * 80)

if __name__ == "__main__":
    main()

