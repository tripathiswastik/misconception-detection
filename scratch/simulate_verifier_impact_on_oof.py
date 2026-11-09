"""Simulate Deterministic Answer Verifier Gating on OOF and Test.

Analyzes:
1. OOF: When verifier says is_correct == True (89.40% clean):
   - What did the champion ensemble predict?
   - Does boosting NONE or gating to NONE improve OOF MAP@3?
2. Test: How many test rows have is_correct == True?
   - What are the current Rank 1 predictions on these rows?
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
from src.metrics.map3 import calculate_map3
from src.reader.diagnostic_verifier import enhanced_safe_verifier

def main():
    print("=" * 80)
    print("SIMULATING VERIFIER GATING IMPACT ON OOF & TEST")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Run verifier on train
    steps_train = train_df["transcription_steps"].fillna("").astype(str).tolist()
    ans_train = train_df["correct_answer"].fillna("").astype(str).tolist()
    train_ver = [enhanced_safe_verifier(s, a)[0] for s, a in zip(steps_train, ans_train)]

    print(f"Train verifier True (Correct answer): {sum(1 for x in train_ver if x is True)} / {len(train_ver)}")

    # Load OOF components for Champion Foundation
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    p_oof_base = 0.88 * oof_m15 + 0.12 * oof_v2

    # Mult calibration
    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_oof_cal = p_oof_base * mult

    def eval_map3(p_mat, beta=0.82):
        cand = p_mat.copy()
        row_idx = np.arange(len(cand))
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(len(cand))]
        return calculate_map3(y_true, preds), preds

    base_map3, oof_preds = eval_map3(p_oof_cal)
    print(f"\nBaseline Champion OOF MAP@3: {base_map3:.5f}")

    # Inspect the verifier True rows on train
    corr_indices = [i for i, x in enumerate(train_ver) if x is True]
    none_as_r1 = sum(1 for i in corr_indices if oof_preds[i][0] == "NONE")
    misc_as_r1 = sum(1 for i in corr_indices if oof_preds[i][0] != "NONE")
    print(f"When Verifier says Correct (151 rows):")
    print(f"  Ensemble already predicts NONE at Rank 1: {none_as_r1} / 151")
    print(f"  Ensemble predicts MISCONCEPTION at Rank 1: {misc_as_r1} / 151")

    # For the misc_as_r1 rows, what is the ground truth?
    misc_but_corr_true_none = sum(1 for i in corr_indices if oof_preds[i][0] != "NONE" and y_true[i] == "NONE")
    misc_but_corr_true_misc = sum(1 for i in corr_indices if oof_preds[i][0] != "NONE" and y_true[i] != "NONE")
    print(f"  Of the {misc_as_r1} misconceptions predicted:")
    print(f"    Actually Clean (NONE): {misc_but_corr_true_none} (False alarms rescued if flipped)")
    print(f"    Actually Misconception: {misc_but_corr_true_misc} (Correct misconception calls lost if flipped)")

    # Simulate gating: if verifier says True, boost NONE probability or force NONE to Rank 1
    for boost in [1.0, 1.1, 1.2, 1.3, 1.5, 2.0]:
        p_gated = p_oof_cal.copy()
        for i in corr_indices:
            p_gated[i, none_idx] *= boost
        m3, _ = eval_map3(p_gated)
        print(f"  Boost NONE by {boost:.2f}x on verifier True -> OOF MAP@3 = {m3:.5f} (delta: {m3 - base_map3:+.5f})")

    # Also test hard gating to NONE
    p_hard = p_oof_cal.copy()
    for i in corr_indices:
        p_hard[i, none_idx] = 999.0
    m3_hard, _ = eval_map3(p_hard)
    print(f"  Hard Force NONE at Rank 1 on verifier True -> OOF MAP@3 = {m3_hard:.5f} (delta: {m3_hard - base_map3:+.5f})")

    print("=" * 80)

if __name__ == "__main__":
    main()

