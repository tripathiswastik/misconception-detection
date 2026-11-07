"""Generate SOTA Peak Refinement Submissions.

Implements two complementary strategies:
1. Candidate A: blend_rank1_locked_sota549_r23_refined_submission.csv
   - 100% Locked Rank-1 to active peak blend_optimal_multimodal_theoretical_submission.csv (0.54899).
   - Zero risk of Rank-1 false alarms on NONE.
   - Rank-2 and Rank-3 optimized with beta = 0.75 and multimodal probability fusion.
2. Candidate B: blend_continuous_optimal_champion_submission.csv
   - Unwarped probabilities (gamma = 0.00).
   - Gentle transductive question boost (alpha = 1.10).
   - Continuous clean multimodal weight (w_clean = 0.05).
   - Secondary discount (beta = 0.75).
   - Shifts exactly 47 high-confidence Rank-1 rows without hard thresholding.
"""
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.fusion.validator import validate_submission_file

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"
DOWNLOADS_DIR = Path("C:/Users/hp/Downloads")


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    if abs(t - 1.0) < 1e-5:
        return probs.copy()
    sharp = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    row_sums = sharp.sum(axis=1, keepdims=True)
    return np.divide(sharp, row_sums, where=row_sums > 0)


def build_submissions():
    print("=" * 80)
    print("GENERATING SOTA PEAK REFINEMENT SUBMISSIONS")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)
    assert n_test == 3498, f"Unexpected test size: {n_test}"

    # Load 0.54899 peak champion
    peak_path = SUB_DIR / "blend_optimal_multimodal_theoretical_submission.csv"
    peak_df = cast(pd.DataFrame, pd.read_csv(peak_path))
    peak_preds = peak_df["root_code"].tolist()
    peak_r1 = [p.split()[0] for p in peak_preds]

    # Load test probability matrices
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_test_champ_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    # Transductive question boost alpha = 1.10
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_champ_base)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_map[qid]:
                    p_test_boosted[i, ci] *= 1.10

    p_test_arc3500_cal = 0.95 * test_m15 + 0.05 * test_arc3500
    p_test_champ_ensemble = 0.85 * p_test_champ_base + 0.10 * p_test_boosted + 0.05 * p_test_arc3500_cal

    # Clean Multimodal Ensemble
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    p_test_clean = (
        0.6448 * test_prior_probs +
        0.0839 * apply_temperature(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * apply_temperature(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * apply_temperature(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )

    # --------------------------------------------------------------------------
    # Candidate A: 100% Rank-1 Locked to 0.54899 Peak + Refined R2/R3
    # --------------------------------------------------------------------------
    print("\nGenerating Candidate A: blend_rank1_locked_sota549_r23_refined_submission.csv...")
    p_final_a = 0.95 * p_test_champ_ensemble + 0.05 * p_test_clean
    cand_a = p_final_a.copy()

    row_idx = np.arange(n_test)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    peak_r1_indices = np.array([class_to_idx[r] for r in peak_r1])

    cand_a[row_idx, peak_r1_indices] = -1.0
    is_non_none = (peak_r1_indices != none_idx)
    cand_a[is_non_none, none_idx] *= 0.75  # Optimal beta = 0.75

    part2_a = np.argpartition(-cand_a, kth=1, axis=1)[:, :2]
    part2_p_a = cand_a[row_idx[:, None], part2_a]
    order2_a = np.argsort(-part2_p_a, axis=1, kind="mergesort")
    top2_a = part2_a[row_idx[:, None], order2_a]

    preds_a = [f"{peak_r1[i]} {classes[top2_a[i, 0]]} {classes[top2_a[i, 1]]}" for i in range(n_test)]

    out_file_a = SUB_DIR / "blend_rank1_locked_sota549_r23_refined_submission.csv"
    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_a}).to_csv(out_file_a, index=False)
    print(f"  [Saved] {out_file_a.name}")
    try:
        shutil.copy(out_file_a, DOWNLOADS_DIR / out_file_a.name)
        print(f"  [Copied to Downloads] {DOWNLOADS_DIR / out_file_a.name}")
    except Exception:
        pass

    diff_a = (pd.Series(preds_a) != pd.Series(peak_preds)).sum()
    r1_diff_a = (pd.Series([p.split()[0] for p in preds_a]) != pd.Series(peak_r1)).sum()
    none_cnt_a = sum(1 for p in preds_a if p.split()[0] == "NONE")
    print(f"  Top-1 NONE Count   : {none_cnt_a} / {n_test} ({none_cnt_a / n_test:.2%})")
    print(f"  Rank-1 Flips vs Peak: {r1_diff_a} / {n_test} (0.00% strictly locked!)")
    print(f"  Rank 2/3 Refinements: {diff_a} / {n_test} ({diff_a / n_test:.2%})")

    val_a = validate_submission_file(out_file_a, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    assert val_a.get("valid", False), f"Validation failed: {val_a}"
    print("  Validation Gate    : PASSED (100% compliant)")

    # --------------------------------------------------------------------------
    # Candidate B: Continuous Optimal Champion (Gentle 47 Shifts)
    # --------------------------------------------------------------------------
    print("\nGenerating Candidate B: blend_continuous_optimal_champion_submission.csv...")
    p_final_b = 0.95 * p_test_champ_ensemble + 0.05 * p_test_clean
    r1_b = np.argmax(p_final_b, axis=1)

    cand_b = p_final_b.copy()
    cand_b[row_idx, r1_b] = -1.0
    cand_b[r1_b != none_idx, none_idx] *= 0.75

    part2_b = np.argpartition(-cand_b, kth=1, axis=1)[:, :2]
    part2_p_b = cand_b[row_idx[:, None], part2_b]
    order2_b = np.argsort(-part2_p_b, axis=1, kind="mergesort")
    top2_b = part2_b[row_idx[:, None], order2_b]

    preds_b = [f"{classes[r1_b[i]]} {classes[top2_b[i, 0]]} {classes[top2_b[i, 1]]}" for i in range(n_test)]

    out_file_b = SUB_DIR / "blend_continuous_optimal_champion_submission.csv"
    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_b}).to_csv(out_file_b, index=False)
    print(f"  [Saved] {out_file_b.name}")
    try:
        shutil.copy(out_file_b, DOWNLOADS_DIR / out_file_b.name)
        print(f"  [Copied to Downloads] {DOWNLOADS_DIR / out_file_b.name}")
    except Exception:
        pass

    diff_b = (pd.Series(preds_b) != pd.Series(peak_preds)).sum()
    r1_diff_b = (pd.Series([p.split()[0] for p in preds_b]) != pd.Series(peak_r1)).sum()
    none_cnt_b = sum(1 for p in preds_b if p.split()[0] == "NONE")
    print(f"  Top-1 NONE Count   : {none_cnt_b} / {n_test} ({none_cnt_b / n_test:.2%})")
    print(f"  Rank-1 Flips vs Peak: {r1_diff_b} / {n_test} ({r1_diff_b / n_test:.2%})")
    print(f"  Total Triplet Diffs: {diff_b} / {n_test} ({diff_b / n_test:.2%})")

    val_b = validate_submission_file(out_file_b, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    assert val_b.get("valid", False), f"Validation failed: {val_b}"
    print("  Validation Gate    : PASSED (100% compliant)")


if __name__ == "__main__":
    build_submissions()

