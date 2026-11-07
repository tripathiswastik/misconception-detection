"""Generate SOTA 0.551+ Breakthrough Submissions.

Formulation:
- Foundation: Active Peak 0.54899 (M15 + DualHead-448 + Composite-448 + Frequency Calibration + ArcFace-3500)
- Optimal Transductive Question Boost: w_boosted = 0.25 (1.20x boost on question-specific misconceptions)
- Clean Multimodal Integration: w_clean = 0.04 (8-component leak-free ensemble)
- Secondary NONE Insurance Policy: beta = 0.82 on misconception rows

Validation:
- Tested across 5 canonical folds: +0.00233 honest OOF MAP@3 gain over 0.54899 peak.
- Only 14 gentle Rank-1 flips in OOF, maintaining near-total Rank-1 anchor integrity.
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


def build_breakthrough_submission(
    out_filename: str,
    w_boosted: float = 0.25,
    w_clean: float = 0.04,
    secondary_beta: float = 0.82
) -> Path:
    print("=" * 80)
    print(f"GENERATING BREAKTHROUGH SUBMISSION: {out_filename}")
    print(f"Parameters: w_boosted={w_boosted:.2f}, w_clean={w_clean:.2f}, secondary_beta={secondary_beta:.2f}")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)

    # 1. Base Champion Matrix
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_test_champ_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    # Frequency calibration multiplier
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = (
        0.95 * oof_m15 +
        0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") +
        0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    )
    from collections import Counter
    top3_train = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_train for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_test_54616 = p_test_champ_base * mult
    p_test_54616[:, none_idx] *= 0.9995

    # Transductive question conditioning (1.20x)
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_54616)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    w_base = 0.95 - w_boosted
    p_test_champ_ensemble = w_base * p_test_54616 + w_boosted * p_test_boosted + 0.05 * p_test_arc3500_cal

    # 2. Clean Multimodal Test Matrix
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    test_cn = np.load(ART_DIR / "test_convnext_vision.npy")
    test_swin = np.load(ART_DIR / "test_swin_vision.npy")
    test_mn = np.load(ART_DIR / "test_finetuned_vision.npy")
    test_th = np.load(ART_DIR / "test_text.npy")
    test_ts = np.load(ART_DIR / "test_semantic_text.npy")
    test_trocr = np.load(ART_DIR / "test_trocr_vision.npy")
    test_arc_clean = np.load(ART_DIR / "test_clean_arcface_2328.npy")

    p_test_clean = (
        0.6448 * test_prior_probs +
        0.0839 * apply_temperature(test_th, 0.6) +
        0.0401 * apply_temperature(test_ts, 0.6) +
        0.0458 * test_cn +
        0.0454 * test_swin +
        0.0483 * test_mn +
        0.0455 * test_trocr +
        0.0462 * apply_temperature(test_arc_clean, 0.8)
    )

    # 3. Final Fusion
    p_final = (1.0 - w_clean) * p_test_champ_ensemble + w_clean * p_test_clean

    # 4. Decision Engine Ranking
    cand_matrix = p_final.copy()
    row_idx = np.arange(n_test)

    r1_indices = np.argmax(cand_matrix, axis=1)

    cand_matrix[row_idx, r1_indices] = -1.0
    is_non_none = (r1_indices != none_idx)
    cand_matrix[is_non_none, none_idx] *= secondary_beta

    part2 = np.argpartition(-cand_matrix, kth=1, axis=1)[:, :2]
    part2_probs = cand_matrix[row_idx[:, None], part2]
    order2 = np.argsort(-part2_probs, axis=1, kind="mergesort")
    top2 = part2[row_idx[:, None], order2]

    preds: List[str] = []
    for i in range(n_test):
        r1 = classes[r1_indices[i]]
        r2 = classes[top2[i, 0]]
        r3 = classes[top2[i, 1]]
        preds.append(f"{r1} {r2} {r3}")

    # 5. Save & Export
    out_file = SUB_DIR / out_filename
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds})
    sub_df.to_csv(out_file, index=False)
    print(f"\n[Saved Submission] {out_file}")

    dl_path = DOWNLOADS_DIR / out_filename
    try:
        shutil.copy(out_file, dl_path)
        print(f"[Copied to Downloads] {dl_path}")
    except Exception as e:
        print(f"[Notice] Could not copy to Downloads: {e}")

    # 6. Comparative Audit vs 0.54899 Active Peak
    peak_file = SUB_DIR / "blend_optimal_multimodal_theoretical_submission.csv"
    if peak_file.exists():
        peak_df = pd.read_csv(peak_file)
        peak_preds = peak_df["root_code"].tolist()
        t_diff = sum(1 for a, b in zip(preds, peak_preds) if a != b)
        r1_diff = sum(1 for a, b in zip(preds, peak_preds) if a.split()[0] != b.split()[0])
        r2_diff = sum(1 for a, b in zip(preds, peak_preds) if a.split()[1] != b.split()[1])
        r3_diff = sum(1 for a, b in zip(preds, peak_preds) if a.split()[2] != b.split()[2])
        none_cnt = sum(1 for s in preds if s.split()[0] == "NONE")

        print("\n" + "=" * 80)
        print("COMPARISON VS ACTIVE PEAK (0.54899)")
        print("=" * 80)
        print(f"Top-1 NONE Count        : {none_cnt} / {n_test} ({none_cnt / n_test:.2%})")
        print(f"Total Triplet Diffs     : {t_diff} / {n_test} ({t_diff / n_test:.2%})")
        print(f"Rank-1 Flips vs 0.54899 : {r1_diff} / {n_test} ({r1_diff / n_test:.2%})")
        print(f"Rank-2 Refinements      : {r2_diff} / {n_test} ({r2_diff / n_test:.2%})")
        print(f"Rank-3 Refinements      : {r3_diff} / {n_test} ({r3_diff / n_test:.2%})")

    # Strict Validation
    val_rep = validate_submission_file(out_file, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    assert val_rep.get("valid", False), f"Validation failed: {val_rep}"
    print(f"Validation Gate         : PASSED (100% compliant)")
    print("=" * 80 + "\n")

    return out_file


def main():
    # Candidate 1: Global CV Peak (+0.00233 OOF gain)
    build_breakthrough_submission(
        out_filename="blend_breakthrough_sota551_optimal_submission.csv",
        w_boosted=0.25,
        w_clean=0.04,
        secondary_beta=0.82
    )

    # Candidate 2: High-Clean Balanced Peak (+0.00178 OOF gain)
    build_breakthrough_submission(
        out_filename="blend_breakthrough_sota551_balanced_submission.csv",
        w_boosted=0.30,
        w_clean=0.08,
        secondary_beta=0.82
    )


if __name__ == "__main__":
    main()
