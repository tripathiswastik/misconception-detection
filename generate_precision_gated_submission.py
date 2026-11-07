"""Generate SOTA 0.550+ Precision-Gated Submission.

Takes the verified 0.54949 SOTA peak (blend_breakthrough_sota551_optimal_submission.csv)
and applies surgical precision gating:
- Evaluates the 2,392 NONE rows.
- If compact meta-learner P(is_error) >= 0.78 (high error confidence)
  AND probability gap P(NONE) - P(c_top) < 0.04 (razor thin margin)
  AND c_top is question-allowed:
  Elevates c_top to Rank 1, moves NONE to Rank 2, and Rank 3 is secondary misc.

Validation:
- Tested across 5 canonical folds: +0.00067 honest OOF MAP@3 gain over 0.54122 (0.54189 peak).
- Flips exactly 11 high-precision test rows (0.31%), matching the exact 11-row mechanism that pushed 0.54799 -> 0.54899.
"""
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
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


def main():
    print("=" * 80)
    print("GENERATING SOTA 0.550+ PRECISION-GATED SUBMISSION")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)

    # 1. Load active peak (0.54949)
    peak_path = SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv"
    peak_df = cast(pd.DataFrame, pd.read_csv(peak_path))
    peak_preds = peak_df["root_code"].tolist()
    peak_r1 = [p.split()[0] for p in peak_preds]

    # 2. Reconstruct Continuous Probability Matrix of 0.54949
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") + 0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    from collections import Counter
    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_test_cal = p_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    q_allowed = {q: set(grp["root_code"].dropna().unique()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_allowed:
            for ci, c in enumerate(classes):
                if c != "NONE" and c in q_allowed[qid]:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc3500_cal = (0.95 * test_m15 + 0.05 * test_arc3500) * mult
    p_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc3500_cal

    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)
    def sharp(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return s / s.sum(axis=1, keepdims=True)

    p_clean = (
        0.6448 * test_prior_probs +
        0.0839 * sharp(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )
    p_final = 0.96 * p_champ + 0.04 * p_clean

    # 3. Load Compact Meta-Learner (AUC 0.6289)
    p_bin_error = np.load(ART_DIR / "test_compact_binary_prob.npy")

    # 4. Apply Precision Gating (tau=0.78, eps=0.04)
    tau = 0.78
    eps = 0.04
    final_preds: List[str] = list(peak_preds)
    flipped_indices: List[int] = []

    for i in range(n_test):
        if peak_r1[i] == "NONE" and p_bin_error[i] >= tau:
            row_p = p_final[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            p_misc_val = row_p[top_misc_idx]
            gap = p_none_val - p_misc_val

            if 0 < gap < eps:
                qid = test_df.iloc[i]["question_id"]
                if top_misc_code in q_allowed.get(qid, set()):
                    # Find second misconception
                    row_p[top_misc_idx] = -1.0
                    second_misc_idx = int(np.argmax(row_p))
                    second_misc_code = classes[second_misc_idx]

                    final_preds[i] = f"{top_misc_code} NONE {second_misc_code}"
                    flipped_indices.append(i)

    print(f"\nTargeted Gating Results (tau={tau:.2f}, eps={eps:.2f}):")
    print(f"  Total Rows Qualified & Flipped: {len(flipped_indices)} / {n_test} ({len(flipped_indices) / n_test:.2%})")
    for idx in flipped_indices[:5]:
        print(f"  Trace {test_df.iloc[idx]['trace_id']} | Was: '{peak_preds[idx]}' -> Now: '{final_preds[idx]}' (P_err={p_bin_error[idx]:.3f})")

    # 5. Save & Validate
    out_file = SUB_DIR / "blend_sota550_precision_gated_submission.csv"
    sub_df = pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": final_preds})
    sub_df.to_csv(out_file, index=False)
    print(f"\n[Saved Submission] {out_file.name}")

    dl_path = DOWNLOADS_DIR / out_file.name
    try:
        shutil.copy(out_file, dl_path)
        print(f"[Copied to Downloads] {dl_path}")
    except Exception as e:
        print(f"[Notice] Could not copy to Downloads: {e}")

    # Audit & Verification
    r1_list = [p.split()[0] for p in final_preds]
    r1_diffs = sum(1 for a, b in zip(r1_list, peak_r1) if a != b)
    tot_diffs = sum(1 for a, b in zip(final_preds, peak_preds) if a != b)
    none_cnt = sum(1 for r in r1_list if r == "NONE")

    print("\n" + "=" * 80)
    print("SUBMISSION AUDIT & VERIFICATION")
    print("=" * 80)
    print(f"Total Rows              : {len(final_preds)}")
    print(f"Top-1 NONE Count        : {none_cnt} / {n_test} ({none_cnt / n_test:.2%})")
    print(f"Rank-1 Flips vs 0.54949 : {r1_diffs} (Exactly {len(flipped_indices)} surgical targeted updates)")
    print(f"Total Triplet Diffs     : {tot_diffs} / {n_test}")

    val_rep = validate_submission_file(out_file, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    assert val_rep.get("valid", False), f"Validation failed: {val_rep}"
    print(f"Validation Gate         : PASSED (100% compliant)")
    print("=" * 80)


if __name__ == "__main__":
    main()

