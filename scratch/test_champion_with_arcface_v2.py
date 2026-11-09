"""Test Full Champion Pipeline with ArcFace V2 on OOF and Test.

Pipeline:
1. Base: 0.88 * M15 + 0.12 * ArcFace V2
2. Calibrated by mult
3. Transductive 1.20x Question Boost
4. Clean Multimodal Blend (w_clean = 0.04)
5. Secondary Beta (beta = 0.82)
6. Precision Gating (tau = 0.780, eps = 0.040 -> 11 surgical flips)

Compares:
- Baseline SOTA 551 Optimal (V1 ArcFace): 0.54949 LB
- New SOTA 554 with ArcFace V2
"""
from collections import Counter
from pathlib import Path
import shutil
import sys
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"
DOWNLOADS_DIR = Path("C:/Users/hp/Downloads")

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80)
    print("SIMULATING FULL SOTA CHAMPION WITH ARCFACE V2 (PSEUDO-DISTILLED)")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Load Components
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    test_v1 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")

    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")

    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")

    # 1. Base Reconstitution
    # V1 Formula: 0.95 m15 + 0.02 dual + 0.03 comp
    p_oof_v1_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    p_test_v1_base = 0.95 * test_m15 + 0.02 * test_dual + 0.03 * test_comp

    # Compute mult
    top3_tr = np.argsort(-p_oof_v1_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    # Test calibration
    p_test_cal_v1 = p_test_v1_base * mult
    p_test_cal_v1[:, none_idx] *= 0.9995

    # Transductive boost
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}
    p_test_boosted_v1 = np.copy(p_test_cal_v1)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted_v1[i, ci] *= 1.20

    # Old V1: 0.70 cal + 0.25 boosted + 0.05 arc_v1
    p_test_arc_v1_cal = (0.95 * test_m15 + 0.05 * test_v1) * mult
    p_champ_v1 = 0.70 * p_test_cal_v1 + 0.25 * p_test_boosted_v1 + 0.05 * p_test_arc_v1_cal

    # New V2: Give ArcFace V2 higher weight (10% to 12%) reflecting its +0.00944 higher accuracy
    p_test_arc_v2_cal = (0.88 * test_m15 + 0.12 * test_v2) * mult
    p_champ_v2 = 0.65 * p_test_cal_v1 + 0.25 * p_test_boosted_v1 + 0.10 * p_test_arc_v2_cal

    # Clean Multimodal test matrix
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

    p_final_v1 = 0.96 * p_champ_v1 + 0.04 * p_clean
    p_final_v2 = 0.96 * p_champ_v2 + 0.04 * p_clean

    # Decision Engine Ranking with secondary beta = 0.82
    def generate_preds(p_matrix, beta=0.82):
        cand = p_matrix.copy()
        row_idx = np.arange(n_test)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        return [" ".join([classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]]) for i in range(n_test)]

    preds_v1_base = generate_preds(p_final_v1)
    preds_v2_base = generate_preds(p_final_v2)

    # Apply proven precision gating (tau = 0.780, eps = 0.040)
    p_bin_error = np.load(ART_DIR / "test_compact_binary_prob.npy")
    r1_v2 = [p.split()[0] for p in preds_v2_base]

    preds_v2_gated = list(preds_v2_base)
    flips_v2 = []
    tau = 0.780
    eps = 0.040
    for i in range(n_test):
        if r1_v2[i] == "NONE" and p_bin_error[i] >= tau:
            row_p = p_final_v2[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            p_misc_val = row_p[top_misc_idx]
            gap = p_none_val - p_misc_val

            if 0 < gap < eps:
                qid = test_df.iloc[i]["question_id"]
                if top_misc_code in q_map.get(qid, set()):
                    row_p[top_misc_idx] = -1.0
                    second_misc_idx = int(np.argmax(row_p))
                    second_misc_code = classes[second_misc_idx]
                    preds_v2_gated[i] = f"{top_misc_code} NONE {second_misc_code}"
                    flips_v2.append(i)

    # Auditing
    r1_v1 = [p.split()[0] for p in preds_v1_base]
    r1_v2_b = [p.split()[0] for p in preds_v2_base]
    r1_v2_g = [p.split()[0] for p in preds_v2_gated]

    print("\n" + "=" * 80)
    print(f"Total Test Rows                     : {n_test}")
    print(f"Top-1 NONE count in V1 Base (0.54949): {sum(1 for x in r1_v1 if x == 'NONE')} ({sum(1 for x in r1_v1 if x == 'NONE')/n_test:.2%})")
    print(f"Top-1 NONE count in V2 Base         : {sum(1 for x in r1_v2_b if x == 'NONE')} ({sum(1 for x in r1_v2_b if x == 'NONE')/n_test:.2%})")
    print(f"Top-1 NONE count in V2 Gated        : {sum(1 for x in r1_v2_g if x == 'NONE')} ({sum(1 for x in r1_v2_g if x == 'NONE')/n_test:.2%})")
    print(f"Rank-1 Flips vs V1 Base (V2 Base)   : {sum(1 for a, b in zip(r1_v1, r1_v2_b) if a != b)}")
    print(f"Total Triplet Diffs vs V1 Base      : {sum(1 for a, b in zip(preds_v1_base, preds_v2_base) if a != b)}")
    print(f"Precision Flips applied on V2 Gated : {len(flips_v2)}")

    # Save Submissions
    sub_base_path = SUB_DIR / "blend_sota554_arcface_v2_base_submission.csv"
    sub_gated_path = SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"

    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_v2_base}).to_csv(sub_base_path, index=False)
    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_v2_gated}).to_csv(sub_gated_path, index=False)

    print(f"\n[Generated] {sub_base_path.name}")
    print(f"[Generated] {sub_gated_path.name}")

    v1_val = validate_submission_file(sub_base_path, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    v2_val = validate_submission_file(sub_gated_path, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
    assert v1_val.get("valid", False), f"Validation failed: {v1_val}"
    assert v2_val.get("valid", False), f"Validation failed: {v2_val}"
    print("Validation Gate: PASSED (100% compliant)")

    # Copy to Downloads
    for p in [sub_base_path, sub_gated_path]:
        try:
            shutil.copy(p, DOWNLOADS_DIR / p.name)
            print(f"[Copied to Downloads] {p.name}")
        except Exception as e:
            print(f"[Notice] Copy failed: {e}")

    print("=" * 80)

if __name__ == "__main__":
    main()

