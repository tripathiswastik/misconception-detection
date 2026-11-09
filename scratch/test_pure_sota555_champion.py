"""Test and Generate Pure SOTA 555 Champion Submission.

Pipeline:
1. Pure V2 Base: 0.82 * M15 + 0.18 * ArcFace V2
2. In-fold Frequency Calibration (mult_v2)
3. Transductive 1.20x Question Boost
4. Clean Multimodal Blend (w_clean = 0.04)
5. Secondary Beta (beta = 0.82)
6. Precision Gating (tau = 0.780, eps = 0.040)

Outputs:
- submissions/blend_sota555_pure_v2_base_submission.csv
- submissions/blend_sota555_pure_v2_precision_gated_submission.csv
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
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80, flush=True)
    print("GENERATING PURE SOTA 555 CHAMPION PIPELINE (w_v2 = 0.18)", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_test = len(test_df)
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")

    w_v2 = 0.18
    p_oof_v2_base = (1.0 - w_v2) * oof_m15 + w_v2 * oof_v2
    p_test_v2_base = (1.0 - w_v2) * test_m15 + w_v2 * test_v2

    # In-fold Mult Calibration
    top3_tr = np.argsort(-p_oof_v2_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_oof_cal = p_oof_v2_base * mult
    p_test_cal = p_test_v2_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    # Transductive 1.20x Question Boost
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    # Compound Champion Ensemble
    p_champ_test = 0.70 * p_test_cal + 0.30 * p_test_boosted

    # Clean Multimodal Blend
    from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
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

    p_final = 0.96 * p_champ_test + 0.04 * p_clean

    # Decision Engine Ranking with secondary beta = 0.82
    def generate_preds(p_matrix, beta=0.82):
        cand = p_matrix.copy()
        row_idx = np.arange(n_test)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        return [" ".join([classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]]) for i in range(n_test)]

    preds_base = generate_preds(p_final)

    # Precision Gating with tau = 0.780, eps = 0.040
    p_bin_error = np.load(ART_DIR / "test_compact_binary_prob.npy")
    r1_base = [p.split()[0] for p in preds_base]
    preds_gated = list(preds_base)
    flips = []
    tau = 0.780
    eps = 0.040
    for i in range(n_test):
        if r1_base[i] == "NONE" and p_bin_error[i] >= tau:
            row_p = p_final[i].copy()
            p_none_val = row_p[none_idx]
            row_p[none_idx] = -1.0
            top_misc_idx = int(np.argmax(row_p))
            top_misc_code = classes[top_misc_idx]
            gap = p_none_val - row_p[top_misc_idx]

            if 0 < gap < eps:
                qid = test_df.iloc[i]["question_id"]
                if top_misc_code in q_map.get(qid, set()):
                    row_p[top_misc_idx] = -1.0
                    second_misc_idx = int(np.argmax(row_p))
                    second_misc_code = classes[second_misc_idx]
                    preds_gated[i] = f"{top_misc_code} NONE {second_misc_code}"
                    flips.append(i)

    print(f"Total Flips Applied               : {len(flips)}")
    s550 = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s550_preds = s550["root_code"].tolist()
    r1_diffs = sum(1 for a, b in zip([p.split()[0] for p in preds_gated], [p.split()[0] for p in s550_preds]) if a != b)
    triplet_diffs = sum(1 for a, b in zip(preds_gated, s550_preds) if a != b)
    print(f"Rank-1 Diffs vs SOTA 550 (0.55049): {r1_diffs}")
    print(f"Triplet Diffs vs SOTA 550         : {triplet_diffs}")

    # Save Submissions
    sub_base_path = SUB_DIR / "blend_sota555_pure_v2_base_submission.csv"
    sub_gated_path = SUB_DIR / "blend_sota555_pure_v2_precision_gated_submission.csv"

    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_base}).to_csv(sub_base_path, index=False)
    pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds_gated}).to_csv(sub_gated_path, index=False)

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

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

