"""Generate Next-Level Precision-Gated Submissions (0.55150 - 0.55300+).

Anchored on the proven 0.55049 winning formula (blend_breakthrough_sota551_optimal_submission.csv + precision gating).
Evaluates fine-tuned thresholds discovered in audit_fine_gating_thresholds.py:

Candidate 1 (Conservative Step):
- tau = 0.775, eps = 0.040
- OOF MAP@3: 0.54222 (+0.00100 over 0.54122 baseline)
- Exactly 21 qualified test flips (vs 11 flips in 0.55049)
- Expected LB: 0.55150 - 0.55200+

Candidate 2 (High-Upside Step):
- tau = 0.770, eps = 0.045
- OOF MAP@3: 0.54256 (+0.00133 over 0.54122 baseline, 2x the CV gain of 0.55049)
- Exactly 29 qualified test flips (0.83% of test set)
- Expected LB: 0.55220 - 0.55300+
"""
import shutil
import sys
from collections import Counter
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
    print("GENERATING NEXT-LEVEL PRECISION-GATED SUBMISSIONS (0.55150 - 0.55300+)")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # 1. Load active peak foundation
    peak_path = SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv"
    peak_df = cast(pd.DataFrame, pd.read_csv(peak_path))
    peak_preds = peak_df["root_code"].tolist()
    peak_r1 = [p.split()[0] for p in peak_preds]

    # 2. Reconstruct Continuous Probability Matrix of Active Peak
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = 0.95 * oof_m15 + 0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") + 0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
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

    # Load Compact Meta-Learner
    p_bin_error = np.load(ART_DIR / "test_compact_binary_prob.npy")

    def generate_gated(tau: float, eps: float, filename: str):
        preds = list(peak_preds)
        flips = []
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
                        row_p[top_misc_idx] = -1.0
                        second_misc_idx = int(np.argmax(row_p))
                        second_misc_code = classes[second_misc_idx]
                        preds[i] = f"{top_misc_code} NONE {second_misc_code}"
                        flips.append(i)

        out_path = SUB_DIR / filename
        pd.DataFrame({"trace_id": test_df["trace_id"], "root_code": preds}).to_csv(out_path, index=False)
        print(f"\n[Generated {filename}]")
        print(f"  Thresholds: tau={tau:.3f}, eps={eps:.3f}")
        print(f"  Total Rank-1 Flips vs Peak: {len(flips)} / {n_test} ({len(flips)/n_test:.2%})")

        val = validate_submission_file(out_path, taxonomy_csv=DATA_DIR / "taxonomy_v1.csv")
        assert val.get("valid", False), f"Validation failed: {val}"
        print(f"  Validation Gate: PASSED (100% compliant)")

        # Copy to Downloads
        dl_path = DOWNLOADS_DIR / filename
        try:
            shutil.copy(out_path, dl_path)
            print(f"  [Copied to Downloads] {dl_path}")
        except Exception as e:
            print(f"  [Notice] Copy failed: {e}")

        return preds, flips

    # 1. Candidate A: tau=0.775, eps=0.040 (21 Flips)
    generate_gated(0.775, 0.040, "blend_sota551_precision_gated_tau775_submission.csv")

    # 2. Candidate B: tau=0.770, eps=0.045 (29 Flips)
    generate_gated(0.770, 0.045, "blend_sota552_precision_gated_tau770_submission.csv")

    # 3. Candidate C: tau=0.770, eps=0.035 (24 Flips, ultra-tight gap)
    generate_gated(0.770, 0.035, "blend_sota551_precision_gated_tau770_tight_submission.csv")

    print("\n" + "=" * 80)
    print("ALL 3 NEXT-LEVEL PRECISION-GATED CANDIDATES SUCCESSFULLY PRODUCED!")
    print("=" * 80)


if __name__ == "__main__":
    main()

