"""Audit 45 High-Confidence Test Error Traces (P(is_error) >= 0.85).

For each of the 45 test rows:
- Question ID, Topic, Grade
- P(is_error) from Detector V2
- Top-1 misconception code
- Gap = P(NONE) - P(Top-1 Misc)
- Is Top-1 Misc in question's allowed set?
- Overlap with Golden 11 Flips from SOTA 550
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
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager

def sharp(probs: np.ndarray, t: float) -> np.ndarray:
    s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    return s / s.sum(axis=1, keepdims=True)

def main():
    print("=" * 80, flush=True)
    print("AUDITING 45 HIGH-CONFIDENCE TEST ERROR TRACES (P_bin_v2 >= 0.85)", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}

    # Reconstruct exact SOTA 551 Optimal Continuous Base Matrix
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    test_v1 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")

    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")

    p_oof_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    p_test_base = 0.95 * test_m15 + 0.02 * test_dual + 0.03 * test_comp

    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(train_df["root_code"].fillna("NONE").values)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_test_cal = p_test_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc_cal = (0.95 * test_m15 + 0.05 * test_v1) * mult
    p_champ = 0.70 * p_test_cal + 0.25 * p_test_boosted + 0.05 * p_test_arc_cal

    from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

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

    cand = p_final.copy()
    row_idx = np.arange(n_test)
    r1 = np.argmax(cand, axis=1)
    cand[row_idx, r1] = -1.0
    cand[r1 != none_idx, none_idx] *= 0.82
    top2 = np.argsort(-cand, axis=1)[:, :2]

    p_bin_v2 = np.load(ART_DIR / "test_compact_binary_prob_v2.npy")
    s550 = pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv")
    s551_opt = pd.read_csv(SUB_DIR / "blend_breakthrough_sota551_optimal_submission.csv")
    golden_550 = set(i for i in range(len(s550)) if s550.iloc[i]["root_code"].split()[0] != s551_opt.iloc[i]["root_code"].split()[0])

    # Filter rows with r1 == NONE and p_bin_v2 >= 0.85
    candidates = []
    for i in range(n_test):
        if r1[i] == none_idx and p_bin_v2[i] >= 0.85:
            r2_idx = top2[i, 0]
            r2_code = classes[r2_idx]
            p_none = p_final[i, none_idx]
            p_r2 = p_final[i, r2_idx]
            gap = p_none - p_r2
            qid = test_df.iloc[i]["question_id"]
            is_allowed = (r2_code in q_map.get(qid, set()))
            is_golden = (i in golden_550)

            candidates.append({
                "idx": i,
                "qid": qid,
                "r2_code": r2_code,
                "conf": p_bin_v2[i],
                "gap": gap,
                "is_allowed": is_allowed,
                "is_golden": is_golden
            })

    cand_df = pd.DataFrame(candidates).sort_values("gap")
    print(f"Total Qualified Rows (P_bin_v2 >= 0.85): {len(cand_df)}")
    print(f"  Allowed by Question Prior: {cand_df['is_allowed'].sum()} / {len(cand_df)}")
    print(f"  Overlaps Golden 550 Flips : {cand_df['is_golden'].sum()} / {len(cand_df)}")

    print("\nCandidates sorted by smallest gap (Most urgent flips):")
    print(f"{'Idx':5s} | {'Conf':6s} | {'Gap':7s} | {'R2 Code':10s} | {'Allowed?':8s} | {'Golden?':8s}")
    print("-" * 55)
    for _, r in cand_df.head(25).iterrows():
        print(f"{int(r['idx']):5d} | {r['conf']:6.4f} | {r['gap']:7.4f} | {r['r2_code']:10s} | {str(r['is_allowed']):8s} | {str(r['is_golden']):8s}")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

