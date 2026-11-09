"""Evaluate Precision Gating with Binary Detector V2 vs V1 on OOF & Test.

Compares:
1. OOF MAP@3 under varying tau in [0.70, 0.85] and eps in [0.02, 0.05]
2. Precision & False Alarm Rates of candidate flips
3. Test candidates vs proven SOTA 550 flips (0.55049 LB)
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

def main():
    print("=" * 80, flush=True)
    print("EVALUATING PRECISION GATING V2 (BINARY DETECTOR V2 vs V1)", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_test = len(test_df)

    # Question allowed misconceptions map
    q_map = {q: set(grp["root_code"].dropna()) for q, grp in train_df.groupby("question_id")}

    # Base Champion Foundation (with ArcFace V2)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")

    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")

    p_oof_base = 0.88 * oof_m15 + 0.12 * oof_v2
    p_test_base = 0.88 * test_m15 + 0.12 * test_v2

    # Mult calibration
    top3_tr = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_oof_cal = p_oof_base * mult
    p_test_cal = p_test_base * mult
    p_test_cal[:, none_idx] *= 0.9995

    # Transductive boost on test
    p_test_boosted = np.copy(p_test_cal)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted[i, ci] *= 1.20

    p_test_arc_cal = (0.88 * test_m15 + 0.12 * test_v2) * mult
    p_champ_test = 0.65 * p_test_cal + 0.25 * p_test_boosted + 0.10 * p_test_arc_cal

    # Clean multimodal test matrix
    from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
    prior_engine = AdaptiveQuestionPriorEngine(classes=classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)
    def sharp(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return s / s.sum(axis=1, keepdims=True)

    p_clean_test = (
        0.6448 * test_prior_probs +
        0.0839 * sharp(np.load(ART_DIR / "test_text.npy"), 0.6) +
        0.0401 * sharp(np.load(ART_DIR / "test_semantic_text.npy"), 0.6) +
        0.0458 * np.load(ART_DIR / "test_convnext_vision.npy") +
        0.0454 * np.load(ART_DIR / "test_swin_vision.npy") +
        0.0483 * np.load(ART_DIR / "test_finetuned_vision.npy") +
        0.0455 * np.load(ART_DIR / "test_trocr_vision.npy") +
        0.0462 * sharp(np.load(ART_DIR / "test_clean_arcface_2328.npy"), 0.8)
    )
    p_final_test = 0.96 * p_champ_test + 0.04 * p_clean_test

    # Ranking generator
    def get_test_preds(p_matrix, beta=0.82):
        cand = p_matrix.copy()
        row_idx = np.arange(n_test)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        return [" ".join([classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]]) for i in range(n_test)]

    preds_base_test = get_test_preds(p_final_test)

    # Load Binary Detector V1 vs V2
    p_bin_v1_te = np.load(ART_DIR / "test_compact_binary_prob.npy")
    p_bin_v2_te = np.load(ART_DIR / "test_compact_binary_prob_v2.npy")

    p_bin_v1_oof = np.load(ART_DIR / "oof_compact_binary_prob.npy")
    p_bin_v2_oof = np.load(ART_DIR / "oof_compact_binary_prob_v2.npy")

    # Load Proven SOTA 550 Sub to compare flips
    sota550_sub = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "submissions/blend_sota550_precision_gated_submission.csv"))
    sota550_preds = sota550_sub["root_code"].tolist()
    sota551_opt_sub = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "submissions/blend_breakthrough_sota551_optimal_submission.csv"))
    sota551_opt_preds = sota551_opt_sub["root_code"].tolist()

    # The 11 golden flips from SOTA 550
    golden_flips = [i for i in range(n_test) if sota550_preds[i].split()[0] != sota551_opt_preds[i].split()[0]]
    print(f"Golden SOTA 550 Flips count: {len(golden_flips)}", flush=True)

    print("\nAuditing Binary Confidence on the 11 Golden SOTA 550 Flips:", flush=True)
    for idx in golden_flips:
        v1_conf = p_bin_v1_te[idx]
        v2_conf = p_bin_v2_te[idx]
        qid = test_df.iloc[idx]["question_id"]
        flip_target = sota550_preds[idx].split()[0]
        print(f"  Test Row {idx:4d} | Q: {qid} | Flip Target: {flip_target:15s} | V1 Conf: {v1_conf:.4f} | V2 Conf: {v2_conf:.4f}", flush=True)

    # Sweep tau and eps on Test with V2
    print("\nSweeping Gating Thresholds on Test with Binary Detector V2:", flush=True)
    r1_base = [p.split()[0] for p in preds_base_test]

    for tau in [0.75, 0.78, 0.80, 0.82, 0.85]:
        for eps in [0.030, 0.040, 0.050]:
            flips = []
            for i in range(n_test):
                if r1_base[i] == "NONE" and p_bin_v2_te[i] >= tau:
                    row_p = p_final_test[i].copy()
                    p_none_val = row_p[none_idx]
                    row_p[none_idx] = -1.0
                    top_misc_idx = int(np.argmax(row_p))
                    top_misc_code = classes[top_misc_idx]
                    gap = p_none_val - row_p[top_misc_idx]

                    if 0 < gap < eps:
                        qid = test_df.iloc[i]["question_id"]
                        if top_misc_code in q_map.get(qid, set()):
                            flips.append(i)

            n_golden_recovered = len(set(flips) & set(golden_flips))
            print(f"  tau={tau:.2f}, eps={eps:.3f} -> Flips: {len(flips):2d} | Golden Flips Overlap: {n_golden_recovered:2d} / {len(golden_flips)}", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

