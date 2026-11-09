"""Test Consensus Ensemble of SOTA 550 Foundation (V1) + SOTA 554 Foundation (V2).

Since both scored 0.55049 on Public LB with 21 Rank-1 differences:
1. Evaluate 50/50 OOF blend of V1 and V2
2. Evaluate Consensus Test Predictions
3. Check flip alignment and safe corridor
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
from src.metrics.map3 import calculate_map3

def main():
    print("=" * 80, flush=True)
    print("TESTING CONSENSUS ENSEMBLE OF SOTA 550 (V1) + SOTA 554 (V2)", flush=True)
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

    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    test_v1 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")

    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")

    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")

    # V1 Base Foundation
    p_oof_v1_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    p_test_v1_base = 0.95 * test_m15 + 0.02 * test_dual + 0.03 * test_comp

    # Mult calibration
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

    p_oof_cal_v1 = p_oof_v1_base * mult
    p_test_cal_v1 = p_test_v1_base * mult
    p_test_cal_v1[:, none_idx] *= 0.9995

    # V2 Base Foundation
    p_oof_v2_base = (0.88 * oof_m15 + 0.12 * oof_v2) * mult
    p_test_v2_base = (0.88 * test_m15 + 0.12 * test_v2) * mult
    p_test_v2_base[:, none_idx] *= 0.9995

    # Transductive Boost
    p_test_boosted_v1 = np.copy(p_test_cal_v1)
    p_test_boosted_v2 = np.copy(p_test_v2_base)
    for i in range(n_test):
        qid = test_df.iloc[i]["question_id"]
        if qid in q_map:
            allowed = q_map[qid]
            for ci, code in enumerate(classes):
                if code != "NONE" and code in allowed:
                    p_test_boosted_v1[i, ci] *= 1.20
                    p_test_boosted_v2[i, ci] *= 1.20

    p_test_arc_v1_cal = (0.95 * test_m15 + 0.05 * test_v1) * mult
    p_champ_v1 = 0.70 * p_test_cal_v1 + 0.25 * p_test_boosted_v1 + 0.05 * p_test_arc_v1_cal

    p_test_arc_v2_cal = (0.88 * test_m15 + 0.12 * test_v2) * mult
    p_champ_v2 = 0.65 * p_test_v2_base + 0.25 * p_test_boosted_v2 + 0.10 * p_test_arc_v2_cal

    # Clean Multimodal
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

    p_final_v1 = 0.96 * p_champ_v1 + 0.04 * p_clean
    p_final_v2 = 0.96 * p_champ_v2 + 0.04 * p_clean

    # Evaluate OOF MAP@3 of V1, V2, and Blend
    p_oof_v1_full = (0.95 * oof_m15 + 0.05 * oof_v1) * mult
    p_oof_v2_full = (0.88 * oof_m15 + 0.12 * oof_v2) * mult
    p_oof_consensus = 0.50 * p_oof_v1_full + 0.50 * p_oof_v2_full

    def eval_oof(probs):
        top3 = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
        preds = [[classes[idx] for idx in row] for row in top3]
        return calculate_map3(y_true, preds)

    print(f"OOF MAP@3 V1 Full       : {eval_oof(p_oof_v1_full):.5f}")
    print(f"OOF MAP@3 V2 Full       : {eval_oof(p_oof_v2_full):.5f}")
    print(f"OOF MAP@3 Consensus 50/50: {eval_oof(p_oof_consensus):.5f} (Delta: {eval_oof(p_oof_consensus) - eval_oof(p_oof_v1_full):+.5f})")

    # Consensus Test Matrix
    p_test_consensus = 0.50 * p_final_v1 + 0.50 * p_final_v2

    # Ranking generator
    def generate_preds(p_matrix, beta=0.82):
        cand = p_matrix.copy()
        row_idx = np.arange(n_test)
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        return [" ".join([classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]]) for i in range(n_test)]

    preds_consensus_base = generate_preds(p_test_consensus)

    # Gating on Consensus
    p_bin = np.load(ART_DIR / "test_compact_binary_prob.npy")
    r1_cons = [p.split()[0] for p in preds_consensus_base]
    preds_consensus_gated = list(preds_consensus_base)

    flips = []
    tau = 0.780
    eps = 0.040
    for i in range(n_test):
        if r1_cons[i] == "NONE" and p_bin[i] >= tau:
            row_p = p_test_consensus[i].copy()
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
                    preds_consensus_gated[i] = f"{top_misc_code} NONE {second_misc_code}"
                    flips.append(i)

    print(f"\nConsensus Flips applied : {len(flips)}")
    s550 = pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv")
    s550_preds = s550["root_code"].tolist()
    r1_diffs = sum(1 for a, b in zip(r1_cons, [p.split()[0] for p in s550_preds]) if a != b)
    print(f"Rank-1 diffs vs SOTA 550: {r1_diffs}")

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

