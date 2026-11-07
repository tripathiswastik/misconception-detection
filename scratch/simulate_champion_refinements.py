"""Simulate Champion Refinements on Test Set.

Evaluates:
- gamma = 0.00 (unwarped probabilities)
- alpha = 1.10 (gentle transductive boost)
- beta = 0.70 (30% secondary NONE discount on error rows)
- w_clean in [0.03, 0.05, 0.08]

Compares against the active SOTA peak: blend_optimal_multimodal_theoretical_submission.csv (0.54899).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"


def apply_temperature(probs: np.ndarray, t: float) -> np.ndarray:
    if abs(t - 1.0) < 1e-5:
        return probs.copy()
    sharp = np.power(np.maximum(probs, 1e-12), 1.0 / t)
    row_sums = sharp.sum(axis=1, keepdims=True)
    return np.divide(sharp, row_sums, where=row_sums > 0)


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    n_test = len(test_df)

    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_dual448 = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp448 = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_arc3500 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    p_test_champ_base = 0.95 * test_m15 + 0.02 * test_dual448 + 0.03 * test_comp448

    # Mild question boost alpha = 1.10
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

    # Clean ensemble
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

    s548 = pd.read_csv(SUB_DIR / "blend_optimal_multimodal_theoretical_submission.csv")
    s548_r1 = s548["root_code"].str.split().str[0]

    print("=" * 80)
    print("SIMULATING REFINED CHAMPION ON TEST SET")
    print("=" * 80)

    for w_clean in [0.00, 0.03, 0.05, 0.08]:
        p_final = (1.0 - w_clean) * p_test_champ_ensemble + w_clean * p_test_clean
        r1 = np.argmax(p_final, axis=1)

        for beta in [0.70, 0.75, 0.80, 0.82]:
            cand = p_final.copy()
            cand[np.arange(n_test), r1] = -1.0
            cand[r1 != none_idx, none_idx] *= beta
            top2 = np.argsort(-cand, axis=1)[:, :2]
            preds = [f"{classes[r1[i]]} {classes[top2[i, 0]]} {classes[top2[i, 1]]}" for i in range(n_test)]

            preds_r1 = pd.Series([p.split()[0] for p in preds])
            none_cnt = sum(1 for r in preds_r1 if r == "NONE")
            r1_diffs = (preds_r1 != s548_r1).sum()
            tot_diffs = (pd.Series(preds) != s548["root_code"]).sum()

            print(f"w_clean={w_clean:.2f} | beta={beta:.2f} -> NONE R1: {none_cnt} ({none_cnt/n_test:.2%}) | R1 Diffs vs 0.54899: {r1_diffs:2d} | Total Triplet Diffs: {tot_diffs:3d}")


if __name__ == "__main__":
    main()

