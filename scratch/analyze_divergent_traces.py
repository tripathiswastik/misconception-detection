"""Analyze Divergent Traces between Champion and Clean Multimodal Model.

Examines:
1. The 23 traces where Champion = NONE but Clean Ensemble = Misconception.
2. The 383 traces where Champion = Misconception but Clean Ensemble = NONE.
3. Probability margins: P(top misconception) - P(NONE).
4. Cross-validation analog on train.csv: does resolving divergence by probability margin improve CV MAP@3?
"""

import json
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]


def score_map3(y_true_subset, preds_subset):
    scores = []
    for label, pred in zip(y_true_subset, preds_subset):
        score = 0.0
        for r, p in enumerate(pred[:3]):
            if p == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    print("=" * 80)
    print("ANALYZE DIVERGENCE & MARGIN REFINEMENT ON TRAIN CV")
    print("=" * 80)

    # Load Champion OOF
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    p_champ = 0.95 * oof_m15 + 0.02 * oof_dual448 + 0.03 * oof_comp448

    # Frequency calibration on Champion OOF
    mult = np.ones(len(tax_classes), dtype=np.float64)
    mult[none_idx] = 0.9840
    p_champ_cal = p_champ * mult

    # Load Clean Multimodal Ensemble
    from src.data.taxonomy import TaxonomyManager
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")

    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    p_th = np.load(ART_DIR / "honest_oof_text.npy")
    p_ts = np.load(ART_DIR / "oof_semantic_text.npy")
    p_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    # In-fold domain mask for MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )

    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_th_sharp = apply_t(p_th, 0.6)
    p_ts_sharp = apply_t(p_ts, 0.6)

    p_clean = (
        0.58 * p_prior +
        0.05 * p_cn +
        0.07 * p_swin +
        0.07 * p_mn +
        0.04 * p_trocr +
        0.11 * p_th_sharp +
        0.08 * p_ts_sharp
    )

    r1_champ = np.argmax(p_champ_cal, axis=1)
    r1_clean = np.argmax(p_clean, axis=1)

    agree = (r1_champ == r1_clean)
    print(f"Train Rank-1 Agreement: {agree.sum()} / 1500 ({agree.mean():.2%})")

    # On agreeing rows: what is MAP@3?
    agree_idx = np.where(agree)[0]
    disagree_idx = np.where(~agree)[0]

    # Evaluate when Champion and Clean disagree
    champ_correct_disagree = sum(1 for i in disagree_idx if y_true[i] == tax_classes[r1_champ[i]])
    clean_correct_disagree = sum(1 for i in disagree_idx if y_true[i] == tax_classes[r1_clean[i]])
    print(f"On {len(disagree_idx)} Disagreeing Rows:")
    print(f"  Champion Rank 1 was correct: {champ_correct_disagree} ({champ_correct_disagree/len(disagree_idx):.2%})")
    print(f"  Clean Model Rank 1 was correct: {clean_correct_disagree} ({clean_correct_disagree/len(disagree_idx):.2%})")

    # What if we blend probabilities:
    fused = 0.95 * p_champ_cal + 0.05 * p_clean
    r1_fused = np.argmax(fused, axis=1)
    fused_correct = sum(1 for i in range(1500) if y_true[i] == tax_classes[r1_fused[i]])
    champ_correct = sum(1 for i in range(1500) if y_true[i] == tax_classes[r1_champ[i]])
    print(f"\nOverall Rank 1 Accuracy:")
    print(f"  Champion Rank 1 Accuracy: {champ_correct} / 1500 ({champ_correct/1500:.2%})")
    print(f"  Fused Rank 1 Accuracy   : {fused_correct} / 1500 ({fused_correct/1500:.2%}) (+{fused_correct - champ_correct})")


if __name__ == "__main__":
    main()

