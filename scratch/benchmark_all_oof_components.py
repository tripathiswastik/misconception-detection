"""Evaluate Standalone OOF MAP@3 of Every Component in artifacts/."""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def score_triplets(y_true: List[str], triplets: List[List[str]]) -> float:
    scores = []
    for true_label, row in zip(y_true, triplets):
        score = 0.0
        for r, pred in enumerate(row[:3]):
            if pred == true_label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    oof_files = [
        "best_adaptive_prior_oof.npy",
        "oof_text.npy",
        "oof_semantic_text.npy",
        "oof_convnext_vision.npy",
        "oof_swin_vision.npy",
        "oof_finetuned_vision.npy",
        "oof_trocr_vision.npy",
        "oof_trivision.npy",
        "oof_arcface_2328d.npy",
        "oof_clean_arcface_2328.npy",
        "oof_arcface_dualhead_448_soft.npy",
        "oof_arcface_composite_448.npy",
        "oof_arcface_pseudo_3500.npy",
        "oof_m15_balanced_fusion.npy",
        "oof_m16_power_calibrated.npy",
        "oof_m18_tuned_convex.npy",
    ]

    print("=" * 80)
    print("STANDALONE OOF MAP@3 BENCHMARK OF ALL COMPONENTS")
    print("=" * 80)

    for fn in oof_files:
        p_path = ART_DIR / fn
        if not p_path.exists():
            continue
        probs = np.load(p_path)
        if probs.shape != (n_samples, n_classes):
            continue

        r1 = np.argmax(probs, axis=1)
        cand = probs.copy()
        cand[np.arange(n_samples), r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]
        triplets = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        sc = score_triplets(y_true, triplets)
        none_r1 = sum(1 for r in r1 if r == none_idx)
        print(f"{fn:35s} | MAP@3: {sc:.5f} | R1 NONE: {none_r1:4d} ({none_r1/n_samples:.1%})")


if __name__ == "__main__":
    main()

