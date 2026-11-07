"""Optimize Honest Multimodal Weights on the 100% Leak-Free Pipeline.

With honest_oof_text (zero gold transcription steps), what are the true optimal weights
between Prior, Text, Vision (ConvNeXt, Swin, TrOCR), Semantic Text, and ArcFace?
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # 100% Honest, Leak-Free Components
    prior_oof = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    honest_text = np.load(ART_DIR / "honest_oof_text.npy")
    sem_text = np.load(ART_DIR / "oof_semantic_text.npy")
    cn_oof = np.load(ART_DIR / "oof_convnext_vision.npy")
    swin_oof = np.load(ART_DIR / "oof_swin_vision.npy")
    trocr_oof = np.load(ART_DIR / "oof_trocr_vision.npy")
    arc_oof = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    clean_arc = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    def sharp(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return s / s.sum(axis=1, keepdims=True)

    print("=" * 80)
    print("STANDALONE HONEST COMPONENT BENCHMARK")
    print("=" * 80)

    comps = {
        "Adaptive Prior": prior_oof,
        "Honest Text (no gold steps)": honest_text,
        "Semantic Text (MiniLM-384)": sem_text,
        "ConvNeXt Vision": cn_oof,
        "Swin Vision": swin_oof,
        "TrOCR Vision": trocr_oof,
        "ArcFace Composite 448": arc_oof,
        "Clean ArcFace 2328": clean_arc,
    }

    for name, mat in comps.items():
        r1 = np.argmax(mat, axis=1)
        cand = mat.copy()
        cand[np.arange(n_samples), r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]
        trips = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        sc = score_triplets(y_true, trips)
        print(f"{name:30s} | Honest OOF MAP@3: {sc:.5f}")

    # Evaluate current honest baseline
    # Grid search across text weight w_text in [0.05, 0.10, 0.15, 0.20, 0.25]
    print("\n" + "=" * 80)
    print("SWEEPING TEXT WEIGHT ON HONEST ENSEMBLE")
    print("=" * 80)

    for w_txt in [0.00, 0.05, 0.08, 0.10, 0.15, 0.20, 0.24]:
        # Remaining weight redistributed to Prior and Vision
        w_rem = 1.0 - w_txt
        w_pr = 0.65 * w_rem
        w_sem = 0.15 * w_rem
        w_vis = 0.20 * w_rem

        fused = (
            w_pr * prior_oof +
            w_txt * honest_text +
            w_sem * sem_text +
            w_vis * (0.4 * cn_oof + 0.4 * swin_oof + 0.2 * trocr_oof)
        )

        r1 = np.argmax(fused, axis=1)
        cand = fused.copy()
        cand[np.arange(n_samples), r1] = -1.0
        cand[r1 != none_idx, none_idx] *= 0.82
        top2 = np.argsort(-cand, axis=1)[:, :2]
        trips = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(n_samples)]
        sc = score_triplets(y_true, trips)
        print(f"w_text={w_txt:4.2f} (w_prior={w_pr:4.2f}, w_sem={w_sem:4.2f}, w_vis={w_vis:4.2f}) | Honest MAP@3: {sc:.5f}")


if __name__ == "__main__":
    main()
