"""Audit Honest OOF Leakage: Compare Leaked oof_text vs Honest honest_oof_text.

Investigation:
1. Does oof_text contain gold transcription_steps (leaked)?
2. What is the real honest OOF score of M15, M18, and Champion when evaluated with honest_oof_text?
3. Does this explain why M18 (40% text) regressed on Kaggle while M15 (23% text) scored 0.54149?
4. Does the Precision Gate's gain hold up on the honest pipeline?
"""
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

    oof_text_leaked = np.load(ART_DIR / "oof_text.npy")
    oof_text_honest = np.load(ART_DIR / "honest_oof_text.npy")

    # Evaluate Text Model standalone
    r1_leak = np.argmax(oof_text_leaked, axis=1)
    trips_leak = [[classes[idx], "NONE", "OTH-00"] for idx in r1_leak]
    sc_leak = score_triplets(y_true, trips_leak)

    r1_honest = np.argmax(oof_text_honest, axis=1)
    trips_honest = [[classes[idx], "NONE", "OTH-00"] for idx in r1_honest]
    sc_honest = score_triplets(y_true, trips_honest)

    print("=" * 80)
    print("STANDALONE TEXT MODEL: LEAKED vs HONEST")
    print("=" * 80)
    print(f"Leaked Text OOF (with gold transcription_steps):  MAP@3: {sc_leak:.5f}")
    print(f"Honest Text OOF (zero gold transcription_steps):  MAP@3: {sc_honest:.5f}")
    print(f"Artificial Leakage Inflation:                     +{sc_leak - sc_honest:.5f}")

    # Reconstruct M15 with Leaked vs Honest Text
    prior_oof = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    vis_cn_oof = np.load(ART_DIR / "oof_convnext_vision.npy")
    vis_mn_oof = np.load(ART_DIR / "oof_finetuned_vision.npy")
    swin_oof = np.load(ART_DIR / "oof_swin_vision.npy")
    text_sem_oof = np.load(ART_DIR / "oof_semantic_text.npy")

    w = np.array([0.4900, 0.0191, 0.0497, 0.0077, 0.2384, 0.1951])
    w = w / w.sum()

    m15_leaked = (w[0] * prior_oof + w[1] * vis_cn_oof + w[2] * vis_mn_oof +
                  w[3] * swin_oof + w[4] * oof_text_leaked + w[5] * text_sem_oof)

    m15_honest = (w[0] * prior_oof + w[1] * vis_cn_oof + w[2] * vis_mn_oof +
                  w[3] * swin_oof + w[4] * oof_text_honest + w[5] * text_sem_oof)

    r1_m15_l = np.argmax(m15_leaked, axis=1)
    cand_l = m15_leaked.copy()
    cand_l[np.arange(n_samples), r1_m15_l] = -1.0
    cand_l[r1_m15_l != none_idx, none_idx] *= 0.82
    top2_l = np.argsort(-cand_l, axis=1)[:, :2]
    sc_m15_leak = score_triplets(y_true, [[classes[r1_m15_l[i]], classes[top2_l[i, 0]], classes[top2_l[i, 1]]] for i in range(n_samples)])

    r1_m15_h = np.argmax(m15_honest, axis=1)
    cand_h = m15_honest.copy()
    cand_h[np.arange(n_samples), r1_m15_h] = -1.0
    cand_h[r1_m15_h != none_idx, none_idx] *= 0.82
    top2_h = np.argsort(-cand_h, axis=1)[:, :2]
    sc_m15_honest = score_triplets(y_true, [[classes[r1_m15_h[i]], classes[top2_h[i, 0]], classes[top2_h[i, 1]]] for i in range(n_samples)])

    print("\n" + "=" * 80)
    print("MODEL M15: LEAKED vs HONEST")
    print("=" * 80)
    print(f"M15 with Leaked Text OOF:                         MAP@3: {sc_m15_leak:.5f}")
    print(f"M15 with Honest Text OOF:                         MAP@3: {sc_m15_honest:.5f}")
    print(f"M15 Leakage Inflation:                           +{sc_m15_leak - sc_m15_honest:.5f}")

    # Reconstruct M18 with Leaked vs Honest Text
    w_prior, w_sgd, w_sem, w_vis = 0.35, 0.40, 0.10, 0.15
    m18_leaked = (w_prior * prior_oof + w_sgd * oof_text_leaked + w_sem * text_sem_oof + w_vis * 0.5 * (vis_cn_oof + swin_oof))
    m18_honest = (w_prior * prior_oof + w_sgd * oof_text_honest + w_sem * text_sem_oof + w_vis * 0.5 * (vis_cn_oof + swin_oof))

    r1_m18_l = np.argmax(m18_leaked, axis=1)
    cand_18l = m18_leaked.copy()
    cand_18l[np.arange(n_samples), r1_m18_l] = -1.0
    cand_18l[r1_m18_l != none_idx, none_idx] *= 0.82
    top2_18l = np.argsort(-cand_18l, axis=1)[:, :2]
    sc_m18_leak = score_triplets(y_true, [[classes[r1_m18_l[i]], classes[top2_18l[i, 0]], classes[top2_18l[i, 1]]] for i in range(n_samples)])

    r1_m18_h = np.argmax(m18_honest, axis=1)
    cand_18h = m18_honest.copy()
    cand_18h[np.arange(n_samples), r1_m18_h] = -1.0
    cand_18h[r1_m18_h != none_idx, none_idx] *= 0.82
    top2_18h = np.argsort(-cand_18h, axis=1)[:, :2]
    sc_m18_honest = score_triplets(y_true, [[classes[r1_m18_h[i]], classes[top2_18h[i, 0]], classes[top2_18h[i, 1]]] for i in range(n_samples)])

    print("\n" + "=" * 80)
    print("MODEL M18: LEAKED vs HONEST")
    print("=" * 80)
    print(f"M18 with Leaked Text OOF:                         MAP@3: {sc_m18_leak:.5f}")
    print(f"M18 with Honest Text OOF:                         MAP@3: {sc_m18_honest:.5f}")
    print(f"M18 Leakage Inflation:                           +{sc_m18_leak - sc_m18_honest:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

