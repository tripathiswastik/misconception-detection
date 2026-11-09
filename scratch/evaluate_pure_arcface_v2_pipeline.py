"""Evaluate Full Pure ArcFace V2 Pipeline on OOF.

Compares:
1. Baseline SOTA 550 Foundation (V1): OOF MAP@3
2. SOTA 554 Partial V2 (where V2 was only 1.2% effective weight)
3. Pure SOTA 555 Pipeline (where ArcFace V2 is 12% across the entire foundation: base, cal, boosted)
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
    print("EVALUATING PURE ARCFACE V2 FOUNDATION ON OOF", flush=True)
    print("=" * 80, flush=True)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")

    def eval_map3(probs, beta=0.82):
        cand = probs.copy()
        row_idx = np.arange(len(cand))
        r1 = np.argmax(cand, axis=1)
        cand[row_idx, r1] = -1.0
        cand[r1 != none_idx, none_idx] *= beta
        top2 = np.argsort(-cand, axis=1)[:, :2]
        preds = [[classes[r1[i]], classes[top2[i, 0]], classes[top2[i, 1]]] for i in range(len(cand))]
        return calculate_map3(y_true, preds)

    # 1. Pipeline V1 (Pure SOTA 550)
    p_oof_v1_base = 0.95 * oof_m15 + 0.02 * oof_dual + 0.03 * oof_comp
    top3_tr = np.argsort(-p_oof_v1_base.astype(np.float64), axis=1)[:, :3]
    pred_counts_v1 = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult_v1 = np.ones(n_classes, dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts_v1.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult_v1[i] = (tc / pc) ** 0.05
    mult_v1[none_idx] = 0.9840
    p_oof_v1_cal = p_oof_v1_base * mult_v1
    print(f"Pipeline V1 (SOTA 550) OOF MAP@3     : {eval_map3(p_oof_v1_cal):.5f}", flush=True)

    # 2. Pipeline SOTA 554 (Partial V2: only 1.2% effective V2)
    p_oof_554 = (0.90 * p_oof_v1_cal + 0.10 * ((0.88 * oof_m15 + 0.12 * oof_v2) * mult_v1))
    print(f"Pipeline SOTA 554 (Partial V2) OOF   : {eval_map3(p_oof_554):.5f}", flush=True)

    # 3. Pipeline Pure V2 across varying weights of V2 (0.05, 0.08, 0.10, 0.12, 0.15)
    print("\nSweeping Pure ArcFace V2 Weight in Foundation:")
    for w_v2 in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        p_base_v2 = (1.0 - w_v2) * oof_m15 + w_v2 * oof_v2
        top3_tr_v2 = np.argsort(-p_base_v2.astype(np.float64), axis=1)[:, :3]
        pred_counts_v2 = Counter(classes[idx] for row in top3_tr_v2 for idx in row)
        mult_v2 = np.ones(n_classes, dtype=np.float64)
        for i, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts_v2.get(c, 1), 1)
                tc = max(true_counts.get(c, 1), 1)
                mult_v2[i] = (tc / pc) ** 0.05
        mult_v2[none_idx] = 0.9840

        p_cal_v2 = p_base_v2 * mult_v2
        m3 = eval_map3(p_cal_v2)
        print(f"  w_v2 = {w_v2:.2f} -> Pure V2 OOF MAP@3 = {m3:.5f} (Delta vs SOTA 550: {m3 - eval_map3(p_oof_v1_cal):+.5f})", flush=True)

    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()

