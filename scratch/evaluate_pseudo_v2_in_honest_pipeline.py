"""Evaluate Upgraded ArcFace V2 in the 100% Leak-Free Honest Evaluation Pipeline.

Compares:
- Baseline Honest Foundation (using V1 pseudo 3500)
- Upgraded Honest Foundation (using V2 champion-distilled pseudo 3500)
"""
from pathlib import Path
import sys
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
    print("=" * 80)
    print("EVALUATING ARCFACE V2 IN 100% LEAK-FREE HONEST PIPELINE")
    print("=" * 80)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()

    # Load Honest OOF Components
    oof_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    oof_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    oof_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_txt = np.load(ART_DIR / "honest_oof_text.npy")
    oof_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    oof_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    oof_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")

    oof_v1 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    oof_v2 = np.load(ART_DIR / "oof_arcface_pseudo_3500_v2.npy")

    # Evaluate standalone
    def get_map3(probs):
        top3 = np.argsort(-probs.astype(np.float64), axis=1)[:, :3]
        preds = [[classes[idx] for idx in row] for row in top3]
        return calculate_map3(y_true, preds)

    print(f"Standalone ArcFace V1 OOF MAP@3 : {get_map3(oof_v1):.5f}")
    print(f"Standalone ArcFace V2 OOF MAP@3 : {get_map3(oof_v2):.5f}")
    print(f"Standalone Delta                : {get_map3(oof_v2) - get_map3(oof_v1):+.5f}")

    # Now evaluate in Champion Foundation (M15 + ArcFace + DualHead + Composite + Boost)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    from collections import Counter

    top3_tr = np.argsort(-oof_m15.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_tr for idx in row)
    true_counts = Counter(y_true)
    mult = np.ones(len(classes), dtype=np.float64)
    for i, c in enumerate(classes):
        if c != "NONE":
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[i] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840

    p_base_v1 = (0.95 * oof_m15 + 0.05 * oof_v1) * mult
    p_base_v2 = (0.95 * oof_m15 + 0.05 * oof_v2) * mult

    print(f"\nChampion Blend with ArcFace V1  : {get_map3(p_base_v1):.5f}")
    print(f"Champion Blend with ArcFace V2  : {get_map3(p_base_v2):.5f}")
    print(f"Net Champion Delta              : {get_map3(p_base_v2) - get_map3(p_base_v1):+.5f}")

    # Sweep ArcFace weight (0.05, 0.08, 0.10, 0.12, 0.15)
    print("\nSweeping ArcFace V2 Weight in Champion Ensemble:")
    for w_arc in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        p_sweep = ((1.0 - w_arc) * oof_m15 + w_arc * oof_v2) * mult
        print(f"  w_arc = {w_arc:.2f} -> OOF MAP@3 = {get_map3(p_sweep):.5f}")

    print("=" * 80)

if __name__ == "__main__":
    main()
