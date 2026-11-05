"""Evaluate M14 + M15 Ensemble Blend."""
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.data.taxonomy import TaxonomyManager
from src.fusion.validator import validate_submission_file
from src.metrics.map3 import calculate_map3

tax = TaxonomyManager(PATHS.taxonomy_csv)
classes = tax.all_codes(include_none=True)

train_df = cast(pd.DataFrame, pd.read_csv(PATHS.data_dir / "train.csv"))
y_true = train_df["root_code"].astype(str).tolist()

art = PROJECT_ROOT / "artifacts"
prior_oof = np.load(art / "best_adaptive_prior_oof.npy")
vis_cn_oof = np.load(art / "oof_convnext_vision.npy")
vis_mn_oof = np.load(art / "oof_finetuned_vision.npy")
swin_oof = np.load(art / "oof_swin_vision.npy")
text_sgd_oof = np.load(art / "oof_text.npy")
text_sem_oof = np.load(art / "oof_semantic_text.npy")

def eval_map3(probs):
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)

# Reconstruct M14 OOF (LB 0.53683)
w_m14 = np.array([0.60, 0.036, 0.060, 0.024, 0.14, 0.14])
oof_m14 = (w_m14[0] * prior_oof + w_m14[1] * vis_cn_oof + w_m14[2] * vis_mn_oof +
           w_m14[3] * swin_oof + w_m14[4] * text_sgd_oof + w_m14[5] * text_sem_oof)

# Reconstruct M15 OOF (LB 0.54149)
w_m15 = np.array([0.4900, 0.0191, 0.0497, 0.0077, 0.2384, 0.1951])
w_m15 = w_m15 / w_m15.sum()
oof_m15 = (w_m15[0] * prior_oof + w_m15[1] * vis_cn_oof + w_m15[2] * vis_mn_oof +
           w_m15[3] * swin_oof + w_m15[4] * text_sgd_oof + w_m15[5] * text_sem_oof)

score_14 = eval_map3(oof_m14)
score_15 = eval_map3(oof_m15)
print(f"M14 OOF: {score_14:.5f} (LB: 0.53683)")
print(f"M15 OOF: {score_15:.5f} (LB: 0.54149)")

# Sweep blending alpha
best_alpha = 0.0
best_blend_score = 0.0

for alpha in np.arange(0.0, 1.05, 0.05):
    # alpha = weight on M15, (1-alpha) = weight on M14
    blend = alpha * oof_m15 + (1.0 - alpha) * oof_m14
    sc = eval_map3(blend)
    print(f"  Alpha M15={alpha:.2f} (M14={1-alpha:.2f}): OOF MAP@3 = {sc:.5f}")
    if sc > best_blend_score:
        best_blend_score = sc
        best_alpha = alpha

print(f"\nBest Blend: {best_alpha:.2f} M15 + {1-best_alpha:.2f} M14 -> OOF MAP@3: {best_blend_score:.5f}")

