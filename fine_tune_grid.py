"""Fine-grained weight exploration around M14."""
import sys
from pathlib import Path
from typing import cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

tax = TaxonomyManager(PROJECT_ROOT / "data" / "taxonomy_v1.csv")
classes = tax.all_codes(include_none=True)

train_df = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "data" / "train.csv"))
y_true = train_df["root_code"].astype(str).tolist()

art = PROJECT_ROOT / "artifacts"
prior = np.load(art / "best_adaptive_prior_oof.npy")
vis_cn = np.load(art / "oof_convnext_vision.npy")
vis_mn = np.load(art / "oof_finetuned_vision.npy")
swin = np.load(art / "oof_swin_vision.npy")
text_sgd = np.load(art / "oof_text.npy")
text_sem = np.load(art / "oof_semantic_text.npy")

def eval_map3(probs):
    top3_idx = np.argsort(-probs, axis=1, kind="mergesort")[:, :3]
    preds = [[classes[idx] for idx in row] for row in top3_idx]
    return calculate_map3(y_true, preds)

# Baseline M14: [0.60, 0.036, 0.060, 0.024, 0.14, 0.14] -> 0.53167
w_base = np.array([0.60, 0.036, 0.060, 0.024, 0.14, 0.14])
base_fused = (w_base[0] * prior + w_base[1] * vis_cn + w_base[2] * vis_mn +
              w_base[3] * swin + w_base[4] * text_sgd + w_base[5] * text_sem)
base_score = eval_map3(base_fused)
print(f"M14 Baseline OOF MAP@3: {base_score:.5f}")

# Sweep w_prior from 0.45 to 0.65 in fine steps
best_score = base_score
best_w = w_base.copy()
count = 0

for w_prior in np.arange(0.45, 0.66, 0.02):
    rem = 1.0 - w_prior
    for vis_frac in np.arange(0.15, 0.45, 0.05):
        w_vis = rem * vis_frac
        w_txt = rem * (1.0 - vis_frac)
        for cn_frac in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for swin_frac in [0.10, 0.15, 0.20, 0.25]:
                mn_frac = 1.0 - cn_frac - swin_frac
                if mn_frac <= 0.05:
                    continue
                for sgd_frac in [0.35, 0.40, 0.45, 0.50, 0.55]:
                    sem_frac = 1.0 - sgd_frac
                    w = np.array([
                        w_prior,
                        w_vis * cn_frac,
                        w_vis * mn_frac,
                        w_vis * swin_frac,
                        w_txt * sgd_frac,
                        w_txt * sem_frac
                    ])
                    fused = (w[0] * prior + w[1] * vis_cn + w[2] * vis_mn +
                             w[3] * swin + w[4] * text_sgd + w[5] * text_sem)
                    score = eval_map3(fused)
                    count += 1
                    if score > best_score:
                        best_score = score
                        best_w = w.copy()
                        print(f"New best: {best_score:.5f} (Delta: {best_score - base_score:+.5f}) with prior={w_prior:.2f}, vis={w_vis:.3f}, txt={w_txt:.3f}")

print(f"\nEvaluated {count} combinations")
print(f"Best OOF MAP@3: {best_score:.5f}")
labels = ["Prior", "ConvNeXt", "MobileNet", "Swin-T", "SGD Text", "Semantic Text"]
for lbl, w in zip(labels, best_w):
    print(f"  {lbl}: {w:.4f}")

