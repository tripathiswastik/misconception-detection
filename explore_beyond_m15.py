"""Exploration beyond M15 (0.54149 LB).

Tests:
1. Ultra-fine grid search around M15 optimal weights (0.47-0.52 prior)
2. Temperature / Power transformation on components
3. Selective Transductive Bayesian refinement for unseen test questions
"""
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

# Baseline M15: OOF 0.53822 (LB 0.54149)
w_m15 = np.array([0.4900, 0.0191, 0.0497, 0.0077, 0.2384, 0.1951])
w_m15 = w_m15 / w_m15.sum()

fused_m15 = (w_m15[0] * prior + w_m15[1] * vis_cn + w_m15[2] * vis_mn +
             w_m15[3] * swin + w_m15[4] * text_sgd + w_m15[5] * text_sem)
m15_score = eval_map3(fused_m15)
print(f"M15 Baseline OOF MAP@3: {m15_score:.5f}")

# Experiment 1: Ultra-fine sweep around M15
print("\n--- Experiment 1: Ultra-fine weight sweep ---")
best_score = m15_score
best_w = w_m15.copy()

for p in np.arange(0.46, 0.52, 0.01):
    rem = 1.0 - p
    for vis_frac in np.arange(0.12, 0.20, 0.02):
        w_v = rem * vis_frac
        w_t = rem * (1.0 - vis_frac)
        for cn_f in [0.20, 0.25, 0.30]:
            for sw_f in [0.08, 0.10, 0.12]:
                mn_f = 1.0 - cn_f - sw_f
                for sgd_f in [0.50, 0.52, 0.54, 0.56, 0.58]:
                    sem_f = 1.0 - sgd_f
                    w_cand = np.array([
                        p,
                        w_v * cn_f,
                        w_v * mn_f,
                        w_v * sw_f,
                        w_t * sgd_f,
                        w_t * sem_f
                    ])
                    cand_fused = (w_cand[0] * prior + w_cand[1] * vis_cn + w_cand[2] * vis_mn +
                                  w_cand[3] * swin + w_cand[4] * text_sgd + w_cand[5] * text_sem)
                    sc = eval_map3(cand_fused)
                    if sc > best_score:
                        best_score = sc
                        best_w = w_cand.copy()
                        print(f"  New best: {best_score:.5f} (Delta: {best_score - m15_score:+.5f}) p={p:.2f}")

print(f"Best fine-tuned score: {best_score:.5f}")

# Experiment 2: Power transformation on text probabilities
print("\n--- Experiment 2: Power sharpening on text and prior ---")
best_t_score = m15_score
best_t_params = None

for t_sgd in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    for t_sem in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
        sgd_p = text_sgd ** (1.0 / t_sgd)
        sgd_p /= sgd_p.sum(axis=1, keepdims=True)
        sem_p = text_sem ** (1.0 / t_sem)
        sem_p /= sem_p.sum(axis=1, keepdims=True)
        
        cand = (best_w[0] * prior + best_w[1] * vis_cn + best_w[2] * vis_mn +
                best_w[3] * swin + best_w[4] * sgd_p + best_w[5] * sem_p)
        sc = eval_map3(cand)
        if sc > best_t_score:
            best_t_score = sc
            best_t_params = (t_sgd, t_sem)
            print(f"  Sharpening best: {best_t_score:.5f} (t_sgd={t_sgd}, t_sem={t_sem})")

print(f"Overall Best OOF MAP@3: {max(best_score, best_t_score):.5f}")

