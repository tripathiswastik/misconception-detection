"""Phase 0 & 5: Completely Leak-Free, In-Fold Honest OOF Pipeline.

Rigorous Invariants Enforced:
1. Zero Gold Transcription Steps: Uses honest_oof_text.npy (test-like reality, MAP@3 ~0.42).
2. In-Fold Frequency Calibration: Frequency multiplier mult is computed strictly per fold using only fold_train labels and predictions.
3. In-Fold Transductive Question Boost: q_allowed is computed strictly from fold_train for each validation fold.
4. In-Fold Binary Error Detector: Uses out-of-fold predictions from train_compact_error_metalearner.py.
5. Exact Canonical Splits: Evaluated strictly over the 5 folds in data/splits.json.

Evaluates:
- Baseline Active Peak (0.54949 architecture).
- Precision Gate (tau=0.78, eps=0.04 -> the 0.55049 winning formula).
- Precision Gate sweep (tau in [0.76, 0.80], eps in [0.03, 0.05]).
- Reassesses whether the precision gate's gains hold up when all leakage is eliminated.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.metrics.map3 import calculate_map3

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


def run_honest_pipeline(tau: float = 0.78, eps: float = 0.04, boost_factor: float = 1.20, beta: float = 0.82):
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    n_classes = len(classes)

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # 1. Load Component OOFs
    # Note: Use honest_oof_text (leak-free) instead of leaked oof_text
    honest_text = np.load(ART_DIR / "honest_oof_text.npy")
    oof_m15_raw = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")

    # Honest M15 reconstruction (replacing leaked text with honest text)
    w_m15 = np.array([0.4900, 0.0191, 0.0497, 0.0077, 0.2384, 0.1951])
    w_m15 = w_m15 / w_m15.sum()
    prior_oof = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    cn_oof = np.load(ART_DIR / "oof_convnext_vision.npy")
    mn_oof = np.load(ART_DIR / "oof_finetuned_vision.npy")
    swin_oof = np.load(ART_DIR / "oof_swin_vision.npy")
    sem_text_oof = np.load(ART_DIR / "oof_semantic_text.npy")
    trocr_oof = np.load(ART_DIR / "oof_trocr_vision.npy")
    clean_arc_oof = np.load(ART_DIR / "oof_clean_arcface_2328.npy")

    oof_m15_honest = (
        w_m15[0] * prior_oof +
        w_m15[1] * cn_oof +
        w_m15[2] * mn_oof +
        w_m15[3] * swin_oof +
        w_m15[4] * honest_text +
        w_m15[5] * sem_text_oof
    )

    oof_dual448 = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    oof_comp448 = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    oof_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    p_bin_error = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    def sharp(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        return s / s.sum(axis=1, keepdims=True)

    p_clean_honest = (
        0.6448 * prior_oof +
        0.0839 * sharp(honest_text, 0.6) +
        0.0401 * sharp(sem_text_oof, 0.6) +
        0.0458 * cn_oof +
        0.0454 * swin_oof +
        0.0483 * mn_oof +
        0.0455 * trocr_oof +
        0.0462 * sharp(clean_arc_oof, 0.8)
    )

    # Matrices to collect across folds
    baseline_triplets = [[] for _ in range(n_samples)]
    gated_triplets = [[] for _ in range(n_samples)]
    flips_per_fold = []

    fold_baseline_scores = []
    fold_gated_scores = []

    for fold_idx, fold in enumerate(splits):
        tr_idx = fold["train_indices"]
        va_idx = fold["val_indices"]
        tr_df = train_df.iloc[tr_idx]
        va_df = train_df.iloc[va_idx]
        y_tr = [y_true[i] for i in tr_idx]
        y_va = [y_true[i] for i in va_idx]

        # --- A. In-Fold Base Matrix ---
        p_base_tr = 0.95 * oof_m15_honest[tr_idx] + 0.02 * oof_dual448[tr_idx] + 0.03 * oof_comp448[tr_idx]
        p_base_va = 0.95 * oof_m15_honest[va_idx] + 0.02 * oof_dual448[va_idx] + 0.03 * oof_comp448[va_idx]

        # --- B. In-Fold mult (computed strictly on tr_idx!) ---
        top3_tr = np.argsort(-p_base_tr.astype(np.float64), axis=1)[:, :3]
        pred_counts_tr = Counter(classes[idx] for row in top3_tr for idx in row)
        true_counts_tr = Counter(y_tr)
        mult_fold = np.ones(n_classes, dtype=np.float64)
        for ci, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts_tr.get(c, 1), 1)
                tc = max(true_counts_tr.get(c, 1), 1)
                mult_fold[ci] = (tc / pc) ** 0.05
        mult_fold[none_idx] = 0.9840

        # Calibrated val probabilities
        p_va_cal = p_base_va * mult_fold
        p_va_cal[:, none_idx] *= 0.9995

        # --- C. In-Fold Transductive Question Boost ---
        q_allowed_tr: Dict[str, Set[str]] = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allowed_tr[q] = set(grp["root_code"].dropna().unique())

        p_va_boosted = p_va_cal.copy()
        for local_i, global_i in enumerate(va_idx):
            qid = train_df.iloc[global_i]["question_id"]
            allowed_q = q_allowed_tr.get(qid, set())
            for ci in range(n_classes):
                if ci != none_idx and classes[ci] in allowed_q:
                    p_va_boosted[local_i, ci] *= boost_factor

        # Pseudo 3500 branch
        p_va_arc3500 = (0.95 * oof_m15_honest[va_idx] + 0.05 * oof_arc3500[va_idx]) * mult_fold

        # Champion combination
        p_champ_va = 0.70 * p_va_cal + 0.25 * p_va_boosted + 0.05 * p_va_arc3500
        p_final_va = 0.96 * p_champ_va + 0.04 * p_clean_honest[va_idx]

        # --- D. Baseline Ranking ---
        local_row_idx = np.arange(len(va_idx))
        r1_va = np.argmax(p_final_va, axis=1)
        cand_va = p_final_va.copy()
        cand_va[local_row_idx, r1_va] = -1.0
        cand_va[r1_va != none_idx, none_idx] *= beta
        top2_va = np.argsort(-cand_va, axis=1)[:, :2]

        triplets_va_base = [[classes[r1_va[i]], classes[top2_va[i, 0]], classes[top2_va[i, 1]]] for i in range(len(va_idx))]
        sc_base = score_triplets(y_va, triplets_va_base)
        fold_baseline_scores.append(sc_base)

        for local_i, global_i in enumerate(va_idx):
            baseline_triplets[global_i] = triplets_va_base[local_i]

        # --- E. Precision Gate Application ---
        triplets_va_gated = [list(t) for t in triplets_va_base]
        fold_flips = 0

        for local_i, global_i in enumerate(va_idx):
            if r1_va[local_i] == none_idx and p_bin_error[global_i] >= tau:
                row_p = p_final_va[local_i].copy()
                p_none_val = row_p[none_idx]
                row_p[none_idx] = -1.0
                top_misc_idx = int(np.argmax(row_p))
                top_misc_code = classes[top_misc_idx]
                p_misc_val = row_p[top_misc_idx]
                gap = p_none_val - p_misc_val

                if 0 < gap < eps:
                    qid = train_df.iloc[global_i]["question_id"]
                    if top_misc_code in q_allowed_tr.get(qid, set()):
                        row_p[top_misc_idx] = -1.0
                        second_misc_idx = int(np.argmax(row_p))
                        second_misc_code = classes[second_misc_idx]
                        triplets_va_gated[local_i] = [top_misc_code, "NONE", second_misc_code]
                        fold_flips += 1

        sc_gated = score_triplets(y_va, triplets_va_gated)
        fold_gated_scores.append(sc_gated)
        flips_per_fold.append(fold_flips)

        for local_i, global_i in enumerate(va_idx):
            gated_triplets[global_i] = triplets_va_gated[local_i]

    overall_baseline_map3 = score_triplets(y_true, baseline_triplets)
    overall_gated_map3 = score_triplets(y_true, gated_triplets)
    total_flips = sum(flips_per_fold)

    return {
        "overall_baseline_map3": overall_baseline_map3,
        "overall_gated_map3": overall_gated_map3,
        "delta": overall_gated_map3 - overall_baseline_map3,
        "total_flips": total_flips,
        "flips_per_fold": flips_per_fold,
        "fold_baseline_scores": fold_baseline_scores,
        "fold_gated_scores": fold_gated_scores,
    }


def main():
    print("=" * 80)
    print("RIGOROUS LEAK-FREE HONEST OOF PIPELINE BENCHMARK")
    print("=" * 80)

    # 1. Evaluate Proven 0.55049 Gating Configuration
    res_0550 = run_honest_pipeline(tau=0.78, eps=0.04)
    print(f"Honest Baseline OOF MAP@3 : {res_0550['overall_baseline_map3']:.5f}")
    print(f"Honest Gated OOF MAP@3    : {res_0550['overall_gated_map3']:.5f} (Delta: {res_0550['delta']:+.5f})")
    print(f"Total Flips across 5 folds: {res_0550['total_flips']} (Per fold: {res_0550['flips_per_fold']})")
    print("Fold scores:")
    for f in range(5):
        print(f"  Fold {f}: Base={res_0550['fold_baseline_scores'][f]:.5f} -> Gated={res_0550['fold_gated_scores'][f]:.5f} (Delta: {res_0550['fold_gated_scores'][f] - res_0550['fold_baseline_scores'][f]:+.5f})")

    # 2. Sweep Thresholds on Honest Pipeline
    print("\n" + "=" * 80)
    print("SWEEPING PRECISION GATE THRESHOLDS ON 100% LEAK-FREE PIPELINE")
    print("=" * 80)
    print(f"{'Tau':>6s} | {'Epsilon':>8s} | {'Honest OOF MAP@3':>17s} | {'Delta vs Base':>14s} | {'Train Flips':>12s}")
    print("-" * 80)

    best_sc = res_0550['overall_baseline_map3']
    best_params = None

    for tau in [0.760, 0.765, 0.770, 0.775, 0.780, 0.785, 0.790]:
        for eps in [0.020, 0.030, 0.035, 0.040, 0.045, 0.050]:
            r = run_honest_pipeline(tau=tau, eps=eps)
            sc = r["overall_gated_map3"]
            delta = r["delta"]
            flips = r["total_flips"]
            flag = " *" if sc > best_sc else ""
            if sc > best_sc:
                best_sc = sc
                best_params = (tau, eps)
            print(f"{tau:6.3f} | {eps:8.3f} | {sc:17.5f} | {delta:+14.5f} | {flips:12d}{flag}")

    print("=" * 80)
    if best_params:
        print(f"OPTIMAL HONEST GATE: tau={best_params[0]:.3f}, eps={best_params[1]:.3f} | Peak OOF: {best_sc:.5f}")
    print("=" * 80)


if __name__ == "__main__":
    main()

