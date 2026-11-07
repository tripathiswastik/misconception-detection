"""Test Precision Gating on top of the Upgraded Honest Foundation (0.52600).

Winning Honest Foundation:
- Prior: 0.40
- ArcFace Pseudo 3500: 0.15
- Semantic Text: 0.16
- ArcFace Composite 448: 0.06
- ArcFace DualHead 448: 0.03
- Honest Text: 0.10
- Swin: 0.03
- ConvNeXt: 0.04
- TrOCR: 0.03

Honest OOF Baseline: 0.52600 (+0.00200 over M15-like base).

Now apply Precision Gating (tau in [0.76, 0.78], eps in [0.03, 0.04]).
What is the new combined Honest Peak?
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
    splits = json.load(open(DATA_DIR / "splits.json"))["folds"]
    y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    c_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    c_arc3500 = np.load(ART_DIR / "oof_arcface_pseudo_3500.npy")
    c_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    c_comp = np.load(ART_DIR / "oof_arcface_composite_448.npy")
    c_dual = np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy")
    c_text = np.load(ART_DIR / "honest_oof_text.npy")
    c_swin = np.load(ART_DIR / "oof_swin_vision.npy")
    c_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    c_trocr = np.load(ART_DIR / "oof_trocr_vision.npy")
    p_bin_error = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    w = np.array([0.40, 0.15, 0.16, 0.06, 0.03, 0.10, 0.03, 0.04, 0.03])
    w = w / w.sum()

    fused_oof = (
        w[0] * c_prior +
        w[1] * c_arc3500 +
        w[2] * c_sem +
        w[3] * c_comp +
        w[4] * c_dual +
        w[5] * c_text +
        w[6] * c_swin +
        w[7] * c_cn +
        w[8] * c_trocr
    )

    # In-fold setup
    base_triplets = [[] for _ in range(n_samples)]
    p_final_matrix = np.zeros((n_samples, n_classes), dtype=np.float64)
    q_allowed_folds = []

    beta = 0.82
    boost_factor = 1.20

    for f in splits:
        tr_idx = f["train_indices"]
        va_idx = f["val_indices"]
        tr_df = train_df.iloc[tr_idx]
        y_tr = [y_true[i] for i in tr_idx]

        q_allow = {}
        for q, grp in tr_df.groupby("question_id"):
            q_allow[q] = set(grp["root_code"].dropna().unique())
        q_allowed_folds.append(q_allow)

        p_base_tr = fused_oof[tr_idx]
        p_base_va = fused_oof[va_idx]

        # In-fold mult
        top3_tr = np.argsort(-p_base_tr, axis=1)[:, :3]
        pred_counts_tr = Counter(classes[idx] for row in top3_tr for idx in row)
        true_counts_tr = Counter(y_tr)
        mult = np.ones(n_classes, dtype=np.float64)
        for ci, c in enumerate(classes):
            if c != "NONE":
                pc = max(pred_counts_tr.get(c, 1), 1)
                tc = max(true_counts_tr.get(c, 1), 1)
                mult[ci] = (tc / pc) ** 0.05
        mult[none_idx] = 0.9840

        p_va_cal = p_base_va * mult
        p_va_cal[:, none_idx] *= 0.9995

        p_va_boosted = p_va_cal.copy()
        for loc_i, glob_i in enumerate(va_idx):
            qid = train_df.iloc[glob_i]["question_id"]
            allow_q = q_allow.get(qid, set())
            for ci in range(n_classes):
                if ci != none_idx and classes[ci] in allow_q:
                    p_va_boosted[loc_i, ci] *= boost_factor

        p_final_va = 0.75 * p_va_cal + 0.25 * p_va_boosted
        p_final_matrix[va_idx] = p_final_va

        loc_rows = np.arange(len(va_idx))
        r1_va = np.argmax(p_final_va, axis=1)
        cand_va = p_final_va.copy()
        cand_va[loc_rows, r1_va] = -1.0
        cand_va[r1_va != none_idx, none_idx] *= beta
        top2_va = np.argsort(-cand_va, axis=1)[:, :2]

        for loc_i, glob_i in enumerate(va_idx):
            base_triplets[glob_i] = [classes[r1_va[loc_i]], classes[top2_va[loc_i, 0]], classes[top2_va[loc_i, 1]]]

    base_score = score_triplets(y_true, base_triplets)
    print("=" * 80)
    print(f"UPGRADED HONEST FOUNDATION BASELINE: MAP@3 = {base_score:.5f}")
    print("=" * 80)

    print("\nAuditing Precision Gating on Upgraded Foundation:")
    print("-" * 80)
    print(f"{'Tau':>6s} | {'Epsilon':>8s} | {'Honest OOF MAP@3':>17s} | {'Delta vs Base':>14s} | {'Train Flips':>12s}")
    print("-" * 80)

    best_sc = base_score
    best_params = None

    for tau in [0.760, 0.765, 0.770, 0.775, 0.780]:
        for eps in [0.020, 0.030, 0.035, 0.040, 0.045]:
            trips_gated = [list(t) for t in base_triplets]
            flips = 0

            for f_idx, f in enumerate(splits):
                va_idx = f["val_indices"]
                q_allow = q_allowed_folds[f_idx]

                for glob_i in va_idx:
                    r1_code = trips_gated[glob_i][0]
                    if r1_code == "NONE" and p_bin_error[glob_i] >= tau:
                        row_p = p_final_matrix[glob_i].copy()
                        p_none_val = row_p[none_idx]
                        row_p[none_idx] = -1.0
                        top_idx = int(np.argmax(row_p))
                        top_code = classes[top_idx]
                        p_misc_val = row_p[top_idx]
                        gap = p_none_val - p_misc_val

                        if 0 < gap < eps:
                            qid = train_df.iloc[glob_i]["question_id"]
                            if top_code in q_allow.get(qid, set()):
                                row_p[top_idx] = -1.0
                                second_idx = int(np.argmax(row_p))
                                trips_gated[glob_i] = [top_code, "NONE", classes[second_idx]]
                                flips += 1

            sc = score_triplets(y_true, trips_gated)
            flag = " *" if sc > best_sc else ""
            if sc > best_sc:
                best_sc = sc
                best_params = (tau, eps)
            print(f"{tau:6.3f} | {eps:8.3f} | {sc:17.5f} | {sc - base_score:+14.5f} | {flips:12d}{flag}")

    print("=" * 80)
    print(f"NEW ALL-TIME HONEST PEAK: {best_params} | MAP@3 = {best_sc:.5f} (Delta vs M15 base: {best_sc - 0.52400:+.5f})")
    print("=" * 80)


if __name__ == "__main__":
    main()

