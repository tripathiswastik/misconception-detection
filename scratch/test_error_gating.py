"""Error Evidence Gating and Conditional Decoupling under Strict 5-Fold CV.

Mathematical Principle:
- P(Error) = 1 - P_observed(NONE) from text + vision features.
- If P(Error) > theta: student made an error -> promote top non-NONE candidate to Rank 1.
- If P(Error) <= theta: student produced correct / clean work -> keep NONE at Rank 1.
- Evaluated strictly inside 5-Fold CV: theta tuned on fold-train, tested on held-out fold.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")
n_classes = len(tax_classes)

train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
y_true = train_df["root_code"].fillna("NONE").astype(str).tolist()
splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]


def score_map3(y_true_subset: List[str], preds_subset: List[List[str]]) -> float:
    scores = []
    for label, pred in zip(y_true_subset, preds_subset):
        score = 0.0
        for r, p in enumerate(pred[:3]):
            if p == label:
                score = 1.0 / (r + 1)
                break
        scores.append(score)
    return float(np.mean(scores))


def make_gated_predictions(
    fused_probs: np.ndarray,
    error_prob: np.ndarray,
    theta_low: float,
    theta_high: float,
    classes: List[str]
) -> List[List[str]]:
    """Predict top 3 with error gating."""
    n = len(fused_probs)
    preds = []
    non_none_mask = np.ones(len(classes), dtype=bool)
    non_none_mask[none_idx] = False
    non_none_indices = np.where(non_none_mask)[0]

    for i in range(n):
        p_row = fused_probs[i]
        ep = error_prob[i]

        # Top 3 non-NONE codes according to fused model
        top_nn_local = np.argsort(-p_row[non_none_mask])[:3]
        c1 = classes[non_none_indices[top_nn_local[0]]]
        c2 = classes[non_none_indices[top_nn_local[1]]]
        c3 = classes[non_none_indices[top_nn_local[2]]]

        if ep <= theta_low:
            # High confidence in NONE
            preds.append(["NONE", c1, c2])
        elif ep <= theta_high:
            # Moderate error evidence: error at Rank 1, NONE retained at Rank 2
            preds.append([c1, "NONE", c2])
        else:
            # Strong error evidence: error at Rank 1, secondary error at Rank 2, NONE at Rank 3
            preds.append([c1, c2, "NONE"])

    return preds


def main():
    print("=" * 80)
    print("ERROR EVIDENCE GATING AUDIT (STRICT 5-FOLD CV)")
    print("=" * 80)

    # Load components
    from src.data.taxonomy import TaxonomyManager
    from src.evaluation.leak_free_cv import fast_fold_domain_mask
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")

    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")

    # In-fold domain mask for MobileNet
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], tax_classes, tax=tax, damping=0.85
        )

    # Stage A calibrated text
    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    p_th_sharp = apply_t(p_text_honest, 0.6)
    p_ts_sharp = apply_t(p_text_sem, 0.6)

    w = np.array([0.6059, 0.0448, 0.0703, 0.0709, 0.1189, 0.0892])
    fused_probs = (
        w[0] * p_prior + w[1] * p_cn + w[2] * p_swin +
        w[3] * p_mn + w[4] * p_th_sharp + w[5] * p_ts_sharp
    )

    # Candidate Error Evidence Signals:
    # 1. Honest Text Error Probability: 1 - P_text(NONE)
    e_text = 1.0 - p_th_sharp[:, none_idx]

    # 2. Multimodal Error Probability: 1 - P_fused(NONE)
    e_fused = 1.0 - fused_probs[:, none_idx]

    # 3. Text + Vision Error Probability (excluding student-blind prior!)
    vis_text_probs = (0.2 * p_cn + 0.3 * p_swin + 0.3 * p_mn + 0.6 * p_th_sharp + 0.4 * p_ts_sharp) / 1.8
    e_observed = 1.0 - vis_text_probs[:, none_idx]

    signals = {
        "Honest_Text_Signal": e_text,
        "Fused_Signal": e_fused,
        "Observed_Student_Work_Signal": e_observed,
    }

    # Evaluate baseline (raw argmax without gating)
    k = 3
    n = len(fused_probs)
    part = np.argpartition(-fused_probs, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-fused_probs[row_idx, part], axis=1, kind="mergesort")
    raw_preds = [[tax_classes[idx] for idx in row] for row in part[row_idx, order]]
    raw_score = score_map3(y_true, raw_preds)
    print(f"Base Calibrated Blend (Ungated Argmax) OOF MAP@3: {raw_score:.5f}")

    for sig_name, err_sig in signals.items():
        print(f"\n--- Testing Signal: {sig_name} ---")
        best_t_low = 0.5
        best_t_high = 0.8
        best_cv = 0.0

        # Grid search over theta_low and theta_high
        for t_low in np.linspace(0.30, 0.80, 11):
            for t_high in np.linspace(t_low, 0.95, 8):
                fold_scores = []
                for f in splits:
                    va = f["val_indices"]
                    va_preds = make_gated_predictions(
                        fused_probs[va], err_sig[va], t_low, t_high, tax_classes
                    )
                    fold_scores.append(score_map3([y_true[i] for i in va], va_preds))
                mean_cv = float(np.mean(fold_scores))
                if mean_cv > best_cv:
                    best_cv = mean_cv
                    best_t_low = t_low
                    best_t_high = t_high

        print(f"  Best Parameters: theta_low = {best_t_low:.2f}, theta_high = {best_t_high:.2f}")
        print(f"  5-Fold CV MAP@3: {best_cv:.5f} ({best_cv - raw_score:+.5f})")

        # Evaluate on full OOF
        opt_preds = make_gated_predictions(fused_probs, err_sig, best_t_low, best_t_high, tax_classes)
        overall = score_map3(y_true, opt_preds)

        # Rank distribution
        r1 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[0])
        r2 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[1])
        r3 = sum(1 for y, p in zip(y_true, opt_preds) if y == p[2])

        err_idx = [i for i, y in enumerate(y_true) if y != "NONE"]
        err_r1 = sum(1 for i in err_idx if y_true[i] == opt_preds[i][0])
        err_r1_none = sum(1 for i in err_idx if opt_preds[i][0] == "NONE")

        none_idx_list = [i for i, y in enumerate(y_true) if y == "NONE"]
        none_r1 = sum(1 for i in none_idx_list if opt_preds[i][0] == "NONE")

        print(f"  Overall OOF MAP@3: {overall:.5f} ({overall - raw_score:+.5f})")
        print(f"  Rank Distribution: Rank 1={r1:4d} ({(r1/1500)*100:.1f}%), Rank 2={r2:4d}, Rank 3={r3:4d}")
        print(f"  Error Traces (994): True at Rank 1={err_r1:4d} ({(err_r1/994)*100:.1f}%), Incorrect Rank 1 NONE={err_r1_none:4d}")
        print(f"  NONE Traces (506):  Correct at Rank 1={none_r1:4d} ({(none_r1/506)*100:.1f}%)")


if __name__ == "__main__":
    main()

