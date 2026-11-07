"""Analyze Test Set Predictions: Active Champion (0.54799) vs Clean Leak-Free Ensemble.

Examines:
1. Exact Rank 1, 2, 3 agreement and divergence on test.csv (3,498 rows).
2. For test rows where Champion predicts NONE at Rank 1:
   What does the Clean Ensemble predict? How confident is it?
3. For test rows where Champion predicts a misconception at Rank 1:
   What does the Clean Ensemble predict?
4. Distribution of Top-1 NONE counts across models.
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

with open(ART_DIR / "taxonomy_classes.json", "r", encoding="utf-8") as f:
    tax_classes = json.load(f)
none_idx = tax_classes.index("NONE")

test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
champ_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_rank1_locked_snr85_champion_submission.csv"))


def main():
    print("=" * 80)
    print("TEST SET AUDIT: ACTIVE CHAMPION (0.54799) VS CLEAN LEAK-FREE ENSEMBLE")
    print("=" * 80)
    print(f"Total Test Traces: {len(test_df)}")

    # 1. Parse Champion triplets
    champ_triplets = [row.split() for row in champ_sub["root_code"]]
    champ_r1 = [t[0] for t in champ_triplets]
    champ_r2 = [t[1] for t in champ_triplets]
    champ_r3 = [t[2] for t in champ_triplets]

    champ_none_r1 = sum(1 for r in champ_r1 if r == "NONE")
    print(f"Champion Rank-1 NONE Count: {champ_none_r1} / {len(test_df)} ({champ_none_r1/len(test_df):.2%})")

    # 2. Load Clean Test Component Predictions
    test_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy") # Wait, prior on test!
    # Let's check how prior is computed on test:
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    from src.diagnoser.adaptive_prior import AdaptiveQuestionPriorEngine
    prior_engine = AdaptiveQuestionPriorEngine(classes=tax_classes).fit(train_df)
    test_prior_probs = prior_engine.predict_proba(test_df)

    test_cn = np.load(ART_DIR / "test_convnext_vision.npy")
    test_swin = np.load(ART_DIR / "test_swin_vision.npy")
    test_mn = np.load(ART_DIR / "test_finetuned_vision.npy")
    test_th = np.load(ART_DIR / "test_text.npy")
    test_ts = np.load(ART_DIR / "test_semantic_text.npy")
    test_trocr = np.load(ART_DIR / "test_trocr_vision.npy")

    # Stage A temperature calibration on text
    def apply_t(probs, t):
        s = np.power(np.maximum(probs, 1e-12), 1.0 / t)
        rs = s.sum(axis=1, keepdims=True)
        return np.divide(s, rs, where=rs > 0)

    test_th_cal = apply_t(test_th, 0.6)
    test_ts_cal = apply_t(test_ts, 0.6)

    # Clean ensemble weights (from nested CV)
    # w = [Prior: 60.59%, CN: 4.48%, Swin: 7.03%, MN: 7.09%, Text_SGD: 11.89%, Text_Sem: 8.92%]
    p_clean_test = (
        0.6059 * test_prior_probs +
        0.0448 * test_cn +
        0.0703 * test_swin +
        0.0709 * test_mn +
        0.1189 * test_th_cal +
        0.0892 * test_ts_cal
    )

    k = 3
    n = len(test_df)
    part = np.argpartition(-p_clean_test, kth=k - 1, axis=1)[:, :k]
    row_idx = np.arange(n)[:, None]
    order = np.argsort(-p_clean_test[row_idx, part], axis=1, kind="mergesort")
    top3_idx = part[row_idx, order]
    clean_triplets = [[tax_classes[idx] for idx in row] for row in top3_idx]
    clean_r1 = [t[0] for t in clean_triplets]

    clean_none_r1 = sum(1 for r in clean_r1 if r == "NONE")
    print(f"Clean Ensemble Rank-1 NONE Count: {clean_none_r1} / {len(test_df)} ({clean_none_r1/len(test_df):.2%})")

    # 3. Agreement vs Champion
    r1_agree = sum(1 for a, b in zip(champ_r1, clean_r1) if a == b)
    print(f"\nRank-1 Agreement: {r1_agree} / {len(test_df)} ({r1_agree/len(test_df):.2%})")
    print(f"Rank-1 Divergence: {len(test_df) - r1_agree} / {len(test_df)} ({(len(test_df) - r1_agree)/len(test_df):.2%})")

    # Confusion matrix of Champion R1 vs Clean R1
    # 4 bins: Champ=NONE & Clean=NONE, Champ=NONE & Clean=Error, Champ=Error & Clean=NONE, Champ=Error & Clean=Error
    c_n_cl_n = sum(1 for a, b in zip(champ_r1, clean_r1) if a == "NONE" and b == "NONE")
    c_n_cl_e = sum(1 for a, b in zip(champ_r1, clean_r1) if a == "NONE" and b != "NONE")
    c_e_cl_n = sum(1 for a, b in zip(champ_r1, clean_r1) if a != "NONE" and b == "NONE")
    c_e_cl_e = sum(1 for a, b in zip(champ_r1, clean_r1) if a != "NONE" and b != "NONE")

    print("\n--- 2x2 Contingency Matrix on Rank 1 (NONE vs Misconception) ---")
    print(f"  Champ=NONE & Clean=NONE : {c_n_cl_n:4d} ({c_n_cl_n/len(test_df):.1%}) [Both Agree Student is Clean]")
    print(f"  Champ=NONE & Clean=Error: {c_n_cl_e:4d} ({c_n_cl_e/len(test_df):.1%}) [Clean Ensemble detected Error]")
    print(f"  Champ=Error & Clean=NONE: {c_e_cl_n:4d} ({c_e_cl_n/len(test_df):.1%}) [Clean Ensemble detected Clean]")
    print(f"  Champ=Error & Clean=Error:{c_e_cl_e:4d} ({c_e_cl_e/len(test_df):.1%}) [Both Agree Student made Error]")

    # Check non-NONE agreement when both agree it's an error
    err_both = [i for i in range(len(test_df)) if champ_r1[i] != "NONE" and clean_r1[i] != "NONE"]
    exact_code_agree = sum(1 for i in err_both if champ_r1[i] == clean_r1[i])
    print(f"\nWhen both predict an error, exact misconception code agreement: {exact_code_agree} / {len(err_both)} ({exact_code_agree/len(err_both):.1%})")


if __name__ == "__main__":
    main()

