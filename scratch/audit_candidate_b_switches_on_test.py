"""Inspect Candidate B switches on the test set 21 Rank-1 diverging rows."""
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Dict, List, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path("d:/ag-ai/misconception-detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"
SUB_DIR = PROJECT_ROOT / "submissions"

from src.data.taxonomy import TaxonomyManager

def main():
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))

    # Training question label distributions
    q_label_dist: Dict[str, Counter] = {
        q: Counter(grp["root_code"].fillna("NONE").values)
        for q, grp in train_df.groupby("question_id")
    }

    # Load test submissions for S550 and S554
    s550_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota550_precision_gated_submission.csv"))
    s554_sub = cast(pd.DataFrame, pd.read_csv(SUB_DIR / "blend_sota554_arcface_v2_precision_gated_submission.csv"))

    # Load test binary error prob
    test_bin_v2 = np.load(ART_DIR / "test_compact_binary_prob_v2.npy")

    # Load test probability matrices
    # S550 test and S554 test matrices
    # Let's inspect available test files or compute margins if test matrices are present
    # Let's check which test matrices we have:
    test_m15 = np.load(ART_DIR / "test_m15_balanced_fusion.npy")
    test_v2 = np.load(ART_DIR / "test_arcface_pseudo_3500_v2.npy")
    test_dual = np.load(ART_DIR / "test_arcface_dualhead_448_soft.npy")
    test_comp = np.load(ART_DIR / "test_arcface_composite_448.npy")
    test_v1 = np.load(ART_DIR / "test_arcface_pseudo_3500.npy")

    print(f"Test shapes: test_m15={test_m15.shape}, test_v2={test_v2.shape}")

    # Rank 1 predictions
    r1_550 = [row.split()[0] for row in s550_sub["root_code"]]
    r1_554 = [row.split()[0] for row in s554_sub["root_code"]]
    r1_diff = [i for i in range(len(test_df)) if r1_550[i] != r1_554[i]]
    print(f"Rank 1 diverging rows: {len(r1_diff)}")

    # Check Candidate B conditions on each of the 21 diverging rows:
    # Rule parameters from report:
    # tau = 0.78 (or 0.75-0.80)
    # supp_min = 1
    # none_rate_max = 0.70
    # m_diff_min = -0.02
    tau = 0.78
    supp_min = 1
    none_rate_max = 0.70

    switches = []
    reasons = []

    for idx in r1_diff:
        qid = test_df.iloc[idx]["question_id"]
        q_counts = q_label_dist.get(qid, Counter())
        tot_q = sum(q_counts.values())
        q_none_rate = (q_counts.get("NONE", 0) / tot_q) if tot_q > 0 else 0.0

        p550_r1 = r1_550[idx]
        p554_r1 = r1_554[idx]
        supp_554 = q_counts.get(p554_r1, 0)
        b2 = float(test_bin_v2[idx])

        is_550_none = (p550_r1 == "NONE")
        is_554_none = (p554_r1 == "NONE")

        # Check conditions
        cond_none_flip = (is_550_none and not is_554_none)
        cond_b2 = (b2 >= tau)
        cond_supp = (supp_554 >= supp_min)
        cond_qnone = (q_none_rate <= none_rate_max)

        switch = cond_none_flip and cond_b2 and cond_supp and cond_qnone
        if switch:
            switches.append(idx)

        print(f"Row {idx:4d} | QID: {qid} | S550: {s550_sub.iloc[idx]['root_code']} | S554: {s554_sub.iloc[idx]['root_code']}")
        print(f"       b2={b2:.3f} (>= {tau}: {cond_b2}), supp_554={supp_554} (>= {supp_min}: {cond_supp}), q_none_rate={q_none_rate:.2f} (<= {none_rate_max}: {cond_qnone}) -> Switch={switch}")

    print(f"\nTotal test switches to S554 under Candidate B: {len(switches)} / {len(r1_diff)}")
    print(f"Switched row indices: {switches}")

if __name__ == "__main__":
    main()

