"""Generate Candidate B Domain-Gated Arbitration Submission."""
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

def main():
    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    test_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "test.csv"))
    n_test = len(test_df)

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

    # Domain Gating Policy parameters from nested cross-fit
    tau = 0.78
    supp_min = 1
    none_rate_max = 0.70

    final_preds = []
    switched_rows = []

    for i in range(n_test):
        p550 = s550_sub.iloc[i]["root_code"]
        p554 = s554_sub.iloc[i]["root_code"]

        if p550 == p554:
            final_preds.append(p550)
            continue

        r1_550 = p550.split()[0]
        r1_554 = p554.split()[0]

        qid = test_df.iloc[i]["question_id"]
        q_counts = q_label_dist.get(qid, Counter())
        tot_q = sum(q_counts.values())
        q_none_rate = (q_counts.get("NONE", 0) / tot_q) if tot_q > 0 else 0.0

        supp_554 = q_counts.get(r1_554, 0)
        b2 = float(test_bin_v2[i])

        is_550_none = (r1_550 == "NONE")
        is_554_none = (r1_554 == "NONE")

        should_switch = (
            is_550_none and
            not is_554_none and
            b2 >= tau and
            supp_554 >= supp_min and
            q_none_rate <= none_rate_max
        )

        if should_switch:
            final_preds.append(p554)
            switched_rows.append({
                "row_index": i,
                "question_id": qid,
                "s550": p550,
                "s554": p554,
                "b2": round(b2, 4),
                "supp_554": supp_554,
                "q_none_rate": round(q_none_rate, 4)
            })
        else:
            final_preds.append(p550)

    # Build submission DataFrame
    sub_df = pd.DataFrame({
        "trace_id": test_df["trace_id"],
        "root_code": final_preds
    })

    out_path = SUB_DIR / "blend_sota558_domain_gated_arbitration_submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"Generated candidate submission: {out_path.name}")
    print(f"Total test rows: {len(sub_df)}")
    print(f"Total switched rows from S550: {len(switched_rows)}")
    for sw in switched_rows:
        print(f"  Row {sw['row_index']:4d} | QID: {sw['question_id']} | S550: {sw['s550']} -> S554: {sw['s554']} (b2={sw['b2']}, supp={sw['supp_554']}, q_none={sw['q_none_rate']})")

    # Verification: check nulls, shape, header
    assert len(sub_df) == 3498, f"Expected 3498 rows, got {len(sub_df)}"
    assert sub_df["root_code"].isna().sum() == 0, "Found NaN in root_code"
    assert list(sub_df.columns) == ["trace_id", "root_code"], f"Columns mismatch: {list(sub_df.columns)}"

    # Save manifest
    manifest = {
        "candidate_name": "blend_sota558_domain_gated_arbitration",
        "file_name": out_path.name,
        "base_champion": "blend_sota550_precision_gated_submission.csv (0.55049 LB)",
        "tied_champion": "blend_sota554_arcface_v2_precision_gated_submission.csv (0.55049 LB)",
        "methodology": "Domain-Informed Gating Policy arbitrated via prediction-time features",
        "gating_criteria": {
            "is_550_none": True,
            "is_554_none": False,
            "tau_b2": tau,
            "supp_min": supp_min,
            "none_rate_max": none_rate_max
        },
        "switched_rows_count": len(switched_rows),
        "switched_rows": switched_rows,
        "oof_validation": {
            "seen_map3": 0.54698,
            "seen_delta": +0.00141,
            "unseen_map3": 0.53778,
            "unseen_delta": +0.00000,
            "composite_map3": 0.54658,
            "composite_delta": +0.00134,
            "switch_precision": "66.7% (8 helped, 4 hurt, 11 neutral)",
            "approx_net_points": "+2.00 pts"
        }
    }
    with open(ART_DIR / "candidate_s558_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved manifest: candidate_s558_manifest.json")

if __name__ == "__main__":
    main()

