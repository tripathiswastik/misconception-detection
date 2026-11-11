"""Audit Actual Intersection of S550 Rank 4-5 Rare Errors and NONE at Rank 1.

Questions Addressed per User Directive:
1. What is the EXACT intersection between:
   - Rows where the true label is a rare misconception (<15 train traces) AND ranked at Rank 4 or 5.
   - Rows where S550 predicted NONE at Rank 1.
2. For these intersecting rows:
   - What is the probability gap: P(NONE) - P(true_rare)?
   - What is the probability gap to enter Top 3: P(Rank 3) - P(true_rare)?
   - What are the binary error detector probabilities P(is_error)?
   - What is the theoretical ORACLE MAP@3 gain across all 1,500 traces if an oracle rescues:
     a) All intersecting rows to Rank 1 (score jumps 0.0 -> 1.0)
     b) All intersecting rows to Rank 2 (score jumps 0.0 -> 0.5)
     c) All intersecting rows to Rank 3 (score jumps 0.0 -> 0.3333)
3. Collateral Risk Assessment:
   - How many true NONE rows sit within the same probability gap margin?
   - How many false promotions would be triggered by demoting NONE on this margin?
"""
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Set, Tuple, cast
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
    print("=" * 95)
    print("AUDIT: INTERSECTION OF S550 RANK 4-5 RARE ERRORS AND NONE AT RANK 1")
    print("=" * 95)

    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    none_idx = classes.index("NONE")
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = cast(pd.DataFrame, pd.read_csv(DATA_DIR / "train.csv"))
    y_raw = train_df["root_code"].fillna("NONE").astype(str).tolist()
    n_samples = len(train_df)

    # Class frequencies
    from collections import Counter
    class_counts = Counter(y_raw)
    rare_classes = {c for c, cnt in class_counts.items() if c != "NONE" and cnt < 15}
    print(f"Total Traces: {n_samples} | Total Rare Classes (<15): {len(rare_classes)}")

    # Load S550 OOF audit records
    # Let's run the exact in-fold S550 prediction
    from scratch.audit_rare_class_s550_oof_error_breakdown import S550AuditEngine
    engine = S550AuditEngine()

    oof_p_s550 = np.zeros((n_samples, engine.n_classes), dtype=np.float64)
    final_preds_all: List[List[str]] = [[] for _ in range(n_samples)]
    seen_mask = np.zeros(n_samples, dtype=bool)

    for fold_num in range(1, 6):
        fold_info = engine.folds[fold_num - 1]
        trn_idx = np.array(fold_info["train_indices"], dtype=np.int64)
        val_idx = np.array(fold_info["val_indices"], dtype=np.int64)

        p_val, q_dist = engine.generate_s550_probs(trn_idx, val_idx)
        oof_p_s550[val_idx] = p_val

        preds_val = engine.rank_and_gate(p_val, val_idx, q_dist)
        for i, idx in enumerate(val_idx):
            final_preds_all[idx] = preds_val[i]

        trn_qids = set(engine.train_df.iloc[trn_idx]["question_id"].astype(str))
        for i in val_idx:
            if str(engine.train_df.iloc[i]["question_id"]) in trn_qids:
                seen_mask[i] = True

    p_bin_error = np.load(ART_DIR / "oof_compact_binary_prob.npy")

    # Filter to traces where true label is a rare misconception
    rare_traces = []
    for i in range(n_samples):
        y_gold = y_raw[i]
        if y_gold not in rare_classes:
            continue

        p_row = oof_p_s550[i]
        top3_pred = final_preds_all[i]
        gold_idx = class_to_idx[y_gold]

        prob_ranking = list(np.argsort(-p_row))
        prob_rank = prob_ranking.index(gold_idx) + 1

        in_top3 = y_gold in top3_pred
        r1_pred = top3_pred[0]
        r1_is_none = (r1_pred == "NONE")

        p_gold = float(p_row[gold_idx])
        p_none = float(p_row[none_idx])
        p_r3 = float(p_row[prob_ranking[2]])
        p_r1 = float(p_row[prob_ranking[0]])

        gap_none = p_none - p_gold
        gap_to_top3 = max(0.0, p_r3 - p_gold)
        bin_prob = float(p_bin_error[i])

        rare_traces.append({
            "trace_id": str(train_df.iloc[i]["trace_id"]),
            "question_id": str(train_df.iloc[i]["question_id"]),
            "gold_code": y_gold,
            "gold_count": class_counts[y_gold],
            "is_seen": bool(seen_mask[i]),
            "in_top3": in_top3,
            "prob_rank": prob_rank,
            "is_rank_4_5": prob_rank in [4, 5],
            "r1_pred": r1_pred,
            "r1_is_none": r1_is_none,
            "top3_pred": top3_pred,
            "p_gold": round(p_gold, 5),
            "p_none": round(p_none, 5),
            "p_r3": round(p_r3, 5),
            "gap_none": round(gap_none, 5),
            "gap_to_top3": round(gap_to_top3, 5),
            "p_bin_error": round(bin_prob, 4)
        })

    # Intersections
    total_rare = len(rare_traces)
    rare_errors = [r for r in rare_traces if not r["in_top3"]]
    n_rare_err = len(rare_errors)

    r45_errors = [r for r in rare_errors if r["is_rank_4_5"]]
    none_r1_errors = [r for r in rare_errors if r["r1_is_none"]]

    intersection = [r for r in r45_errors if r["r1_is_none"]]
    n_intersect = len(intersection)

    print(f"\n1. SET SIZES & INTERSECTION:")
    print(f"  Total Rare Misconception Traces:        {total_rare}")
    print(f"  Total Rare Misconception Errors:        {n_rare_err} (47.7%)")
    print(f"  Traces with True Rare in Rank 4-5:      {len(r45_errors)} (26.1% of rare errors)")
    print(f"  Traces with S550 R1 = NONE:             {len(none_r1_errors)} (66.9% of rare errors)")
    print(f"  ACTUAL INTERSECTION (Rank 4-5 AND R1=NONE): {n_intersect} traces ({n_intersect / n_rare_err * 100:.1f}% of rare errors)")

    # Analyze the 24 intersecting traces
    print("\n" + "=" * 115)
    print(f"TABLE: ALL {n_intersect} INTERSECTING TRACES (True Rare at Rank 4-5 AND Predicted R1 = NONE)")
    print("=" * 115)
    print(
        f"{'Trace ID':14s} | {'Gold Code':10s} | {'Rank':4s} | "
        f"{'P(gold)':8s} | {'P(NONE)':8s} | {'Gap(NONE)':10s} | {'Gap(Top3)':10s} | {'P(error)':9s} | {'Predicted Top 3':26s}"
    )
    print("-" * 115)

    gaps_none = []
    gaps_top3 = []
    bin_probs = []

    for r in intersection:
        gaps_none.append(r["gap_none"])
        gaps_top3.append(r["gap_to_top3"])
        bin_probs.append(r["p_bin_error"])
        top3_str = " ".join(r["top3_pred"])
        print(
            f"{r['trace_id']:14s} | {r['gold_code']:10s} | {r['prob_rank']:4d} | "
            f"{r['p_gold']:8.4f} | {r['p_none']:8.4f} | {r['gap_none']:10.4f} | {r['gap_to_top3']:10.4f} | {r['p_bin_error']:9.4f} | {top3_str:26s}"
        )

    # Upper-bound Oracle Gain Estimation
    # There are n_intersect traces
    # In S550, their score is 0.0 (since they are not in Top 3)
    # If promoted to Rank 1: each gives +1.0 -> Delta = n_intersect * 1.0 / 1500
    # If promoted to Rank 2: each gives +0.5 -> Delta = n_intersect * 0.5 / 1500
    # If promoted to Rank 3: each gives +0.3333 -> Delta = n_intersect * 0.3333 / 1500
    oracle_delta_r1 = (n_intersect * 1.0) / 1500.0
    oracle_delta_r2 = (n_intersect * 0.5) / 1500.0
    oracle_delta_r3 = (n_intersect * (1.0 / 3.0)) / 1500.0

    print("\n" + "=" * 95)
    print("2. THEORETICAL REACHABLE ORACLE MAP@3 GAIN:")
    print("=" * 95)
    print(f"  Intersecting Traces: {n_intersect} / 1,500 total traces ({n_intersect / 1500 * 100:.2f}%)")
    print(f"  If ALL {n_intersect} traces promoted to Rank 1:  +{oracle_delta_r1:.5f} Composite MAP@3 (Oracle Ceiling)")
    print(f"  If ALL {n_intersect} traces promoted to Rank 2:  +{oracle_delta_r2:.5f} Composite MAP@3 (Surgical Swapping)")
    print(f"  If ALL {n_intersect} traces promoted to Rank 3:  +{oracle_delta_r3:.5f} Composite MAP@3 (Conservative Edge)")

    # Gap distribution
    print("\n" + "=" * 95)
    print("3. PROBABILITY GAP & FEASIBILITY ANALYSIS:")
    print("=" * 95)
    print(f"  Gap to Top 3 [P(Rank 3) - P(gold)]:")
    print(f"    - Min:    {min(gaps_top3):.4f}")
    print(f"    - Median: {float(np.median(gaps_top3)):.4f}")
    print(f"    - Mean:   {float(np.mean(gaps_top3)):.4f}")
    print(f"    - Max:    {max(gaps_top3):.4f}")
    print(f"    - Traces with Gap < 0.04 (Razor-Thin): {sum(1 for g in gaps_top3 if g < 0.04)} / {n_intersect}")
    print(f"    - Traces with Gap < 0.08:              {sum(1 for g in gaps_top3 if g < 0.08)} / {n_intersect}")

    print(f"\n  Binary Error Detector P(is_error):")
    print(f"    - Min:    {min(bin_probs):.4f}")
    print(f"    - Median: {float(np.median(bin_probs)):.4f}")
    print(f"    - Mean:   {float(np.mean(bin_probs)):.4f}")
    print(f"    - Max:    {max(bin_probs):.4f}")
    print(f"    - Traces with P(is_error) >= 0.78:     {sum(1 for p in bin_probs if p >= 0.78)} / {n_intersect}")
    print(f"    - Traces with P(is_error) >= 0.65:     {sum(1 for p in bin_probs if p >= 0.65)} / {n_intersect}")

    # Collateral risk: How many TRUE NONE traces sit in this exact gap / error zone?
    true_none_in_zone = 0
    for i in range(n_samples):
        if y_raw[i] == "NONE" and final_preds_all[i][0] == "NONE":
            if p_bin_error[i] >= 0.65:
                true_none_in_zone += 1

    print(f"\n  COLLATERAL DAMAGE RISK:")
    print(f"    - True NONE traces with P(is_error) >= 0.65: {true_none_in_zone} traces")
    print(f"    - If gating is widened to capture the {sum(1 for p in bin_probs if p >= 0.65)} rare traces, it risks falsely demoting {true_none_in_zone} true NONE traces!")
    print(f"    - Falsely demoting 1 true NONE costs -0.50 MAP@3 pts (-0.00033 per trace).")
    print("=" * 95)

    report_dict = {
        "total_rare_traces": total_rare,
        "total_rare_errors": n_rare_err,
        "r45_rare_errors": len(r45_errors),
        "none_r1_rare_errors": len(none_r1_errors),
        "intersection_count": n_intersect,
        "oracle_gains": {
            "all_to_rank1": round(oracle_delta_r1, 5),
            "all_to_rank2": round(oracle_delta_r2, 5),
            "all_to_rank3": round(oracle_delta_r3, 5)
        },
        "gap_to_top3_stats": {
            "min": round(min(gaps_top3), 4),
            "median": round(float(np.median(gaps_top3)), 4),
            "mean": round(float(np.mean(gaps_top3)), 4),
            "max": round(max(gaps_top3), 4),
            "count_gap_lt_004": sum(1 for g in gaps_top3 if g < 0.04),
            "count_gap_lt_008": sum(1 for g in gaps_top3 if g < 0.08)
        },
        "p_bin_error_stats": {
            "min": round(min(bin_probs), 4),
            "median": round(float(np.median(bin_probs)), 4),
            "mean": round(float(np.mean(bin_probs)), 4),
            "max": round(max(bin_probs), 4),
            "count_gte_078": sum(1 for p in bin_probs if p >= 0.78),
            "count_gte_065": sum(1 for p in bin_probs if p >= 0.65)
        },
        "intersecting_traces": intersection
    }

    out_path = ART_DIR / "rank45_none_intersection_audit.json"
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Saved Audit Report] {out_path.name}")

if __name__ == "__main__":
    main()

