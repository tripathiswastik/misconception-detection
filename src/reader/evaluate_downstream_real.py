"""Evaluate Real Downstream SOTA MAP@3 Impact of the Ablated Verifier on Fresh 150 Holdout."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reader.comparator_ablation import run_ablation_study
from src.reader.verification_gate import simulate_downstream_gating_map3

def main():
    study = run_ablation_study()["Variant E — Conservative combined"]
    decisions = study["decisions"]
    y_pred = [d["decision"] for d in decisions]

    with open(PROJECT_ROOT / "artifacts" / "taxonomy_classes.json") as f:
        classes = json.load(f)

    train_df = cast(pd.DataFrame, pd.read_csv(PROJECT_ROOT / "data" / "train.csv"))
    # Map trace IDs to indices in train.csv
    trace_to_idx = {str(tid): i for i, tid in enumerate(train_df["trace_id"])}
    indices_150 = [trace_to_idx[str(d["trace_id"])] for d in decisions]

    # Load honest OOF probabilities for our SOTA model
    oof_prior = np.load(PROJECT_ROOT / "artifacts" / "oof_prior_target_encoding.npy")
    oof_convnext = np.load(PROJECT_ROOT / "artifacts" / "oof_convnext_vision.npy")
    oof_text = np.load(PROJECT_ROOT / "artifacts" / "oof_text.npy")

    oof_sota = 0.70 * oof_prior + 0.20 * oof_convnext + 0.10 * oof_text
    probs_150 = oof_sota[indices_150]
    labels_150 = train_df.iloc[indices_150]["root_code"].tolist()

    impact = simulate_downstream_gating_map3(
        ground_truth_codes=labels_150,
        base_probabilities=probs_150,
        classes=classes,
        verifier_predictions=y_pred,
        none_class_code="NONE",
        boost_weight=0.90,
        suppress_weight=0.01,
        run_bootstrap=True
    )

    print("\n" + "=" * 65)
    print("REAL DOWNSTREAM SOTA IMPACT: HARD GATING (suppress=0.01, boost=1.9)")
    print("=" * 65)
    print(f"Baseline SOTA MAP@3: {impact['baseline_map3']:.4f}")
    print(f"Gated SOTA MAP@3:    {impact['gated_map3']:.4f}")
    print(f"Delta MAP@3:         {impact['delta_map3']:+.4f}")
    print(f"Improved Traces:     {impact['n_improved']}")
    print(f"Degraded Traces:     {impact['n_degraded']}")
    print(f"Unchanged Traces:    {impact['n_unchanged']}")
    print(f"Abstained Traces:    {impact['n_abstained']} (safety fallback to baseline)")
    if "bootstrap" in impact:
        b = impact["bootstrap"]
        print(f"Bootstrap 95% CI:    [{b['ci_95'][0]:+.4f}, {b['ci_95'][1]:+.4f}]")
        print(f"Bootstrap p-value:   {b['p_value']:.4f}")
        print(f"Stat. Significant:   {b['is_significant_improvement']}")
    print("=" * 65)

    # Soft Bayesian Calibrated Gating (matching verifier 33% false rejection rate)
    soft_impact = simulate_downstream_gating_map3(
        ground_truth_codes=labels_150,
        base_probabilities=probs_150,
        classes=classes,
        verifier_predictions=y_pred,
        none_class_code="NONE",
        boost_weight=0.50, # p[none] *= 1.50
        suppress_weight=0.50, # p[none] *= 0.50
        run_bootstrap=True
    )
    print("\n" + "=" * 65)
    print("REAL DOWNSTREAM SOTA IMPACT: SOFT BAYESIAN (suppress=0.50, boost=1.50)")
    print("=" * 65)
    print(f"Baseline SOTA MAP@3: {soft_impact['baseline_map3']:.4f}")
    print(f"Calibrated MAP@3:    {soft_impact['gated_map3']:.4f}")
    print(f"Delta MAP@3:         {soft_impact['delta_map3']:+.4f}")
    print(f"Improved Traces:     {soft_impact['n_improved']}")
    print(f"Degraded Traces:     {soft_impact['n_degraded']}")
    print(f"Unchanged Traces:    {soft_impact['n_unchanged']}")
    if "bootstrap" in soft_impact:
        sb = soft_impact["bootstrap"]
        print(f"Bootstrap 95% CI:    [{sb['ci_95'][0]:+.4f}, {sb['ci_95'][1]:+.4f}]")
        print(f"Bootstrap p-value:   {sb['p_value']:.4f}")
        print(f"Stat. Significant:   {sb['is_significant_improvement']}")
    print("=" * 65)

    out_file = PROJECT_ROOT / "artifacts" / "dataset_150_fresh" / "real_downstream_impact.json"
    with open(out_file, "w") as f:
        json.dump(impact, f, indent=2)
    print(f"Saved downstream audit report to: {out_file}")

    # Forensic breakdown of degraded traces
    from src.reader.verification_gate import compute_map3_row
    none_idx = classes.index("NONE")
    degraded_details = []
    for i in range(150):
        pred = y_pred[i]
        base_top3 = [classes[idx] for idx in np.argsort(-probs_150[i])[:3]]
        base_score = compute_map3_row(labels_150[i], base_top3)

        g_prob = probs_150[i].copy()
        if pred is True:
            g_prob[none_idx] = 1.0 + 0.90 * np.max(g_prob)
        elif pred is False:
            g_prob[none_idx] = g_prob[none_idx] * 0.01

        gated_top3 = [classes[idx] for idx in np.argsort(-g_prob)[:3]]
        gated_score = compute_map3_row(labels_150[i], gated_top3)

        if gated_score < base_score:
            degraded_details.append({
                "trace_id": decisions[i]["trace_id"],
                "gold": decisions[i]["gold_label"],
                "true_label": labels_150[i],
                "pred": pred,
                "base_top3": base_top3,
                "gated_top3": gated_top3,
                "base_score": base_score,
                "gated_score": gated_score
            })

    print(f"\nBreakdown of {len(degraded_details)} degraded traces:")
    for d in degraded_details[:10]:
        print(f"  {d['trace_id']} | Gold: {d['gold']} | True: {d['true_label']} | Pred: {d['pred']} | Base: {d['base_top3']} ({d['base_score']:.2f}) -> Gated: {d['gated_top3']} ({d['gated_score']:.2f})")

if __name__ == "__main__":
    main()
