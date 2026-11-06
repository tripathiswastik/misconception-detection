"""Standard Benchmark & Diagnostic Runner for Misconception Detection.

Executes leakage-free evaluation across canonical 5-fold CV and Question-Grouped CV.
Generates comprehensive diagnostic reports distinguishing Stage A (base model representation)
from Stage B (reranking and calibration) headroom.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.taxonomy import TaxonomyManager
from src.evaluation.cross_validator import CrossValidationResult, ModelCrossValidator
from src.evaluation.leak_free_cv import fast_fold_domain_mask

DATA_DIR = PROJECT_ROOT / "data"
ART_DIR = PROJECT_ROOT / "artifacts"


def load_benchmark_components(
    tax: TaxonomyManager,
    train_df: pd.DataFrame,
    classes: List[str],
    splits: List[Dict]
) -> Dict[str, np.ndarray]:
    """Load and prepare all leak-free and champion probability matrices."""
    # 1. Individual clean components
    p_prior = np.load(ART_DIR / "best_adaptive_prior_oof.npy")
    p_text_honest = np.load(ART_DIR / "honest_oof_text.npy")
    p_text_sem = np.load(ART_DIR / "oof_semantic_text.npy")
    p_cn = np.load(ART_DIR / "oof_convnext_vision.npy")
    p_swin = np.load(ART_DIR / "oof_swin_vision.npy")

    # MobileNet with strict in-fold domain masking
    mn_raw = np.load(ART_DIR / "oof_finetuned_vision.npy")
    p_mn = np.zeros_like(mn_raw)
    for f in splits:
        tr = f["train_indices"]
        va = f["val_indices"]
        p_mn[va] = fast_fold_domain_mask(
            mn_raw[va], train_df.iloc[tr], train_df.iloc[va], classes, tax=tax, damping=0.85
        )

    # 2. Vision Ensemble (Foundation Vision)
    p_vision_ens = (p_cn + p_swin + p_mn) / 3.0

    # 3. Clean Locked Ensemble Blend (Nested CV weights from leak_free_cv)
    # Weights: Prior 60.59%, ConvNeXt 4.48%, Swin 7.03%, MobileNet 7.09%, Honest SGD 11.89%, Semantic Text 8.92%
    p_clean_blend = (
        0.6059 * p_prior +
        0.0448 * p_cn +
        0.0703 * p_swin +
        0.0709 * p_mn +
        0.1189 * p_text_honest +
        0.0892 * p_text_sem
    )

    # 4. Active Public LB Champion (SNR85 / Rank-1 Locked pipeline applied to OOF)
    oof_m15 = np.load(ART_DIR / "oof_m15_balanced_fusion.npy")
    p_oof_base = (
        0.95 * oof_m15 +
        0.02 * np.load(ART_DIR / "oof_arcface_dualhead_448_soft.npy") +
        0.03 * np.load(ART_DIR / "oof_arcface_composite_448.npy")
    )

    # Frequency calibration multiplier on base OOF
    from collections import Counter
    y_train = train_df["root_code"].fillna("NONE").values
    top3_base = np.argsort(-p_oof_base.astype(np.float64), axis=1)[:, :3]
    pred_counts = Counter(classes[idx] for row in top3_base for idx in row)
    true_counts = Counter(y_train)
    mult = np.ones(len(classes), dtype=np.float64)
    none_idx = classes.index("NONE")
    for ci, c in enumerate(classes):
        if ci != none_idx:
            pc = max(pred_counts.get(c, 1), 1)
            tc = max(true_counts.get(c, 1), 1)
            mult[ci] = (tc / pc) ** 0.05
    mult[none_idx] = 0.9840
    p_champ_cal = p_oof_base * mult

    return {
        "1_Baseline_Text_Honest": p_text_honest,
        "2_Question_Prior_Adaptive": p_prior,
        "3_Clean_Vision_Ensemble": p_vision_ens,
        "4_Semantic_Text_384d": p_text_sem,
        "5_Locked_Clean_Blend": p_clean_blend,
        "6_Public_LB_Champion_OOF": p_champ_cal,
    }


def run_benchmark(
    save_report_path: Optional[Path] = None,
    eval_grouped: bool = True
) -> Dict[str, Dict]:
    """Execute standard benchmark across all core pipelines."""
    tax = TaxonomyManager(DATA_DIR / "taxonomy_v1.csv")
    classes = tax.all_codes(include_none=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    splits = json.loads((DATA_DIR / "splits.json").read_text(encoding="utf-8"))["folds"]

    components = load_benchmark_components(tax, train_df, classes, splits)

    cv_stratified = ModelCrossValidator(
        n_splits=5,
        taxonomy=tax,
        use_canonical_splits=True
    )

    cv_grouped = ModelCrossValidator(
        n_splits=5,
        taxonomy=tax,
        grouped_by="question_id"
    ) if eval_grouped else None

    results: Dict[str, Dict] = {}

    print("=" * 80)
    print("MISCONCEPTION DETECTION: CANONICAL BENCHMARK & DIAGNOSTIC AUDIT")
    print("=" * 80)
    print(f"Total Samples: {len(train_df)} | Total Classes: {len(classes)}")
    print(f"Canonical Splits: 5 Folds (data/splits.json)")
    if eval_grouped:
        print(f"Guardrail: 5-Fold Question-Grouped CV (GroupKFold on question_id)")
    print("=" * 80)

    for name, probs in components.items():
        print(f"\n[Evaluating {name}]...")
        strat_res = cv_stratified.evaluate_probabilities(probs, train_df, model_name=name, classes=classes)
        print(strat_res.diagnostic_report())

        res_dict = {
            "stratified_oof_map3": strat_res.overall_oof_map3,
            "stratified_mean": strat_res.mean_map3,
            "stratified_std": strat_res.std_map3,
            "stratified_folds": strat_res.fold_scores,
            "rank1_pct": strat_res.diagnostics.rank1_pct if strat_res.diagnostics else 0.0,
            "rank2_pct": strat_res.diagnostics.rank2_pct if strat_res.diagnostics else 0.0,
            "rank3_pct": strat_res.diagnostics.rank3_pct if strat_res.diagnostics else 0.0,
            "missed_pct": strat_res.diagnostics.missed_pct if strat_res.diagnostics else 0.0,
            "none_map3": strat_res.diagnostics.none_map3 if strat_res.diagnostics else 0.0,
            "error_map3": strat_res.diagnostics.error_map3 if strat_res.diagnostics else 0.0,
            "error_rank1_is_none": strat_res.diagnostics.error_rank1_is_none if strat_res.diagnostics else 0,
            "stage_b_headroom": (
                (strat_res.diagnostics.error_rank2 * 0.5 + strat_res.diagnostics.error_rank3 * 0.6667) / 1500.0
                if strat_res.diagnostics else 0.0
            ),
        }

        if cv_grouped is not None:
            group_res = cv_grouped.evaluate_probabilities(probs, train_df, model_name=f"{name}_Grouped", classes=classes)
            print(f"  --> Question-Grouped Guardrail OOF MAP@3: {group_res.overall_oof_map3:.5f} (Mean: {group_res.mean_map3:.5f} +/- {group_res.std_map3:.5f})")
            res_dict["grouped_oof_map3"] = group_res.overall_oof_map3
            res_dict["grouped_mean"] = group_res.mean_map3
            res_dict["grouped_std"] = group_res.std_map3

        results[name] = res_dict

    # Summary Leaderboard
    print("\n" + "=" * 80)
    print("OFFICIAL BENCHMARK SUMMARY TABLE")
    print("=" * 80)
    header = f"{'Pipeline Name':<30} | {'Strat OOF':<10} | {'Grouped OOF':<12} | {'Error MAP@3':<11} | {'NONE Saturation':<15} | {'Stage B Gain'}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        grp_str = f"{r.get('grouped_oof_map3', 0.0):.5f}" if "grouped_oof_map3" in r else "N/A"
        print(
            f"{name:<30} | {r['stratified_oof_map3']:<10.5f} | {grp_str:<12} | {r['error_map3']:<11.5f} | "
            f"{r['error_rank1_is_none']:>4d} / 994 ({r['error_rank1_is_none']/994*100:>4.1f}%) | "
            f"+{r['stage_b_headroom']:.4f}"
        )
    print("=" * 80)

    if save_report_path:
        save_report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[Saved JSON Report to {save_report_path}]")

    return results


def main():
    parser = argparse.ArgumentParser(description="Standard Benchmark & Diagnostic Runner")
    parser.add_argument("--save-report", type=str, default=str(ART_DIR / "standard_benchmark_report.json"),
                        help="Path to save output JSON report")
    parser.add_argument("--no-grouped", action="store_true", help="Skip Question-Grouped CV evaluation")
    args = parser.parse_args()

    report_path = Path(args.save_report) if args.save_report else None
    run_benchmark(save_report_path=report_path, eval_grouped=not args.no_grouped)


if __name__ == "__main__":
    main()

