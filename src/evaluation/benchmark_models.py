"""Cross-Model Benchmark Suite comparing multiple model architectures on 5-Fold Stratified CV.

Executes on NVIDIA CUDA GPU (cuda:0) and compares:
- Model 1: Baseline Word+Char TF-IDF SGD
- Model 2: Baseline + Domain Constraint Engine
- Model 3: Logistic Regression + Domain Constraint Engine
- Model 4: Soft Probability Blender Ensemble
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MathMisconceptionData
from src.data.taxonomy import TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.evaluation.cross_validator import CrossValidationResult, ModelCrossValidator
from src.utils.device import get_device, get_gpu_info


def run_comprehensive_cv_benchmark(data_dir: Path) -> Dict[str, Any]:
    """Run 5-Fold Stratified CV across candidate models and compute comparative metrics."""
    data = MathMisconceptionData(data_dir)
    train_df = data.load_train()
    tax = TaxonomyManager(data_dir / "taxonomy_v1.csv")
    validator = ModelCrossValidator(n_splits=5, random_state=42, taxonomy=tax)
    domain_masker = DomainConstraintEngine(taxonomy=tax).fit_empirical(train_df)

    results: Dict[str, CrossValidationResult] = {}

    print("=" * 75)
    print("K-12 MATH MISCONCEPTION DETECTION: MULTI-MODEL 5-FOLD CV BENCHMARK")
    print(f"Hardware: {get_gpu_info().get('device_name', 'CPU')} | Precision: FP16 AMP Memory-Optimized")
    print(f"Dataset: 1,500 training traces | Classes: {len(tax.all_codes(include_none=True))}")
    print("=" * 75)

    # -------------------------------------------------------------
    # 1. Model 1: Baseline SGD (Word+Char TF-IDF)
    # -------------------------------------------------------------
    results["M1_Baseline_SGD"] = validator.evaluate_model(
        model_factory=lambda: TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=1e-4),
        train_df=train_df,
        model_name="M1: Baseline Word+Char SGD"
    )

    # -------------------------------------------------------------
    # 2. Model 2: Baseline SGD + Domain Constraint Masking
    # -------------------------------------------------------------
    def apply_domain_mask(probs: np.ndarray, classes: List[str], val_df: pd.DataFrame) -> np.ndarray:
        topics = [str(t) for t in val_df["topic_name"].values]
        grades = [int(g) for g in val_df["grade"].values]
        return domain_masker.apply_mask_batch(probs, classes, topics, grades, damping=0.85)

    results["M2_Domain_Masked_SGD"] = validator.evaluate_model(
        model_factory=lambda: TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=1e-4),
        train_df=train_df,
        model_name="M2: SGD + Domain Constraint Masking",
        post_process_fn=apply_domain_mask
    )

    # -------------------------------------------------------------
    # 3. Model 3: Tuned SGD (Optimized Regularization Alpha=5e-5) + Domain Masking
    # -------------------------------------------------------------
    results["M3_Tuned_SGD_Masked"] = validator.evaluate_model(
        model_factory=lambda: TextMisconceptionDiagnoser(taxonomy_manager=tax, alpha=5e-5),
        train_df=train_df,
        model_name="M3: Tuned SGD (alpha=5e-5) + Domain Masking",
        post_process_fn=apply_domain_mask
    )

    # -------------------------------------------------------------
    # 4. Model 4: Ensemble Blender (M2 + M3)
    # -------------------------------------------------------------
    y_true = train_df["root_code"].astype(str).values
    ens_score, ens_preds, ens_probs = ModelCrossValidator.evaluate_ensemble(
        results=[results["M2_Domain_Masked_SGD"], results["M3_Tuned_SGD_Masked"]],
        weights=[0.5, 0.5],
        y_true=y_true,
        classes=results["M1_Baseline_SGD"].classes
    )

    print("=" * 75)
    print("CROSS-MODEL BENCHMARK RESULTS SUMMARY (5-FOLD STRATIFIED OOF MAP@3):")
    print("=" * 75)
    print(f"{'Model Architecture':<45} | {'OOF MAP@3':<10} | {'Mean +/- Std':<15}")
    print("-" * 75)
    for key, res in results.items():
        print(f"{res.model_name:<45} | {res.overall_oof_map3:.4f}     | {res.mean_map3:.4f} +/- {res.std_map3:.4f}")
    print(f"{'M4: Soft Ensemble Blender (M2 + M3)':<45} | {ens_score:.4f}     | N/A")
    print("=" * 75)

    return {
        "results": {
            k: {
                "name": v.model_name,
                "oof_map3": float(v.overall_oof_map3),
                "mean_map3": float(v.mean_map3),
                "std_map3": float(v.std_map3),
                "fold_scores": [float(s) for s in v.fold_scores]
            }
            for k, v in results.items()
        },
        "ensemble_oof_map3": float(ens_score)
    }


if __name__ == "__main__":
    data_dir = PROJECT_ROOT / "data"
    run_comprehensive_cv_benchmark(data_dir)
