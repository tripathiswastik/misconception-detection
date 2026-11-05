"""Multi-Model Cross-Validation and Benchmarking Suite.

Provides leakage-free 5-Fold Stratified Cross-Validation on NVIDIA CUDA GPU / CPU
with exact Top-3 Mean Average Precision (MAP@3) computation.
"""
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.metrics.map3 import calculate_map3 as compute_map3
from src.utils.device import clear_gpu_memory, get_device


@dataclass
class CrossValidationResult:
    """Out-of-fold evaluation metrics and predictions across cross-validation splits."""
    model_name: str
    fold_scores: List[float]
    mean_map3: float
    std_map3: float
    overall_oof_map3: float
    oof_predictions: List[List[str]]
    oof_probabilities: np.ndarray
    classes: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        folds_str = ", ".join([f"F{i+1}: {score:.4f}" for i, score in enumerate(self.fold_scores)])
        return (
            f"[{self.model_name}] OOF MAP@3: {self.overall_oof_map3:.4f} "
            f"(Mean: {self.mean_map3:.4f} +/- {self.std_map3:.4f}) | {folds_str}"
        )


class ModelCrossValidator:
    """Leakage-free Stratified Cross-Validator with rare-class protection."""

    def __init__(
        self,
        n_splits: int = 5,
        random_state: int = 42,
        taxonomy: Optional[TaxonomyManager] = None
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.taxonomy = taxonomy or TaxonomyManager()

    def create_stratified_folds(self, df: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate stratified indices pooling rare classes (< n_splits samples)."""
        y = df["root_code"].astype(str).values
        class_counts = pd.Series(y).value_counts()

        strat_key = y.copy()
        for label, count in class_counts.items():
            if count < self.n_splits:
                strat_key[strat_key == label] = "RARE_CLASS"

        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state
        )
        return list(skf.split(df, strat_key))

    def evaluate_model(
        self,
        model_factory: Callable[[], Any],
        train_df: pd.DataFrame,
        model_name: str = "Model",
        post_process_fn: Optional[Callable[[np.ndarray, List[str], pd.DataFrame], np.ndarray]] = None
    ) -> CrossValidationResult:
        """Run 5-Fold Stratified CV on a candidate model architecture.
        
        Args:
            model_factory: Callable returning a fresh, un-trained model instance.
            train_df: Complete training dataframe (1,500 rows).
            model_name: Identifier label for experiment logs.
            post_process_fn: Optional function (probabilities, classes, val_df) -> calibrated_probabilities.
        """
        folds = self.create_stratified_folds(train_df)
        n_samples = len(train_df)
        y_true = train_df["root_code"].astype(str).values

        all_classes = self.taxonomy.all_codes(include_none=True)
        oof_probabilities = np.zeros((n_samples, len(all_classes)), dtype=np.float32)
        oof_predictions: List[Optional[List[str]]] = [None] * n_samples
        fold_scores: List[float] = []

        print(f"\n[CrossValidator] Starting {self.n_splits}-Fold CV for '{model_name}' on device: {get_device(prefer_gpu=True)}")

        for fold_idx, (train_idx, val_idx) in enumerate(folds, start=1):
            fold_train = train_df.iloc[train_idx].reset_index(drop=True)
            fold_val = train_df.iloc[val_idx].reset_index(drop=True)

            model = model_factory()
            model.fit(fold_train)

            raw_val_probs = model.predict_proba(fold_val)
            model_classes = list(model.classes_)

            # Map model output probabilities to master taxonomy class index
            aligned_probs = np.zeros((len(val_idx), len(all_classes)), dtype=np.float32)
            for m_idx, class_name in enumerate(model_classes):
                if class_name in all_classes:
                    master_idx = all_classes.index(class_name)
                    aligned_probs[:, master_idx] = raw_val_probs[:, m_idx]

            # Normalize row sums
            row_sums = aligned_probs.sum(axis=1, keepdims=True)
            aligned_probs = np.divide(aligned_probs, row_sums, where=row_sums > 0)

            # Apply post-processing / domain masking if provided
            if post_process_fn is not None:
                aligned_probs = post_process_fn(aligned_probs, all_classes, fold_val)

            fold_preds_top3: List[List[str]] = []
            for i, global_idx in enumerate(val_idx):
                prob_row = aligned_probs[i]
                top_indices = np.argsort(prob_row)[::-1][:3]
                top_3 = [all_classes[idx] for idx in top_indices]
                oof_predictions[global_idx] = top_3
                oof_probabilities[global_idx] = prob_row
                fold_preds_top3.append(top_3)

            fold_y_true = [str(y_true[g_idx]) for g_idx in val_idx]
            fold_map3 = compute_map3(fold_y_true, fold_preds_top3)
            fold_scores.append(fold_map3)
            print(f"  Fold {fold_idx}/{self.n_splits} MAP@3: {fold_map3:.4f}")

            clear_gpu_memory()

        # Compute overall Out-Of-Fold (OOF) MAP@3 across all samples
        overall_oof_map3 = compute_map3(list(y_true), [p for p in oof_predictions if p is not None])
        mean_map3 = float(np.mean(fold_scores))
        std_map3 = float(np.std(fold_scores))

        result = CrossValidationResult(
            model_name=model_name,
            fold_scores=fold_scores,
            mean_map3=mean_map3,
            std_map3=std_map3,
            overall_oof_map3=overall_oof_map3,
            oof_predictions=[p for p in oof_predictions if p is not None],
            oof_probabilities=oof_probabilities,
            classes=all_classes
        )
        print(f"[CrossValidator] {result.summary()}\n")
        return result

    @staticmethod
    def evaluate_ensemble(
        results: Sequence[CrossValidationResult],
        weights: Optional[Sequence[float]] = None,
        y_true: Optional[Sequence[str]] = None,
        classes: Optional[Sequence[str]] = None
    ) -> Tuple[float, List[List[str]], np.ndarray]:
        """Compute ensemble OOF MAP@3 by weighted probability averaging."""
        n_models = len(results)
        if weights is None:
            w = np.ones(n_models) / n_models
        else:
            w = np.array(weights) / np.sum(weights)

        master_classes = classes or results[0].classes
        blended_probs = np.zeros_like(results[0].oof_probabilities)
        for i, res in enumerate(results):
            blended_probs += w[i] * res.oof_probabilities

        blended_preds: List[List[str]] = []
        for i in range(len(blended_probs)):
            top_indices = np.argsort(blended_probs[i])[::-1][:3]
            blended_preds.append([master_classes[idx] for idx in top_indices])

        if y_true is not None:
            score = compute_map3(list(y_true), blended_preds)
        else:
            score = 0.0

        return score, blended_preds, blended_probs
