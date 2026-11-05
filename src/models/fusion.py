"""Fold-Safe Multimodal Late Fusion Engine.

Calibrates and fuses out-of-fold predictions from text and direct vision models
using competition MAP@3 evaluation with strict leakage controls.
"""
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.metrics.map3 import calculate_map3


class MultimodalLateFusion:
    """Fuses text and vision probability distributions via out-of-fold calibration."""

    @staticmethod
    def align_predictions(
        source_probs: np.ndarray,
        source_classes: Sequence[str],
        target_classes: Sequence[str]
    ) -> np.ndarray:
        """Re-align class probability columns to target master taxonomy ordering."""
        n_samples = len(source_probs)
        aligned = np.zeros((n_samples, len(target_classes)), dtype=np.float32)
        source_map = {c: idx for idx, c in enumerate(source_classes)}

        for t_idx, code in enumerate(target_classes):
            if code in source_map:
                s_idx = source_map[code]
                aligned[:, t_idx] = source_probs[:, s_idx]

        # Row-normalize
        row_sums = aligned.sum(axis=1, keepdims=True)
        return np.divide(aligned, row_sums, where=row_sums > 0)

    @classmethod
    def tune_fusion_weight(
        cls,
        y_true: Sequence[str],
        text_scores: np.ndarray,
        vision_scores: np.ndarray,
        classes: Sequence[str],
        candidate_weights: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
    ) -> Dict[str, float]:
        """Find optimal vision weight on out-of-fold predictions without leakage."""
        if text_scores.shape != vision_scores.shape:
            raise ValueError(f"Shape mismatch: text {text_scores.shape} vs vision {vision_scores.shape}")

        best_score = -1.0
        best_weight = 0.0

        for w_vis in candidate_weights:
            fused = (1.0 - w_vis) * text_scores + w_vis * vision_scores
            top_indices = np.argsort(-fused, axis=1, kind="mergesort")[:, :3]
            preds = [[classes[idx] for idx in row] for row in top_indices]

            score = calculate_map3(list(y_true), preds)
            if score > best_score:
                best_score = score
                best_weight = float(w_vis)

        return {
            "best_oof_map3": best_score,
            "optimal_vision_weight": best_weight,
            "baseline_text_map3": float(calculate_map3(
                list(y_true),
                [[classes[idx] for idx in row] for row in np.argsort(-text_scores, axis=1, kind="mergesort")[:, :3]]
            ))
        }

    @classmethod
    def apply_fusion(
        cls,
        text_scores: np.ndarray,
        vision_scores: np.ndarray,
        vision_weight: float = 0.15
    ) -> np.ndarray:
        """Combine text and vision scores using validated weight."""
        return (1.0 - vision_weight) * text_scores + vision_weight * vision_scores
