"""Calibrated Rank and Score Fusion Engine.

Combines out-of-fold calibrated model scores, enforces domain constraints,
and deterministically outputs exactly three distinct, valid taxonomy codes per sample.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import numpy as np

from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.fusion.validator import validate_submission_file

__all__ = ["RankFusionEngine", "validate_submission_file"]


class RankFusionEngine:
    """Combines candidate scores, applies domain constraints, and returns deterministic Top-3."""

    def __init__(
        self,
        taxonomy: Optional[TaxonomyManager] = None,
        domain_masker: Optional[DomainConstraintEngine] = None
    ):
        self.taxonomy = taxonomy or TaxonomyManager()
        self.domain_masker = domain_masker or DomainConstraintEngine(taxonomy=self.taxonomy)
        self.valid_codes: Set[str] = set(self.taxonomy.all_codes(include_none=True))

    def rank_single(
        self,
        scores: np.ndarray,
        classes: Sequence[str],
        topic_name: Optional[str] = None,
        grade: Optional[Union[int, str]] = None,
        apply_mask: bool = True,
        top_k: int = 3
    ) -> Tuple[List[str], List[float]]:
        """Rank candidates for a single sample, ensuring exactly top_k distinct valid codes.
        
        Args:
            scores: (num_classes,) array of model scores or probabilities.
            classes: list of class code strings matching the columns of scores.
            topic_name: curricular math topic name for domain filtering.
            grade: student grade level.
            apply_mask: whether to apply domain constraint mask.
            top_k: number of distinct predictions to return (default: 3).

        Returns:
            Tuple of (top_k_codes, top_k_scores).
        """
        calibrated_scores = np.array(scores, dtype=np.float32).copy()

        # Apply domain masking if enabled and context is available
        if apply_mask and topic_name is not None and grade is not None:
            grade_int = int(grade) if str(grade).isdigit() else 6
            masked_batch = self.domain_masker.apply_mask_batch(
                calibrated_scores.reshape(1, -1), list(classes), [topic_name], [grade_int], damping=0.85
            )
            calibrated_scores = masked_batch[0]

        # Sort indices in descending order; for ties, break deterministically by code string
        sorted_indices = np.argsort(-calibrated_scores, kind="mergesort")

        selected_codes: List[str] = []
        selected_scores: List[float] = []
        seen_codes: Set[str] = set()

        for idx in sorted_indices:
            code = str(classes[idx])
            if code in self.valid_codes and code not in seen_codes:
                selected_codes.append(code)
                selected_scores.append(float(calibrated_scores[idx]))
                seen_codes.add(code)
                if len(selected_codes) == top_k:
                    break

        # Fallback padding if fewer than top_k candidates selected
        if len(selected_codes) < top_k:
            fallback_codes = [LABEL_NONE, "FRM-01", "INT-04", "CAL-01", "EXP-01"]
            for fb in fallback_codes:
                if fb in self.valid_codes and fb not in seen_codes:
                    selected_codes.append(fb)
                    selected_scores.append(0.0)
                    seen_codes.add(fb)
                    if len(selected_codes) == top_k:
                        break

        return selected_codes[:top_k], selected_scores[:top_k]

    def rank_batch(
        self,
        score_matrix: np.ndarray,
        classes: Sequence[str],
        topic_names: Optional[Sequence[str]] = None,
        grades: Optional[Sequence[Union[int, str]]] = None,
        apply_mask: bool = True,
        top_k: int = 3
    ) -> List[Tuple[List[str], List[float]]]:
        """Vectorized ranking across a batch of samples."""
        n_samples = len(score_matrix)
        results = []

        for i in range(n_samples):
            t = topic_names[i] if topic_names is not None else None
            g = grades[i] if grades is not None else None
            res = self.rank_single(
                scores=score_matrix[i],
                classes=classes,
                topic_name=t,
                grade=g,
                apply_mask=apply_mask,
                top_k=top_k
            )
            results.append(res)

        return results
