"""Mean Average Precision at 3 (MAP@3) Metric Implementation.

In the CoLearn/Kaggle Misconception Detection competition:
- Ground truth is a single root misconception code or 'NONE'.
- Predictions are ranked lists of candidates (up to 3).
- If the true label appears at rank 1: score = 1.0
- If the true label appears at rank 2: score = 0.5 (1/2)
- If the true label appears at rank 3: score = 0.3333333333 (1/3)
- Otherwise: score = 0.0
"""
from typing import List, Union, Sequence
import numpy as np


def calculate_apk(actual: str, predicted: Sequence[str], k: int = 3) -> float:
    """Calculate Average Precision at K for a single query with a single ground truth.
    
    Args:
        actual: Ground truth string (e.g. 'FRM-01' or 'NONE').
        predicted: Sequence of predicted strings in descending order of confidence.
        k: Maximum rank depth (default: 3).
        
    Returns:
        float: Precision at the rank of the true label if <= k, else 0.0.
    """
    if not actual:
        return 0.0

    actual_str = actual.strip()
    
    seen = set()
    unique_preds = []
    for p in predicted:
        p_clean = p.strip()
        if p_clean not in seen:
            seen.add(p_clean)
            unique_preds.append(p_clean)
        if len(unique_preds) == k:
            break
            
    for rank, p in enumerate(unique_preds, start=1):
        if p == actual_str:
            return 1.0 / rank

    return 0.0


def calculate_map3(
    actuals: Sequence[str],
    predictions: Sequence[Sequence[str]],
    k: int = 3
) -> float:
    """Compute Mean Average Precision at 3 across a dataset.
    
    Args:
        actuals: List/Array of true labels.
        predictions: List of sequences of predicted candidates.
        k: Maximum rank cutoff (default: 3).
        
    Returns:
        float: Mean Average Precision at K.
    """
    if len(actuals) == 0:
        return 0.0
    if len(actuals) != len(predictions):
        raise ValueError(
            f"Length mismatch: {len(actuals)} actuals vs {len(predictions)} predictions."
        )

    scores = [calculate_apk(act, pred, k=k) for act, pred in zip(actuals, predictions)]
    return float(np.mean(scores))


def compute_map3(
    actuals: Sequence[str],
    predictions: Sequence[Sequence[str]],
    k: int = 3
) -> float:
    """Compute Mean Average Precision at 3 (semantic alias for calculate_map3)."""
    return calculate_map3(actuals, predictions, k=k)


__all__ = ["calculate_apk", "calculate_map3", "compute_map3"]


if __name__ == "__main__":
    # Smoke test verification
    acts = ["FRM-01", "CAL-01", "NONE", "WOP-01"]
    preds = [
        ["FRM-01", "CAL-01", "OTH-00"], # rank 1 -> 1.0
        ["FRM-01", "CAL-01", "OTH-00"], # rank 2 -> 0.5
        ["OTH-00", "CAL-01", "NONE"],   # rank 3 -> 0.3333333333333333
        ["OTH-00", "CAL-01", "FRM-01"]  # not in top 3 -> 0.0
    ]
    map_score = calculate_map3(acts, preds, k=3)
    expected = (1.0 + 0.5 + (1.0 / 3.0) + 0.0) / 4.0
    print(f"Computed MAP@3: {map_score:.6f} | Expected: {expected:.6f}")
    assert abs(map_score - expected) < 1e-9, "Verification failed!"
    print("Self-check passed!")
