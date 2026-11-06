"""Paired Bootstrap Significance Testing for Out-Of-Fold Model Comparisons.

Computes exact per-row MAP@3 differences between model A and model B,
resamples 10,000 times, and outputs:
- Mean delta: E[MAP@3_A - MAP@3_B]
- Bootstrap Standard Error (SE)
- 95% Bootstrap Confidence Interval
- Empirical p-value (H0: Model A <= Model B)
- Row-level win / loss / tie counts
"""
from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def compute_row_map3(true_label: str, top3_preds: Sequence[str]) -> float:
    """Compute MAP@3 score for a single sample (0.0, 0.333, 0.5, or 1.0)."""
    for rank, pred in enumerate(top3_preds[:3], start=1):
        if pred == true_label:
            return 1.0 / rank
    return 0.0


def paired_bootstrap_test(
    y_true: Sequence[str],
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    classes: Sequence[str],
    n_resamples: int = 10000,
    random_state: int = 42,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """Execute paired bootstrap test comparing model A against model B."""
    assert len(y_true) == len(probs_a) == len(probs_b)
    n_samples = len(y_true)

    # Extract top 3 predictions via argpartition followed by sort on top-3 only
    k = min(3, probs_a.shape[1])
    row_idx = np.arange(n_samples)[:, None]

    part_a = np.argpartition(-probs_a, kth=k - 1, axis=1)[:, :k]
    order_a = np.argsort(-probs_a[row_idx, part_a], axis=1, kind="mergesort")
    top_a = part_a[row_idx, order_a]

    part_b = np.argpartition(-probs_b, kth=k - 1, axis=1)[:, :k]
    order_b = np.argsort(-probs_b[row_idx, part_b], axis=1, kind="mergesort")
    top_b = part_b[row_idx, order_b]

    # Vectorized MAP@3 calculation
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_indices = np.array([class_to_idx.get(str(y), -1) for y in y_true], dtype=np.int64)

    m0_a = top_a[:, 0] == y_indices
    m1_a = top_a[:, 1] == y_indices
    m2_a = top_a[:, 2] == y_indices
    scores_a = np.where(m0_a, 1.0, np.where(m1_a, 0.5, np.where(m2_a, 1.0 / 3.0, 0.0)))

    m0_b = top_b[:, 0] == y_indices
    m1_b = top_b[:, 1] == y_indices
    m2_b = top_b[:, 2] == y_indices
    scores_b = np.where(m0_b, 1.0, np.where(m1_b, 0.5, np.where(m2_b, 1.0 / 3.0, 0.0)))

    # Paired row differences
    deltas = scores_a - scores_b
    mean_delta = float(np.mean(deltas))

    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    ties = int(np.sum(deltas == 0))

    # Vectorized chunked bootstrap resampling (avoids 10,000 array allocations)
    rng = np.random.RandomState(random_state)
    boot_means = np.empty(n_resamples, dtype=np.float64)
    batch_size = 1000

    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        batch_indices = rng.randint(0, n_samples, size=(end - start, n_samples))
        boot_means[start:end] = np.mean(deltas[batch_indices], axis=1)

    se_delta = float(np.std(boot_means, ddof=1))
    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    p_value = float(np.mean(boot_means <= 0.0))

    mean_a = float(np.mean(scores_a))
    mean_b = float(np.mean(scores_b))

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta": mean_delta,
        "se_delta": se_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_value,
        "is_significant": ci_lower > 0.0,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "n_samples": n_samples
    }


def print_bootstrap_report(res: Dict[str, Any], name_a: str, name_b: str):
    """Format and print paired bootstrap report."""
    print(f"\n=======================================================")
    print(f"Paired Bootstrap Test: {name_a} vs. {name_b}")
    print(f"=======================================================")
    print(f"  {name_a} MAP@3: {res['mean_a']:.4f}")
    print(f"  {name_b} MAP@3: {res['mean_b']:.4f}")
    print(f"  Mean Paired Difference (Delta): {res['mean_delta']:+.4f}")
    print(f"  Bootstrap SE: {res['se_delta']:.4f}")
    print(f"  95% Confidence Interval: [{res['ci_lower']:+.4f}, {res['ci_upper']:+.4f}]")
    print(f"  Empirical p-value (H0: Delta <= 0): {res['p_value']:.4f}")
    print(f"  Row-Level Breakdown: {res['wins']} wins, {res['losses']} losses, {res['ties']} ties (N={res['n_samples']})")
    print(f"  Statistically Significant at 95%? {'YES (CI excludes 0)' if res['is_significant'] else 'NO (CI includes 0)'}")
    print(f"=======================================================\n")
