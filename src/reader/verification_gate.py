"""Verification Quality Gate and Downstream Metric Evaluator.

Enforces strict statistical guardrails for handwritten answer verifiers:
1. Single-class degeneracy rejection (rejects all-negative or all-positive collapses).
2. Class-specific performance auditing (FPR, FNR, Recall for both True and False).
3. Downstream MAP@3 impact simulation on held-out or out-of-fold validation sets.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


def parse_bool(value: Any) -> bool:
    """Safely parse boolean values, preventing 'False' string from evaluating as True."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError("Missing or NaN final_answer_correct label")
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Unexpected boolean label: {value!r}")


class VerificationQualityGate:
    """Statistical gatekeeper determining whether an answer-verification engine is safe to deploy."""

    def __init__(
        self,
        min_accuracy: float = 0.85,
        min_class_recall: float = 0.50,
        min_samples: int = 20,
        max_abstain_rate: float = 0.20,
    ) -> None:
        self.min_accuracy = float(min_accuracy)
        self.min_class_recall = float(min_class_recall)
        self.min_samples = min_samples
        self.max_abstain_rate = float(max_abstain_rate)

    def evaluate(
        self,
        y_true: Sequence[bool],
        y_pred: Sequence[Any],
    ) -> Dict[str, Any]:
        """Audit binary predictions against ground-truth final_answer_correct.
        
        Args:
            y_true: Ground truth boolean values (True = student answer correct).
            y_pred: Model predicted boolean values.
            
        Returns:
            Dict containing accuracy, confusion matrix metrics, degeneracy status,
            and whether the verifier passes the gate.
        """
        n = len(y_true)
        if n != len(y_pred):
            raise ValueError(f"Length mismatch: {n} y_true vs {len(y_pred)} y_pred")
        if n < self.min_samples:
            return {
                "passed_gate": False,
                "n_samples": n,
                "rejection_reason": f"INSUFFICIENT_SAMPLES: {n} < {self.min_samples}",
                "is_degenerate": False,
            }

        yt = np.array(y_true, dtype=bool)

        # Convert y_pred: True (correct), False (incorrect), None (abstain)
        clean_preds: List[Optional[bool]] = []
        for p in y_pred:
            if p is None or (isinstance(p, str) and p.upper() in ["ABSTAIN", "UNREADABLE", "AMBIGUOUS"]):
                clean_preds.append(None)
            else:
                clean_preds.append(bool(p))

        n_abstain = sum(1 for p in clean_preds if p is None)
        abstain_rate = float(n_abstain / n)

        # Non-abstained evaluation
        non_abstained_indices = [i for i, p in enumerate(clean_preds) if p is not None]
        n_evaluated = len(non_abstained_indices)

        if n_evaluated == 0:
            return {
                "passed_gate": False,
                "n_samples": n,
                "n_evaluated": 0,
                "abstain_rate": 1.0,
                "is_degenerate": False,
                "rejection_reason": "ALL_SAMPLES_ABSTAINED: Zero decisions emitted",
            }

        yt_eval = np.array([yt[i] for i in non_abstained_indices], dtype=bool)
        yp_eval = np.array([clean_preds[i] for i in non_abstained_indices], dtype=bool)

        # 1. Degeneracy Check on evaluated decisions
        n_pred_true = int(np.sum(yp_eval))
        n_pred_false = int(np.sum(~yp_eval))
        is_degenerate = (n_pred_true == 0) or (n_pred_false == 0)

        # Confusion Matrix
        # Positive = Correct answer (True), Negative = Incorrect answer (False)
        tp = int(np.sum((yt_eval == True) & (yp_eval == True)))
        tn = int(np.sum((yt_eval == False) & (yp_eval == False)))
        fp = int(np.sum((yt_eval == False) & (yp_eval == True)))
        fn = int(np.sum((yt_eval == True) & (yp_eval == False)))

        # Selective metrics (decided cases only)
        selective_accuracy = float((tp + tn) / n_evaluated) if n_evaluated > 0 else 0.0
        pos_decided = int(np.sum(yt_eval == True))
        neg_decided = int(np.sum(yt_eval == False))
        selective_recall_pos = float(tp / pos_decided) if pos_decided > 0 else 0.0
        selective_recall_neg = float(tn / neg_decided) if neg_decided > 0 else 0.0

        # Conservative operational metrics (full population, abstentions count against recall)
        pos_total = int(np.sum(yt == True))
        neg_total = int(np.sum(yt == False))
        operational_accuracy = float((tp + tn) / n)
        operational_recall_pos = float(tp / pos_total) if pos_total > 0 else 0.0
        operational_recall_neg = float(tn / neg_total) if neg_total > 0 else 0.0
        false_acceptance_rate = float(fp / neg_total) if neg_total > 0 else 0.0
        false_rejection_rate = float((pos_total - tp) / pos_total) if pos_total > 0 else 0.0
        coverage = float(n_evaluated / n)

        reasons: List[str] = []
        if is_degenerate:
            collapsed_to = "ALL_FALSE" if n_pred_true == 0 else "ALL_TRUE"
            reasons.append(f"DEGENERATE_SINGLE_CLASS_PREDICTOR: Model predicted {collapsed_to} for 100% of samples")

        if abstain_rate > self.max_abstain_rate:
            reasons.append(f"HIGH_ABSTAIN_RATE: {abstain_rate:.4f} > {self.max_abstain_rate:.4f}")

        if selective_accuracy < self.min_accuracy:
            reasons.append(f"ACCURACY_BELOW_GATE: {selective_accuracy:.4f} < {self.min_accuracy:.4f}")

        if operational_recall_pos < self.min_class_recall:
            reasons.append(f"LOW_POSITIVE_RECALL: {operational_recall_pos:.4f} < {self.min_class_recall:.4f}")

        if operational_recall_neg < self.min_class_recall:
            reasons.append(f"LOW_NEGATIVE_RECALL: {operational_recall_neg:.4f} < {self.min_class_recall:.4f}")

        passed = (len(reasons) == 0)

        return {
            "passed_gate": passed,
            "is_degenerate": is_degenerate,
            "accuracy": selective_accuracy,
            "selective_accuracy": selective_accuracy,
            "operational_accuracy": operational_accuracy,
            "coverage": coverage,
            "n_samples": n,
            "n_evaluated": n_evaluated,
            "n_abstained": n_abstain,
            "abstain_rate": abstain_rate,
            "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            "recall_positive": operational_recall_pos,
            "recall_negative": operational_recall_neg,
            "selective_recall_positive": selective_recall_pos,
            "selective_recall_negative": selective_recall_neg,
            "false_rejection_rate": false_rejection_rate,
            "false_acceptance_rate": false_acceptance_rate,
            "reasons": reasons,
            "rejection_reason": "; ".join(reasons) if not passed else None,
        }


def compute_map3_row(actual: str, pred_top3: Sequence[str]) -> float:
    """Compute MAP@3 score for a single prediction row."""
    for rank, p in enumerate(pred_top3[:3]):
        if p == actual:
            return 1.0 / (rank + 1)
    return 0.0


def paired_bootstrap_map3_test(
    baseline_scores: Sequence[float],
    gated_scores: Sequence[float],
    n_resamples: int = 10000,
    seed: int = 42
) -> Dict[str, Any]:
    """Calculate paired bootstrap significance test (10,000 resamples) for MAP@3 delta.
    
    Delta_i = MAP@3_gated,i - MAP@3_base,i.
    Cancels out cross-sample difficulty variance and yields tight, honest CI.
    """
    diffs = np.array(gated_scores) - np.array(baseline_scores)
    n = len(diffs)
    observed_delta = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(0, n, size=(n_resamples, n))
    boot_means = np.mean(diffs[boot_indices], axis=1)

    ci_low = float(np.percentile(boot_means, 2.5))
    ci_high = float(np.percentile(boot_means, 97.5))

    # Two-tailed p-value against null hypothesis (delta == 0)
    p_value = float(np.mean(np.abs(boot_means) >= np.abs(observed_delta))) if observed_delta != 0 else 1.0
    is_statistically_significant = (ci_low > 0.0) and (p_value < 0.05)

    return {
        "observed_delta": observed_delta,
        "ci_95": [ci_low, ci_high],
        "p_value": p_value,
        "is_significant_improvement": is_statistically_significant,
        "n_resamples": n_resamples
    }


def simulate_downstream_gating_map3(
    ground_truth_codes: Sequence[str],
    base_probabilities: np.ndarray,
    classes: Sequence[str],
    verifier_predictions: Sequence[Any],
    none_class_code: str = "NONE",
    boost_weight: float = 0.90,
    suppress_weight: float = 0.01,
    run_bootstrap: bool = True
) -> Dict[str, Any]:
    """Simulate downstream MAP@3 impact of applying verifier gating to probability matrices.
    
    If verifier predicts TRUE (student correct):
        Promote `none_class_code` by scaling its probability up.
    If verifier predicts FALSE (student incorrect):
        Suppress `none_class_code` to allow specific error codes to rank higher.
    If verifier predicts NONE / 'ABSTAIN':
        SAFETY FALLBACK: Leave baseline probabilities completely unaltered (zero modification).
    
    Returns:
        Dict with baseline_map3, gated_map3, delta_map3, degradation breakdown, and bootstrap stats.
    """
    n = len(ground_truth_codes)
    if base_probabilities.shape[0] != n:
        raise ValueError(f"Shape mismatch: {n} labels vs {base_probabilities.shape[0]} probs")
    
    class_list = list(classes)
    if none_class_code not in class_list:
        raise ValueError(f"none_class_code '{none_class_code}' not in class list")
    
    none_idx = class_list.index(none_class_code)
    
    # 1. Baseline MAP@3
    base_top3_indices = np.argsort(-base_probabilities, axis=1)[:, :3]
    baseline_scores: List[float] = []
    for i in range(n):
        top3_classes = [class_list[idx] for idx in base_top3_indices[i]]
        baseline_scores.append(compute_map3_row(ground_truth_codes[i], top3_classes))
    baseline_map3 = float(np.mean(baseline_scores))

    # 2. Gated probabilities with Safety Fallback on ABSTAIN
    gated_probs = base_probabilities.copy()
    n_abstained = 0
    for i in range(n):
        pred_corr = verifier_predictions[i]
        
        # Safety fallback check: abstain if None or string marker
        if pred_corr is None or (isinstance(pred_corr, str) and pred_corr.upper() in ["ABSTAIN", "UNREADABLE", "AMBIGUOUS"]):
            n_abstained += 1
            continue  # Row preserved exactly as baseline!

        if bool(pred_corr) is True:
            # Promote NONE
            gated_probs[i, none_idx] = max(gated_probs[i, none_idx], boost_weight)
        else:
            # Suppress NONE
            gated_probs[i, none_idx] *= suppress_weight
        
        # Renormalize row
        s = np.sum(gated_probs[i])
        if s > 0:
            gated_probs[i] /= s

    # 3. Gated MAP@3
    gated_top3_indices = np.argsort(-gated_probs, axis=1)[:, :3]
    gated_scores: List[float] = []
    degraded_count = 0
    improved_count = 0
    for i in range(n):
        top3_classes = [class_list[idx] for idx in gated_top3_indices[i]]
        score = compute_map3_row(ground_truth_codes[i], top3_classes)
        gated_scores.append(score)
        if score > baseline_scores[i]:
            improved_count += 1
        elif score < baseline_scores[i]:
            degraded_count += 1

    gated_map3 = float(np.mean(gated_scores))
    delta_map3 = float(gated_map3 - baseline_map3)

    result: Dict[str, Any] = {
        "baseline_map3": baseline_map3,
        "gated_map3": gated_map3,
        "delta_map3": delta_map3,
        "is_improvement": delta_map3 > 0,
        "n_improved": improved_count,
        "n_degraded": degraded_count,
        "n_unchanged": n - (improved_count + degraded_count),
        "n_abstained": n_abstained,
        "abstain_rate": float(n_abstained / n) if n > 0 else 0.0
    }

    if run_bootstrap and n >= 10:
        boot_res = paired_bootstrap_map3_test(baseline_scores, gated_scores)
        result["bootstrap"] = boot_res

    return result
