"""Unit and regression tests for VerificationQualityGate and Downstream MAP@3 Gating."""
import numpy as np
import pytest

from src.reader.verification_gate import (
    VerificationQualityGate,
    simulate_downstream_gating_map3,
)


def test_quality_gate_rejects_all_negative_degenerate_predictor():
    """Regression test: An all-negative verifier (like Qwen2-VL pilot) must be rejected."""
    gate = VerificationQualityGate(min_accuracy=0.85, min_class_recall=0.50, min_samples=20)
    
    # 50 samples: 25 True, 25 False
    y_true = [True] * 25 + [False] * 25
    y_pred = [False] * 50  # Degenerate all-negative collapse
    
    audit = gate.evaluate(y_true, y_pred)
    
    assert audit["passed_gate"] is False
    assert audit["is_degenerate"] is True
    assert "DEGENERATE_SINGLE_CLASS_PREDICTOR" in audit["rejection_reason"]
    assert audit["accuracy"] == 0.50
    assert audit["recall_positive"] == 0.0
    assert audit["false_rejection_rate"] == 1.0  # 100% of correct answers were falsely rejected!


def test_quality_gate_rejects_all_positive_degenerate_predictor():
    """Regression test: An all-positive verifier must also be rejected."""
    gate = VerificationQualityGate(min_accuracy=0.85, min_class_recall=0.50, min_samples=20)
    
    y_true = [True] * 25 + [False] * 25
    y_pred = [True] * 50  # Degenerate all-positive collapse
    
    audit = gate.evaluate(y_true, y_pred)
    
    assert audit["passed_gate"] is False
    assert audit["is_degenerate"] is True
    assert "DEGENERATE_SINGLE_CLASS_PREDICTOR" in audit["rejection_reason"]
    assert audit["recall_negative"] == 0.0
    assert audit["false_acceptance_rate"] == 1.0


def test_quality_gate_passes_high_quality_balanced_verifier():
    """A verifier meeting >= 85% accuracy and healthy two-class recall passes."""
    gate = VerificationQualityGate(min_accuracy=0.85, min_class_recall=0.50, min_samples=20)
    
    # 90% accuracy: 2 mismatches in 20
    y_true = [True] * 10 + [False] * 10
    y_pred = [True] * 9 + [False] * 1 + [False] * 9 + [True] * 1
    
    audit = gate.evaluate(y_true, y_pred)
    
    assert audit["passed_gate"] is True
    assert audit["is_degenerate"] is False
    assert audit["accuracy"] == 0.90
    assert audit["recall_positive"] == 0.90
    assert audit["recall_negative"] == 0.90
    assert audit["rejection_reason"] is None


def test_simulate_downstream_gating_proves_degenerate_damage():
    """Simulating an all-negative gate on data with genuine NONE labels degrades MAP@3."""
    classes = ["NONE", "ERR_A", "ERR_B"]
    
    # 4 samples: two have true label NONE, two have true label ERR_A
    ground_truth = ["NONE", "NONE", "ERR_A", "ERR_B"]
    
    # Baseline probabilities where NONE is rank 1 for samples 0 and 1
    base_probs = np.array([
        [0.70, 0.20, 0.10],  # true: NONE (baseline rank 1)
        [0.65, 0.25, 0.10],  # true: NONE (baseline rank 1)
        [0.10, 0.80, 0.10],  # true: ERR_A (baseline rank 1)
        [0.10, 0.20, 0.70],  # true: ERR_B (baseline rank 1)
    ])
    
    # Degenerate all-negative verifier (claims no student is correct)
    degenerate_preds = [False, False, False, False]
    
    sim = simulate_downstream_gating_map3(
        ground_truth_codes=ground_truth,
        base_probabilities=base_probs,
        classes=classes,
        verifier_predictions=degenerate_preds,
        none_class_code="NONE",
    )
    
    # Baseline was 1.0 MAP@3
    assert sim["baseline_map3"] == 1.0
    # Degenerate gating suppressed NONE, ruining predictions for genuinely correct students
    assert sim["gated_map3"] < sim["baseline_map3"]
    assert sim["is_improvement"] is False
    assert sim["n_degraded"] == 2


def test_deterministic_answer_verifier_normalization():
    """Verify math normalization logic."""
    from src.reader.diagnostic_verifier import normalize_math_answer, deterministic_answer_verifier
    assert normalize_math_answer("Jawaban: 42.0") == "42"
    assert normalize_math_answer("$3,5$") == "3.5"
    assert normalize_math_answer("Maka x = 7") == "7"

    is_match, ans = deterministic_answer_verifier("L1: Diketahui\nL2: x = 5", "5")
    assert is_match is True
    assert ans == "5"

    is_match, ans = deterministic_answer_verifier("L1: Diketahui\nL2: x = 4", "5")
    assert is_match is False

