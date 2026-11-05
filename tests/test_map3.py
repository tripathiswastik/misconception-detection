"""Unit tests for MAP@3 evaluation metric."""
import pytest
from src.metrics.map3 import calculate_apk, calculate_map3


def test_apk_rank1():
    assert calculate_apk("FRM-01", ["FRM-01", "CAL-01", "NONE"]) == 1.0


def test_apk_rank2():
    assert calculate_apk("CAL-01", ["FRM-01", "CAL-01", "NONE"]) == 0.5


def test_apk_rank3():
    assert abs(calculate_apk("NONE", ["FRM-01", "CAL-01", "NONE"]) - (1.0 / 3.0)) < 1e-9


def test_apk_rank4_or_missing():
    assert calculate_apk("WOP-01", ["FRM-01", "CAL-01", "NONE"]) == 0.0
    assert calculate_apk("WOP-01", ["FRM-01", "CAL-01", "NONE", "WOP-01"]) == 0.0


def test_apk_duplicate_predictions():
    # If the model emits duplicates, the duplicate at higher rank should not steal lower ranks
    assert calculate_apk("CAL-01", ["FRM-01", "FRM-01", "CAL-01"]) == 0.5


def test_map3_batch():
    actuals = ["A", "B", "C", "D"]
    preds = [
        ["A", "X", "Y"], # 1.0
        ["X", "B", "Y"], # 0.5
        ["X", "Y", "C"], # 1/3
        ["X", "Y", "Z"]  # 0.0
    ]
    expected = (1.0 + 0.5 + (1.0 / 3.0) + 0.0) / 4.0
    assert abs(calculate_map3(actuals, preds) - expected) < 1e-9


def test_map3_empty():
    assert calculate_map3([], []) == 0.0


def test_map3_mismatch_raises():
    with pytest.raises(ValueError):
        calculate_map3(["A"], [["A"], ["B"]])
