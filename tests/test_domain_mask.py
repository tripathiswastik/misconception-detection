"""Unit tests for DomainConstraintEngine."""
import numpy as np
import pandas as pd
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.diagnoser.domain_mask import DomainConstraintEngine


def test_domain_constraint_affinity():
    engine = DomainConstraintEngine()

    # NONE is always 1.0
    assert engine.compute_affinity(LABEL_NONE, "Operasi pecahan", 6) == 1.0

    # Fraction error on fraction topic has high affinity (1.0)
    frac_aff = engine.compute_affinity("FRA-01", "Operasi pecahan", 5)
    assert frac_aff == 1.0

    # Fraction error on Geometry topic has low affinity (< 0.15)
    geom_aff = engine.compute_affinity("FRA-01", "Luas Permukaan Bangun Ruang", 9)
    assert geom_aff < 0.15

    # Universal calculation error on geometry is allowed
    cal_aff = engine.compute_affinity("CAL-01", "Luas Permukaan Bangun Ruang", 9)
    assert cal_aff >= 0.8


def test_domain_mask_batch_calibration():
    engine = DomainConstraintEngine()
    classes = ["FRA-01", "GEO-01", "NONE"]
    # 2 samples: 1 fraction topic, 1 geometry topic
    raw_probs = np.array([
        [0.45, 0.45, 0.10],  # sample 1: fraction topic
        [0.45, 0.45, 0.10],  # sample 2: geometry topic
    ])
    topics = ["Operasi pecahan", "Luas Bangun Datar Segitiga"]
    grades = [5, 7]

    calibrated = engine.apply_mask_batch(raw_probs, classes, topics, grades)

    # For sample 1 (fraction), FRA-01 should have higher probability than GEO-01
    assert calibrated[0, 0] > calibrated[0, 1]

    # For sample 2 (geometry), GEO-01 should have higher probability than FRA-01
    assert calibrated[1, 1] > calibrated[1, 0]

    # Each row sums to 1.0
    assert np.isclose(calibrated[0].sum(), 1.0)
    assert np.isclose(calibrated[1].sum(), 1.0)
