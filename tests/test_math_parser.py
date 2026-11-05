"""Unit tests for MathFeatureExtractor and symbolic pattern parsing."""
import pandas as pd
from src.diagnoser.math_parser import MathFeatureExtractor, build_enhanced_feature_text


def test_fraction_addition_error_detection():
    # 1/3 + 1/4 = 2/7 (adding numerators and denominators)
    q_text = "Hitunglah hasil dari 1/3 + 1/4"
    w_text = "L1 | 1/3 + 1/4 = 2/7"
    assert MathFeatureExtractor.check_fraction_addition_error(q_text, w_text) is True

    # Correct addition (not error)
    w_correct = "L1 | 1/3 + 1/4 = 4/12 + 3/12 = 7/12"
    assert MathFeatureExtractor.check_fraction_addition_error(q_text, w_correct) is False


def test_sign_flip_error_detection():
    # Key is 5, student got -5
    assert MathFeatureExtractor.check_sign_flip_error("5", "L1 | -2 - 3 = -5") is True
    # Key is -8, student got 8
    assert MathFeatureExtractor.check_sign_flip_error("-8", "L1 | -3 - 5 = 8") is True
    # Student matches key
    assert MathFeatureExtractor.check_sign_flip_error("-8", "L1 | -3 - 5 = -8") is False


def test_exponent_error_detection():
    # 2^3 = 6 (base * power error)
    q_text = "Berapakah nilai dari 2^3?"
    w_text = "L1 | 2^3 = 2 * 3 = 6"
    assert MathFeatureExtractor.check_exponent_multiplication_error(q_text, w_text) is True


def test_build_enhanced_feature_text():
    row = pd.Series({
        "grade": 6,
        "topic_name": "fractions",
        "question_text": "Hitung 1/3 + 1/4",
        "correct_answer": "7/12",
        "transcription_steps": "L1 | 1/3 + 1/4 = 2/7"
    })
    feat_text = build_enhanced_feature_text(row)
    assert "GRADE_6" in feat_text
    assert "MATH_FRA_ADD_TOPS_BOTTOMS" in feat_text
    assert "MATH_MISMATCHES_KEY" in feat_text
    assert "WORKING_HAS_EQUALS" in feat_text
