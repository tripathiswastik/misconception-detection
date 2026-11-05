"""Unit and regression tests for conservative TranscriptionNormalizer and Answer Verification."""
import pytest
from src.reader.transcription_normalizer import TranscriptionNormalizer


def test_indonesian_answer_label_normalization():
    """Requirement: 'Jawaban: 5' vs '5' -> Equivalent extracted answer."""
    extracted, status = TranscriptionNormalizer.extract_final_conclusion("L1 | Langkah 1\nL2 | Jawaban: 5")
    assert status == "EXTRACTED"
    assert extracted == "5"

    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent(extracted, "5")
    assert is_equiv is True


def test_indonesian_jadi_equality_normalization():
    """Requirement: 'Jadi, x = 3' vs 'x=3' -> Equivalent extracted answer."""
    extracted, status = TranscriptionNormalizer.extract_final_conclusion("L1 | 2x = 6\nL2 | Jadi, x = 3")
    assert status == "EXTRACTED"
    assert extracted == "3"

    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent(extracted, "x=3")
    assert is_equiv is True


def test_indonesian_decimal_comma_normalization():
    """Requirement: '0,5' vs '0.5' -> Equivalent only when decimal notation is unambiguous."""
    extracted, status = TranscriptionNormalizer.extract_final_conclusion("L1 | Hasil = 0,5")
    assert status == "EXTRACTED"
    assert extracted == "0.5"

    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent(extracted, "0.5")
    assert is_equiv is True

    # Test raw unnormalized comma input against dot decimal
    is_equiv_raw, _ = TranscriptionNormalizer.are_answers_equivalent("0,5", "0.5")
    assert is_equiv_raw is True


def test_negative_sign_preservation():
    """Requirement: '-3' vs '3' -> Different answers (must preserve signs!)."""
    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent("-3", "3")
    assert is_equiv is False
    assert reason == "NUMERIC_INEQUALITY"

    # Also Unicode minus sign normalization
    is_equiv_unicode, _ = TranscriptionNormalizer.are_answers_equivalent("\u22123", "-3")
    assert is_equiv_unicode is True


def test_fraction_direction_preservation():
    """Requirement: '1/2' vs '2/1' -> Different answers (must preserve order!)."""
    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent("1/2", "2/1")
    assert is_equiv is False
    assert reason == "NUMERIC_INEQUALITY"

    # But 1/2 vs 0.5 must be equivalent
    is_equiv_float, reason_float = TranscriptionNormalizer.are_answers_equivalent("1/2", "0.5")
    assert is_equiv_float is True
    assert reason_float == "NUMERIC_EQUIVALENCE"


def test_unrecognizable_answer_returns_uncertain():
    """Requirement: No recognizable final answer -> Uncertain, not automatically incorrect."""
    extracted, status = TranscriptionNormalizer.extract_final_conclusion("L1 | [scribbles and crossed out working]")
    assert status == "UNCERTAIN"
    assert extracted is None

    # Verification must abstain on uncertain/None
    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent(extracted, "42")
    assert is_equiv is None
    assert "ABSTAIN" in reason


def test_variable_guard_prevents_false_matches():
    """Strict guard: '2x' vs '2' must NOT match."""
    is_equiv, reason = TranscriptionNormalizer.are_answers_equivalent("2x", "2")
    assert is_equiv is False
    assert reason == "VARIABLE_MISMATCH"
