"""Test Suite for AnswerNormalizer.

Covers all 33+ critical cases:
- Indonesian thousands vs decimals
- False-accept guards (e.g. 10 vs 10.000 rejected)
- Units and currency stripping
- Fractions and mixed numbers
- Polynomial expressions and superscripts
- Permuted item lists
- Abstention and failure codes
"""
import pytest
from src.reader.answer_normalizer import AnswerNormalizer


class TestAnswerNormalizer:
    """Test suite for AnswerNormalizer."""

    # 1. Indonesian Thousands Separator
    def test_thousands_separator_basic(self):
        eq, reason = AnswerNormalizer.are_equivalent("10.000", "10000")
        assert eq is True
        assert reason == "NUMERIC_EQUIVALENT"

    def test_thousands_separator_large(self):
        eq, reason = AnswerNormalizer.are_equivalent("1.250.000", "1250000")
        assert eq is True
        assert reason == "NUMERIC_EQUIVALENT"

    def test_thousands_separator_both_dotted(self):
        eq, reason = AnswerNormalizer.are_equivalent("25.000", "25.000")
        assert eq is True

    # 2. False-Accept Guards (CRITICAL)
    def test_false_accept_guard_10_vs_10000(self):
        eq, reason = AnswerNormalizer.are_equivalent("10", "10.000")
        assert eq is False
        assert reason == "NUMERIC_DIFFERENCE"

    def test_false_accept_guard_reverse(self):
        eq, reason = AnswerNormalizer.are_equivalent("10.000", "10")
        assert eq is False
        assert reason == "NUMERIC_DIFFERENCE"

    def test_false_accept_guard_sign_mismatch(self):
        eq, reason = AnswerNormalizer.are_equivalent("-18", "18")
        assert eq is False

    # 3. Indonesian Comma Decimals
    def test_comma_decimal_basic(self):
        eq, reason = AnswerNormalizer.are_equivalent("54,5", "54.5")
        assert eq is True
        assert reason == "NUMERIC_EQUIVALENT"

    def test_comma_decimal_with_leading_zero(self):
        eq, reason = AnswerNormalizer.are_equivalent("0,125", "0.125")
        assert eq is True

    def test_ambiguous_comma_group(self):
        # 1,250 can be read as 1250 or 1.25
        eq1, _ = AnswerNormalizer.are_equivalent("1,250", "1250")
        eq2, _ = AnswerNormalizer.are_equivalent("1,250", "1.25")
        assert eq1 is True
        assert eq2 is True

    # 4. Currency Normalization
    def test_currency_rp_prefix(self):
        eq, _ = AnswerNormalizer.are_equivalent("Rp10.000", "10000")
        assert eq is True

    def test_currency_case_insensitive(self):
        eq, _ = AnswerNormalizer.are_equivalent("rp 25.000", "25000")
        assert eq is True

    def test_currency_rupiah_suffix(self):
        eq, _ = AnswerNormalizer.are_equivalent("50.000 rupiah", "50000")
        assert eq is True

    # 5. Units Stripping
    def test_unit_tahun(self):
        eq, _ = AnswerNormalizer.are_equivalent("13 tahun", "13")
        assert eq is True

    def test_unit_cm_squared(self):
        eq, _ = AnswerNormalizer.are_equivalent("120 cm²", "120")
        assert eq is True

    def test_unit_percentage(self):
        eq, _ = AnswerNormalizer.are_equivalent("25%", "25")
        assert eq is True

    def test_unit_jam(self):
        eq, _ = AnswerNormalizer.are_equivalent("4 jam", "4")
        assert eq is True

    def test_unit_kg(self):
        eq, _ = AnswerNormalizer.are_equivalent("80 kg", "80")
        assert eq is True

    def test_unit_degrees(self):
        eq, _ = AnswerNormalizer.are_equivalent("110°", "110")
        assert eq is True

    # 6. Fractions & LaTeX
    def test_plain_fraction_to_decimal(self):
        eq, _ = AnswerNormalizer.are_equivalent("3/4", "0.75")
        assert eq is True

    def test_latex_fraction_conversion(self):
        eq, _ = AnswerNormalizer.are_equivalent(r"\frac{3}{4}", "0.75")
        assert eq is True

    def test_fraction_with_parentheses_unmangled(self):
        eq, _ = AnswerNormalizer.are_equivalent("(3)/(4)", "0.75")
        assert eq is True

    def test_mixed_fraction(self):
        eq, _ = AnswerNormalizer.are_equivalent("23 1/3", "23.33333")
        assert eq is True

    # 7. Preambles and Labels
    def test_preamble_jawaban(self):
        eq, _ = AnswerNormalizer.are_equivalent("Jawaban: 45", "45")
        assert eq is True

    def test_preamble_jadi_with_variable(self):
        eq, _ = AnswerNormalizer.are_equivalent("Jadi x = 60", "60")
        assert eq is True

    def test_line_marker_prefix(self):
        eq, _ = AnswerNormalizer.are_equivalent("L6 | 80", "80")
        assert eq is True

    # 8. Polynomials & Algebraic Expressions
    def test_polynomial_superscript_unification(self):
        eq, _ = AnswerNormalizer.are_equivalent("2x² + 8x - 7", "2x^2+8x-7")
        assert eq is True

    def test_polynomial_variable_expression(self):
        eq, _ = AnswerNormalizer.are_equivalent("x^2 - 6", "x^2-6")
        assert eq is True

    # 9. Permuted Lists
    def test_permuted_names_list(self):
        eq, reason = AnswerNormalizer.are_equivalent(
            "Budi, Ahmad, Cici",
            "Ahmad, Budi, Cici"
        )
        assert eq is True
        assert reason == "PERMUTED_SET_MATCH"

    def test_permuted_names_case_and_spaces(self):
        eq, _ = AnswerNormalizer.are_equivalent(
            "ahmad , budi",
            "Budi, Ahmad"
        )
        assert eq is True

    # 10. Abstentions & Coordinates
    def test_abstain_unreadable(self):
        eq, reason = AnswerNormalizer.are_equivalent("UNREADABLE", "100")
        assert eq is None
        assert reason == "ABSTAIN_UNREADABLE"

    def test_abstain_empty(self):
        eq, reason = AnswerNormalizer.are_equivalent("", "100")
        assert eq is None
        assert reason == "EMPTY_STUDENT"

    def test_coordinates_parentheses_preserved(self):
        eq, _ = AnswerNormalizer.are_equivalent("(-7, 13)", "(-7, 13)")
        assert eq is True

    def test_coordinates_whitespace_tolerant(self):
        eq, _ = AnswerNormalizer.are_equivalent("(3, -2)", "(3,-2)")
        assert eq is True
