"""Unit and Invariant Tests for Comparator Variants and Normalizer Guardrails."""
import pytest
from src.reader.comparator_ablation import (
    ComparatorConfig,
    ModularAnswerNormalizer,
)


@pytest.fixture
def full_config() -> ComparatorConfig:
    return ComparatorConfig(
        name="V5_Cumulative_All",
        description="All safe candidate rules active",
        strip_latex_delimiters=True,
        expand_units=True,
        terminal_equality_extraction=True,
        compound_alternatives=True,
    )


class TestGroupASafeNormalization:
    """Group A: Unambiguous formatting, LaTeX wrapping, and unit nouns."""

    def test_fraction_numeric_equivalence_ffe05e6614d9(self, full_config):
        # 30/15 vs $2$
        eq, reason = ModularAnswerNormalizer.are_equivalent("30/15", "$2$", full_config)
        assert eq is True
        assert "NUMERIC_EQUIVALENT" in reason

    def test_latex_fraction_environment_b391dd9fd91a(self, full_config):
        # \(\frac{3b}{b+1}\) vs $\dfrac{3b}{b + 1}$
        eq, reason = ModularAnswerNormalizer.are_equivalent(r"\(\frac{3b}{b+1}\)", r"$\dfrac{3b}{b + 1}$", full_config)
        assert eq is True

    def test_terminal_calculation_with_latex_unit_e3b2f6f06a06(self, full_config):
        # Total Luas = 60 cm² + 216 cm² = 276 cm² vs $276\ \text{cm}^2$
        eq, reason = ModularAnswerNormalizer.are_equivalent(
            "Total Luas = 60 cm² + 216 cm² = 276 cm²",
            r"$276\ \text{cm}^2$",
            full_config
        )
        assert eq is True

    def test_latex_area_unit_e994eca801ad(self, full_config):
        # 990 m² vs $990\text{ m}^2$
        eq, reason = ModularAnswerNormalizer.are_equivalent("990 m²", r"$990\text{ m}^2$", full_config)
        assert eq is True

    def test_terminal_equality_calculation_63fa16e9115b(self, full_config):
        # FPBnya 2 x 2 x 2 = 8 vs 8
        eq, reason = ModularAnswerNormalizer.are_equivalent("FPBnya 2 x 2 x 2 = 8", "8", full_config)
        assert eq is True

    def test_unit_noun_siswa_c1493c95da73(self, full_config):
        # 19 vs 19 siswa
        eq, reason = ModularAnswerNormalizer.are_equivalent("19", "19 siswa", full_config)
        assert eq is True

    def test_unit_abbreviation_org_d2acb5106baf(self, full_config):
        # 156 org vs 156
        eq, reason = ModularAnswerNormalizer.are_equivalent("156 org", "156", full_config)
        assert eq is True

    def test_unit_noun_kantong_a38da9f85ceb(self, full_config):
        # 14 kantong vs 14
        eq, reason = ModularAnswerNormalizer.are_equivalent("14 kantong", "14", full_config)
        assert eq is True

    def test_unit_noun_susunan_763cb5feee58(self, full_config):
        # 72 susunan. vs 72
        eq, reason = ModularAnswerNormalizer.are_equivalent("72 susunan.", "72", full_config)
        assert eq is True

    def test_plain_integer_wrapped_in_latex_2089fd944870(self, full_config):
        # 1 kg vs $1$
        eq, reason = ModularAnswerNormalizer.are_equivalent("1 kg", "$1$", full_config)
        assert eq is True


class TestGroupBMathematicalEquivalence:
    """Group B: Alternative answer forms, equality chains, and expressions."""

    def test_compound_alternatives_atau_523b69adbb4c(self, full_config):
        # 0,625 atau 62,5% vs $\dfrac{5}{8}$
        eq, reason = ModularAnswerNormalizer.are_equivalent(
            "0,625 atau 62,5%",
            r"$\dfrac{5}{8}$",
            full_config
        )
        assert eq is True
        assert "COMPOUND_ALTERNATIVE_MATCH" in reason

    def test_compound_alternatives_p_32484c492b52(self, full_config):
        # P = 9/10 atau 0,9 atau 90% vs $\dfrac{9}{10}$
        eq, reason = ModularAnswerNormalizer.are_equivalent(
            "P = 9/10 atau 0,9 atau 90%",
            r"$\dfrac{9}{10}$",
            full_config
        )
        assert eq is True
        assert "COMPOUND_ALTERNATIVE_MATCH" in reason

    def test_equality_chain_ba6698cc5c41(self, full_config):
        # = 2/6 - 1/3 = 9/6 - 2/6 = 7/6 vs $\frac{7}{6}$
        eq, reason = ModularAnswerNormalizer.are_equivalent(
            "= 2/6 - 1/3 = 9/6 - 2/6 = 7/6",
            r"$\frac{7}{6}$",
            full_config
        )
        assert eq is True

    def test_regression_line_equation_688bf78fe501(self, full_config):
        # y = -10x + 100 vs $\hat{y} = -10x + 100$
        eq, reason = ModularAnswerNormalizer.are_equivalent(
            "y = -10x + 100",
            r"$\hat{y} = -10x + 100$",
            full_config
        )
        assert eq is True


class TestGroupCDoNotAutoAcceptGuards:
    """Group C: Genuine arithmetic/conceptual differences that MUST be rejected."""

    def test_arithmetic_inequivalence_59a8c34f3e14(self, full_config):
        # 72/15 (= 4.8) vs $\dfrac{4}{5}$ (= 0.8) -> MUST BE REJECTED
        eq, reason = ModularAnswerNormalizer.are_equivalent("= 72/15 m³", r"$\\dfrac{4}{5}$", full_config)
        assert eq is False

    def test_power_slip_18b50e6dda88(self, full_config):
        # 266 vs $2^8$ (= 256) -> MUST BE REJECTED
        eq, reason = ModularAnswerNormalizer.are_equivalent("266", r"$2^8$", full_config)
        assert eq is False

    def test_mixed_fraction_ambiguity_7f5c3b0ce553(self, full_config):
        # 11/2 (= 5.5) vs $1\dfrac{1}{2}$ (= 1.5) -> MUST BE REJECTED
        eq, reason = ModularAnswerNormalizer.are_equivalent("11/2", r"$1\dfrac{1}{2}$", full_config)
        assert eq is False

    def test_monetary_magnitude_error_a41716ed2eab(self, full_config):
        # Rp50.000 vs Rp250.000 -> MUST BE REJECTED
        eq, reason = ModularAnswerNormalizer.are_equivalent("=Rp50.000 Rupiah", "Rp250.000", full_config)
        assert eq is False

    def test_false_accept_guard_10_vs_10000(self, full_config):
        # 10 vs 10.000 -> MUST BE REJECTED
        eq, _ = ModularAnswerNormalizer.are_equivalent("10", "10.000", full_config)
        assert eq is False
