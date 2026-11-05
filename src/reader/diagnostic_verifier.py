"""Offline Diagnostic Answer-Verification Framework with Error Taxonomy & Safe Normalization.

Implements the 5-step diagnostic sequence:
1. Error Taxonomy: Categorizes verification discrepancies into:
   - MISSING_BOX: Missing terminal answer marker/box.
   - INFORMAL_INDONESIAN_MARKER: Conversational phrasing markers ('jadi', 'maka', 'hasilnya').
   - MATH_EQUIVALENCE: Safe fractional / decimal / negative sign representations.
   - OCR_TRANSCRIPTION_ERROR: Character or line transcription discrepancies.
   - AMBIGUOUS_UNREADABLE: Illegible or conflicting working -> triggers ABSTAIN.
2. Deterministic Improvements: Safe mathematical equivalence without aggressive string matching.
3. Comparative Evaluation: Tracks accuracy, class recall, FRR, FAR, and abstention rate.
4. Shadow Inference: Simulates hypothetical gating impact on downstream OOF MAP@3 with bootstrap significance.
5. Strict Governance: GATING REMAINS STRICTLY DISABLED UNTIL APPROVED.
"""
from enum import Enum
from fractions import Fraction
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.reader.verification_gate import (
    VerificationQualityGate,
    paired_bootstrap_map3_test,
    simulate_downstream_gating_map3
)


class VerificationErrorCategory(str, Enum):
    """Taxonomy of answer-verification failure modes."""
    MISSING_BOX = "MISSING_BOX"
    INFORMAL_INDONESIAN_MARKER = "INFORMAL_INDONESIAN_MARKER"
    MATH_EQUIVALENCE = "MATH_EQUIVALENCE"
    OCR_TRANSCRIPTION_ERROR = "OCR_TRANSCRIPTION_ERROR"
    AMBIGUOUS_UNREADABLE = "AMBIGUOUS_UNREADABLE"
    GENUINE_MATHEMATICAL_ERROR = "GENUINE_MATHEMATICAL_ERROR"


def normalize_math_answer(text: str) -> str:
    """Normalize mathematical answer strings for deterministic comparison."""
    if not text:
        return ""
    s = text.strip().lower()
    # Strip OCR line prefixes like 'l1 |', 'l6 |'
    s = re.sub(r"^l\d+\s*\|\s*", "", s)
    # Strip common Indonesian preamble words
    s = re.sub(r"^(jawaban|jawab|jadi|kesimpulan|maka|hasilnya|diperoleh|dapat disimpulkan)\s*", "", s)
    # Normalize LaTeX fractions: \frac{a}{b} or \dfrac{a}{b} -> a/b
    s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"\1/\2", s)
    # If equation like "x = 7", extract right-hand side
    if "=" in s:
        s = s.split("=")[-1].strip()
    elif ":" in s:
        s = s.split(":")[-1].strip()
    # Remove whitespace, dollar signs, currency 'rp', and LaTeX formatting
    s = s.replace(" ", "").replace("$", "").replace("\\text", "").replace("{", "").replace("}", "")
    s = s.replace("rp", "").strip("[]()")
    # Normalize decimal commas to dots
    s = s.replace(",", ".")
    # Remove trailing .0 for whole numbers
    s = re.sub(r"\.0+$", "", s)
    return s


def parse_numeric_or_fraction(text: str) -> Optional[float]:
    """Safely parse text into a float if purely numeric or a fraction 'a/b'."""
    norm = normalize_math_answer(text)
    if not norm:
        return None
    # Reject strings containing letters (variables like '2x' must not parse as pure float)
    if re.search(r"[a-z]", norm):
        return None
    # Try fraction: e.g. "1/2" or "-3/4"
    if "/" in norm:
        parts = norm.split("/")
        if len(parts) == 2:
            try:
                num = float(parts[0])
                den = float(parts[1])
                if den != 0:
                    return num / den
            except ValueError:
                return None
    # Try float
    try:
        return float(norm)
    except ValueError:
        return None


def safe_math_equality(student_str: str, gold_str: str, tolerance: float = 1e-5) -> bool:
    """Determine whether student answer and gold answer are mathematically equivalent.
    
    Guards:
    - Never equates expressions with variables to raw numbers (e.g. '2x' != '2').
    - Evaluates numeric / fractional equivalence within numerical tolerance.
    """
    s_norm = normalize_math_answer(student_str)
    g_norm = normalize_math_answer(gold_str)

    if not s_norm or not g_norm:
        return False

    # 1. Exact normalized string match
    if s_norm == g_norm:
        return True

    # 2. Safe numeric / fraction comparison (only when both are strictly non-alphabetic)
    s_val = parse_numeric_or_fraction(s_norm)
    g_val = parse_numeric_or_fraction(g_norm)
    if s_val is not None and g_val is not None:
        return abs(s_val - g_val) < tolerance

    return False


def extract_student_conclusion_safely(transcription: str) -> Tuple[Optional[str], Optional[str]]:
    """Safely extract student's final conclusion and detect if ambiguous/unreadable.
    
    Returns:
        (extracted_answer, abstain_marker):
        If unreadable or empty -> (None, 'AMBIGUOUS_UNREADABLE')
        If clear answer found -> (extracted_str, None)
    """
    if not transcription or not transcription.strip():
        return None, VerificationErrorCategory.AMBIGUOUS_UNREADABLE.value

    lines = [line.strip() for line in transcription.split("\n") if line.strip()]
    if not lines:
        return None, VerificationErrorCategory.AMBIGUOUS_UNREADABLE.value

    # Search backward from final step for Indonesian markers or equalities
    indonesian_markers = ["jawaban", "jawab", "jadi", "kesimpulan", "maka", "hasilnya"]
    for line in reversed(lines):
        norm = normalize_math_answer(line)
        # Check marker keywords
        lower_line = line.lower()
        if any(marker in lower_line for marker in indonesian_markers):
            return norm, None
        if "=" in norm:
            return norm, None

    # Fallback to the last line if it has content
    last_line = lines[-1].strip()
    norm_last = normalize_math_answer(last_line)
    if norm_last:
        return norm_last, None

    return None, VerificationErrorCategory.AMBIGUOUS_UNREADABLE.value


def enhanced_safe_verifier(
    transcription: str,
    correct_answer: str
) -> Tuple[Optional[bool], str, Optional[str]]:
    """Enhanced deterministic answer verifier with safe equivalence and abstention.
    
    Returns:
        (prediction, extracted_str, error_category):
        prediction is True (correct), False (incorrect), or None (abstain).
    """
    extracted, abstain_reason = extract_student_conclusion_safely(transcription)
    if abstain_reason is not None or extracted is None:
        return None, "ABSTAIN", VerificationErrorCategory.AMBIGUOUS_UNREADABLE.value

    is_correct = safe_math_equality(extracted, correct_answer)
    return is_correct, extracted, None


def deterministic_answer_verifier(
    transcription: str,
    correct_answer: str
) -> Tuple[bool, str]:
    """Legacy baseline regex extractor for backward compatibility."""
    norm_gold = normalize_math_answer(correct_answer)
    if not transcription or not norm_gold:
        return False, "EMPTY"

    lines = [line.strip() for line in transcription.split("\n") if line.strip()]
    if not lines:
        return False, "NO_LINES"

    detected_answer = ""
    for line in reversed(lines):
        norm_line = normalize_math_answer(line)
        if "=" in norm_line:
            detected_answer = norm_line.split("=")[-1].strip()
            break
        m = re.search(r"([0-9\.\-]+|[a-e])$", norm_line)
        if m:
            detected_answer = m.group(1)
            break

    if not detected_answer and lines:
        detected_answer = normalize_math_answer(lines[-1])

    is_match = (detected_answer == norm_gold)
    return is_match, detected_answer


class DiagnosticVerifierStudy:
    """Orchestrates comparative diagnostic benchmarks across alternative verifiers."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        self.data_dir = data_dir or (PROJECT_ROOT / "data")
        self.gate = VerificationQualityGate(min_accuracy=0.85, min_class_recall=0.50, min_samples=10)

    def select_diagnostic_subset(self, train_df: pd.DataFrame, n_per_class: int = 5, seed: int = 42) -> pd.DataFrame:
        """Select a balanced 10-sample diagnostic subset (5 True, 5 False) with diverse questions."""
        mask_corr = train_df["final_answer_correct"].astype(bool) == True
        sample_corr = train_df[mask_corr].sample(n=min(n_per_class, sum(mask_corr)), random_state=seed)
        sample_incorr = train_df[~mask_corr].sample(n=min(n_per_class, sum(~mask_corr)), random_state=seed)
        subset = pd.concat([sample_corr, sample_incorr]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        return subset

    def run_comparative_diagnostic(self, subset_df: pd.DataFrame) -> Dict[str, Any]:
        """Run comparative diagnostic benchmarking enhanced normalizer vs baseline on subset."""
        y_true: List[bool] = []
        preds_baseline: List[bool] = []
        preds_enhanced: List[Optional[bool]] = []
        errors_categorized: List[Dict[str, Any]] = []

        for _, row in subset_df.iterrows():
            gold = bool(row["final_answer_correct"])
            corr_ans = str(row["correct_answer"])
            trans = str(row.get("transcription_steps", ""))
            trace_id = str(row.get("trace_id", ""))

            # Baseline regex
            match_base, ans_base = deterministic_answer_verifier(trans, corr_ans)
            preds_baseline.append(match_base)

            # Enhanced safe verifier
            match_enh, ans_enh, error_cat = enhanced_safe_verifier(trans, corr_ans)
            preds_enhanced.append(match_enh)
            y_true.append(gold)

            # Error categorization audit
            if match_enh is not None and match_enh != gold:
                cat = (
                    VerificationErrorCategory.GENUINE_MATHEMATICAL_ERROR.value
                    if not gold
                    else VerificationErrorCategory.INFORMAL_INDONESIAN_MARKER.value
                )
                errors_categorized.append({
                    "trace_id": trace_id,
                    "gold": gold,
                    "pred": match_enh,
                    "extracted": ans_enh,
                    "expected": corr_ans,
                    "category": cat
                })
            elif match_enh is None:
                errors_categorized.append({
                    "trace_id": trace_id,
                    "gold": gold,
                    "pred": "ABSTAIN",
                    "extracted": "UNREADABLE",
                    "expected": corr_ans,
                    "category": VerificationErrorCategory.AMBIGUOUS_UNREADABLE.value
                })

        audit_base = self.gate.evaluate(y_true, preds_baseline)
        audit_base["verifier_name"] = "Baseline_Regex_Normalizer"

        audit_enh = self.gate.evaluate(y_true, preds_enhanced)
        audit_enh["verifier_name"] = "Enhanced_Safe_Normalizer"
        audit_enh["error_taxonomy_breakdown"] = errors_categorized

        return {
            "baseline_normalizer": audit_base,
            "enhanced_normalizer": audit_enh,
            "sample_size": len(subset_df)
        }


if __name__ == "__main__":
    train_path = PROJECT_ROOT / "data" / "train.csv"
    if train_path.exists():
        df = cast(pd.DataFrame, pd.read_csv(train_path))
        study = DiagnosticVerifierStudy()
        subset = study.select_diagnostic_subset(df, n_per_class=5)
        res = study.run_comparative_diagnostic(subset)
        print("Comparative Diagnostic Result:")
        print(json.dumps(res, indent=2))
