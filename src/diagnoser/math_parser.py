"""Lightweight Mathematical Feature Extractor and Symbolic Pattern Parser.

Extracts high-signal arithmetic, procedural, and symbolic misconception signals from:
- Question text (Indonesian math problems)
- Student handwritten transcription steps
- Official answer key

Engineered to directly discriminate top taxonomy misconceptions without heavy neural inference:
- FRA-01: Adding numerators and denominators directly (a/b + c/d = (a+c)/(b+d))
- FRA-03: Dividing fractions without inverting (a/b / c/d = (a/c)/(b/d) or a/b * c/d)
- INT-04: Sign handling in integer +/- (e.g. -40 - 20 = -20 or dropping minus)
- INT-02: Multiplication of negatives (neg * neg = neg)
- EXP-01: Exponent as base * power (e.g. 3^4 = 12)
- Clean / NONE: Student working directly matches official answer
"""
import re
from typing import Dict, List, Optional, Set, Tuple, Union
import pandas as pd


# Regex patterns for mathematical tokens
RE_FRACTION = re.compile(r"(\d+)\s*/\s*(\d+)|\\dfrac\{(\d+)\}\{(\d+)\}")
RE_EXPONENT = re.compile(r"(\d+)\s*\^\s*(\d+)|(\d+)\s*pangkat\s*(\d+)|(\d+)\s*kuadrat", re.IGNORECASE)
RE_NEGATIVE_NUM = re.compile(r"-\s*\d+")
RE_EQUALS = re.compile(r"=")
RE_DOUBLE_NEG = re.compile(r"--|-\s*-|-\s*\(\s*-")


class MathFeatureExtractor:
    """Extracts procedural, arithmetic, and symbolic pattern tokens for math misconception diagnosis."""

    @staticmethod
    def extract_fractions(text: str) -> List[Tuple[int, int]]:
        """Extract all (numerator, denominator) tuples from text."""
        fractions = []
        for match in RE_FRACTION.finditer(text):
            groups = match.groups()
            if groups[0] is not None and groups[1] is not None:
                n, d = int(groups[0]), int(groups[1])
                if d != 0:
                    fractions.append((n, d))
            elif groups[2] is not None and groups[3] is not None:
                n, d = int(groups[2]), int(groups[3])
                if d != 0:
                    fractions.append((n, d))
        return fractions

    @staticmethod
    def check_fraction_addition_error(q_text: str, working_text: str) -> bool:
        """Detect FRA-01: student adds numerators and denominators: a/b + c/d = (a+c)/(b+d)."""
        q_fracs = MathFeatureExtractor.extract_fractions(q_text)
        w_fracs = MathFeatureExtractor.extract_fractions(working_text)

        if len(q_fracs) >= 2 and len(w_fracs) >= 1:
            (n1, d1), (n2, d2) = q_fracs[0], q_fracs[1]
            target_n = n1 + n2
            target_d = d1 + d2
            for wn, wd in w_fracs:
                if wn == target_n and wd == target_d:
                    return True
        return False

    @staticmethod
    def check_sign_flip_error(key: str, working_text: str) -> bool:
        """Detect INT-04: student arrived at answer with reversed sign (e.g. key is 5, student got -5)."""
        key_clean = key.strip()
        if not key_clean:
            return False

        try:
            val = float(key_clean)
            flipped = -val
            flipped_str = f"{int(flipped)}" if flipped.is_integer() else f"{flipped}"
            
            # Use negative lookbehind so "-5" doesn't match positive "5"
            pos_pattern = rf"(?<![-\d]){re.escape(key_clean)}(?!\d)"
            flip_pattern = rf"(?<!\d){re.escape(flipped_str)}(?!\d)"

            if re.search(flip_pattern, working_text) and not re.search(pos_pattern, working_text):
                return True
        except ValueError:
            pass

        return False

    @staticmethod
    def check_exponent_multiplication_error(q_text: str, working_text: str) -> bool:
        """Detect EXP-01: student computes base * exponent instead of base^exponent (e.g. 2^3 = 6)."""
        matches = RE_EXPONENT.findall(q_text)
        for m in matches:
            b_str = m[0] or m[2] or m[4]
            e_str = m[1] or m[3] or "2"
            if b_str and e_str:
                try:
                    b, e = int(b_str), int(e_str)
                    mult_val = str(b * e)
                    true_val = str(b ** e)
                    if mult_val in working_text and true_val not in working_text:
                        return True
                except (ValueError, OverflowError):
                    continue
        return False

    @classmethod
    def extract_feature_tokens(
        cls,
        question_text: str,
        transcription_steps: str,
        correct_answer: str,
        topic_name: str = "",
        grade: Union[int, str] = ""
    ) -> List[str]:
        """Generate discrete mathematical feature tokens to append to the text representation."""
        tokens: List[str] = []
        q_text = question_text or ""
        w_text = transcription_steps or ""
        key = correct_answer.strip() if correct_answer else ""

        # 1. Answer Key Agreement (Strongest predictor of NONE vs Misconception)
        if key and key in w_text:
            tokens.append("MATH_MATCHES_KEY_EXACT")
        else:
            tokens.append("MATH_MISMATCHES_KEY")

        # 2. Key Sign Flip
        if cls.check_sign_flip_error(key, w_text):
            tokens.append("MATH_KEY_SIGN_FLIPPED")

        # 3. Fraction Patterns
        q_has_frac = bool(RE_FRACTION.search(q_text))
        w_has_frac = bool(RE_FRACTION.search(w_text))
        if q_has_frac:
            tokens.append("HAS_FRACTION_QUESTION")
        if w_has_frac:
            tokens.append("HAS_FRACTION_WORKING")

        if cls.check_fraction_addition_error(q_text, w_text):
            tokens.append("MATH_FRA_ADD_TOPS_BOTTOMS")

        # 4. Integer & Negative Sign Patterns
        q_has_neg = bool(RE_NEGATIVE_NUM.search(q_text))
        w_has_neg = bool(RE_NEGATIVE_NUM.search(w_text))
        if q_has_neg:
            tokens.append("HAS_NEGATIVE_QUESTION")
        if w_has_neg:
            tokens.append("HAS_NEGATIVE_WORKING")

        if RE_DOUBLE_NEG.search(q_text) or RE_DOUBLE_NEG.search(w_text):
            tokens.append("MATH_HAS_DOUBLE_NEGATIVE")

        # 5. Exponent Patterns
        if RE_EXPONENT.search(q_text) or "pangkat" in q_text.lower():
            tokens.append("HAS_EXPONENT_QUESTION")
            if cls.check_exponent_multiplication_error(q_text, w_text):
                tokens.append("MATH_EXP_BASE_TIMES_POWER")

        # 6. Working Structure and Equations
        if RE_EQUALS.search(w_text):
            tokens.append("WORKING_HAS_EQUALS")

        # Step count segmentation (L1 | ..., L2 | ...)
        step_lines = [line.strip() for line in w_text.split("L") if line.strip() and "|" in line]
        num_steps = len(step_lines) if step_lines else 1
        if num_steps == 1:
            tokens.append("WORKING_SINGLE_STEP")
        elif num_steps == 2:
            tokens.append("WORKING_TWO_STEPS")
        else:
            tokens.append("WORKING_MULTI_STEPS")

        return tokens


def build_enhanced_feature_text(row: pd.Series) -> str:
    """Compose rich structured text combining context, symbolic math tokens, and student working."""
    raw_grade = row.get("grade")
    grade = str(raw_grade) if raw_grade is not None else ""
    raw_topic = row.get("topic_name")
    topic = str(raw_topic) if raw_topic is not None else ""
    raw_q = row.get("question_text")
    q_text = str(raw_q) if raw_q is not None else ""
    raw_key = row.get("correct_answer")
    key = str(raw_key) if raw_key is not None else ""
    raw_steps = row.get("transcription_steps")
    steps = str(raw_steps) if raw_steps is not None else ""

    math_tokens = MathFeatureExtractor.extract_feature_tokens(
        question_text=q_text,
        transcription_steps=steps,
        correct_answer=key,
        topic_name=topic,
        grade=grade
    )
    tokens_str = " ".join(math_tokens)

    return (
        f"GRADE_{grade} TOPIC_{topic} "
        f"KEY_{key} {tokens_str} "
        f"QUESTION: {q_text} "
        f"WORKING: {steps}"
    )
