"""Mathematical Transcription Normalizer and Conservative Answer Verifier.

Standardizes mathematical notation, LaTeX symbols, Indonesian decimal commas,
and step delimiters from OCR outputs without erasing student procedural errors.
Provides conservative answer extraction and equivalence checks:
- Unicode whitespace, minus signs, and equivalent decimal formatting.
- Recognizes Indonesian answer labels (Jawaban, Jadi, Maka, Kesimpulan) without requiring boxes.
- Preserves negative signs, fractions, units, and operators.
- Never infers correctness from the presence of a conclusion marker alone.
- Returns explicit UNCERTAIN/ABSTAIN when an answer cannot be reliably extracted.
"""
from fractions import Fraction
import re
from typing import List, Optional, Tuple


class TranscriptionNormalizer:
    """Normalizes raw handwriting OCR text and extracts/verifies mathematical conclusions."""

    @staticmethod
    def normalize_latex(text: str) -> str:
        """Standardize LaTeX fractions, exponents, and multiplication symbols."""
        if not text:
            return ""

        s = text
        # Convert Unicode minus, en-dash, em-dash, horizontal bar to standard ASCII minus
        s = s.replace("\u2212", "-").replace("\u2013", "-").replace("\u2014", "-").replace("\u2015", "-")
        # Convert Unicode non-breaking spaces and ideographic spaces
        s = s.replace("\u00a0", " ").replace("\u200b", "").replace("\u3000", " ")
        # Convert Unicode multiplication and division
        s = s.replace("\u00d7", "*").replace("\u00f7", "/")
        # Convert \dfrac{a}{b} and \frac{a}{b} to a/b
        s = re.sub(r"\\d?frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)
        # Convert \times and \cdot to *
        s = re.sub(r"\\times|\\cdot", "*", s)
        # Convert \div and \colon to /
        s = re.sub(r"\\div", "/", s)
        # Convert \pm to +/-
        s = re.sub(r"\\pm", "+/-", s)
        # Convert \sqrt{x} to sqrt(x)
        s = re.sub(r"\\sqrt\{([^}]+)\}", r"sqrt(\1)", s)
        # Remove LaTeX text wrappers and dollar signs
        s = s.replace("$", "").replace("\\text", "")
        return s

    @staticmethod
    def normalize_decimal_commas(text: str) -> str:
        """Convert Indonesian decimal commas (e.g. 0,5 or 3,14) to standard decimal points."""
        if not text:
            return ""
        # Match digits followed by comma and digits (avoid matching comma in lists like 1, 2, 3)
        return re.sub(r"(\d+),(\d+)", r"\1.\2", text)

    @staticmethod
    def format_step_lines(lines: List[str]) -> str:
        """Format step lines into standard L1 | ..., L2 | ... notation."""
        cleaned = [line.strip() for line in lines if line and line.strip()]
        if not cleaned:
            return "L1 | [No working steps detected]"
        return ", ".join([f"L{idx} | {line}" for idx, line in enumerate(cleaned, start=1)])

    @classmethod
    def normalize(cls, raw_text: str) -> str:
        """Full normalization pipeline for transcribed student working."""
        if not raw_text:
            return ""

        text = cls.normalize_latex(raw_text)
        text = cls.normalize_decimal_commas(text)

        # Standardize multiple spaces and operator whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @classmethod
    def clean_answer_token(cls, token: str) -> str:
        """Normalize answer token while preserving negative signs, fractions, and symbols."""
        if not token:
            return ""
        s = cls.normalize_latex(token)
        s = cls.normalize_decimal_commas(s)
        # Strip OCR line prefixes e.g. 'L6 | ', 'L1: '
        s = re.sub(r"^l\d+\s*[:|]\s*", "", s, flags=re.IGNORECASE)
        # Strip Indonesian preamble labels
        preambles = [
            r"^jawaban\s*(akhir)?\s*[:=]\s*",
            r"^jawaban\s*(akhir)?\s*",
            r"^jawab\s*[:=]\s*",
            r"^jawab\s*",
            r"^jadi\s*[:=,]?\s*",
            r"^kesimpulan(nya)?\s*[:=]?\s*",
            r"^maka\s*[:=,]?\s*",
            r"^hasil(nya)?\s*[:=]?\s*",
            r"^diperoleh\s*[:=]?\s*"
        ]
        for p in preambles:
            s = re.sub(p, "", s, flags=re.IGNORECASE)

        # If equation like "x = 3", extract right-hand side
        if "=" in s:
            s = s.split("=")[-1].strip()
        elif ":" in s:
            s = s.split(":")[-1].strip()

        # Remove surrounding whitespace, currency 'rp', and brackets
        s = re.sub(r"^\s*rp\.?\s*", "", s, flags=re.IGNORECASE)
        s = s.strip(" \t\n\r[](){}")
        # Remove internal unnecessary whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # Remove trailing .0 for whole numbers e.g. 5.0 -> 5 (only if not followed by more digits)
        s = re.sub(r"\.0+$", "", s)
        return s

    @classmethod
    def extract_final_conclusion(cls, transcription: str) -> Tuple[Optional[str], str]:
        """Conservatively extract student final conclusion.
        
        Returns:
            (extracted_answer, status):
            status is 'EXTRACTED', 'UNCERTAIN', or 'EMPTY'.
        """
        if not transcription or not transcription.strip():
            return None, "EMPTY"

        # Split into lines or step segments (do NOT split on comma, to preserve decimal commas like 0,5)
        raw_lines = [l.strip() for l in re.split(r"[\n;]|L\d+\s*[:|]", transcription) if l.strip()]
        if not raw_lines:
            return None, "EMPTY"

        # Search backward from the last line for conclusion markers
        markers = [
            r"jawaban\s*(akhir)?\s*[:=]?",
            r"jawab\s*[:=]?",
            r"jadi\b",
            r"kesimpulan\b",
            r"maka\b",
            r"hasilnya\b"
        ]
        marker_pattern = re.compile("|".join(markers), re.IGNORECASE)

        for line in reversed(raw_lines):
            if marker_pattern.search(line):
                cleaned = cls.clean_answer_token(line)
                if cleaned:
                    return cleaned, "EXTRACTED"

        # Fallback to last line if it contains an equality or a standalone answer
        last_line = raw_lines[-1]
        cleaned_last = cls.clean_answer_token(last_line)
        if cleaned_last:
            # Check if it has an answer-like shape (digit, fraction, negative number, or variable equality)
            if re.search(r"[\d]|=", cleaned_last):
                return cleaned_last, "EXTRACTED"

        return None, "UNCERTAIN"

    @classmethod
    def parse_numeric(cls, text: str) -> Optional[float]:
        """Safely parse text as a float or fraction, rejecting expressions with variables."""
        clean = cls.clean_answer_token(text).replace(" ", "")
        if not clean:
            return None
        # Strict guard: If there are letters (e.g. '2x' or '3a'), do not parse as pure float
        if re.search(r"[a-zA-Z]", clean):
            return None
        # Handle fractions like "1/2" or "-3/4" or "(1)/(2)"
        clean_fraction = clean.replace("(", "").replace(")", "")
        if "/" in clean_fraction:
            parts = clean_fraction.split("/")
            if len(parts) == 2:
                try:
                    num = float(parts[0])
                    den = float(parts[1])
                    if den != 0:
                        return num / den
                except ValueError:
                    return None
        # Handle regular float
        try:
            return float(clean)
        except ValueError:
            return None

    @classmethod
    def are_answers_equivalent(
        cls,
        student_ans: Optional[str],
        gold_ans: str,
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        """Conservatively verify if student answer matches gold answer.
        
        Guards:
        - Never equate expressions with different variables or numbers.
        - Preserves negative signs (-3 != 3).
        - Preserves fraction direction (1/2 != 2/1).
        - Returns (None, 'ABSTAIN') when student answer is None or uncertain.
        """
        if student_ans is None:
            return None, "ABSTAIN_UNCERTAIN"

        s_clean = cls.clean_answer_token(student_ans)
        g_clean = cls.clean_answer_token(gold_ans)

        if not s_clean or not g_clean:
            return None, "ABSTAIN_EMPTY"

        # 1. Exact string match (handles words, equations, letters)
        if s_clean.lower() == g_clean.lower():
            return True, "EXACT_STRING_MATCH"

        # 2. Strict variable check: if one has variables and the other doesn't, cannot match
        s_has_var = bool(re.search(r"[a-zA-Z]", s_clean))
        g_has_var = bool(re.search(r"[a-zA-Z]", g_clean))
        if s_has_var != g_has_var:
            return False, "VARIABLE_MISMATCH"

        # 3. Numeric / Fraction equivalence
        s_num = cls.parse_numeric(s_clean)
        g_num = cls.parse_numeric(g_clean)

        if s_num is not None and g_num is not None:
            if abs(s_num - g_num) < tolerance:
                return True, "NUMERIC_EQUIVALENCE"
            else:
                return False, "NUMERIC_INEQUALITY"

        # If non-numeric and strings differ, they do not match
        return False, "NON_NUMERIC_MISMATCH"
