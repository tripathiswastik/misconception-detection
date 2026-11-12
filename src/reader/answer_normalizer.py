"""Robust Indonesian and Mathematical Answer Normalizer for Misconception Detection.

Implements reference-aware normalization:
1. LaTeX environment and delimiter stripping ($...$, dfrac, hat, text, left/right).
2. Indonesian thousands vs decimal grouping (1.250.000 -> 1250000, 0,125 -> 0.125).
3. Expanded Indonesian unit noun normalization (siswa, kantong, susunan, org, ton, °C).
4. Terminal equality extraction (FPB = 2x2x2 = 8 -> 8).
5. Compound alternative evaluation ('atau' / 'or').
6. Polynomial canonicalization and permuted list matching.
7. Strict false-accept guards (e.g. 10 != 10.000).
"""
import re
from typing import List, Optional, Set, Tuple


class AnswerNormalizer:
    """Robust normalizer for comparing student handwritten answers against gold answers."""

    BASE_UNITS = [
        r"cm\^?[23]?", r"m\^?[23]?", r"km", r"kg", r"gram", r"tahun", r"jam",
        r"menit", r"detik", r"kali", r"derajat", r"orang",
        r"buah", r"buku", r"lembar", r"butir", r"ekor", r"rupiah", r"rp"
    ]
    EXPANDED_UNITS = BASE_UNITS + [
        r"siswa", r"murid", r"kantong", r"susunan", r"lele", r"brantong",
        r"ton", r"hektar", r"ha", r"org", r"°c", r"°", r"c"
    ]

    UNITS_REGEX = re.compile(
        r"(?i)(?<=\d|\))\s*(?:%|°|" + "|".join(EXPANDED_UNITS) + r"\b)\.?"
    )
    CURRENCY_REGEX = re.compile(r"(?i)\b(?:rp|idr|rupiah)\.?\s*")

    @classmethod
    def strip_latex(cls, text: str) -> str:
        """Strip LaTeX syntax markers and environments without corrupting mathematical content."""
        s = text
        # Strip outer math delimiters: $...$, $$...$$, \(...\), \[...\]
        s = re.sub(r"^\s*(\$\$|\$|\\\(|\\\[)\s*", "", s)
        s = re.sub(r"\s*(\$\$|\$|\\\)|\\\])\s*$", "", s)
        s = s.replace("$", "")

        # Unify fractions: \dfrac -> \frac
        s = s.replace(r"\dfrac", r"\frac")
        s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

        # Strip font and text formatting: \text{...}, \mathrm{...}, \mathbf{...}
        s = re.sub(r"\\(?:text|mathrm|mathbf)\{([^{}]+)\}", r"\1", s)

        # Strip accents like \hat{y} -> y, \hat{x} -> x
        s = re.sub(r"\\hat\{([^{}]+)\}", r"\1", s)
        s = re.sub(r"\\hat\s*([a-zA-Z])", r"\1", s)

        # Strip \left and \right delimiters
        s = s.replace(r"\left", "").replace(r"\right", "")

        # Normalize operators
        s = s.replace(r"\times", "*").replace(r"\cdot", "*")
        s = s.replace(r"\pi", "pi")
        s = s.replace("\\ ", " ").replace(r"\,", " ").replace(r"\;", " ")

        return s.strip()

    @classmethod
    def clean(cls, token: Optional[str], is_gold: bool = False) -> str:
        """Perform conservative, unambiguous normalization on answer string."""
        if token is None:
            return ""
        s = token.strip()
        if not s:
            return ""

        # Normalize unicode dashes, spaces, and superscripts
        s = (
            s.replace("\u2212", "-")
            .replace("\u2013", "-")
            .replace("\u2014", "-")
            .replace("\u00a0", " ")
            .replace("\u202f", " ")
            .replace("²", "^2")
            .replace("³", "^3")
            .replace("√", "sqrt")
            .replace("ŷ", "y")
        )

        # LaTeX fraction conversion & LaTeX delimiter stripping
        s = cls.strip_latex(s)

        # Remove currency prefixes
        s = cls.CURRENCY_REGEX.sub("", s).strip()

        # Remove preambles (Jawaban, Jadi, Maka, Kesimpulan)
        s = re.sub(
            r"^(?:jawaban(?:\s+akhir)?|jadi|maka|hasil(?:nya)?|kesimpulan)\s*[:=,]?\s*",
            "",
            s,
            flags=re.IGNORECASE
        )

        # Remove line/step markers (e.g., "L6 | ", "1: ", "A: ")
        s = re.sub(r"^\s*[A-Za-z]?\d+\s*[:|]\s*", "", s)

        # Terminal equality handling (e.g., "Luas = 60 + 216 = 276 cm²" or "FPB = 2x2x2 = 8")
        if not is_gold and "=" in s:
            parts = [p.strip() for p in s.split("=") if p.strip()]
            if parts:
                terminal = parts[-1]
                if len(terminal.split()) <= 3:
                    s = terminal
        elif "=" in s:
            parts = s.split("=")
            left = parts[0].strip()
            if len(left.split()) <= 2 and not any(op in left for op in ["+", "*", "/", "^"]):
                s = parts[-1].strip()

        # Remove units at the end of numbers
        s = cls.UNITS_REGEX.sub("", s).strip()
        s = s.rstrip(".")

        # Strip outer quotes and redundant whitespace
        s = s.strip("\"' \t\n\r")
        return s

    @classmethod
    def parse_numeric_candidates(cls, text: str) -> List[float]:
        """Parse potential numeric interpretations of text (handling Indonesian thousands vs decimals)."""
        candidates: List[float] = []
        raw = text.strip()
        if not raw:
            return candidates

        # 1. LaTeX/Unicode fraction: (a)/(b) or a/b
        frac_match = re.fullmatch(r"\(?([+-]?\d+(?:\.\d+)?)\)?\s*/\s*\(?([+-]?\d+(?:\.\d+)?)\)?", raw)
        if frac_match:
            num = float(frac_match.group(1))
            den = float(frac_match.group(2))
            if abs(den) > 1e-9:
                candidates.append(num / den)
                return candidates

        # 2. Mixed fraction: "11 39/45"
        mixed_match = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", raw)
        if mixed_match:
            whole = float(mixed_match.group(1))
            num = float(mixed_match.group(2))
            den = float(mixed_match.group(3))
            if abs(den) > 1e-9:
                sign = -1.0 if whole < 0 else 1.0
                candidates.append(whole + sign * (num / den))
                return candidates

        # 3. Indonesian thousands grouping: "1.250.000" or "10.000" (dot followed by exactly 3 digits)
        if re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+", raw):
            candidates.append(float(raw.replace(".", "")))
            return candidates

        # 4. Indonesian decimal with comma: "54,5" or "0,125"
        if re.fullmatch(r"[+-]?\d+,\d+", raw):
            comma_parts = raw.split(",")
            if len(comma_parts) == 2 and len(comma_parts[1]) == 3 and not comma_parts[0].startswith("0"):
                candidates.append(float(raw.replace(",", "")))
            candidates.append(float(raw.replace(",", ".")))
            return candidates

        # 5. Standard decimal or integer: "0.125" or "120"
        if re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", raw):
            candidates.append(float(raw))

        return candidates

    @classmethod
    def clean_polynomial(cls, text: str) -> str:
        """Canonicalize algebraic polynomial expressions by removing whitespace and unifying terms."""
        s = text.replace(" ", "").lower()
        s = s.replace("²", "^2").replace("³", "^3")
        s = s.replace("×", "*").replace("\\times", "*").replace("\\cdot", "*")
        return s

    @classmethod
    def compare_single(
        cls,
        st_clean: str,
        gt_clean: str,
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        """Compare single clean student answer against single clean gold answer."""
        if not st_clean:
            return None, "EMPTY_STUDENT"

        if st_clean.lower() in ("unreadable", "tidak jelas", "coret", "ragu", "uncertain", "illegible"):
            return None, "ABSTAIN_UNREADABLE"

        # 1. Exact string match
        if st_clean.lower() == gt_clean.lower():
            return True, "EXACT_NORMALIZED_MATCH"

        # 2. Permuted comma-separated list (e.g. "Budi, Ahmad, Cici" == "Ahmad, Budi, Cici")
        if "," in st_clean and "," in gt_clean:
            st_tokens = sorted([t.strip().lower() for t in st_clean.split(",") if t.strip()])
            gt_tokens = sorted([t.strip().lower() for t in gt_clean.split(",") if t.strip()])
            if st_tokens and st_tokens == gt_tokens:
                return True, "PERMUTED_SET_MATCH"

        # 3. Polynomial / Algebraic match
        st_poly = cls.clean_polynomial(st_clean)
        gt_poly = cls.clean_polynomial(gt_clean)
        if st_poly and gt_poly and st_poly == gt_poly:
            return True, "EXPRESSION_MATCH"

        # 4. Numeric equivalence with false-accept guards
        st_nums = cls.parse_numeric_candidates(st_clean)
        gt_nums = cls.parse_numeric_candidates(gt_clean)

        if st_nums and gt_nums:
            for s_val in st_nums:
                for g_val in gt_nums:
                    if (s_val < 0) != (g_val < 0) and abs(s_val - g_val) > tolerance:
                        continue
                    if abs(s_val - g_val) <= tolerance:
                        return True, "NUMERIC_EQUIVALENT"

            return False, "NUMERIC_DIFFERENCE"

        # 5. String mismatch fallback
        return False, "STRING_MISMATCH"

    @classmethod
    def are_equivalent(
        cls,
        student: Optional[str],
        gold: Optional[str],
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        """Determine whether student's extracted answer matches gold answer."""
        if student is None or not student.strip():
            return None, "EMPTY_STUDENT"

        raw_student = student.strip()

        # Check compound alternative answers (e.g. "0,625 atau 62,5%")
        if re.search(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE):
            sub_candidates = [c.strip() for c in re.split(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE) if c.strip()]
            gt_clean = cls.clean(gold, is_gold=True)
            for sub_cand in sub_candidates:
                sub_clean = cls.clean(sub_cand, is_gold=False)
                match, reason = cls.compare_single(sub_clean, gt_clean, tolerance=tolerance)
                if match is True:
                    return True, f"COMPOUND_ALTERNATIVE_MATCH ({reason})"

        st_clean = cls.clean(student, is_gold=False)
        gt_clean = cls.clean(gold, is_gold=True)
        return cls.compare_single(st_clean, gt_clean, tolerance=tolerance)
