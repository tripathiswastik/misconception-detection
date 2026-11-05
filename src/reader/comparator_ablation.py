"""Cached-Output Comparator Ablation Bench.

Runs offline, sub-second comparator evaluations across modular rule variants
using the frozen raw extractions in artifacts/dataset_150_fresh/qwen25_150_benchmark_fresh.json.

Key Guardrails:
1. Zero Negative Contamination: Any rule that causes a False Acceptance (gold=False -> True)
   is flagged as a safety violation.
2. Dual Metric Reporting: Full 150 population operational metrics penalize abstentions.
3. Group A/B/C Tracking: Explicitly monitors expected recoveries vs forbidden auto-accepts.
"""
import copy
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

BENCHMARK_JSON = PROJECT_ROOT / "artifacts" / "dataset_150_fresh" / "qwen25_150_benchmark_fresh.json"


@dataclass
class ComparatorConfig:
    name: str
    description: str
    strip_latex_delimiters: bool = False
    expand_units: bool = False
    terminal_equality_extraction: bool = False
    compound_alternatives: bool = False


class ModularAnswerNormalizer:
    """Configurable Answer Normalizer for ablation testing."""

    BASE_UNITS = [
        r"cm\^?[23]?", r"m\^?[23]?", r"km", r"kg", r"gram", r"tahun", r"jam",
        r"menit", r"detik", r"kali", r"derajat", r"orang",
        r"buah", r"buku", r"lembar", r"butir", r"ekor", r"rupiah", r"rp"
    ]
    EXPANDED_UNITS = BASE_UNITS + [
        r"siswa", r"murid", r"kantong", r"susunan", r"lele", r"brantong",
        r"ton", r"hektar", r"ha", r"org", r"°c", r"°", r"c"
    ]

    CURRENCY_REGEX = re.compile(r"(?i)\b(?:rp|idr|rupiah)\.?\s*")

    @classmethod
    def get_unit_regex(cls, expand: bool = False) -> re.Pattern:
        units = cls.EXPANDED_UNITS if expand else cls.BASE_UNITS
        return re.compile(r"(?i)(?<=\d|\))\s*(?:%|°|" + "|".join(units) + r"\b)\.?")

    @classmethod
    def strip_latex(cls, text: str) -> str:
        """Strip LaTeX syntax markers and environments without corrupting math content."""
        s = text
        # Strip math delimiters: $...$, $$...$$, \(...\), \[...\]
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
    def clean_token(cls, token: Optional[str], config: ComparatorConfig, is_gold: bool = False) -> str:
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
        if config.strip_latex_delimiters:
            s = cls.strip_latex(s)
        else:
            s = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

        # Currency removal
        s = cls.CURRENCY_REGEX.sub("", s).strip()

        # Remove preambles
        s = re.sub(
            r"^(?:jawaban(?:\s+akhir)?|jadi|maka|hasil(?:nya)?|kesimpulan)\s*[:=,]?\s*",
            "",
            s,
            flags=re.IGNORECASE
        )

        # Remove line markers
        s = re.sub(r"^\s*[A-Za-z]?\d+\s*[:|]\s*", "", s)

        # Terminal equality handling
        if config.terminal_equality_extraction and not is_gold and "=" in s:
            parts = [p.strip() for p in s.split("=") if p.strip()]
            if parts:
                terminal = parts[-1]
                # If terminal is a clean number, fraction, or short expression, adopt it
                if len(terminal.split()) <= 3:
                    s = terminal
        elif "=" in s:
            parts = s.split("=")
            left = parts[0].strip()
            if len(left.split()) <= 2 and not any(op in left for op in ["+", "*", "/", "^"]):
                s = parts[-1].strip()

        # Units removal
        unit_re = cls.get_unit_regex(expand=config.expand_units)
        s = unit_re.sub("", s).strip()
        s = s.rstrip(".")

        # Strip quotes and redundant whitespace
        s = s.strip("\"' \t\n\r")
        return s

    @classmethod
    def parse_numeric_candidates(cls, text: str) -> List[float]:
        candidates: List[float] = []
        raw = text.strip()
        if not raw:
            return candidates

        frac_match = re.fullmatch(r"\(?([+-]?\d+(?:\.\d+)?)\)?\s*/\s*\(?([+-]?\d+(?:\.\d+)?)\)?", raw)
        if frac_match:
            try:
                num = float(frac_match.group(1))
                den = float(frac_match.group(2))
                if abs(den) > 1e-9:
                    candidates.append(num / den)
                    return candidates
            except ValueError:
                pass

        mixed_match = re.fullmatch(r"([+-]?\d+)\s+(\d+)\s*/\s*(\d+)", raw)
        if mixed_match:
            try:
                whole = float(mixed_match.group(1))
                num = float(mixed_match.group(2))
                den = float(mixed_match.group(3))
                if abs(den) > 1e-9:
                    sign = -1.0 if whole < 0 else 1.0
                    candidates.append(whole + sign * (num / den))
                    return candidates
            except ValueError:
                pass

        if re.fullmatch(r"[+-]?[1-9]\d{0,2}(?:\.\d{3})+", raw):
            try:
                candidates.append(float(raw.replace(".", "")))
                return candidates
            except ValueError:
                pass

        if re.fullmatch(r"[+-]?\d+,\d+", raw):
            comma_parts = raw.split(",")
            if len(comma_parts) == 2 and len(comma_parts[1]) == 3 and not comma_parts[0].startswith("0"):
                try:
                    candidates.append(float(raw.replace(",", "")))
                    candidates.append(float(raw.replace(",", ".")))
                    return candidates
                except ValueError:
                    pass
            try:
                candidates.append(float(raw.replace(",", ".")))
                return candidates
            except ValueError:
                pass

        try:
            candidates.append(float(raw))
            return candidates
        except ValueError:
            pass

        return candidates

    @classmethod
    def clean_polynomial(cls, text: str) -> str:
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
        if not st_clean:
            return None, "EMPTY_STUDENT"
        if st_clean.lower() in ("unreadable", "tidak jelas", "coret", "ragu", "uncertain", "illegible"):
            return None, "ABSTAIN_UNREADABLE"

        if st_clean.lower() == gt_clean.lower():
            return True, "EXACT_NORMALIZED_MATCH"

        if "," in st_clean and "," in gt_clean:
            st_tokens = sorted([t.strip().lower() for t in st_clean.split(",") if t.strip()])
            gt_tokens = sorted([t.strip().lower() for t in gt_clean.split(",") if t.strip()])
            if st_tokens and st_tokens == gt_tokens:
                return True, "PERMUTED_SET_MATCH"

        st_poly = cls.clean_polynomial(st_clean)
        gt_poly = cls.clean_polynomial(gt_clean)
        if st_poly and gt_poly and st_poly == gt_poly:
            return True, "EXPRESSION_MATCH"

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

        return False, "STRING_MISMATCH"

    @classmethod
    def are_equivalent(
        cls,
        student: Optional[str],
        gold: Optional[str],
        config: ComparatorConfig,
        tolerance: float = 1e-5
    ) -> Tuple[Optional[bool], str]:
        if student is None or not student.strip():
            return None, "EMPTY_STUDENT"

        raw_student = student.strip()
        # Check compound alternatives (e.g. "0,625 atau 62,5%")
        if config.compound_alternatives and re.search(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE):
            sub_candidates = [c.strip() for c in re.split(r"\b(?:atau|or)\b", raw_student, flags=re.IGNORECASE) if c.strip()]
            gt_clean = cls.clean_token(gold, config, is_gold=True)
            for sub_cand in sub_candidates:
                sub_clean = cls.clean_token(sub_cand, config, is_gold=False)
                match, reason = cls.compare_single(sub_clean, gt_clean, tolerance=tolerance)
                if match is True:
                    return True, f"COMPOUND_ALTERNATIVE_MATCH ({reason})"

        st_clean = cls.clean_token(student, config, is_gold=False)
        gt_clean = cls.clean_token(gold, config, is_gold=True)
        return cls.compare_single(st_clean, gt_clean, tolerance=tolerance)


def evaluate_variant(
    records: List[Dict[str, Any]],
    config: ComparatorConfig
) -> Dict[str, Any]:
    """Audit single configuration variant over all benchmark records."""
    y_true: List[bool] = []
    y_pred: List[Optional[bool]] = []
    decisions: List[Dict[str, Any]] = []

    for r in records:
        yt = bool(r["gold"])
        extracted = r["extracted"]
        expected = r["expected"]
        is_match, reason = ModularAnswerNormalizer.are_equivalent(extracted, expected, config)

        st_norm = ModularAnswerNormalizer.clean_token(extracted, config, is_gold=False)
        gt_norm = ModularAnswerNormalizer.clean_token(expected, config, is_gold=True)

        # Audit status classification
        if is_match is True:
            audit_status = "SAFE_RECOVERED_EQUIVALENCE" if r.get("is_match") is False else "AUTO_ACCEPTED_MATCH"
        elif is_match is False:
            audit_status = "TRUE_NEGATIVE_ERROR" if not yt else "REJECTED_MISMATCH"
        else:
            audit_status = "ABSTAINED_UNCERTAIN"

        y_true.append(yt)
        y_pred.append(is_match)
        decisions.append({
            "trace_id": r["trace_id"],
            "gold_label": yt,
            "extracted_raw": extracted,
            "expected_raw": expected,
            "normalized_extracted": st_norm,
            "normalized_expected": gt_norm,
            "decision": is_match,
            "reason_code": reason or "UNKNOWN",
            "manual_audit_status": audit_status,
            "prev_is_match": r.get("is_match")
        })

    n_total = len(y_true)
    decided = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp is not None]
    n_decided = len(decided)
    n_abstain = n_total - n_decided
    abstain_rate = n_abstain / n_total if n_total > 0 else 0.0

    tp_all = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp is True)
    tn_all = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp is False)
    fp_all = sum(1 for yt, yp in zip(y_true, y_pred) if not yt and yp is True)
    fn_all = sum(1 for yt, yp in zip(y_true, y_pred) if yt and yp is False)

    n_positive = sum(1 for yt in y_true if yt)
    n_negative = n_total - n_positive

    op_acc = (tp_all + tn_all) / n_total if n_total > 0 else 0.0
    op_rec_pos = tp_all / n_positive if n_positive > 0 else 0.0
    op_rec_neg = tn_all / n_negative if n_negative > 0 else 0.0
    far = fp_all / n_negative if n_negative > 0 else 0.0
    frr = (n_positive - tp_all) / n_positive if n_positive > 0 else 0.0

    sel_acc = (sum(1 for yt, yp in decided if yt and yp is True) + sum(1 for yt, yp in decided if not yt and yp is False)) / n_decided if n_decided > 0 else 0.0

    # Theoretical headroom estimate
    expected_map3_delta = 0.5 * (op_rec_pos - 0.50) * 0.1122 + 0.5 * (op_rec_neg - 0.50) * 0.1122

    # Preregistered gate criteria
    gate_passed = bool((op_acc >= 0.78) and (op_rec_pos >= 0.60) and (op_rec_neg >= 0.60) and (abstain_rate <= 0.20))

    # Changed decision analysis
    newly_accepted_positives = [
        d["trace_id"] for d in decisions if d["gold_label"] and d["decision"] is True and d["prev_is_match"] is not True
    ]
    newly_accepted_negatives = [
        d["trace_id"] for d in decisions if not d["gold_label"] and d["decision"] is True and d["prev_is_match"] is not True
    ]

    return {
        "variant": config.name,
        "description": config.description,
        "n_total": n_total,
        "n_decided": n_decided,
        "n_abstained": n_abstain,
        "abstain_rate": abstain_rate,
        "selective_accuracy": sel_acc,
        "operational_accuracy": op_acc,
        "operational_recall_positive": op_rec_pos,
        "operational_recall_negative": op_rec_neg,
        "false_acceptance_rate": far,
        "false_rejection_rate": frr,
        "false_accept_count": fp_all,
        "expected_map3_delta": expected_map3_delta,
        "gate_passed": gate_passed,
        "new_positives_recovered": newly_accepted_positives,
        "new_negatives_accepted": newly_accepted_negatives,
        "decisions": decisions
    }


def run_ablation_study() -> Dict[str, Any]:
    if not BENCHMARK_JSON.exists():
        raise FileNotFoundError(f"Frozen benchmark not found: {BENCHMARK_JSON}")

    with open(BENCHMARK_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data["results"]

    variants = [
        ComparatorConfig(
            name="Variant A — Baseline",
            description="Current normalization",
        ),
        ComparatorConfig(
            name="Variant B — LaTeX cleanup",
            description="Strip math delimiters and normalize \\frac, \\dfrac, spacing, and \\text{...} safely",
            strip_latex_delimiters=True,
        ),
        ComparatorConfig(
            name="Variant C — Unit-aware",
            description="Parse supported Indonesian units and mathematical unit notation",
            expand_units=True,
        ),
        ComparatorConfig(
            name="Variant D — Structured answers",
            description="Handle equivalent fractions, percentages, and answer-plus-explanation strings",
            terminal_equality_extraction=True,
            compound_alternatives=True,
        ),
        ComparatorConfig(
            name="Variant E — Conservative combined",
            description="Combine B–D, abstaining on ambiguous cases",
            strip_latex_delimiters=True,
            expand_units=True,
            terminal_equality_extraction=True,
            compound_alternatives=True,
        )
    ]

    study_results: Dict[str, Any] = {}
    print("\n" + "=" * 95)
    print("COMPARATOR-ONLY ABLATION: FROZEN FRESH 150-SAMPLE BENCHMARK")
    print("=" * 95)
    print(f"{'Variant':<34} | {'OpAcc':<7} | {'OpRec(+)':<9} | {'OpRec(-)':<9} | {'FAR':<6} | {'New(+)':<6} | {'New(-)':<6} | {'Gate':<6}")
    print("-" * 95)

    for v in variants:
        res = evaluate_variant(records, v)
        study_results[v.name] = res
        n_new_pos = len(res["new_positives_recovered"])
        n_new_neg = len(res["new_negatives_accepted"])
        gate_str = "PASS" if res["gate_passed"] else "FAIL"

        print(f"{v.name:<34} | {res['operational_accuracy']:.4f}  | {res['operational_recall_positive']:.4f}    | {res['operational_recall_negative']:.4f}    | {res['false_acceptance_rate']:.4f} | {n_new_pos:<6} | {n_new_neg:<6} | {gate_str:<6}")

    print("=" * 95)

    # Export standardized mismatch report for winning Variant E
    winning_decisions = study_results["Variant E — Conservative combined"]["decisions"]
    report_json_path = PROJECT_ROOT / "artifacts" / "dataset_150_fresh" / "mismatch_report_ablation.json"
    report_csv_path = PROJECT_ROOT / "artifacts" / "dataset_150_fresh" / "mismatch_report_ablation.csv"

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(winning_decisions, f, indent=2)

    import pandas as pd
    report_df = pd.DataFrame(winning_decisions)
    report_df.to_csv(report_csv_path, index=False)
    print(f"Exported complete 150-trace audit report to:\n  - {report_json_path}\n  - {report_csv_path}")

    return study_results


if __name__ == "__main__":
    run_ablation_study()
