"""Domain Constraint and Curriculum Topic/Grade Masking Engine.

Enforces pedagogical domain boundaries on model class probability distributions:
1. Topic Affinity: Boosts misconceptions conceptually relevant to the mathematical topic
   while softly attenuating out-of-domain misconceptions (e.g. Fraction errors on Geometry).
2. Grade Band Calibration: Filters out misconceptions that do not belong to the student's grade tier.
3. Cross-Cutting Invariance: Preserves universal misconceptions (NONE, CAL-*, WOP-*, ANS-*, VIS-*, UNT-*).
"""
import re
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from src.data.taxonomy import LABEL_NONE, TaxonomyManager


# Universally applicable misconceptions across all topics
UNIVERSAL_PREFIXES = ("CAL-", "WOP-", "ANS-", "VIS-", "UNT-", "OTH-")

# Topic keywords to taxonomy prefix mapping
TOPIC_KEYWORD_MAP = {
    "pecahan": ["FRA-", "FRM-", "Dec-", "RAT-"],
    "fraksi": ["FRA-", "FRM-"],
    "bilangan bulat": ["INT-", "OPS-", "OOO-"],
    "negatif": ["INT-"],
    "aljabar": ["ALG-", "EQN-", "EXP-", "FUN-"],
    "persamaan": ["EQN-", "ALG-"],
    "pertidaksamaan": ["EQN-", "ALG-"],
    "sistem persamaan": ["EQN-", "ALG-"],
    "fungsi": ["FUN-", "ALG-"],
    "relasi": ["FUN-", "ALG-"],
    "pangkat": ["EXP-"],
    "akar": ["EXP-"],
    "eksponen": ["EXP-"],
    "kpk": ["KPF-", "OPS-"],
    "fpb": ["KPF-", "OPS-"],
    "faktor": ["KPF-"],
    "kelipatan": ["KPF-"],
    "desimal": ["Dec-", "PV-", "FRA-"],
    "nilai tempat": ["PV-", "OPS-"],
    "sudut": ["GEO-"],
    "segitiga": ["GEO-", "MEA-"],
    "lingkaran": ["GEO-", "MEA-"],
    "pythagoras": ["GEO-", "MEA-", "EXP-"],
    "bangun datar": ["GEO-", "MEA-"],
    "bangun ruang": ["GEO-", "MEA-"],
    "dimensi": ["GEO-", "MEA-"],
    "luas": ["MEA-", "GEO-", "UNT-"],
    "keliling": ["MEA-", "GEO-", "UNT-"],
    "volume": ["MEA-", "GEO-", "UNT-"],
    "permukaan": ["MEA-", "GEO-", "UNT-"],
    "statistika": ["STA-"],
    "rata-rata": ["STA-", "CAL-"],
    "mean": ["STA-"],
    "median": ["STA-"],
    "modus": ["STA-"],
    "diagram": ["STA-", "VIS-"],
    "data": ["STA-"],
    "peluang": ["PRO-", "RAT-"],
    "kombinasi": ["PRO-"],
    "permutasi": ["PRO-"],
    "perbandingan": ["RAT-", "FRA-"],
    "skala": ["RAT-", "MEA-", "UNT-"],
    "aritmetika sosial": ["RAT-", "CAL-", "WOP-"],
    "pola bilangan": ["SEQ-", "ALG-"],
    "barisan": ["SEQ-", "ALG-"],
    "deret": ["SEQ-", "ALG-"]
}


class DomainConstraintEngine:
    """Soft domain masking and curriculum compatibility calibrator."""

    def __init__(self, taxonomy: Optional[TaxonomyManager] = None):
        self.taxonomy = taxonomy or TaxonomyManager()
        self.empirical_topic_codes: Dict[str, Set[str]] = {}
        self.grade_ranges: Dict[str, Tuple[int, int]] = {}
        self._build_grade_ranges()

    def _build_grade_ranges(self):
        """Parse typical_grade_band from taxonomy definitions."""
        for code in self.taxonomy.all_codes(include_none=False):
            item = self.taxonomy.get(code)
            if item and item.typical_grade_band:
                band_str = item.typical_grade_band.replace("–", "-").strip()
                match = re.match(r"(\d+)(?:\s*-\s*(\d+))?", band_str)
                if match:
                    min_g = int(match.group(1))
                    max_g = int(match.group(2)) if match.group(2) else min_g
                    self.grade_ranges[code] = (min_g, max_g)
                else:
                    self.grade_ranges[code] = (4, 12)
            else:
                self.grade_ranges[code] = (4, 12)

    def fit_empirical(self, train_df: pd.DataFrame) -> "DomainConstraintEngine":
        """Learn empirical topic-to-code occurrences from labeled training traces."""
        for _, row in train_df.iterrows():
            topic = str(row["topic_name"]).strip().lower()
            code = str(row["root_code"]).strip()
            if topic not in self.empirical_topic_codes:
                self.empirical_topic_codes[topic] = set()
            self.empirical_topic_codes[topic].add(code)
        return self

    def compute_affinity(self, code: str, topic_name: str, grade: int) -> float:
        """Compute compatibility multiplier in [0.05, 1.0] for a given (code, topic, grade)."""
        # NONE is always 100% valid
        if code == LABEL_NONE:
            return 1.0

        topic_clean = topic_name.strip().lower()

        # Universal codes are always allowed with at least 0.8 affinity
        if any(code.startswith(prefix) for prefix in UNIVERSAL_PREFIXES):
            base_affinity = 0.85
        else:
            base_affinity = 0.10

        # Check empirical occurrence in training set
        if topic_clean in self.empirical_topic_codes:
            if code in self.empirical_topic_codes[topic_clean]:
                base_affinity = 1.0

        # Check keyword-based semantic matching
        matched_prefixes = []
        for kw, prefixes in TOPIC_KEYWORD_MAP.items():
            if kw in topic_clean:
                matched_prefixes.extend(prefixes)

        if matched_prefixes:
            if any(code.startswith(p) for p in matched_prefixes):
                base_affinity = max(base_affinity, 1.0)
            else:
                # If topic has specific prefixes, slightly dampen unrelated specific prefixes
                if not any(code.startswith(prefix) for prefix in UNIVERSAL_PREFIXES):
                    base_affinity = min(base_affinity, 0.08)

        # Grade-band penalty
        if code in self.grade_ranges:
            min_g, max_g = self.grade_ranges[code]
            if grade < min_g - 2 or grade > max_g + 3:
                base_affinity *= 0.25

        return max(0.05, base_affinity)

    def apply_mask_batch(
        self,
        probabilities: np.ndarray,
        classes: List[str],
        topics: List[str],
        grades: List[int],
        damping: float = 0.85
    ) -> np.ndarray:
        """Apply domain compatibility masking across batch probability matrix.
        
        Args:
            probabilities: (N, C) probability matrix
            classes: list of C class names
            topics: list of N topic names
            grades: list of N grade integers
            damping: exponent control (1.0 = full masking, 0.0 = no effect)
            
        Returns:
            Calibrated (N, C) probability matrix summing to 1.0 per row.
        """
        calibrated = np.zeros_like(probabilities)
        n_samples = len(probabilities)

        for i in range(n_samples):
            topic = topics[i]
            grade = grades[i]
            row_probs = probabilities[i]

            affinities = np.array([
                self.compute_affinity(classes[c_idx], topic, grade) ** damping
                for c_idx in range(len(classes))
            ])

            masked = row_probs * affinities
            prob_sum = np.sum(masked)
            if prob_sum > 1e-12:
                calibrated[i] = masked / prob_sum
            else:
                calibrated[i] = row_probs

        return calibrated

    def apply_mask_single(
        self,
        probabilities: np.ndarray,
        classes: List[str],
        topic: str,
        grade: int,
        damping: float = 0.85
    ) -> np.ndarray:
        """Apply domain compatibility masking to a single 1D probability/score vector."""
        batch_input = probabilities.reshape(1, -1)
        masked_batch = self.apply_mask_batch(batch_input, classes, [topic], [grade], damping=damping)
        return masked_batch[0]

