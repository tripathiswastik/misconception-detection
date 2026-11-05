"""Taxonomy Manager for K-12 Student Math Misconceptions.

Covers the 75 misconception codes defined by CoLearn's teachers plus the 'NONE' class.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv


@dataclass
class MisconceptionDef:
    code: str
    name: str
    description: str
    nature: str  # conceptual, procedural, careless, meta
    default_severity: str
    typical_grade_band: str
    section: str


# Master class label for work with no major findings
LABEL_NONE = "NONE"


class TaxonomyManager:
    """Manages misconception taxonomy definitions, code lookups, and metadata."""

    def __init__(self, csv_path: Optional[Path] = None):
        self._items: Dict[str, MisconceptionDef] = {}
        if csv_path is None:
            default_path = Path(__file__).resolve().parent.parent.parent / "data" / "taxonomy_v1.csv"
            if default_path.exists():
                csv_path = default_path
        if csv_path and Path(csv_path).exists():
            self.load_from_csv(Path(csv_path))

    def load_from_csv(self, csv_path: Path) -> None:
        """Parse taxonomy_v1.csv into memory."""
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code") or row.get("root_code") or row.get("id") or ""
                if not code:
                    continue
                self._items[code] = MisconceptionDef(
                    code=code,
                    name=row.get("name", ""),
                    description=row.get("description", ""),
                    nature=row.get("nature", "conceptual"),
                    default_severity=row.get("default_severity", "major"),
                    typical_grade_band=row.get("typical_grade_band") or row.get("grade_band", ""),
                    section=row.get("section", "")
                )

    def get(self, code: str) -> Optional[MisconceptionDef]:
        """Lookup a definition by taxonomy code."""
        return self._items.get(code)

    def all_codes(self, include_none: bool = True) -> List[str]:
        """Return sorted list of all available taxonomy codes."""
        codes = sorted(list(self._items.keys()))
        if include_none and LABEL_NONE not in codes:
            return [LABEL_NONE] + codes
        return codes

    def code_to_id(self, include_none: bool = True) -> Dict[str, int]:
        """Generate mapping from string code to integer class index."""
        codes = self.all_codes(include_none=include_none)
        return {c: i for i, c in enumerate(codes)}

    def id_to_code(self, include_none: bool = True) -> Dict[int, str]:
        """Generate mapping from integer class index to string code."""
        codes = self.all_codes(include_none=include_none)
        return {i: c for i, c in enumerate(codes)}

    def get_name(self, code: str) -> str:
        """Lookup name or return clean description for NONE."""
        if code == LABEL_NONE:
            return "Clean Working (No Misconception)"
        item = self.get(code)
        return item.name if item and item.name else f"Misconception {code}"

    def get_category(self, code: str) -> str:
        """Lookup category or section."""
        if code == LABEL_NONE:
            return "Correct"
        item = self.get(code)
        return item.section if item and item.section else "General"

    def get_all_categories(self) -> List[str]:
        """Return distinct category sections."""
        cats = sorted(list(set(item.section for item in self._items.values() if item.section)))
        return ["Correct"] + cats if "Correct" not in cats else cats

    def __len__(self) -> int:
        return len(self._items)
