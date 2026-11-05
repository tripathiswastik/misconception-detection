"""Unit tests for TaxonomyManager."""
import tempfile
from pathlib import Path
from src.data.taxonomy import TaxonomyManager, LABEL_NONE


def test_taxonomy_manager_manual_and_csv():
    # Create temporary taxonomy_v1.csv
    csv_content = """code,name,description,nature,default_severity,typical_grade_band,section
FRM-01,Adds denominators,Student adds denominators directly,procedural,major,6-8,fractions
CAL-01,Basic arithmetic slip,Simple addition or multiplication error,careless,minor,4-12,calculation
"""
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write(csv_content)
        temp_path = Path(f.name)

    try:
        mgr = TaxonomyManager(temp_path)
        assert len(mgr) == 2
        frm = mgr.get("FRM-01")
        assert frm is not None
        assert frm.name == "Adds denominators"
        assert frm.nature == "procedural"
        
        all_codes = mgr.all_codes(include_none=True)
        assert LABEL_NONE in all_codes
        assert "FRM-01" in all_codes
        assert "CAL-01" in all_codes
        assert len(all_codes) == 3

        code_to_id = mgr.code_to_id(include_none=True)
        id_to_code = mgr.id_to_code(include_none=True)
        assert id_to_code[code_to_id["FRM-01"]] == "FRM-01"
    finally:
        if temp_path.exists():
            temp_path.unlink()
