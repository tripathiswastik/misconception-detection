"""Unit tests for TwoStageDiagnosticPipeline and Submission Validator."""
import tempfile
from pathlib import Path
import pandas as pd
from src.data.dataset import StudentTrace
from src.pipeline import TwoStageDiagnosticPipeline, DiagnosticResult
from src.submission import validate_submission_file


def test_two_stage_pipeline():
    train_data = pd.DataFrame([
        {
            "grade": 6,
            "topic_name": "fractions",
            "question_text": "1/3 + 1/4",
            "correct_answer": "7/12",
            "transcription_steps": "L1 | 1/3 + 1/4 = 2/7",
            "root_code": "FRM-01"
        },
        {
            "grade": 6,
            "topic_name": "fractions",
            "question_text": "1/3 + 1/4",
            "correct_answer": "7/12",
            "transcription_steps": "L1 | 1/3 + 1/4 = 7/12",
            "root_code": "NONE"
        },
        {
            "grade": 7,
            "topic_name": "integers",
            "question_text": "-3 - 5",
            "correct_answer": "-8",
            "transcription_steps": "L1 | -3 - 5 = -2",
            "root_code": "INT-04"
        }
    ] * 3)

    pipeline = TwoStageDiagnosticPipeline()
    pipeline.fit_diagnoser(train_data)

    trace = StudentTrace(
        trace_id="test_tr_001",
        question_id="q1",
        grade=6,
        topic_name="fractions",
        question_text="1/5 + 1/3",
        question_image_file=None,
        correct_answer="8/15",
        image_file="mock.jpeg",
        transcription_steps="L1 | 1/5 + 1/3 = 2/8"
    )

    result = pipeline.diagnose_trace(trace)
    assert isinstance(result, DiagnosticResult)
    assert result.trace_id == "test_tr_001"
    assert len(result.top_3_codes) == 3
    assert len(result.top_3_probabilities) == 3
    sub_str = result.to_submission_string()
    assert len(sub_str.split()) == 3


def test_submission_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Create sample_submission.csv
        sample_path = tmp_path / "sample_submission.csv"
        sample_df = pd.DataFrame({
            "trace_id": ["t1", "t2", "t3"],
            "root_code": ["NONE", "NONE", "NONE"]
        })
        sample_df.to_csv(sample_path, index=False)

        # 2. Create valid submission
        valid_sub_path = tmp_path / "valid_submission.csv"
        valid_df = pd.DataFrame({
            "trace_id": ["t1", "t2", "t3"],
            "root_code": ["FRM-01 NONE CAL-01", "NONE CAL-01 EXP-01", "INT-04 NONE FRM-01"]
        })
        valid_df.to_csv(valid_sub_path, index=False)

        val_report = validate_submission_file(valid_sub_path, sample_path)
        assert val_report["is_valid"] is True
        assert val_report["total_rows"] == 3
        assert len(val_report["errors"]) == 0

        # 3. Create invalid submission (missing rows & wrong columns)
        invalid_sub_path = tmp_path / "invalid_submission.csv"
        invalid_df = pd.DataFrame({
            "trace_id": ["t1"],
            "wrong_col": ["NONE"]
        })
        invalid_df.to_csv(invalid_sub_path, index=False)

        inv_report = validate_submission_file(invalid_sub_path, sample_path)
        assert inv_report["is_valid"] is False
        assert len(inv_report["errors"]) > 0
