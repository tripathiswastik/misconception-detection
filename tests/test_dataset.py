"""Unit tests for StudentTrace and MathMisconceptionData."""
import tempfile
from pathlib import Path
import pandas as pd
from src.data.dataset import StudentTrace, MathMisconceptionData


def test_student_trace_prompt_formatting():
    trace = StudentTrace(
        trace_id="t1",
        question_id="q1",
        grade=7,
        topic_name="fractions",
        question_text="2/3 + 1/4 = ?",
        question_image_file=None,
        correct_answer="11/12",
        image_file="t1.jpeg",
        root_code="FRM-01",
        transcription_steps="L1 | 2/3 + 1/4, L2 | 3/7"
    )
    prompt = trace.format_diagnostic_prompt()
    assert "Grade 7" in prompt
    assert "fractions" in prompt
    assert "11/12" in prompt
    assert "L1 | 2/3 + 1/4" in prompt


def test_math_misconception_data_loading():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        train_df = pd.DataFrame([{
            "trace_id": "tr_1",
            "question_id": "q_1",
            "grade": 6,
            "topic_name": "arithmetic",
            "question_text": "3 * 4",
            "question_image_file": None,
            "correct_answer": "12",
            "image_file": "img1.jpeg",
            "root_code": "NONE",
            "transcription_steps": "L1 | 3*4 = 12",
            "final_answer_correct": True,
            "clean": True,
            "misconceptions": "[]"
        }])
        train_df.to_csv(data_dir / "train.csv", index=False)

        loader = MathMisconceptionData(data_dir)
        traces = loader.get_train_traces()
        assert len(traces) == 1
        assert traces[0].trace_id == "tr_1"
        assert traces[0].root_code == "NONE"
        assert traces[0].clean is True
