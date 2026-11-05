"""Dataset loaders, trace structures, and data preprocessing for Misconception Detection."""
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator, cast
import pandas as pd


@dataclass
class StudentTrace:
    trace_id: str
    question_id: str
    grade: int
    topic_name: str
    question_text: str
    question_image_file: Optional[str]
    correct_answer: str
    image_file: str
    image_path: Optional[Path] = None
    question_image_path: Optional[Path] = None
    root_code: Optional[str] = None
    transcription_steps: Optional[str] = None
    final_answer_correct: Optional[bool] = None
    clean: Optional[bool] = None
    misconceptions: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_labelled(self) -> bool:
        return self.root_code is not None

    def format_diagnostic_prompt(self) -> str:
        """Format the trace as a pedagogical diagnostic prompt for language models."""
        transcription = self.transcription_steps or "[Handwriting not transcribed]"
        return (
            f"Question (Grade {self.grade} - {self.topic_name}):\n"
            f"{self.question_text}\n\n"
            f"Correct Answer: {self.correct_answer}\n\n"
            f"Student Working (Line-by-line transcription):\n"
            f"{transcription}\n"
        )


class MathMisconceptionData:
    """Manages train, test, and taxonomy data tables."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.train_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None
        self.taxonomy_df: Optional[pd.DataFrame] = None

    def load_train(self) -> pd.DataFrame:
        train_path = self.data_dir / "train.csv"
        if not train_path.exists():
            raise FileNotFoundError(f"Missing train.csv at: {train_path}")
        df = cast(pd.DataFrame, pd.read_csv(train_path))
        self.train_df = df
        return df

    def load_test(self) -> pd.DataFrame:
        test_path = self.data_dir / "test.csv"
        if not test_path.exists():
            raise FileNotFoundError(f"Missing test.csv at: {test_path}")
        df = cast(pd.DataFrame, pd.read_csv(test_path))
        self.test_df = df
        return df

    def load_taxonomy(self) -> pd.DataFrame:
        tax_path = self.data_dir / "taxonomy_v1.csv"
        if not tax_path.exists():
            raise FileNotFoundError(f"Missing taxonomy_v1.csv at: {tax_path}")
        df = cast(pd.DataFrame, pd.read_csv(tax_path))
        self.taxonomy_df = df
        return df

    def get_train_traces(self) -> List[StudentTrace]:
        if self.train_df is None:
            self.load_train()
        assert self.train_df is not None
        traces = []
        for _, row in self.train_df.iterrows():
            misc_raw = row.get("misconceptions")
            parsed_misc = []
            if isinstance(misc_raw, str) and misc_raw.strip():
                try:
                    parsed_misc = json.loads(misc_raw)
                except Exception:
                    parsed_misc = []

            img_file = str(row["image_file"])
            img_path = self.data_dir / "images" / "train" / img_file

            q_img_file = row.get("question_image_file")
            q_img_path = None
            if pd.notna(q_img_file) and str(q_img_file).strip():
                q_img_path = self.data_dir / "images" / "questions" / str(q_img_file)

            traces.append(StudentTrace(
                trace_id=str(row["trace_id"]),
                question_id=str(row["question_id"]),
                grade=int(row["grade"]),
                topic_name=str(row["topic_name"]),
                question_text=str(row["question_text"]),
                question_image_file=str(q_img_file) if pd.notna(q_img_file) else None,
                correct_answer=str(row["correct_answer"]),
                image_file=img_file,
                image_path=img_path,
                question_image_path=q_img_path,
                root_code=str(row["root_code"]),
                transcription_steps=str(row["transcription_steps"]) if pd.notna(row.get("transcription_steps")) else None,
                final_answer_correct=bool(row["final_answer_correct"]) if pd.notna(row.get("final_answer_correct")) else None,
                clean=bool(row["clean"]) if pd.notna(row.get("clean")) else None,
                misconceptions=parsed_misc
            ))
        return traces

    def get_test_traces(self) -> List[StudentTrace]:
        if self.test_df is None:
            self.load_test()
        assert self.test_df is not None
        traces = []
        for _, row in self.test_df.iterrows():
            img_file = str(row["image_file"])
            img_path = self.data_dir / "images" / "test" / img_file

            q_img_file = row.get("question_image_file")
            q_img_path = None
            if pd.notna(q_img_file) and str(q_img_file).strip():
                q_img_path = self.data_dir / "images" / "questions" / str(q_img_file)

            traces.append(StudentTrace(
                trace_id=str(row["trace_id"]),
                question_id=str(row["question_id"]),
                grade=int(row["grade"]),
                topic_name=str(row["topic_name"]),
                question_text=str(row["question_text"]),
                question_image_file=str(q_img_file) if pd.notna(q_img_file) else None,
                correct_answer=str(row["correct_answer"]),
                image_file=img_file,
                image_path=img_path,
                question_image_path=q_img_path,
                root_code=None
            ))
        return traces
