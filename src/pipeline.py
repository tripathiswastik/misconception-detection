"""End-to-End Diagnostic Pipeline connecting Vision Reader and Reasoning Diagnoser.

Executes 2-stage inference on GPU:
Image -> [GPU Handwriting Reader in FP16] -> Transcribed Steps -> [Math Diagnoser] -> Top-3 Root Codes
"""
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
import numpy as np
import pandas as pd

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import StudentTrace
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.diagnoser.baseline_text import TextMisconceptionDiagnoser
from src.diagnoser.domain_mask import DomainConstraintEngine
from src.reader.ocr_engine import HandwritingReaderEngine
from src.utils.device import clear_gpu_memory, get_device, get_gpu_info


@dataclass
class DiagnosticResult:
    trace_id: str
    transcription: str
    top_3_codes: List[str]
    top_3_probabilities: List[float]
    clean_predicted: bool

    def to_submission_string(self) -> str:
        """Format as space-delimited top-3 ranking for Kaggle submission."""
        return " ".join(self.top_3_codes)


class TwoStageDiagnosticPipeline:
    """Full End-to-End Pipeline: Vision OCR -> Semantic Math Reasoner with Domain Masking."""

    def __init__(
        self,
        reader: Optional[HandwritingReaderEngine] = None,
        diagnoser: Optional[TextMisconceptionDiagnoser] = None,
        taxonomy: Optional[TaxonomyManager] = None,
        domain_masker: Optional[DomainConstraintEngine] = None,
        use_gpu: bool = True
    ):
        self.device = get_device(prefer_gpu=use_gpu)
        self.taxonomy = taxonomy or TaxonomyManager()
        self.domain_masker = domain_masker or DomainConstraintEngine(taxonomy=self.taxonomy)
        self.reader = reader or HandwritingReaderEngine(use_gpu=use_gpu)
        self.diagnoser = diagnoser or TextMisconceptionDiagnoser(taxonomy_manager=self.taxonomy)

    def fit_diagnoser(self, train_df: pd.DataFrame) -> "TwoStageDiagnosticPipeline":
        """Train internal text diagnoser and fit empirical domain constraint mask."""
        self.diagnoser.fit(train_df)
        self.domain_masker.fit_empirical(train_df)
        return self

    def diagnose_trace(self, trace: StudentTrace) -> DiagnosticResult:
        """Run full 2-stage diagnostic inference on a single student trace."""
        # Stage 1: Handwriting Reading & Step Transcription
        if trace.transcription_steps and trace.transcription_steps.strip():
            transcription = trace.transcription_steps
        elif trace.image_path:
            transcription = self.reader.transcribe_image(
                Path(trace.image_path),
                trace_id=trace.trace_id
            )
        else:
            transcription = "[Handwriting photo missing]"

        # Stage 2: Reasoning Diagnoser
        input_row = pd.Series({
            "trace_id": trace.trace_id,
            "grade": trace.grade,
            "topic_name": trace.topic_name,
            "question_text": trace.question_text,
            "correct_answer": trace.correct_answer,
            "transcription_steps": transcription
        })
        input_df = pd.DataFrame([input_row])
        raw_probs = self.diagnoser.predict_proba(input_df)
        probs = self.domain_masker.apply_mask_batch(
            raw_probs,
            list(self.diagnoser.classes_),
            [trace.topic_name],
            [trace.grade]
        )[0]
        top_indices = np.argsort(probs)[::-1][:3]

        top_codes = [str(self.diagnoser.classes_[idx]) for idx in top_indices]
        top_probs = [float(probs[idx]) for idx in top_indices]

        return DiagnosticResult(
            trace_id=trace.trace_id,
            transcription=transcription,
            top_3_codes=top_codes,
            top_3_probabilities=top_probs,
            clean_predicted=(top_codes[0] == LABEL_NONE)
        )

    def diagnose_batch(
        self,
        traces: Sequence[StudentTrace],
        batch_size: int = 500,
        show_progress: bool = True
    ) -> List[DiagnosticResult]:
        """Run high-throughput 2-stage batch inference with GPU acceleration and domain constraint masking."""
        total = len(traces)
        if show_progress:
            print(f"[Pipeline] Diagnosing {total} traces on device {self.device}...")

        # Stage 1: Fast Vision & Transcription Extraction
        transcriptions: List[str] = []
        for idx, trace in enumerate(traces, start=1):
            if trace.transcription_steps and trace.transcription_steps.strip():
                t = trace.transcription_steps
            elif trace.image_path:
                t = self.reader.transcribe_image(
                    Path(trace.image_path),
                    trace_id=trace.trace_id
                )
            else:
                t = "[Handwriting photo missing]"
            transcriptions.append(t)

            if idx % batch_size == 0 or idx == total:
                clear_gpu_memory()
                if show_progress:
                    print(f"  [Stage 1: Vision OCR] Processed {idx}/{total} traces on GPU...")

        # Persist transcription cache to disk
        self.reader._save_cache()

        # Stage 2: Vectorized Reasoning Prediction across all traces
        if show_progress:
            print(f"[Pipeline] Executing Stage 2 semantic math reasoning for {total} traces...")

        batch_df = pd.DataFrame({
            "trace_id": [t.trace_id for t in traces],
            "grade": [t.grade for t in traces],
            "topic_name": [t.topic_name for t in traces],
            "question_text": [t.question_text for t in traces],
            "correct_answer": [t.correct_answer for t in traces],
            "transcription_steps": transcriptions
        })

        all_probs = self.diagnoser.predict_proba(batch_df)
        classes = list(self.diagnoser.classes_)

        # Apply domain constraint masking
        topics = [t.topic_name for t in traces]
        grades = [t.grade for t in traces]
        calibrated_probs = self.domain_masker.apply_mask_batch(
            all_probs, classes, topics, grades, damping=0.85
        )

        results: List[DiagnosticResult] = []
        for idx in range(total):
            prob_row = calibrated_probs[idx]
            top_indices = np.argsort(prob_row)[::-1][:3]
            top_codes = [str(classes[i]) for i in top_indices]
            top_probs = [float(prob_row[i]) for i in top_indices]

            results.append(DiagnosticResult(
                trace_id=traces[idx].trace_id,
                transcription=transcriptions[idx],
                top_3_codes=top_codes,
                top_3_probabilities=top_probs,
                clean_predicted=(top_codes[0] == LABEL_NONE)
            ))

        if show_progress:
            print(f"[Pipeline] Successfully completed diagnostic inference for {total} traces.")

        return results
