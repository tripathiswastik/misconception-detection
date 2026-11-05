"""FastAPI REST Service for Student Math Misconception Diagnostics.

Endpoints:
- POST /api/v1/diagnose: Run 2-stage vision + semantic reasoning inference on student math workings.
- GET /health: Healthcheck and GPU hardware telemetry status.
- GET /api/v1/taxonomy: Query taxonomy categories and root misconception codes.
"""
import base64
import io
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.schemas import DiagnosticRequest, DiagnosticResponse, MisconceptionCandidate
from src.config import CONFIG, PATHS
from src.data.dataset import MathMisconceptionData, StudentTrace
from src.data.taxonomy import LABEL_NONE, TaxonomyManager
from src.fusion.ranker import RankFusionEngine
from src.pipeline import TwoStageDiagnosticPipeline
from src.utils.device import clear_gpu_memory, get_device, get_gpu_info

app = FastAPI(
    title="K-12 Student Math Misconception Diagnostic API",
    version="1.0.0",
    description="Diagnostic AI service detecting root mathematical misconceptions in handwritten student workings."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global singletons
_TAXONOMY: Optional[TaxonomyManager] = None
_PIPELINE: Optional[TwoStageDiagnosticPipeline] = None
_RANKER: Optional[RankFusionEngine] = None


def get_taxonomy_manager() -> TaxonomyManager:
    global _TAXONOMY
    if _TAXONOMY is None:
        tax_path = PATHS.taxonomy_csv
        _TAXONOMY = TaxonomyManager(tax_path if tax_path.exists() else None)
    return _TAXONOMY


def get_pipeline() -> TwoStageDiagnosticPipeline:
    global _PIPELINE
    if _PIPELINE is None:
        tax = get_taxonomy_manager()
        _PIPELINE = TwoStageDiagnosticPipeline(taxonomy=tax, use_gpu=True)
        # Train on training set if available
        if PATHS.train_csv.exists():
            data_mgr = MathMisconceptionData(PATHS.data_dir)
            train_df = data_mgr.load_train()
            _PIPELINE.fit_diagnoser(train_df)
    return _PIPELINE


def get_ranker() -> RankFusionEngine:
    global _RANKER
    if _RANKER is None:
        tax = get_taxonomy_manager()
        pipeline = get_pipeline()
        _RANKER = RankFusionEngine(taxonomy=tax, domain_masker=pipeline.domain_masker)
    return _RANKER


@app.get("/health", summary="Health and Telemetry Check")
def healthcheck() -> Dict[str, Any]:
    gpu_info = get_gpu_info()
    return {
        "status": "HEALTHY",
        "service": "k12-math-misconception-diagnostics",
        "version": "1.0.0",
        "device": str(get_device(prefer_gpu=True)),
        "gpu_telemetry": gpu_info
    }


@app.get("/api/v1/taxonomy", summary="Retrieve Misconception Taxonomy")
def list_taxonomy() -> Dict[str, Any]:
    tax = get_taxonomy_manager()
    return {
        "total_codes": len(tax.all_codes(include_none=True)),
        "categories": tax.get_all_categories(),
        "misconceptions": [
            {
                "root_code": code,
                "name": tax.get_name(code),
                "category": tax.get_category(code)
            }
            for code in tax.all_codes(include_none=True)
        ]
    }


def _generate_pedagogical_feedback(top_candidates: List[MisconceptionCandidate], is_clean: bool) -> str:
    if is_clean:
        return "Student working is mathematically sound with no fundamental conceptual misconceptions detected."
    primary = top_candidates[0]
    return (
        f"Primary diagnostic finding indicates misconception [{primary.root_code}]: '{primary.name}'. "
        f"Recommended intervention: Reinforce conceptual foundation and targeted practice contrasting this "
        f"error with the canonical mathematical operation."
    )


@app.post("/api/v1/diagnose", response_model=DiagnosticResponse, summary="Diagnose Student Math Working")
def diagnose_student_working(request: DiagnosticRequest) -> DiagnosticResponse:
    # 1. Decode base64 image with strict validation
    try:
        image_bytes = base64.b64decode(request.image_base64)
        if len(image_bytes) == 0:
            raise ValueError("Decoded image data is empty.")
        pil_img = Image.open(io.BytesIO(image_bytes))
        pil_img.verify()  # Verify integrity
        # Reopen after verify
        pil_img = Image.open(io.BytesIO(image_bytes))
        width, height = pil_img.size
        if width < 10 or height < 10:
            raise ValueError(f"Image dimensions too small ({width}x{height}). Minimum is 10x10.")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid image payload: {str(e)}"
        )

    pipeline = get_pipeline()
    tax = get_taxonomy_manager()
    ranker = get_ranker()

    # 2. Stage 1: Vision OCR feature extraction & step transcription
    trace_id = request.trace_id or f"trace_{uuid.uuid4().hex[:8]}"
    transcription = pipeline.reader.transcribe_image(Path("in_memory.jpeg"), trace_id=trace_id)
    _ = pipeline.reader.extract_visual_features(pil_img)

    # 3. Stage 2: Reasoning Diagnoser
    input_df = pd.DataFrame([{
        "trace_id": trace_id,
        "grade": request.grade or 6,
        "topic_name": request.topic_name or "mathematics",
        "question_text": request.question_text or "",
        "correct_answer": request.correct_answer or "",
        "transcription_steps": transcription
    }])

    raw_probs = pipeline.diagnoser.predict_proba(input_df)[0]
    classes = list(pipeline.diagnoser.classes_)

    # Use RankFusionEngine to ensure exactly 3 distinct valid codes
    top_codes, top_scores = ranker.rank_single(
        scores=raw_probs,
        classes=classes,
        topic_name=request.topic_name,
        grade=request.grade,
        apply_mask=True,
        top_k=3
    )

    candidates: List[MisconceptionCandidate] = []
    for code, sc in zip(top_codes, top_scores):
        name = tax.get_name(code)
        candidates.append(MisconceptionCandidate(
            root_code=code,
            name=name,
            nature="conceptual" if code != LABEL_NONE else "none",
            score=round(float(sc), 4)
        ))

    is_clean = (candidates[0].root_code == LABEL_NONE)
    feedback = _generate_pedagogical_feedback(candidates, is_clean)

    clear_gpu_memory()

    return DiagnosticResponse(
        trace_id=trace_id,
        clean_work=is_clean,
        transcribed_steps=transcription,
        top_3_predictions=candidates,
        pedagogical_feedback=feedback
    )
