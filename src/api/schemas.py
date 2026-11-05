"""Pydantic V2 Schemas for Misconception Diagnostic API.

Adheres strictly to the updated OpenAPI 3.0 specification in docs/ARCHITECTURE.md.
"""
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class DiagnosticRequest(BaseModel):
    """Input payload for student math diagnostic inference."""
    trace_id: Optional[str] = Field(default=None, description="Unique trace identifier (auto-generated if omitted)")
    grade: Optional[int] = Field(default=6, ge=1, le=12, description="Student grade band (1 to 12)")
    topic_name: Optional[str] = Field(default="mathematics", description="Curricular math topic name")
    question_text: Optional[str] = Field(default="", description="Question text with Indonesian and LaTeX notation")
    correct_answer: Optional[str] = Field(default=None, description="Teacher's expected solution (optional at inference)")
    image_base64: str = Field(..., description="Base64 encoded string of student handwritten math photo")

    @field_validator("image_base64")
    def validate_base64_payload(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("image_base64 payload cannot be empty.")
        # Check payload size (10MB base64 corresponds to ~14MB text length)
        if len(v) > 15 * 1024 * 1024:
            raise ValueError("Image payload exceeds maximum allowable size of 10 MB.")
        return v


class MisconceptionCandidate(BaseModel):
    """Ranked misconception candidate with taxonomy metadata and model score."""
    root_code: str = Field(..., description="Taxonomy misconception code or NONE")
    name: str = Field(..., description="Human-readable description / name of misconception")
    nature: Optional[str] = Field(default=None, description="Nature of misconception (e.g. conceptual, procedural)")
    score: float = Field(..., description="Model confidence score / calibrated ranking metric")


class DiagnosticResponse(BaseModel):
    """Diagnostic response with top-3 predictions and pedagogical feedback."""
    trace_id: str
    clean_work: bool = Field(..., description="True if no misconception detected (NONE is rank 1)")
    transcribed_steps: str = Field(..., description="Extracted handwritten mathematical steps")
    top_3_predictions: List[MisconceptionCandidate] = Field(
        ..., min_length=3, max_length=3, description="Exactly 3 distinct ranked root misconception candidates"
    )
    pedagogical_feedback: str = Field(..., description="Actionable pedagogical feedback for teacher intervention")

    @field_validator("top_3_predictions")
    def validate_distinct_codes(cls, v: List[MisconceptionCandidate]) -> List[MisconceptionCandidate]:
        codes = [c.root_code for c in v]
        if len(set(codes)) != 3:
            raise ValueError(f"top_3_predictions must contain 3 distinct root codes, got: {codes}")
        return v
