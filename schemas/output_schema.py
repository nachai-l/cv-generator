"""
Output schema definitions for CV generation responses.

This module defines all response models for the CV generation API,
including section content, metadata, evidence tracking, and error responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class GenerationStatus(str, Enum):
    """Status of CV generation process."""

    COMPLETED = "completed"
    FAILED = "failed"
    PROCESSING = "processing"
    PARTIAL = "partial"  # Some sections generated, some failed


class SectionContent(BaseModel):
    """Generated content for a single CV section."""

    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="Generated text content for the section",
    )
    word_count: int = Field(..., ge=0, description="Number of words in the text")
    matched_jd_skills: list[str] = Field(
        default_factory=list,
        description="Skills from job description that appear in this section",
    )
    confidence_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence in content quality (0-1)",
    )

    @field_validator("text")
    @classmethod
    def validate_text_content(cls, v: str) -> str:
        """Ensure text doesn't contain markdown artifacts."""
        if "```" in v:
            raise ValueError("Text contains code block markers")
        return v.strip()

    model_config = {"extra": "forbid"}


class EvidenceMapping(BaseModel):
    """Maps a generated sentence to its supporting evidence."""

    section: str = Field(
        ...,
        description="Section identifier (e.g., 'experience[0].text')",
        examples=["profile_summary.text", "experience[0].text"],
    )
    sentence: str = Field(
        ..., max_length=1000, description="Sentence from generated content"
    )
    evidence_ids: list[str] = Field(
        ...,
        min_length=1,
        description="IDs of evidence that support this sentence",
        examples=[["work_exp#1", "skill#python_L3"]],
    )
    match_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How well evidence supports the sentence",
    )

    model_config = {"extra": "forbid"}


class UnsupportedClaim(BaseModel):
    """A claim that couldn't be verified against input data."""

    section: str = Field(..., description="Section where claim appears")
    claim: str = Field(..., max_length=500, description="The unsupported claim text")
    reason: str = Field(
        default="No matching evidence found",
        description="Why claim is unsupported",
    )
    severity: str = Field(
        default="warning",
        pattern="^(info|warning|error)$",
        description="Severity level",
    )


class Justification(BaseModel):
    """Evidence tracking and justification for generated content."""

    evidence_map: list[EvidenceMapping] = Field(
        default_factory=list,
        description="Mappings from generated content to source evidence",
    )
    unsupported_claims: list[UnsupportedClaim] = Field(
        default_factory=list,
        description="Claims that couldn't be verified",
    )
    coverage_score: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Percentage of content with evidence (0-1)",
    )
    total_claims_analyzed: int = Field(
        default=0, ge=0, description="Total number of claims analyzed"
    )

    @property
    def has_issues(self) -> bool:
        """Check if there are any unsupported claims."""
        return len(self.unsupported_claims) > 0

    model_config = {"extra": "forbid"}

class Metadata(BaseModel):
    """Generation metadata and performance metrics."""

    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp of generation",
    )
    model_version: str = Field(
        default="gemini-2.5-flash",
        description="LLM model version used",
    )
    generation_time_ms: int = Field(
        ...,
        ge=0,
        description="Total generation time in milliseconds",
    )
    retry_count: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Number of retry attempts",
    )
    cache_hit: bool = Field(
        default=False,
        description="Whether result was cached",
    )
    sections_requested: int = Field(
        default=0,
        ge=0,
        description="Number of sections requested",
    )
    sections_generated: int = Field(
        default=0,
        ge=0,
        description="Number of sections successfully generated",
    )
    tokens_used: int = Field(
        default=0,
        ge=0,
        description="Total tokens consumed (if available)",
    )

    # 🔹 NEW — fine-grained token accounting
    input_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total input tokens sent to the LLM",
    )
    output_tokens: int | None = Field(
        default=None,
        ge=0,
        description="Total output tokens produced by the LLM",
    )
    section_breakdown: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Per-section token breakdown: "
            "[{'section_name': str, 'section_input_tokens': int, 'section_output_tokens': int}, ...]"
        ),
    )

    cost_estimate_thb: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in THB",
    )
    cost_estimate_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Estimated cost in USD",
    )

    # 🔹 For renderer header construction
    profile_info: dict[str, Any] | None = Field(
        default=None,
        description="Original profile_info payload used for header/contact rendering",
    )

    # 🔹 Correlation ID, aligned with Stage D
    request_id: str | None = Field(
        default=None,
        description="Request correlation ID (mirrors Stage D request_id when available).",
    )

    # 🔹 Pipeline flags
    stage_c_validated: bool = Field(
        default=False,
        description="True once Stage C validation has run successfully.",
    )
    stage_d_completed: bool = Field(
        default=False,
        description="True once Stage D packaging has completed successfully.",
    )

    @field_validator("generation_time_ms")
    @classmethod
    def validate_reasonable_time(cls, v: int) -> int:
        """Ensure generation time is reasonable."""
        if v > 180000:  # 180 seconds
            raise ValueError("Generation time exceeds maximum allowed (180s)")
        return v

    model_config = {"extra": "forbid"}


class QualityMetrics(BaseModel):
    """Quality assessment metrics for generated CV."""

    clarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Readability/clarity score (0-100)",
    )
    jd_alignment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Alignment with job description (0-100)",
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="How complete the CV is (0-100)",
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Internal consistency (0-100)",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall quality score (0-100)",
    )
    feedback: list[str] = Field(
        default_factory=list, description="Specific quality feedback items"
    )

    model_config = {"extra": "forbid"}

class QualityMetrics(BaseModel):
    """Quality assessment metrics for generated CV."""

    clarity_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Readability / clarity score (0-100)",
    )
    jd_alignment_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Alignment with job description (0-100)",
    )
    completeness_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="How complete the CV is (0-100)",
    )
    consistency_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Internal consistency (0-100)",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Overall quality score (0-100)",
    )

    feedback: list[str] = Field(
        default_factory=list,
        description="List of feedback items describing strengths or issues",
    )

    model_config = {"extra": "forbid"}


class ValidationWarning(BaseModel):
    """Warning about potential issues in generated content."""

    section: str = Field(..., description="Section with the warning")
    warning_type: str = Field(
        ...,
        description="Type of warning",
        examples=["word_count", "missing_evidence", "formatting"],
    )
    message: str = Field(..., max_length=500, description="Warning message")
    suggestion: str | None = Field(
        None, max_length=500, description="Suggested fix"
    )


# ---------------------------------------------------------------------------
# Structured skills output (taxonomy-preserving)
# ---------------------------------------------------------------------------


class OutputSkillItem(BaseModel):
    """Structured representation of a skill in the final CV.

    This mirrors the taxonomy-preserving constraints enforced in Stage B/C:
    - `name` must match a known taxonomy skill.
    - The LLM may reorder skills, drop some (handled upstream), or adjust
      `level` within allowed business rules, but must not rename skills.
    """

    name: str = Field(
        ...,
        description="Canonical skill name; must match taxonomy-provided label.",
    )
    level: str | None = Field(
        default=None,
        description=(
            "Optional skill level code or label (e.g. 'L3_Advanced', 'Advanced'). "
            "Missing if no level is available."
        ),
    )
    source: str | None = Field(
        default=None,
        description="Origin of the skill in the CV (taxonomy, inferred, user_input, etc.).",
    )

    model_config = {"extra": "forbid"}


class CVGenerationResponse(BaseModel):
    """Complete CV generation response."""

    job_id: str = Field(
        ...,
        pattern=r"^JOB_[a-zA-Z0-9_-]+$",
        description="Unique job identifier",
        examples=["JOB_abc123xyz"],
    )
    template_id: str = Field(
        ..., description="Template used for generation", examples=["T_EMPLOYER_STD_V3"]
    )
    language: str = Field(
        ...,
        pattern="^(en|th)$",
        description="Language of generated content",
        examples=["en", "th"],
    )
    status: GenerationStatus = Field(..., description="Generation status")

    sections: dict[str, SectionContent] = Field(
        default_factory=dict, description="Generated CV sections"
    )

    # 🔹 NEW: structured, taxonomy-preserving skills, primarily for the skills section.
    skills: list[OutputSkillItem] | None = Field(
        default=None,
        description=(
            "Structured skills selected for the CV. "
            "Used to enforce taxonomy-preserving constraints and to render "
            "skill chips with levels. If None, the skills section is only "
            "available as plain text under `sections['skills']`."
        ),
    )

    metadata: Metadata = Field(..., description="Generation metadata")
    justification: Justification = Field(
        ..., description="Evidence tracking and justification"
    )
    quality_metrics: QualityMetrics | None = Field(
        None, description="Quality assessment metrics"
    )
    warnings: list[ValidationWarning] = Field(
        default_factory=list, description="Validation warnings"
    )
    error: str | None = Field(
        None, max_length=1000, description="Error message if status is failed"
    )
    error_details: dict[str, Any] | None = Field(
        None, description="Detailed error information"
    )

    @property
    def is_successful(self) -> bool:
        """Check if generation was successful."""
        return self.status in [GenerationStatus.COMPLETED, GenerationStatus.PARTIAL]

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    @property
    def section_names(self) -> list[str]:
        """Get list of generated section names."""
        return list(self.sections.keys())

    def get_section(self, section_name: str) -> SectionContent | None:
        """Get a specific section by name."""
        return self.sections.get(section_name)

    def get_total_word_count(self) -> int:
        """Calculate total word count across all sections."""
        return sum(section.word_count for section in self.sections.values())

    model_config = {"extra": "forbid"}


class ErrorResponse(BaseModel):
    """Error response for failed requests."""

    status: str = Field(default="error", description="Always 'error'")
    error_code: str = Field(
        ...,
        description="Machine-readable error code",
        examples=[
            "VALIDATION_ERROR",
            "SECURITY_VIOLATION",
            "GENERATION_FAILED",
            "TIMEOUT",
        ],
    )
    message: str = Field(..., max_length=500, description="Human-readable error message")
    details: dict[str, Any] | None = Field(
        None, description="Additional error details"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="When error occurred",
    )
    request_id: str | None = Field(
        None, description="Request ID for tracking", examples=["REQ_abc123"]
    )
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions to fix the error"
    )

    model_config = {"extra": "forbid"}


class BatchGenerationRequest(BaseModel):
    """Request to generate multiple CVs in batch."""

    batch_id: str = Field(..., pattern=r"^BATCH_[a-zA-Z0-9_-]+$")
    requests: list[dict[str, Any]] = Field(
        ..., min_length=1, max_length=100, description="List of CV generation requests"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Batch priority (1-10)")
    callback_url: str | None = Field(
        None, description="Webhook URL for completion notification"
    )

    model_config = {"extra": "forbid"}


class BatchGenerationResponse(BaseModel):
    """Response for batch CV generation."""

    batch_id: str = Field(..., pattern=r"^BATCH_[a-zA-Z0-9_-]+$")
    status: str = Field(
        ...,
        pattern="^(queued|processing|completed|failed|partial)$",
        description="Batch processing status",
    )
    total_requests: int = Field(..., ge=1, description="Total number of requests")
    completed: int = Field(default=0, ge=0, description="Number completed")
    failed: int = Field(default=0, ge=0, description="Number failed")
    results: list[CVGenerationResponse] = Field(
        default_factory=list, description="Individual CV results"
    )
    started_at: datetime | None = Field(None, description="When processing started")
    completed_at: datetime | None = Field(None, description="When processing finished")
    estimated_completion: datetime | None = Field(
        None, description="Estimated completion time"
    )

    @property
    def progress_percentage(self) -> float:
        """Calculate progress percentage."""
        if self.total_requests == 0:
            return 0.0
        return ((self.completed + self.failed) / self.total_requests) * 100

    model_config = {"extra": "forbid"}


class HealthCheckResponse(BaseModel):
    """Health check response."""

    status: str = Field(
        ..., pattern="^(healthy|degraded|unhealthy)$", description="Service health status"
    )
    checks: dict[str, str] = Field(
        ..., description="Individual component health checks"
    )
    version: str = Field(default="0.1.0", description="Service version")
    uptime_seconds: int = Field(default=0, ge=0, description="Service uptime")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = {"extra": "forbid"}


# Export all models
__all__ = [
    "GenerationStatus",
    "SectionContent",
    "EvidenceMapping",
    "UnsupportedClaim",
    "Justification",
    "Metadata",
    "QualityMetrics",
    "ValidationWarning",
    "OutputSkillItem",
    "CVGenerationResponse",
    "ErrorResponse",
    "BatchGenerationRequest",
    "BatchGenerationResponse",
    "HealthCheckResponse",
]
