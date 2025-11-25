# functions/stage_d_packaging.py

"""
Stage D – Response Packaging & Delivery

This module is the final packaging layer of the CV Generation pipeline.

Responsibilities:
- Take validated CV output from Stage C (CVGenerationResponse).
- Enrich metadata with timing, token usage, cost, and profile header info.
- Attach / propagate request_id and user_id for observability.
- Mark pipeline completion flags (e.g., stage_d_completed) without mutating
  sections or skills that were already finalized in Stage C.
- Normalize and construct error responses (ErrorResponse) in a way that is
  HTTP-agnostic so FastAPI (or any other framework) can use it directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import structlog

from schemas.output_schema import (
    CVGenerationResponse,
    ErrorResponse,
    GenerationStatus,
)
from functions.utils.quality_metrics import compute_quality_metrics
from schemas.output_schema import QualityMetrics, ValidationWarning

logger = structlog.get_logger(__name__).bind(module="stage_d_packaging")


# ---------------------------------------------------------------------------
# Lightweight helper structs (non-Pydantic, internal to Stage D)
# ---------------------------------------------------------------------------


@dataclass
class LLMUsageSummary:
    """
    Aggregated LLM usage stats for a single CV generation pipeline run.

    This should be created upstream (e.g., in llm_metrics or the orchestrator)
    and passed into Stage D. Stage D only consumes it to update metadata
    and logs; it does not call any LLMs itself.
    """

    model: str = "unknown"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_thb: float = 0.0
    total_cost_usd: float = 0.0
    section_breakdown: Optional[list[dict[str, Any]]] = None

# ---------------------------------------------------------------------------
# Core packaging helpers
# ---------------------------------------------------------------------------


def _generate_request_id() -> str:
    """Generate a simple request ID if upstream did not provide one."""
    now = datetime.now(timezone.utc)
    return f"REQ_{int(now.timestamp() * 1000)}"


def _compute_generation_time_ms(
    generation_start: Optional[datetime],
    generation_end: Optional[datetime],
) -> Optional[int]:
    """Compute generation time in ms from start/end datetimes if available."""
    if generation_start is None or generation_end is None:
        return None

    # Ensure timezone-aware datetimes; if naive, assume UTC.
    if generation_start.tzinfo is None:
        generation_start = generation_start.replace(tzinfo=timezone.utc)
    if generation_end.tzinfo is None:
        generation_end = generation_end.replace(tzinfo=timezone.utc)

    delta = generation_end - generation_start
    ms = int(delta.total_seconds() * 1000)
    return max(ms, 0)


def finalize_cv_response(
    cv: CVGenerationResponse,
    *,
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    profile_info: Optional[Dict[str, Any]] = None,
    llm_usage: Optional[LLMUsageSummary] = None,
    total_tokens: Optional[int] = None,
    total_cost_thb: Optional[float] = None,
    total_cost_usd: Optional[float] = None,
    generation_start: Optional[datetime] = None,
    generation_end: Optional[datetime] = None,
) -> Tuple[CVGenerationResponse, str]:
    """
    Finalize and enrich a successful CVGenerationResponse for delivery.

    This function:
    - Ensures a request_id exists (generate one if missing) and attaches it
      to metadata when possible.
    - Optionally updates metadata.generation_time_ms from start/end timestamps.
    - Optionally updates metadata.tokens_used and metadata.cost_estimate_thb
      from LLMUsageSummary or explicit primitives.
    - Optionally injects profile_info into metadata.profile_info for renderer.
    - Recomputes sections_generated if needed for consistency.
    - Marks metadata.stage_d_completed=True (best-effort).
    - Logs a summary event for observability.

    NOTE:
      Stage D MUST NOT modify sections or skills content; Stage C is the
      last stage allowed to change text or skills.
    """
    # Ensure we have a request_id for logging and headers.
    req_id = request_id or _generate_request_id()

    # Defensive: ensure metadata exists (it *should* per schema, but be safe).
    meta = getattr(cv, "metadata", None)
    if meta is None:
        logger.warning("stage_d_missing_metadata_object")

    # Attach request_id to metadata if possible (useful for clients / renderer).
    if meta is not None:
        try:
            # Only set if empty / missing to avoid clobbering upstream value.
            existing_req_id = getattr(meta, "request_id", None)
            if not existing_req_id:
                meta.request_id = req_id
        except Exception:
            # Non-fatal
            pass

    # Derive generation_time_ms from timestamps if provided.
    computed_ms = _compute_generation_time_ms(generation_start, generation_end)
    if computed_ms is not None and meta is not None:
        try:
            meta.generation_time_ms = computed_ms
        except Exception:
            pass

    # Attach profile_info (for header rendering) if provided.
    if profile_info is not None and meta is not None:
        try:
            meta.profile_info = profile_info
        except Exception:
            pass

    # Recompute sections_generated from actual sections, if zero or inconsistent.
    num_sections = len(getattr(cv, "sections", {}) or {})
    if meta is not None:
        try:
            if getattr(meta, "sections_generated", 0) == 0 or meta.sections_generated < num_sections:
                meta.sections_generated = num_sections

            if getattr(meta, "sections_requested", 0) == 0:
                meta.sections_requested = meta.sections_generated
        except Exception:
            pass

    # Update metadata from LLM usage summary if provided.
    if llm_usage is not None and meta is not None:
        try:
            # Aggregate tokens & cost
            meta.tokens_used = llm_usage.total_tokens
            meta.cost_estimate_thb = llm_usage.total_cost_thb
            meta.cost_estimate_usd = llm_usage.total_cost_usd

            # These represent total prompt/completion tokens across the pipeline.
            try:
                meta.input_tokens = llm_usage.prompt_tokens
            except Exception:
                pass

            try:
                meta.output_tokens = llm_usage.completion_tokens
            except Exception:
                pass

            # Ppass section breakdown from Stage B → Stage D → Metadata
            try:
                if getattr(llm_usage, "section_breakdown", None):
                    meta.section_breakdown = llm_usage.section_breakdown
            except Exception:
                pass

            # Optionally refine model_version from usage info
            current_model = getattr(meta, "model_version", "unknown")
            if current_model in ("unknown", "", "gemini-2.5-flash"):
                if llm_usage.model not in ("", "unknown"):
                    meta.model_version = llm_usage.model

            # ---- NEW OPTIONAL POLISH ----
            # Fill in detailed usage breakdown
            setattr(meta, "llm_model", llm_usage.model)
            setattr(meta, "llm_prompt_tokens", llm_usage.prompt_tokens)
            setattr(meta, "llm_completion_tokens", llm_usage.completion_tokens)
            setattr(meta, "llm_total_tokens", llm_usage.total_tokens)
            # --------------------------------

        except Exception:
            # Metadata best-effort; never break Stage D
            pass

    # Or update tokens/cost explicitly if provided as primitives.
    if meta is not None:
        try:
            if total_tokens is not None:
                meta.tokens_used = total_tokens
            if total_cost_thb is not None:
                meta.cost_estimate_thb = total_cost_thb
            if total_cost_usd is not None:
                meta.cost_estimate_usd = total_cost_usd
        except Exception:
            pass

        meta.stage_d_completed = True

        # -----------------------------------------------------------------------
        # Compute Quality Metrics (Clarity, JD alignment, completeness, etc.)
        # -----------------------------------------------------------------------
        try:
            # The orchestrator or request may provide a list of JD-required skills.
            # If absent, pass empty list – compute_quality_metrics handles it gracefully.
            jd_required_skills = []

            # cv.quality_metrics is optional; always compute a fresh one here.
            qm: QualityMetrics = compute_quality_metrics(
                cv,
                jd_required_skills=jd_required_skills,
            )
            cv.quality_metrics = qm

            # Convert quality-metrics feedback into ValidationWarning objects
            for fb in qm.feedback:
                cv.warnings.append(
                    ValidationWarning(
                        section="quality_metrics",
                        warning_type="quality_feedback",
                        message=fb,
                        suggestion=None,
                    )
                )

        except Exception as e:
            logger.exception("stage_d_quality_metrics_failed", error=str(e))

    # Ensure status is set (default COMPLETED if missing/None)
    try:
        status = getattr(cv, "status", None)
        if status is None:
            cv.status = GenerationStatus.COMPLETED
    except Exception:
        pass

    # Log a compact summary for metrics/observability.
    try:
        # Pull values defensively so logging never breaks the response.
        status_value = (
            cv.status.value
            if isinstance(getattr(cv, "status", None), GenerationStatus)
            else getattr(cv, "status", None)
        )

        sections_list = (
            list((cv.sections or {}).keys())
            if getattr(cv, "sections", None)
            else []
        )

        tokens_used = getattr(meta, "tokens_used", None) if meta is not None else None
        cost_estimate_thb = getattr(meta, "cost_estimate_thb", None) if meta is not None else None
        cost_estimate_usd = getattr(meta, "cost_estimate_usd", None) if meta is not None else None
        gen_time_ms = getattr(meta, "generation_time_ms", None) if meta is not None else None

        # Pull fine-grained tokens for logging
        input_tokens = getattr(meta, "input_tokens", None) if meta is not None else None
        output_tokens = getattr(meta, "output_tokens", None) if meta is not None else None


        logger.info(
            "cv_generation_packaged",
            request_id=req_id,
            user_id=user_id,
            job_id=getattr(cv, "job_id", None),
            template_id=getattr(cv, "template_id", None),
            status=status_value,
            language=getattr(cv, "language", None),
            sections=sections_list,
            sections_generated=getattr(meta, "sections_generated", None) if meta is not None else None,
            tokens_used=tokens_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_estimate_thb=cost_estimate_thb,
            cost_estimate_usd=cost_estimate_usd,
            generation_time_ms=gen_time_ms,
            # LLM usage breakdown (if provided)
            llm_model=llm_usage.model if llm_usage is not None else None,
            llm_prompt_tokens=llm_usage.prompt_tokens if llm_usage is not None else None,
            llm_completion_tokens=llm_usage.completion_tokens if llm_usage is not None else None,
            llm_total_tokens=llm_usage.total_tokens if llm_usage is not None else None,
        )
    except Exception:
        # Logging should never break the response.
        logger.exception("stage_d_logging_failed", request_id=req_id)

    return cv, req_id


def build_error_response(
    *,
    error_code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    suggestions: Optional[list[str]] = None,
    http_status: int = 400,
) -> Tuple[ErrorResponse, int]:
    """
    Build a standardized ErrorResponse for failed requests.

    This is the Stage D entrypoint for packaging fatal errors coming from any
    stage (A/B/C) or from the orchestrator itself. The caller is responsible
    for mapping (ErrorResponse, http_status) into HTTP response objects.
    """
    req_id = request_id or _generate_request_id()
    err = ErrorResponse(
        error_code=error_code,
        message=message,
        details=details or {},
        request_id=req_id,
        suggestions=suggestions or [],
    )

    logger.warning(
        "cv_generation_error",
        request_id=req_id,
        error_code=error_code,
        message=message,
        details=details or {},
        http_status=http_status,
    )

    return err, http_status


__all__ = [
    "LLMUsageSummary",
    "finalize_cv_response",
    "build_error_response",
]
