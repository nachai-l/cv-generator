# tests/test_stage_d_packaging.py

"""
Unit tests for Stage D – Response Packaging & Delivery.

Covers:
- finalize_cv_response(): enrichment of metadata, request_id generation,
  LLM usage aggregation, and basic consistency adjustments.
- build_error_response(): standardized ErrorResponse construction and
  HTTP status propagation.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from typing import Any, Dict

from schemas.output_schema import (
    CVGenerationResponse,
    EvidenceMapping,
    GenerationStatus,
    Justification,
    Metadata,
    OutputSkillItem,
    QualityMetrics,
    SectionContent,
    UnsupportedClaim,
    ValidationWarning,
)
from functions.stage_d_packaging import (
    LLMUsageSummary,
    build_error_response,
    finalize_cv_response,
)


def _build_minimal_cv_response(
    *,
    job_id: str = "JOB_test_123",
    template_id: str = "T_TEST_TEMPLATE",
    language: str = "en",
    status: GenerationStatus = GenerationStatus.COMPLETED,
    sections: Dict[str, SectionContent] | None = None,
) -> CVGenerationResponse:
    """
    Helper to construct a minimal-but-valid CVGenerationResponse
    for Stage D tests.
    """
    if sections is None:
        sections = {
            "profile_summary": SectionContent(
                text="This is a sample profile summary for testing.",
                word_count=7,
                matched_jd_skills=["testing", "python"],
                confidence_score=0.95,
            )
        }

    justification = Justification(
        evidence_map=[
            EvidenceMapping(
                section="profile_summary.text",
                sentence="This is a sample profile summary for testing.",
                evidence_ids=["work_exp#1"],
                match_score=0.9,
            )
        ],
        unsupported_claims=[
            UnsupportedClaim(
                section="profile_summary.text",
                claim="Led a team of 1000 engineers.",
                reason="Claim not found in profile evidence.",
                severity="warning",
            )
        ],
        coverage_score=0.9,
        total_claims_analyzed=2,
    )

    metadata = Metadata(
        generation_time_ms=10,  # Will be overridden by finalize_cv_response if timestamps supplied
        retry_count=0,
        cache_hit=False,
        sections_requested=0,
        sections_generated=0,
        tokens_used=0,
        cost_estimate_thb=0.0,
        profile_info=None,
    )

    quality_metrics = QualityMetrics(
        clarity_score=80.0,
        jd_alignment_score=75.0,
        completeness_score=70.0,
        consistency_score=85.0,
        overall_score=78.0,
        feedback=["Looks good overall."],
    )

    return CVGenerationResponse(
        job_id=job_id,
        template_id=template_id,
        language=language,
        status=status,
        sections=sections,
        skills=[
            OutputSkillItem(name="Python", level="L3_Advanced", source="taxonomy"),
        ],
        metadata=metadata,
        justification=justification,
        quality_metrics=quality_metrics,
        warnings=[],
        error=None,
        error_details=None,
    )


class TestStageDPackaging(unittest.TestCase):
    def test_finalize_cv_response_populates_request_id_and_metadata(self) -> None:
        """
        finalize_cv_response() should:
        - Generate a request_id when missing.
        - Compute generation_time_ms from start/end timestamps.
        - Update sections_generated / sections_requested to match sections.
        - Inject profile_info into metadata.profile_info.
        """
        cv = _build_minimal_cv_response()
        profile_info = {
            "name": "Test User",
            "email": "test@example.com",
        }

        start = datetime.now(timezone.utc)
        end = start + timedelta(milliseconds=500)

        finalized, req_id = finalize_cv_response(
            cv,
            request_id=None,
            user_id="user-123",
            profile_info=profile_info,
            llm_usage=None,
            generation_start=start,
            generation_end=end,
        )

        # Request ID should be generated and start with REQ_
        self.assertIsInstance(req_id, str)
        self.assertTrue(req_id.startswith("REQ_"))

        # generation_time_ms should be >= 0 and <= 60000 (Metadata validator)
        self.assertGreaterEqual(finalized.metadata.generation_time_ms, 0)
        self.assertLessEqual(finalized.metadata.generation_time_ms, 60000)

        # sections_generated and sections_requested should reflect actual sections
        num_sections = len(finalized.sections)
        self.assertEqual(finalized.metadata.sections_generated, num_sections)
        self.assertEqual(finalized.metadata.sections_requested, num_sections)

        # profile_info should be attached
        self.assertIsNotNone(finalized.metadata.profile_info)
        self.assertEqual(finalized.metadata.profile_info["name"], "Test User")
        self.assertEqual(finalized.metadata.profile_info["email"], "test@example.com")

    def test_finalize_cv_response_uses_llm_usage_summary(self) -> None:
        """
        finalize_cv_response() should:
        - Use LLMUsageSummary to populate tokens_used and cost_estimate_thb.
        - Override model_version if default and llm_usage.model is set.
        """
        cv = _build_minimal_cv_response()
        # Default model_version in Metadata is "gemini-2.5-flash"
        self.assertEqual(cv.metadata.model_version, "gemini-2.5-flash")

        usage = LLMUsageSummary(
            model="gemini-2.5-flash-internal",
            prompt_tokens=120,
            completion_tokens=80,
            total_tokens=200,
            total_cost_thb=0.42,
        )

        finalized, req_id = finalize_cv_response(
            cv,
            request_id="REQ_test_1",
            user_id="user-456",
            profile_info=None,
            llm_usage=usage,
            generation_start=None,
            generation_end=None,
        )

        self.assertEqual(req_id, "REQ_test_1")
        self.assertEqual(finalized.metadata.tokens_used, 200)
        self.assertAlmostEqual(finalized.metadata.cost_estimate_thb, 0.42, places=6)
        self.assertEqual(finalized.metadata.model_version, "gemini-2.5-flash-internal")

    def test_finalize_cv_response_explicit_token_cost_override(self) -> None:
        """
        Explicit total_tokens / total_cost_thb arguments should override
        values derived from LLMUsageSummary if both are provided.
        """
        cv = _build_minimal_cv_response()
        usage = LLMUsageSummary(
            model="gemini-2.5-flash-internal",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            total_cost_thb=0.30,
        )

        finalized, _ = finalize_cv_response(
            cv,
            request_id="REQ_test_2",
            user_id="user-789",
            profile_info=None,
            llm_usage=usage,
            total_tokens=999,
            total_cost_thb=9.99,
            generation_start=None,
            generation_end=None,
        )

        # Overrides should win
        self.assertEqual(finalized.metadata.tokens_used, 999)
        self.assertAlmostEqual(finalized.metadata.cost_estimate_thb, 9.99, places=6)

    def test_build_error_response_generates_request_id_and_status(self) -> None:
        """
        build_error_response() should:
        - Generate an ErrorResponse with the provided error_code/message.
        - Generate a request_id when missing.
        - Return the provided http_status unchanged.
        """
        err, status = build_error_response(
            error_code="GENERATION_FAILED",
            message="Stage B generation failed.",
            details={"section": "profile_summary"},
            request_id=None,
            suggestions=["Retry the request later."],
            http_status=500,
        )

        self.assertEqual(status, 500)
        self.assertEqual(err.status, "error")
        self.assertEqual(err.error_code, "GENERATION_FAILED")
        self.assertEqual(err.message, "Stage B generation failed.")
        self.assertIn("section", err.details or {})
        self.assertEqual(err.suggestions, ["Retry the request later."])
        self.assertIsNotNone(err.request_id)
        self.assertTrue(err.request_id.startswith("REQ_"))

    def test_build_error_response_uses_provided_request_id(self) -> None:
        """
        build_error_response() should respect a provided request_id
        instead of generating a new one.
        """
        err, status = build_error_response(
            error_code="VALIDATION_ERROR",
            message="Invalid input payload.",
            details=None,
            request_id="REQ_fixed_123",
            suggestions=[],
            http_status=422,
        )

        self.assertEqual(status, 422)
        self.assertEqual(err.request_id, "REQ_fixed_123")
        self.assertEqual(err.error_code, "VALIDATION_ERROR")
        self.assertEqual(err.message, "Invalid input payload.")


if __name__ == "__main__":
    unittest.main()
