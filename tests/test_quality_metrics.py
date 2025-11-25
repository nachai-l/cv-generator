"""
Unit Tests for quality_metrics.py
=================================

Comprehensive suite validating heuristic scoring logic in
`functions.utils.quality_metrics`.

Each test prints a concise one-line summary indicating key outcomes:
   [PASS] test_name → scores={...}
   [CHECK] test_name → clarity=.. jd=.. complete=.. consistent=.. overall=..

Run with:
    python -m unittest -v tests/test_quality_metrics.py
"""

import sys
import unittest
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from functions.utils.quality_metrics import compute_quality_metrics
from schemas.output_schema import (
    CVGenerationResponse,
    SectionContent,
    Metadata,
    Justification,
    EvidenceMapping,
    GenerationStatus,
    QualityMetrics,
)


# ---------------------------------------------------------------------------
# Pretty test base class (same as security tests)
# ---------------------------------------------------------------------------

class PrettyTestCase(unittest.TestCase):
    """Print consistent headers/dividers for readability."""

    def setUp(self):
        test_name = self._testMethodName
        print("\n" + "=" * 90, file=sys.stderr)
        print(f"🧪 STARTING TEST: {test_name}", file=sys.stderr)
        print("=" * 90, file=sys.stderr)

    def tearDown(self):
        print("-" * 90 + "\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Utilities for creating minimal CV responses for testing
# ---------------------------------------------------------------------------

def _make_section(text: str, wc: int = None):
    """Helper to create a SectionContent with word_count auto-calculated."""
    if wc is None:
        wc = len(text.split())
    return SectionContent(
        text=text,
        word_count=wc,
        matched_jd_skills=[],
        confidence_score=1.0,
    )


def _make_meta(sections_req=4, sections_gen=4, total_words=200):
    """Helper for Metadata."""
    return Metadata(
        generated_at=datetime.now(timezone.utc),
        model_version="gemini-2.5-flash",
        generation_time_ms=1000,
        retry_count=0,
        cache_hit=False,
        sections_requested=sections_req,
        sections_generated=sections_gen,
        tokens_used=0,
        input_tokens=500,
        output_tokens=300,
        section_breakdown=[],
        cost_estimate_thb=0.0,
        cost_estimate_usd=0.0,
        profile_info={"skills_count": 5},
        request_id="REQ_TEST",
        stage_c_validated=True,
        stage_d_completed=True,
    )


def _make_justification(cov=1.0, unsupported=0):
    """Helper for Justification."""
    evidence = [
        EvidenceMapping(
            section="skills",
            sentence="example sentence",
            evidence_ids=["profile_info.skills[0]"],
            match_score=1.0,
        )
    ]
    unsupported_claims = []
    for i in range(unsupported):
        unsupported_claims.append(
            {
                "section": "skills",
                "claim": f"bad{i}",
                "reason": "none",
            }
        )

    return Justification(
        evidence_map=evidence,
        unsupported_claims=unsupported_claims,
        coverage_score=cov,
        total_claims_analyzed=1 + unsupported,
    )


def _make_response(sections: dict, meta: Metadata, justif: Justification):
    """Helper to create CVGenerationResponse."""
    return CVGenerationResponse(
        job_id="JOB_TEST",
        template_id="T_TEMPLATE",
        language="en",
        status=GenerationStatus.COMPLETED,
        sections=sections,
        skills=[],  # structured skills optional
        metadata=meta,
        justification=justif,
        quality_metrics=None,
        warnings=[],
        error=None,
        error_details=None,
    )


# ---------------------------------------------------------------------------
# Test suite
# ---------------------------------------------------------------------------

class TestQualityMetricsBasic(PrettyTestCase):
    """Core tests for quality_metrics.py."""

    @classmethod
    def setUpClass(cls):
        print("\n=== Core quality_metrics tests ===")

    def test_full_good_cv(self):
        """A normal, well-structured CV should yield reasonably high scores."""
        sections = {
            "profile_summary": _make_section(
                "This is a well structured profile summary. It has clear sentences."
            ),
            "skills": _make_section(
                "Python, Machine Learning, Leadership."
            ),
            "experience": _make_section(
                "Led projects. Managed teams. Delivered results."
            ),
            "education": _make_section("BSc in Computer Science."),
        }
        meta = _make_meta()
        justif = _make_justification(cov=1.0)

        resp = _make_response(sections, meta, justif)
        metrics: QualityMetrics = compute_quality_metrics(
            resp,
            jd_required_skills=["python", "leadership"],
        )

        print(
            f"[PASS] test_full_good_cv → clarity={metrics.clarity_score}, "
            f"jd={metrics.jd_alignment_score}, complete={metrics.completeness_score}, "
            f"consistent={metrics.consistency_score}, overall={metrics.overall_score}"
        )

        # Actual clarity from current heuristics is ~60
        self.assertGreaterEqual(metrics.clarity_score, 60.0)
        self.assertGreaterEqual(metrics.completeness_score, 80.0)
        self.assertGreater(metrics.overall_score, 50.0)

    def test_no_jd_skills(self):
        """If JD skills = None, JD alignment score should be 0."""
        sections = {
            "profile_summary": _make_section("Short CV example text."),
            "skills": _make_section("Python, AI, Data Science."),
        }
        meta = _make_meta(sections_req=2, sections_gen=2)
        justif = _make_justification(cov=0.9)

        resp = _make_response(sections, meta, justif)
        metrics = compute_quality_metrics(resp, jd_required_skills=None)

        print(
            f"[CHECK] test_no_jd_skills → jd={metrics.jd_alignment_score}, "
            f"feedback={metrics.feedback}"
        )

        self.assertEqual(metrics.jd_alignment_score, 0.0)
        self.assertTrue(
            any("not applicable" in f.lower() for f in metrics.feedback)
        )

    def test_low_completeness_missing_sections(self):
        """Missing common sections should reduce completeness."""
        sections = {
            "profile_summary": _make_section("Summary only minimal content."),
        }
        meta = _make_meta(sections_req=4, sections_gen=1)
        justif = _make_justification(cov=0.8)

        resp = _make_response(sections, meta, justif)
        metrics = compute_quality_metrics(resp, jd_required_skills=["python"])

        print(
            f"[CHECK] test_low_completeness_missing_sections → "
            f"completeness={metrics.completeness_score}, feedback={metrics.feedback}"
        )

        self.assertLess(metrics.completeness_score, 60.0)
        self.assertTrue(
            any("missing" in f.lower() for f in metrics.feedback)
        )

    def test_consistency_penalty_unsupported_claims(self):
        """Unsupported claims should apply penalties."""
        sections = {
            "profile_summary": _make_section("Basic summary content here."),
            "skills": _make_section("Python skills."),
        }
        meta = _make_meta(sections_req=2, sections_gen=2)
        justif = _make_justification(cov=0.9, unsupported=3)

        resp = _make_response(sections, meta, justif)
        metrics = compute_quality_metrics(resp, jd_required_skills=["python"])

        print(
            f"[CHECK] test_consistency_penalty_unsupported_claims → "
            f"consistency={metrics.consistency_score}, feedback={metrics.feedback}"
        )

        # Base 0.9 * 100 = 90, minus 3 * 5 = 15 → 75 with current heuristics.
        # We just need to assert a penalty is applied but not destroyed.
        self.assertLess(metrics.consistency_score, 90.0)
        self.assertGreaterEqual(metrics.consistency_score, 70.0)
        self.assertTrue(
            any("claims" in f.lower() for f in metrics.feedback)
        )

    def test_clarity_penalty_very_short_sections(self):
        """Very short sections should reduce clarity score."""
        sections = {
            "profile_summary": _make_section("Hi short text."),
            "skills": _make_section("Python skills."),
        }
        meta = _make_meta(sections_req=2, sections_gen=2)
        justif = _make_justification(cov=1.0)

        resp = _make_response(sections, meta, justif)
        metrics = compute_quality_metrics(resp, jd_required_skills=["python"])

        print(
            f"[CHECK] test_clarity_penalty_very_short_sections → "
            f"clarity={metrics.clarity_score}"
        )

        self.assertLess(metrics.clarity_score, 70.0)

    def test_overall_score_computation(self):
        """Ensure overall score is within a reasonable bounded range."""
        sections = {
            "profile_summary": _make_section(
                "Moderate clarity sentence. Another one for balance."
            ),
            "skills": _make_section("Python, ML, Data Science."),
        }
        meta = _make_meta(sections_req=2, sections_gen=2)
        justif = _make_justification(cov=0.5)

        resp = _make_response(sections, meta, justif)
        metrics = compute_quality_metrics(resp, jd_required_skills=["python"])

        print(
            f"[PASS] test_overall_score_computation → overall={metrics.overall_score}"
        )

        self.assertGreaterEqual(metrics.overall_score, 30.0)
        self.assertLessEqual(metrics.overall_score, 100.0)


if __name__ == "__main__":
    unittest.main()
