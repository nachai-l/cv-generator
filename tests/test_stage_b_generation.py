"""Unittest suite for Stage B generation engine.

This module tests the core responsibilities of Stage B:

- Internal helpers:
    - _truncate_text
    - _get_section_char_limits

- Prompt construction for a single section:
    - _build_section_prompt

- LLM retry logic:
    - _call_llm_with_retries

- End-to-end behaviour of:
    - CVGenerationEngine.generate_cv

The tests use lightweight dummy objects instead of the full
CVGenerationRequest / template_info implementations to keep
the Stage B contract focused on the *logical* fields used
by the engine.
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import Any, Dict, List, cast

import functions.stage_b_generation as stage_b_generation
from functions.stage_b_generation import (
    CVGenerationEngine,
    _truncate_text,
    _get_section_char_limits,
)
from schemas.output_schema import CVGenerationResponse, SectionContent, OutputSkillItem
from schemas.input_schema import CVGenerationRequest
from schemas.internal_schema import EvidencePlan, SkillsSectionPlan, CanonicalSkill

import tempfile
from unittest.mock import patch

# ---------------------------------------------------------------------------
# Helper dummies
# ---------------------------------------------------------------------------


class DummyTemplateInfo:
    """Minimal template_info-like object for Stage B tests.

    Only the fields actually consumed by Stage B are provided:
    - template_id
    - max_chars_per_section
    """

    def __init__(self, template_id: str = "T_TEST", max_chars_per_section: Dict[str, int] | None = None):
        self.template_id = template_id
        self.max_chars_per_section: Dict[str, int] = max_chars_per_section or {}


class DummyRequest:
    """Duck-typed request object for Stage B tests.

    Only contains attributes accessed by CVGenerationEngine:

    - sections
    - language
    - user_id
    - template_info
    - profile_info
    - job_role_info
    - job_position_info
    - company_info
    - user_input_cv_text_by_section
    """

    def __init__(
        self,
        sections: List[str] | None = None,
        language: str = "en",
        user_id: str = "U_TEST",
        template_info: DummyTemplateInfo | None = None,
        profile_info: Dict[str, Any] | None = None,
        job_role_info: Dict[str, Any] | None = None,
        job_position_info: Dict[str, Any] | None = None,
        company_info: Dict[str, Any] | None = None,
        user_input_cv_text_by_section: Dict[str, str] | None = None,
    ) -> None:
        self.sections: List[str] = sections or ["profile_summary"]
        self.language: str = language
        self.user_id: str = user_id
        self.template_info: DummyTemplateInfo | None = template_info or DummyTemplateInfo()
        self.profile_info: Dict[str, Any] = profile_info or {"name": "Test User"}
        self.job_role_info: Dict[str, Any] | None = job_role_info
        self.job_position_info: Dict[str, Any] | None = job_position_info
        self.company_info: Dict[str, Any] | None = company_info
        self.user_input_cv_text_by_section: Dict[str, str] = user_input_cv_text_by_section or {}


class DummyEvidence:
    """Minimal Evidence-like object for prompt construction tests."""

    def __init__(self, evidence_id: str, fact: str) -> None:
        self.evidence_id = evidence_id
        self.fact = fact


class DummyEvidencePlan:
    """Duck-typed EvidencePlan for prompt construction tests.

    Attributes:
        section_hints: mapping from section_id to a list of evidence IDs.
        evidences: list of DummyEvidence objects.
    """

    def __init__(self, section_hints: Dict[str, List[str]], evidences: List[DummyEvidence]) -> None:
        self.section_hints = section_hints
        self.evidences = evidences


# ---------------------------------------------------------------------------
# Base test case with simple logging headers (for readability in CI)
# ---------------------------------------------------------------------------


class LoggingTestCase(unittest.TestCase):
    """Base TestCase that prints a readable header per test."""

    @classmethod
    def setUpClass(cls):
        # Patch llm_metrics CSV path → temp file for the whole suite
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls._csv_path = os.path.join(cls._tmpdir.name, "llm_call_logs_stage_b_tests.csv")

        # Clear caches first so the patch is effective
        try:
            from functions.utils import llm_metrics
            llm_metrics._load_config.cache_clear()
            llm_metrics._load_pricing.cache_clear()
            llm_metrics._get_csv_path.cache_clear()
        except Exception:
            pass

        cls._csv_patch = patch(
            "functions.utils.llm_metrics._get_csv_path",
            return_value=cls._csv_path,
        )
        cls._csv_patch.start()

    @classmethod
    def tearDownClass(cls):
        # Stop patch and cleanup temp dir
        cls._csv_patch.stop()
        cls._tmpdir.cleanup()

    def setUp(self) -> None:
        test_name = self._testMethodName
        print("\n" + "=" * 90, file=sys.stderr)
        print(f"🧪 STARTING TEST: {self.__class__.__name__}.{test_name}", file=sys.stderr)
        print("=" * 90, file=sys.stderr)

    def tearDown(self) -> None:
        print("-" * 90 + "\n", file=sys.stderr)



# ---------------------------------------------------------------------------
# Tests for internal helpers
# ---------------------------------------------------------------------------


class TestHelpers(LoggingTestCase):
    """Tests for Stage B internal helper functions."""

    def test_truncate_text_no_limit(self) -> None:
        """_truncate_text should return text unchanged when max_chars is None."""
        text = "hello world"
        self.assertEqual(_truncate_text(text, None), text)

    def test_truncate_text_shorter_than_limit(self) -> None:
        """_truncate_text should return text unchanged when len(text) < limit."""
        text = "short"
        self.assertEqual(_truncate_text(text, 10), text)

    def test_truncate_text_equal_to_limit(self) -> None:
        """_truncate_text should not add ellipsis when len(text) == limit."""
        text = "1234567890"  # exactly 10 chars
        result = _truncate_text(text, 10)
        self.assertEqual(result, text)

    def test_truncate_text_longer_than_limit_adds_ellipsis(self) -> None:
        """_truncate_text should truncate and append ellipsis when len(text) > limit."""
        text = "this is a long text that should be truncated"
        max_chars = 10
        result = _truncate_text(text, max_chars)

        self.assertTrue(result.endswith("…"))
        self.assertLessEqual(len(result), max_chars + 1)
        self.assertTrue(result[:-1].startswith(text[:max_chars].rstrip()))

    def test_get_section_char_limits_from_template_info(self) -> None:
        """_get_section_char_limits should return the template's max_chars_per_section dict."""
        tmpl = DummyTemplateInfo(
            template_id="T_TEST",
            max_chars_per_section={"profile_summary": 200, "skills": 100},
        )
        req = DummyRequest(template_info=tmpl)
        limits = _get_section_char_limits(cast(CVGenerationRequest, cast(object, req)))

        self.assertEqual(limits, {"profile_summary": 200, "skills": 100})

    def test_get_section_char_limits_without_template_info(self) -> None:
        """_get_section_char_limits should return empty dict when template_info is None."""
        req = DummyRequest()
        req.template_info = None  # type: ignore[assignment]
        limits = _get_section_char_limits(cast(CVGenerationRequest, cast(object, req)))
        self.assertEqual(limits, {})


# ---------------------------------------------------------------------------
# Tests for prompt building
# ---------------------------------------------------------------------------


class TestPromptBuilding(LoggingTestCase):
    """Tests for _build_section_prompt behaviour."""

    def test_build_section_prompt_includes_profile_role_company_and_evidence(self) -> None:
        """Prompt should include profile info, JD info, company, evidence, and user draft."""
        evidences = [
            DummyEvidence("ev1", "Did something important."),
            DummyEvidence("ev2", "Another fact."),
        ]
        ep = DummyEvidencePlan(
            section_hints={"profile_summary": ["ev1"]},
            evidences=evidences,
        )

        user_draft = {
            "profile_summary": "This is my draft summary.",
        }

        req = DummyRequest(
            sections=["profile_summary"],
            language="en",
            template_info=DummyTemplateInfo(template_id="T_EMPLOYER_STD_V3"),
            profile_info={"name": "Test User", "education": ["BSc"]},
            job_role_info={"role_name": "Data Scientist"},
            job_position_info={"title": "Junior Data Scientist"},
            company_info={"name": "Example Corp"},
            user_input_cv_text_by_section=user_draft,
        )

        engine = CVGenerationEngine(
            llm_client=lambda *_args, **_kwargs: "ignored",  # we only test the prompt
            generation_params={"log_prompts": False},
        )

        req_typed = cast(CVGenerationRequest, cast(object, req))
        ep_typed = cast(EvidencePlan, cast(object, ep))

        prompt = engine._build_section_prompt(
            request=req_typed,
            evidence_plan=ep_typed,
            section_id="profile_summary",
        )

        # Profile info
        self.assertIn("Test User", prompt)
        # Role and position info
        self.assertIn("Data Scientist", prompt)
        self.assertIn("Junior Data Scientist", prompt)
        # Company info
        self.assertIn("Example Corp", prompt)
        # Evidence fact
        self.assertIn("Did something important.", prompt)
        # User draft
        self.assertIn("This is my draft summary.", prompt)
        # Output requirements marker
        self.assertIn("=== Output Requirements ===", prompt)

    def test_cross_section_evidence_sharing_applies_correctly(self) -> None:
        """Evidence facts should be shared across sections according to parameters.yaml config."""
        # Fake parameters with cross-section sharing: profile_summary gets skills facts too.
        def fake_load_parameters() -> Dict[str, Any]:
            return {
                "generation": {
                    "prompts": {
                        "default": "DEFAULT PROMPT",
                    },
                },
                "cross_section_evidence_sharing": {
                    "default": [],
                    "profile_summary": ["skills"],
                },
            }

        # Patch the load_parameters *used inside stage_b_generation*
        original_load_parameters = stage_b_generation.load_parameters
        stage_b_generation.load_parameters = fake_load_parameters  # type: ignore[assignment]

        try:
            evidences = [
                DummyEvidence("ev_profile", "FACT_PROFILE"),
                DummyEvidence("ev_skill", "FACT_SKILL"),
            ]
            ep = DummyEvidencePlan(
                section_hints={
                    "profile_summary": ["ev_profile"],
                    "skills": ["ev_skill"],
                },
                evidences=evidences,
            )

            req = DummyRequest(
                sections=["profile_summary"],
                language="en",
                template_info=DummyTemplateInfo(template_id="T_EMPLOYER_STD_V3"),
                profile_info={"name": "Test User"},
            )

            engine = CVGenerationEngine(
                llm_client=lambda *_args, **_kwargs: "ignored",
                generation_params={"log_prompts": False},
            )

            req_typed = cast(CVGenerationRequest, cast(object, req))
            ep_typed = cast(EvidencePlan, cast(object, ep))

            prompt_profile = engine._build_section_prompt(
                request=req_typed,
                evidence_plan=ep_typed,
                section_id="profile_summary",
            )

            # Should include both its own fact and the shared skills fact
            self.assertIn("FACT_PROFILE", prompt_profile)
            self.assertIn("FACT_SKILL", prompt_profile)
        finally:
            # Restore original load_parameters to avoid side effects on other tests
            stage_b_generation.load_parameters = original_load_parameters  # type: ignore[assignment]

    def test_section_specific_prompt_override_from_parameters(self) -> None:
        """Section-specific prompt overrides (generation.prompts[section_id]) should be used."""
        def fake_load_parameters() -> Dict[str, Any]:
            return {
                "generation": {
                    "prompts": {
                        "default": "DEFAULT PROMPT",
                        "experience": "EXPERIENCE PROMPT OVERRIDE",
                    },
                },
                "cross_section_evidence_sharing": {
                    "default": [],
                },
            }

        original_load_parameters = stage_b_generation.load_parameters
        stage_b_generation.load_parameters = fake_load_parameters  # type: ignore[assignment]

        try:
            ep = DummyEvidencePlan(section_hints={}, evidences=[])

            req = DummyRequest(
                sections=["experience"],
                language="en",
                template_info=DummyTemplateInfo(template_id="T_EMPLOYER_STD_V3"),
            )

            engine = CVGenerationEngine(
                llm_client=lambda *_args, **_kwargs: "ignored",
                generation_params={"log_prompts": False},
            )

            req_typed = cast(CVGenerationRequest, cast(object, req))
            ep_typed = cast(EvidencePlan, cast(object, ep))

            prompt_exp = engine._build_section_prompt(
                request=req_typed,
                evidence_plan=ep_typed,
                section_id="experience",
            )

            self.assertIn("=== Output Requirements ===", prompt_exp)
            self.assertIn("EXPERIENCE PROMPT OVERRIDE", prompt_exp)
            self.assertNotIn("DEFAULT PROMPT", prompt_exp)
        finally:
            stage_b_generation.load_parameters = original_load_parameters  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Tests for LLM call retry logic
# ---------------------------------------------------------------------------


class TestLLMRetries(LoggingTestCase):
    """Tests for _call_llm_with_retries behaviour."""

    def test_call_llm_with_retries_succeeds_after_transient_failure(self) -> None:
        """LLM call should succeed on third attempt when first two fail.

        Note: in the current implementation, `max_retries` is the *total*
        number of attempts (not extra retries), so we set it to 3 here.
        """
        calls: Dict[str, int] = {"count": 0}

        def flaky_llm(_prompt: str, **_kwargs: Any) -> str:
            calls["count"] += 1
            if calls["count"] < 3:
                raise RuntimeError("Temporary failure")
            return "final result"

        engine = CVGenerationEngine(
            llm_client=flaky_llm,
            generation_params={
                "max_retries": 3,  # total attempts = 3
            },
        )

        result = engine._call_llm_with_retries(
            prompt="test prompt",
            section_id="profile_summary",
        )

        self.assertEqual(result, "final result")
        self.assertEqual(calls["count"], 3)

    def test_call_llm_with_retries_raises_after_all_attempts_fail(self) -> None:
        """If all attempts fail, _call_llm_with_retries should raise RuntimeError."""

        def always_fail_llm(_prompt: str, **_kwargs: Any) -> str:
            raise RuntimeError("Permanent failure")

        engine = CVGenerationEngine(
            llm_client=always_fail_llm,
            generation_params={"max_retries": 1},
        )

        with self.assertRaises(RuntimeError) as ctx:
            engine._call_llm_with_retries(
                prompt="test prompt",
                section_id="skills",
            )

        self.assertIn("LLM generation failed for section 'skills'", str(ctx.exception))


# ---------------------------------------------------------------------------
# End-to-end generation test (happy path)
# ---------------------------------------------------------------------------


class TestEndToEndGeneration(LoggingTestCase):
    """End-to-end tests for CVGenerationEngine.generate_cv."""

    def test_generate_cv_end_to_end_single_section_with_truncation(self) -> None:
        """End-to-end: LLM is called once, text is truncated, response schema is correct."""
        llm_calls: Dict[str, int] = {"count": 0}

        def fake_llm(_prompt: str, **_kwargs: Any) -> str:
            llm_calls["count"] += 1
            return "X" * 100  # long text that should be truncated

        tmpl = DummyTemplateInfo(
            template_id="T_EMPLOYER_STD_V3",
            max_chars_per_section={"profile_summary": 20},
        )
        req = DummyRequest(
            sections=["profile_summary"],
            language="en",
            user_id="U123",
            template_info=tmpl,
            profile_info={
                "name": "Test User",
                "summary": "Short summary used as source data.",
            },
        )

        engine = CVGenerationEngine(
            llm_client=fake_llm,
            generation_params={
                "model": "test-model",
                "temperature": 0.1,
                "top_p": 0.9,
                "max_output_tokens": 128,
                "max_retries": 1,  # 🔹 one total attempt, no extra retries
                "timeout_seconds": 5,
                "log_prompts": False,
            },
        )

        req_typed = cast(CVGenerationRequest, cast(object, req))

        response = engine.generate_cv(req_typed, evidence_plan=None)

        # LLM was called exactly once
        self.assertEqual(llm_calls["count"], 1)

        # Response type and key fields
        self.assertIsInstance(response, CVGenerationResponse)
        self.assertTrue(hasattr(response, "sections"))
        self.assertEqual(response.language, "en")

        # template_id and user_id are only set if they exist in the schema
        if hasattr(response, "template_id"):
            self.assertEqual(response.template_id, "T_EMPLOYER_STD_V3")
        if hasattr(response, "job_id"):
            # job_id pattern is enforced by the schema; here we only check suffix
            self.assertIn("U123", response.job_id)

        # Sections should be a dict keyed by section_id
        self.assertIsInstance(response.sections, dict)
        self.assertIn("profile_summary", response.sections)
        sec = response.sections["profile_summary"]
        self.assertIsInstance(sec, SectionContent)

        # Content field is "text" in SectionContent
        content = sec.text
        self.assertIsInstance(content, str)

        # Content should be truncated to <= 21 chars (20 + ellipsis)
        self.assertLessEqual(len(content), 21)
        self.assertTrue(content.endswith("…"))


# ---------------------------------------------------------------------------
# Skills-specific tests (taxonomy-preserving behaviour)
# ---------------------------------------------------------------------------

class TestSkillsStructuredGeneration(LoggingTestCase):
    """Tests for structured skills generation in Stage B."""

    def test_generate_structured_skills_respects_keep_flag_and_allows_new_skills(self) -> None:
        """_generate_structured_skills should respect keep=false and allow new inferred skills."""
        # Build a simple skills plan with one canonical skill
        skills_plan = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Molecular Biology", level="L4_Expert", taxonomy_id=None),
            ],
            # With new behaviour we don't rely on allowed_additional_skills as a whitelist
            allowed_additional_skills=[],
        )

        # Fake JSON from LLM:
        # - canonical skill (keep=true) -> should be kept
        # - new inferred skill (keep=true) -> should be kept
        # - skill with keep=false -> should be dropped
        fake_json = """
        {
          "items": [
            { "name": "Molecular Biology", "level": "L4_Expert", "keep": true, "source": "taxonomy" },
            { "name": "New Inferred Skill", "level": "L3_Advanced", "keep": true, "source": "inferred" },
            { "name": "Drop Me", "level": null, "keep": false, "source": "inferred" }
          ]
        }
        """

        # Engine with dummy llm client; we override _call_llm_with_retries to return fake_json
        engine = CVGenerationEngine(
            llm_client=lambda *_args, **_kwargs: fake_json,
            generation_params={"max_retries": 1},
        )

        # Patch instance method for this engine only so it always returns fake_json
        engine._call_llm_with_retries = lambda prompt, section_id: fake_json  # type: ignore[assignment]

        req = DummyRequest(
            sections=["skills"],
            language="en",
            profile_info={"name": "Test User"},
        )
        req_typed = cast(CVGenerationRequest, cast(object, req))

        result = engine._generate_structured_skills(req_typed, skills_plan)

        # We expect two skills: canonical + new inferred; the keep=false one is dropped
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(s, OutputSkillItem) for s in result))

        names = [s.name for s in result]
        self.assertIn("Molecular Biology", names)
        self.assertIn("New Inferred Skill", names)
        self.assertNotIn("Drop Me", names)

        # Canonical skill should still be tagged as taxonomy
        canonical = next(s for s in result if s.name == "Molecular Biology")
        self.assertEqual(canonical.level, "L4_Expert")
        self.assertEqual(canonical.source, "taxonomy")

    def test_generate_cv_populates_structured_skills_from_profile(self) -> None:
        """generate_cv should populate response.skills when 'skills' section is requested."""
        # Dummy LLM that returns some filler text for all sections
        def fake_llm(_prompt: str, **_kwargs: Any) -> str:
            return "Generated section text."

        # Profile info contains skills in taxonomy form (name + level)
        profile_info = {
            "name": "Test User",
            "skills": [
                {"name": "Molecular Biology", "level": "L4_Expert"},
                {"name": "Analytical Thinking", "level": "L3_Advanced"},
            ],
        }

        req = DummyRequest(
            sections=["profile_summary", "skills"],
            language="en",
            profile_info=profile_info,
        )
        req_typed = cast(CVGenerationRequest, cast(object, req))

        engine = CVGenerationEngine(
            llm_client=fake_llm,
            generation_params={"max_retries": 1, "log_prompts": False},
        )

        # Stub structured skills generation to avoid relying on JSON parsing in this test
        stub_skills = [
            OutputSkillItem(name="Molecular Biology", level="L4_Expert", source="taxonomy"),
            OutputSkillItem(name="Analytical Thinking", level="L3_Advanced", source="taxonomy"),
        ]
        engine._generate_structured_skills = (  # type: ignore[assignment]
            lambda _request, _plan: stub_skills
        )

        response = engine.generate_cv(req_typed, evidence_plan=None)

        # Ensure sections still exist (including 'skills' as text section)
        self.assertIn("skills", response.sections)
        self.assertIsInstance(response.sections["skills"], SectionContent)

        # Ensure structured skills list is populated
        self.assertIsNotNone(response.skills)
        self.assertEqual(len(response.skills or []), 2)
        names = [s.name for s in response.skills or []]
        self.assertIn("Molecular Biology", names)
        self.assertIn("Analytical Thinking", names)

    def test_generate_cv_skips_sections_without_source_data(self) -> None:
        """Sections with no backing data in profile_info or user drafts should be skipped.

        Requested sections:
        - profile_summary
        - volunteering

        profile_info only contains a name, no 'summary' or 'volunteering' fields.
        user_input_cv_text_by_section is empty.

        Expected:
        - No LLM calls are made.
        - response.sections is empty (no generated sections).
        """
        llm_calls: Dict[str, int] = {"count": 0}

        def fake_llm(_prompt: str, **_kwargs: Any) -> str:
            llm_calls["count"] += 1
            return "This should never be called."

        tmpl = DummyTemplateInfo(
            template_id="T_EMPLOYER_STD_V3",
            max_chars_per_section={"profile_summary": 200, "volunteering": 200},
        )

        # profile_info has only a name → no mapped fields for profile_summary / volunteering
        req = DummyRequest(
            sections=["profile_summary", "volunteering"],
            language="en",
            user_id="U_SKIP",
            template_info=tmpl,
            profile_info={"name": "Test User"},
            user_input_cv_text_by_section={},  # no drafts
        )

        engine = CVGenerationEngine(
            llm_client=fake_llm,
            generation_params={
                "model": "test-model",
                "max_retries": 1,
                "log_prompts": False,
            },
        )

        req_typed = cast(CVGenerationRequest, cast(object, req))

        response = engine.generate_cv(req_typed, evidence_plan=None)

        # ✅ No LLM calls because no effective sections
        self.assertEqual(llm_calls["count"], 0)

        # ✅ No sections generated
        self.assertIsInstance(response, CVGenerationResponse)
        self.assertIsInstance(response.sections, dict)
        self.assertEqual(len(response.sections), 0)

        # ✅ Metadata still consistent: 2 requested, 0 generated
        self.assertEqual(response.metadata.sections_requested, 2)
        self.assertEqual(response.metadata.sections_generated, 0)

if __name__ == "__main__":
    unittest.main()
