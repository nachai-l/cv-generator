# tests_utils/test_stage_c_validation.py

"""
Unittest suite for Stage C validation.

This module tests_utils the core responsibilities of Stage C:

- Internal helpers:
    - _compute_allowed_sections
    - _clean_text
    - _sanitize_skills (via run_stage_c_validation)

- End-to-end behaviour of:
    - run_stage_c_validation

Stage C is responsible for taking a CVGenerationResponse from Stage B
and enforcing:
    - section set & ordering according to template_info
    - basic text cleanup and truncation rules
    - skills normalization, deduplication, and max length
    - lightweight metadata annotations (warnings, stage_c_validated flag)
"""

from __future__ import annotations

import sys
import unittest
from typing import Any, Dict, List, Tuple, cast
from unittest.mock import patch

from schemas.output_schema import CVGenerationResponse, SectionContent, OutputSkillItem
import functions.stage_c_validation as stage_c
from functions.stage_c_validation import run_stage_c_validation, StageCValidationError


# ---------------------------------------------------------------------------
# Helper dummy metadata / request objects
# ---------------------------------------------------------------------------

class DummyNewRequestForStageC:
    """
    Minimal 'new-shape' request-like object for Stage C tests_utils.

    It has:
    - template_id
    - sections
    - language
    - job_id

    and NO .template_info attribute (to exercise the new resolver path).
    """

    def __init__(
        self,
        template_id: str = "T_NEW",
        sections: List[str] | None = None,
        language: Any = "en",
        job_id: str = "JOB_NEW",
    ) -> None:
        self.template_id = template_id
        self.sections = sections or ["profile_summary"]
        self.language = language
        self.job_id = job_id


class DummyMeta:
    """Minimal metadata-like object used in Stage C tests_utils."""

    def __init__(self) -> None:
        self.validation_warnings: List[str] = []
        # Optional profile_info used by _get_meta_field in Stage C
        self.profile_info: Dict[str, Any] | None = None


class DummyTemplateInfoForReq:
    """Minimal template_info-like object for cross-check tests_utils."""

    def __init__(self, template_id: str) -> None:
        self.template_id = template_id


class DummyRequestForStageC:
    """Minimal request-like object for cross-check tests_utils."""

    def __init__(
        self,
        template_id: str = "T_REQ",
        job_id: str = "JOB_REQ",
        language: Any = "en",
    ) -> None:
        self.template_info = DummyTemplateInfoForReq(template_id)
        self.job_id = job_id
        self.language = language


class DummyLangEnum:
    """Simple enum-like object with a .value attribute used to test language normalization."""

    def __init__(self, value: str) -> None:
        self.value = value


# ---------------------------------------------------------------------------
# Base test case with simple logging headers (for readability in CI)
# ---------------------------------------------------------------------------


class LoggingTestCase(unittest.TestCase):
    """Base TestCase that prints a readable header per test."""

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
    """Tests for Stage C internal helper functions."""

    def test_compute_allowed_sections_respects_template_order(self) -> None:
        """_compute_allowed_sections should respect sections_order and ignore duplicates."""
        template_info = {
            "sections_order": [
                "profile_summary",
                {"id": "experience"},
                "skills",
                "experience",  # duplicate
                {"id": None},  # invalid
                42,  # invalid
            ]
        }

        allowed = stage_c._compute_allowed_sections(template_info)

        self.assertEqual(
            allowed,
            ["profile_summary", "experience", "skills"],
        )

    def test_clean_text_normalizes_and_strips_artifacts(self) -> None:
        """_clean_text should normalize newlines, strip code fences and extra blank lines."""
        raw = '  """\r\nline1\r\n\r\n\r\n```code```  \n\nline2\n\n\n  "'
        cleaned = stage_c._clean_text(raw, enable_cleaning=True)

        # No CRLF, no ``` or surrounding quotes, no triple blank lines
        self.assertNotIn("\r", cleaned)
        self.assertNotIn("```", cleaned)
        self.assertFalse(cleaned.startswith('"'))
        self.assertFalse(cleaned.endswith('"'))

        # At most double newlines
        self.assertNotIn("\n\n\n", cleaned)


# ---------------------------------------------------------------------------
# Tests for run_stage_c_validation – sections behaviour
# ---------------------------------------------------------------------------


class TestSectionValidation(LoggingTestCase):
    """Tests for section sanitization via run_stage_c_validation."""

    def _fake_validation_cfg(self) -> Dict[str, Any]:
        return {
            "validation": {
                "max_section_chars_default": 50,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 10,
                # leave other keys defaulted by _load_validation_params
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
        }
    def test_json_section_never_truncated_even_if_over_limit(self) -> None:
        """
        JSON-like sections (e.g., skills_structured) must NEVER be truncated,
        even when their length exceeds the per-section max_chars_per_section.
        This relies on smart_truncate_markdown's JSON fast-path.
        """
        meta = DummyMeta()

        json_text = '{"skills": [{"name": "Python", "level": "Advanced"}, {"name": "SQL", "level": "Intermediate"}]}'
        self.assertGreater(len(json_text), 20)  # sanity check

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "skills_structured": SectionContent.model_construct(text=json_text),
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["skills_structured"],
            "max_chars_per_section": {
                # Intentionally tiny limit to force truncation *if* JSON were not protected
                "skills_structured": 20,
            },
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        out_text = validated.sections["skills_structured"].text

        # JSON should be returned EXACTLY as-is (no truncation)
        self.assertEqual(out_text, json_text)
        # And there should be no truncation warning mentioning this section
        self.assertFalse(
            any("skills_structured" in msg and "truncated" in msg for msg in meta.validation_warnings),
            msg=f"JSON section should not be truncated, warnings: {meta.validation_warnings}",
        )

    def test_template_order_respected_and_extra_sections_kept(self) -> None:
        """
        run_stage_c_validation should respect template_info order but
        keep any extra sections that Stage B already generated.
        """
        # Response from Stage B with an allowed section and an extra one
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(text="Valid content"),
                "extra_section": SectionContent.model_construct(text="Should be kept"),
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 100},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        # Order: template-defined first, then extra sections
        self.assertEqual(list(validated.sections.keys()), ["profile_summary", "extra_section"])
        self.assertIn("profile_summary", validated.sections)
        self.assertIn("extra_section", validated.sections)

    def test_empty_section_dropped_with_warning(self) -> None:
        """Sections that become empty after cleaning should be dropped and logged."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "empty_sec": SectionContent.model_construct(text="   \n\n "),
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["empty_sec"],
            "max_chars_per_section": {"empty_sec": 100},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        # Section removed
        self.assertEqual(len(validated.sections), 0)

        # Warning attached to metadata
        self.assertTrue(
            any("empty_sec" in msg and "removed as empty" in msg for msg in meta.validation_warnings),
            msg=f"Expected warning about empty_sec removal, got: {meta.validation_warnings}",
        )

    def test_section_truncated_to_template_limit(self) -> None:
        """run_stage_c_validation should truncate sections according to template limits."""
        long_text = "X" * 50
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(text=long_text),
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 10},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        text = validated.sections["profile_summary"].text
        self.assertEqual(len(text), 10)
        self.assertTrue(
            any("profile_summary" in msg and "truncated" in msg for msg in meta.validation_warnings),
            msg=f"Expected truncation warning, got: {meta.validation_warnings}",
        )


# ---------------------------------------------------------------------------
# Tests for run_stage_c_validation – skills behaviour
# ---------------------------------------------------------------------------


class TestSkillsValidation(LoggingTestCase):
    """Tests for skills sanitization and capping via run_stage_c_validation."""

    def _fake_validation_cfg(self) -> Dict[str, Any]:
        return {
            "validation": {
                "max_section_chars_default": 1000,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 2,  # small cap for testing
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
        }

    def test_skills_normalized_deduplicated_and_capped(self) -> None:
        """
        Skills should be stripped, deduplicated by name (case-insensitive),
        and capped to max_skills, without changing canonical levels.
        """
        meta = DummyMeta()

        skills_raw: List[Any] = [
            OutputSkillItem.model_construct(name=" Skill ", level="L1", source="profile"),
            OutputSkillItem.model_construct(name="skill", level="L1", source="profile"),  # duplicate name
            "Other",  # raw string → treated as name
        ]

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=skills_raw,
        )

        template_info: Dict[str, Any] = {
            "sections_order": [],
            "max_chars_per_section": {},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        skills = validated.skills or []
        self.assertLessEqual(len(skills), 2)

        names = {s.name for s in skills}
        # Dedup should keep only one "Skill" variant
        self.assertIn("Skill", names)
        self.assertEqual(
            sum(1 for n in names if n.lower() == "skill"),
            1,
        )

        # Capping should produce a warning
        self.assertTrue(
            any("Skills list truncated" in msg for msg in meta.validation_warnings),
            msg=f"Expected skills truncation warning, got: {meta.validation_warnings}",
        )

    def test_skills_dedup_by_name_even_if_levels_differ(self) -> None:
        """
        If the same skill name appears with different levels, Stage C should
        still emit only one entry for that skill name (defensive behaviour),
        without trying to override a non-empty level chosen upstream.
        """
        meta = DummyMeta()

        skills_raw: List[Any] = [
            OutputSkillItem.model_construct(name="Python", level="L3", source="profile"),
            OutputSkillItem.model_construct(name="Python", level="L4", source="profile"),  # different level
        ]

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=skills_raw,
        )

        template_info: Dict[str, Any] = {
            "sections_order": [],
            "max_chars_per_section": {},
        }

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                }
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        skills = validated.skills or []
        python_skills = [s for s in skills if s.name.lower() == "python"]
        self.assertEqual(len(python_skills), 1, msg=f"Expected only one Python skill, got: {skills}")


# ---------------------------------------------------------------------------
# Tests for metadata behaviour
# ---------------------------------------------------------------------------


class TestMetadataBehaviour(LoggingTestCase):
    """Tests for metadata flags added by Stage C."""

    def _fake_validation_cfg(self) -> Dict[str, Any]:
        return {
            "validation": {
                "max_section_chars_default": 1000,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 10,
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
        }

    def test_metadata_stage_c_validated_flag_set(self) -> None:
        """run_stage_c_validation should set metadata.stage_c_validated=True if metadata exists."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {
            "sections_order": [],
            "max_chars_per_section": {},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        self.assertTrue(
            getattr(validated.metadata, "stage_c_validated", False),
            msg="metadata.stage_c_validated should be True after Stage C",
        )


# ---------------------------------------------------------------------------
# Tests for global validation rules (name/email, min skills/education, strict mode)
# ---------------------------------------------------------------------------


class TestGlobalValidationRules(LoggingTestCase):
    """Tests for new global validation knobs from parameters.yaml."""

    def test_require_name_email_soft_mode_adds_warning(self) -> None:
        """Missing name/email should add warnings but not raise when strict_mode=False."""
        meta = DummyMeta()
        # profile_info exists but without name/email
        meta.profile_info = {}

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {"sections_order": [], "max_chars_per_section": {}}

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "require_name": True,
                    "require_email": True,
                    "strict_mode": False,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                }
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        self.assertIs(validated, resp)
        joined = " ".join(meta.validation_warnings)
        self.assertIn("Missing required metadata fields", joined)
        self.assertIn("name", joined)
        self.assertIn("email", joined)

    def test_require_name_email_strict_mode_raises(self) -> None:
        """Missing name/email should raise StageCValidationError when strict_mode=True."""
        meta = DummyMeta()
        meta.profile_info = {}

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {"sections_order": [], "max_chars_per_section": {}}

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "require_name": True,
                    "require_email": True,
                    "strict_mode": True,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                }
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            with self.assertRaises(StageCValidationError):
                run_stage_c_validation(
                    resp,
                    template_info=template_info,
                    original_request=None,
                )

    def test_min_skills_required_soft_mode_adds_warning(self) -> None:
        """If skills < min_skills_required, add warning but don't raise in non-strict mode."""
        meta = DummyMeta()
        skills = [
            OutputSkillItem.model_construct(name="Molecular Biology", level="L4", source="taxonomy"),
        ]  # only 1 skill

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=skills,
        )

        template_info: Dict[str, Any] = {"sections_order": [], "max_chars_per_section": {}}

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "min_skills_required": 3,
                    "strict_mode": False,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 50,
                }
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        self.assertIs(validated, resp)
        self.assertTrue(
            any("min_skills_required=3" in msg for msg in meta.validation_warnings),
            msg=f"Expected min_skills warning, got: {meta.validation_warnings}",
        )

    def test_missing_education_section_strict_mode_raises(self) -> None:
        """If min_education_required>0 and no education section, strict_mode=True should raise."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},  # no 'education'
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {
            "sections_order": [],
            "max_chars_per_section": {},
        }

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "min_education_required": 1,
                    "strict_mode": True,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                }
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            with self.assertRaises(StageCValidationError):
                run_stage_c_validation(
                    resp,
                    template_info=template_info,
                    original_request=None,
                )


# ---------------------------------------------------------------------------
# Tests for cross-checks vs original_request
# ---------------------------------------------------------------------------


class TestCrossChecksAgainstRequest(LoggingTestCase):
    """Tests that template_id, job_id, and language are reconciled with the original request."""

    def _base_cfg(self) -> Dict[str, Any]:
        return {
            "validation": {
                "max_section_chars_default": 1000,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 10,
                "strict_mode": False,
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
        }

    def test_template_id_mismatch_adds_warning_non_strict(self) -> None:
        """If template_id differs between request and response, Stage C should warn (non-strict)."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_RESP",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        req = DummyRequestForStageC(template_id="T_REQ", job_id="JOB1", language="en")

        with patch("functions.stage_c_validation.load_parameters", lambda: self._base_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info={},
                original_request=req,
            )

        self.assertIs(validated, resp)
        joined = " ".join(meta.validation_warnings)
        self.assertIn("template_id mismatch", joined)

    def test_missing_template_id_in_response_is_filled_from_request(self) -> None:
        """If response.template_id is missing, Stage C should populate it from request."""
        meta = DummyMeta()
        # template_id intentionally omitted
        resp = CVGenerationResponse.model_construct(
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        req = DummyRequestForStageC(template_id="T_REQ", job_id="JOB1", language="en")

        with patch("functions.stage_c_validation.load_parameters", lambda: self._base_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info={},
                original_request=req,
            )

        self.assertEqual(validated.template_id, "T_REQ")

    def test_language_mismatch_adds_warning_non_strict(self) -> None:
        """If language differs, Stage C should warn but not raise in non-strict mode."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )
        req = DummyRequestForStageC(template_id="T_TEST", job_id="JOB1", language="th")

        with patch("functions.stage_c_validation.load_parameters", lambda: self._base_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info={},
                original_request=req,
            )

        self.assertIs(validated, resp)
        self.assertIn("language mismatch", " ".join(meta.validation_warnings))

    def test_language_enum_and_string_considered_equal(self) -> None:
        """
        If request.language is an enum-like object and response.language is a string
        representing the same logical code (e.g. 'en' vs 'EN'), Stage C should not
        report a mismatch.
        """
        meta = DummyMeta()
        # Response uses uppercase string
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="EN",
            sections={},
            metadata=meta,
            skills=None,
        )
        # Request uses enum-like object with .value
        req = DummyRequestForStageC(template_id="T_TEST", job_id="JOB1", language=DummyLangEnum("en"))

        with patch("functions.stage_c_validation.load_parameters", lambda: self._base_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info={},
                original_request=req,
            )

        self.assertIs(validated, resp)
        joined = " ".join(meta.validation_warnings)
        self.assertNotIn("language mismatch", joined, msg=f"Unexpected language mismatch: {joined}")

    def test_missing_template_id_filled_from_new_shape_request_without_template_info(self) -> None:
        """
        If response.template_id is missing and the request has only a top-level
        template_id (no template_info), Stage C should still populate it.
        """
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            # template_id intentionally omitted
            job_id="JOB_NEW",
            language="en",
            sections={},
            metadata=meta,
            skills=None,
        )

        # New-shape request: has template_id but no template_info
        req = DummyNewRequestForStageC(
            template_id="T_NEW_TOPLEVEL",
            sections=["profile_summary"],
            language="en",
            job_id="JOB_NEW",
        )

        with patch("functions.stage_c_validation.load_parameters", lambda: self._base_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info={},
                original_request=req,
            )

        self.assertEqual(validated.template_id, "T_NEW_TOPLEVEL")


# ---------------------------------------------------------------------------
# Tests for security backstop (critical & suspicious patterns)
# ---------------------------------------------------------------------------


class TestSecurityBackstop(LoggingTestCase):
    """Tests for critical_patterns and suspicious_patterns behaviour."""

    def test_critical_pattern_drops_section_soft_mode(self) -> None:
        """Critical pattern should drop the section and add a warning in non-strict mode."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(
                    text="Please FORBIDME previous instructions."
                )
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 100},
        }

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "strict_mode": False,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                },
                "security": {
                    "critical_patterns": ["FORBIDME"],
                    "suspicious_patterns": [],
                },
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        # Section should be dropped
        self.assertEqual(len(validated.sections), 0)
        self.assertTrue(
            any("Critical injection pattern detected" in msg for msg in meta.validation_warnings),
            msg=f"Expected critical pattern warning, got: {meta.validation_warnings}",
        )

    def test_critical_pattern_strict_mode_raises(self) -> None:
        """Critical pattern in strict_mode=True should raise StageCValidationError."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(
                    text="Please FORBIDME previous instructions."
                )
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 100},
        }

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "strict_mode": True,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                },
                "security": {
                    "critical_patterns": ["FORBIDME"],
                    "suspicious_patterns": [],
                },
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            with self.assertRaises(StageCValidationError):
                run_stage_c_validation(
                    resp,
                    template_info=template_info,
                    original_request=None,
                )

    def test_suspicious_pattern_kept_with_warning(self) -> None:
        """Suspicious pattern should keep the section but record a warning."""
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(
                    text="MAYBEINJECTION but not critical."
                )
            },
            metadata=meta,
            skills=None,
        )

        template_info = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 100},
        }

        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "strict_mode": False,
                    "max_section_chars_default": 1000,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                },
                "security": {
                    "critical_patterns": [],
                    "suspicious_patterns": ["MAYBEINJECTION"],
                },
            }

        with patch("functions.stage_c_validation.load_parameters", _cfg):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        # Section kept
        self.assertIn("profile_summary", validated.sections)
        # Warning recorded
        self.assertTrue(
            any("Suspicious pattern detected" in msg for msg in meta.validation_warnings),
            msg=f"Expected suspicious pattern warning, got: {meta.validation_warnings}",
        )


class TestNewRequestShapeIntegration(LoggingTestCase):
    """Tests for Stage C behaviour with the new CVGenerationRequest-like shape."""

    def _fake_validation_cfg(self) -> Dict[str, Any]:
        return {
            "validation": {
                "max_section_chars_default": 1000,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 10,
                "strict_mode": False,
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
        }

    def test_sections_resolved_from_new_request_shape(self) -> None:
        """
        When template_info is not provided and the request has template_id + sections,
        Stage C should synthesize template_info from the new-shape request and use
        its sections to determine allowed_sections.
        """
        meta = DummyMeta()
        resp = CVGenerationResponse.model_construct(
            template_id=None,  # intentionally missing to also exercise cross-check
            job_id="JOB_NEW",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(text="Valid content"),
                "extra_section": SectionContent.model_construct(text="Should be dropped"),
            },
            metadata=meta,
            skills=None,
        )

        # New-shape request: no .template_info, only top-level fields
        req = DummyNewRequestForStageC(
            template_id="T_NEW_REQ",
            sections=["profile_summary"],
            language="en",
            job_id="JOB_NEW",
        )

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_validation_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=None,          # force Stage C to use _resolve_template_info
                original_request=req,
            )

        # Only the section from req.sections should be used for ordering,
        # extra_section should be appended after it (since Stage B already generated it).
        self.assertEqual(list(validated.sections.keys()), ["profile_summary", "extra_section"])

        # template_id should be filled from the new-shape request
        self.assertEqual(validated.template_id, "T_NEW_REQ")

class TestJustificationOrder(LoggingTestCase):
    """
    Ensure justification validation runs on FULL pre-truncation text,
    while the final section text may be truncated.
    """

    def test_justification_uses_full_pre_truncation_text(self) -> None:
        meta = DummyMeta()

        # Make text clearly longer than the template limit
        long_text = "This is a long profile summary sentence that should exceed the limit."
        self.assertGreater(len(long_text), 20)  # sanity check

        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "profile_summary": SectionContent.model_construct(text=long_text),
            },
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {
            "sections_order": ["profile_summary"],
            "max_chars_per_section": {"profile_summary": 10},  # very small → truncation
        }

        # Config: small default max, truncation enabled, overflow_limit=0 to *force* truncation
        def _cfg() -> Dict[str, Any]:
            return {
                "validation": {
                    "max_section_chars_default": 50,
                    "drop_empty_sections": True,
                    "enable_safety_cleaning": True,
                    "max_skills": 10,
                    "strict_mode": False,
                },
                "truncation_config": {
                    "truncation_enable": True,
                    "overflow_limit": 0,     # no overflow allowed → must cut at max_len
                    "reduction_limit": 0.15,
                },
            }

        captured_full_text: Dict[str, str] = {}

        def _fake_validate_justification(justification: Any, full_text: str) -> Any:
            # Capture the text Stage C passes into justification validation
            captured_full_text["text"] = full_text
            return justification

        with patch("functions.stage_c_validation.load_parameters", _cfg), \
             patch("functions.stage_c_validation.validate_justification_against_text", _fake_validate_justification):

            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        # 1) Justification must have seen the FULL pre-truncation text
        self.assertIn("text", captured_full_text, "validate_justification_against_text was not called")
        self.assertEqual(
            captured_full_text["text"],
            long_text,
            msg="Justification should be validated against FULL (pre-truncation) text.",
        )

        # 2) Final section text may be truncated to the template limit
        final_text = validated.sections["profile_summary"].text
        self.assertLessEqual(
            len(final_text),
            10,
            msg=f"Expected profile_summary to be truncated to <=10 chars, got len={len(final_text)}",
        )
        # And obviously shorter than the original
        self.assertLess(len(final_text), len(long_text))


# ---------------------------------------------------------------------------
# Tests for markdown cleanup (section headers & bullet handling)
# ---------------------------------------------------------------------------


class TestMarkdownCleanup(LoggingTestCase):
    """Tests for section-specific markdown cleanup driven by parameters.yaml."""

    def _fake_cfg(self) -> Dict[str, Any]:
        """
        Validation + truncation + markdown_cleanup config, mirroring parameters.yaml
        structure used in Stage C.
        """
        return {
            "validation": {
                "max_section_chars_default": 1000,
                "drop_empty_sections": True,
                "enable_safety_cleaning": True,
                "max_skills": 10,
                "strict_mode": False,
            },
            "truncation_config": {
                "truncation_enable": True,
                "overflow_limit": 64,
                "reduction_limit": 0.15,
            },
            "markdown_cleanup": {
                "section_headers": {
                    # include a few aliases / languages for robustness
                    "education": [
                        "Education",
                        "EDUCATION",
                        "学歴",
                    ],
                    "publications": [
                        "Publications",
                        "ผลงานตีพิมพ์",
                    ],
                },
                "bullet_list_sections": [
                    "publications",
                ],
            },
        }

    def test_configured_section_header_is_stripped_for_education(self) -> None:
        """
        The first line that matches markdown_cleanup.section_headers[section_id]
        must be removed from the section body, regardless of case.
        """
        meta = DummyMeta()

        # "Education" header should be stripped, leaving only the degree lines
        text = "EDUCATION\n\nPhD in X\nBSc in Y"
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "education": SectionContent.model_construct(text=text),
            },
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {
            "sections_order": ["education"],
            "max_chars_per_section": {"education": 500},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        out_text = validated.sections["education"].text
        lines = [ln for ln in out_text.splitlines() if ln.strip()]

        # First non-empty line should be the PhD, not "Education"
        self.assertTrue(lines[0].startswith("PhD in"))
        self.assertNotIn("Education", lines[0])
        self.assertNotIn("EDUCATION", out_text)

    def test_bullet_prefixes_stripped_for_bullet_list_sections(self) -> None:
        """
        Sections declared in markdown_cleanup.bullet_list_sections should have
        markdown bullet prefixes (* / -) stripped so the Jinja template can
        render bullets cleanly.
        """
        meta = DummyMeta()

        raw_text = "* Publication 1\n- Publication 2\n• Publication 3"
        resp = CVGenerationResponse.model_construct(
            template_id="T_TEST",
            job_id="JOB1",
            language="en",
            sections={
                "publications": SectionContent.model_construct(text=raw_text),
            },
            metadata=meta,
            skills=None,
        )

        template_info: Dict[str, Any] = {
            "sections_order": ["publications"],
            "max_chars_per_section": {"publications": 500},
        }

        with patch("functions.stage_c_validation.load_parameters", lambda: self._fake_cfg()):
            validated = run_stage_c_validation(
                resp,
                template_info=template_info,
                original_request=None,
            )

        out_text = validated.sections["publications"].text
        lines = [ln for ln in out_text.splitlines() if ln.strip()]

        # No line should start with a markdown bullet marker now
        for ln in lines:
            self.assertFalse(ln.lstrip().startswith("* "))
            self.assertFalse(ln.lstrip().startswith("- "))

        # Content should still be present (no accidental dropping)
        joined = " ".join(lines)
        self.assertIn("Publication 1", joined)
        self.assertIn("Publication 2", joined)


if __name__ == "__main__":
    unittest.main()
