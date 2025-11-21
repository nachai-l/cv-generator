"""Unittest suite for Stage A guardrails (validate_and_sanitize + evidence plan).

These tests_utils use the YAML fixtures that represent the expected input
format for the Stage A contract:

- profile_info.yaml                  → request.profile_info (required)
- job_role_info.yaml                 → request.job_role_info (optional)
- job_position_info.yaml             → request.job_position_info (optional)
- company_info.yaml                  → request.company_info (optional)
- template_info.yaml                 → request.template_info (required)
- user_input_cv_text_by_section.yaml → request.user_input_cv_text_by_section (optional)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, cast

import yaml

from functions.stage_a_guardrails import GuardrailsProcessor


FIXTURES_DIR = Path(__file__).parent / "json_test_inputs"


# ---------------------------------------------------------------------------
# Helper objects to simulate the Stage A contract without touching the
# real CVGenerationRequest schema (which still uses student_profile, etc.)
# ---------------------------------------------------------------------------


class DotObj:
    """Simple object wrapper that exposes dict keys as attributes."""

    _data: Dict[str, Any]  # for static type checkers / IDEs

    def __init__(self, data: Dict[str, Any]):
        object.__setattr__(self, "_data", data)
        for key, value in data.items():
            object.__setattr__(self, key, self._convert(value))

    def _convert(self, value: Any) -> Any:
        if isinstance(value, dict):
            return DotObj(value)
        if isinstance(value, list):
            return [self._convert(v) for v in value]
        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert back to a plain dict recursively."""

        def conv(v: Any) -> Any:
            if isinstance(v, DotObj):
                return v.to_dict()
            if isinstance(v, list):
                return [conv(i) for i in v]
            return v

        # dict comprehension avoids “could be rewritten as literal” warning
        return {k: conv(getattr(self, k)) for k in self._data.keys()}


class DummyRequest:
    """Minimal request object matching the *logical* Stage A contract."""

    STRUCTURED_FIELDS = {
        "profile_info",
        "job_role_info",
        "job_position_info",
        "company_info",
        "template_info",
        "user_input_cv_text_by_section",
    }

    def __init__(self, payload: Dict[str, Any]):
        object.__setattr__(self, "_base_payload", payload)
        self.user_id: str = payload.get("user_id", "test-user")
        self.sections: List[str] = payload.get("sections", [])

        for field in self.STRUCTURED_FIELDS:
            if field in payload:
                setattr(self, field, payload[field])
            else:
                setattr(self, field, None)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in self.STRUCTURED_FIELDS and isinstance(value, dict):
            value = DotObj(value)
        object.__setattr__(self, name, value)

    def model_dump(self, mode: str = "python") -> Dict[str, Any]:
        """Return a plain dict representation (like pydantic's model_dump).

        The `mode` parameter is accepted for API compatibility with Pydantic,
        but is not used for anything in this dummy implementation.
        """
        _ = mode  # silence 'unused parameter' warnings

        def to_plain(obj: Any) -> Any:
            if isinstance(obj, DotObj):
                return obj.to_dict()
            if isinstance(obj, list):
                return [to_plain(v) for v in obj]
            return obj

        base = {"user_id": self.user_id, "sections": list(self.sections)}

        structured = {
            field: to_plain(getattr(self, field))
            for field in self.STRUCTURED_FIELDS
            if getattr(self, field, None) is not None
        }

        return {**base, **structured}


# ---------------------------------------------------------------------------
# YAML loading helpers
# ---------------------------------------------------------------------------


def _load_yaml(name: str) -> Any:
    """Helper to load a YAML file from the json_test_inputs directory."""
    path = FIXTURES_DIR / name
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _extract_sections_from_template(template_info: Dict[str, Any]) -> List[str]:
    """Derive section IDs from template_info.

    Supports:
    - template_info["sections"] as a mapping
    - template_info["max_lengths"] as a mapping
    - template_info["section_max_lengths"] as a mapping
    """
    sections_source = (
        template_info.get("sections")
        or template_info.get("max_lengths")
        or template_info.get("section_max_lengths")
        or {}
    )
    if isinstance(sections_source, dict):
        return list(sections_source.keys())
    return []


def _build_request_from_yaml(
    include_job_role: bool = True,
    include_job_position: bool = True,
    include_company: bool = True,
    include_user_text: bool = True,
) -> DummyRequest:
    """Construct a DummyRequest from the YAML fixtures."""
    profile_info = _load_yaml("profile_info.yaml")
    template_info = _load_yaml("template_info.yaml")

    payload = dict(
        user_id="test-user-123",
        profile_info=profile_info,
        template_info=template_info,
    )

    payload["sections"] = _extract_sections_from_template(template_info) or [
        "profile_summary",
        "skills",
        "experience",
        "education",
        "awards",
        "extracurricular",
    ]

    if include_job_role:
        payload["job_role_info"] = _load_yaml("job_role_info.yaml")
    if include_job_position:
        payload["job_position_info"] = _load_yaml("job_position_info.yaml")
    if include_company:
        payload["company_info"] = _load_yaml("company_info.yaml")
    if include_user_text:
        payload["user_input_cv_text_by_section"] = _load_yaml(
            "user_input_cv_text_by_section.yaml"
        )

    return DummyRequest(payload)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStageAGuardrails(unittest.TestCase):
    """Test suite for Stage A GuardrailsProcessor."""

    def setUp(self):
        """Create processor and print a readable header before each test."""
        # Initialise the Stage A processor for each test
        self.processor = GuardrailsProcessor()

        # Pretty logging header
        test_name = self._testMethodName
        print("\n" + "=" * 90, file=sys.stderr)
        print(f"🧪 STARTING TEST: {test_name}", file=sys.stderr)
        print("=" * 90, file=sys.stderr)

    def tearDown(self):
        """Print a divider after each test."""
        print("-" * 90 + "\n", file=sys.stderr)

    # ------------------------------------------------------------------
    # Validation tests_utils
    # ------------------------------------------------------------------

    def test_validate_and_sanitize_full_yaml_payload_is_valid(self):
        """Full payload with all YAML inputs should validate successfully."""
        request = _build_request_from_yaml(
            include_job_role=True,
            include_job_position=True,
            include_company=True,
            include_user_text=True,
        )

        result = self.processor.validate_and_sanitize(cast(Any, request))

        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])

        missing_jd_warning = (
            "No job role / job position information provided "
            "(expected `job_role_info` and/or `job_position_info`). "
            "Targeting quality may be reduced."
        )
        self.assertNotIn(missing_jd_warning, result.warnings or [])

    def test_validate_and_sanitize_without_jd_info_warns_not_error(self):
        """If no job_role_info / job_position_info are provided, warn but don't fail."""
        request = _build_request_from_yaml(
            include_job_role=False,
            include_job_position=False,
            include_company=True,
            include_user_text=True,
        )

        result = self.processor.validate_and_sanitize(cast(Any, request))

        self.assertTrue(result.is_valid)
        self.assertEqual(result.errors, [])

        missing_jd_warning = (
            "No job role / job position information provided "
            "(expected `job_role_info` and/or `job_position_info`). "
            "Targeting quality may be reduced."
        )
        self.assertIn(missing_jd_warning, result.warnings or [])

    def test_missing_profile_info_fails_validation(self):
        """profile_info is required; omitting it should cause validation failure."""
        template_info = _load_yaml("template_info.yaml")
        sections = _extract_sections_from_template(template_info)

        payload = {
            "user_id": "test-user-123",
            "template_info": template_info,
            "sections": sections,
        }

        request = DummyRequest(payload)
        result = self.processor.validate_and_sanitize(cast(Any, request))

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("Profile info is required (missing `profile_info`)." in e for e in result.errors)
        )

    def test_missing_template_info_fails_validation(self):
        """template_info is required; omitting it should cause validation failure."""
        profile_info = _load_yaml("profile_info.yaml")

        payload = {
            "user_id": "test-user-123",
            "profile_info": profile_info,
            "sections": [
                "profile_summary",
                "skills",
                "experience",
                "education",
                "awards",
                "extracurricular",
            ],
        }

        request = DummyRequest(payload)
        result = self.processor.validate_and_sanitize(cast(Any, request))

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("Template info is required (missing `template_info`)." in e for e in result.errors)
        )

    def test_template_info_basic_validation_rules_applied(self):
        """template_info must include template_id and sections/length map."""
        profile_info = _load_yaml("profile_info.yaml")
        template_info = _load_yaml("template_info.yaml")

        broken_template = dict(template_info)
        # Remove all ways Stage A can see section metadata
        broken_template.pop("template_id", None)
        broken_template.pop("sections", None)
        broken_template.pop("max_lengths", None)
        broken_template.pop("section_max_lengths", None)
        broken_template.pop("max_chars_per_section", None)

        payload = {
            "user_id": "test-user-123",
            "profile_info": profile_info,
            "template_info": broken_template,
            "sections": [],  # missing sections is what we're testing
        }

        request = DummyRequest(payload)
        result = self.processor.validate_and_sanitize(cast(Any, request))

        self.assertFalse(result.is_valid)
        self.assertTrue(
            any("template_info: `template_id` is required." in e for e in result.errors)
        )
        self.assertTrue(
            any("template_info: at least one section must be specified." in e for e in result.errors)
        )

    # ------------------------------------------------------------------
    # Evidence plan tests_utils
    # ------------------------------------------------------------------

    def test_build_evidence_plan_from_profile_info(self):
        """Evidence plan should be built correctly from profile_info."""
        request = _build_request_from_yaml(
            include_job_role=True,
            include_job_position=True,
            include_company=True,
            include_user_text=True,
        )

        validation = self.processor.validate_and_sanitize(cast(Any, request))
        self.assertTrue(validation.is_valid)

        plan = self.processor.build_evidence_plan(cast(Any, request))

        self.assertGreater(len(plan.evidences), 0)
        self.assertIn("education", plan.section_hints)
        self.assertIn("skills", plan.section_hints)
        self.assertTrue(
            any(
                ev.source_type == "education" and ev.fact.startswith("Holds ")
                for ev in plan.evidences
            )
        )

    def test_evidence_plan_empty_if_profile_missing(self):
        """If profile_info missing, build_evidence_plan should return empty plan."""
        template_info = _load_yaml("template_info.yaml")
        sections = _extract_sections_from_template(template_info)

        payload = {
            "user_id": "test-user-123",
            "template_info": template_info,
            "sections": sections,
        }
        request = DummyRequest(payload)

        plan = self.processor.build_evidence_plan(cast(Any, request))

        self.assertEqual(plan.evidences, [])
        self.assertEqual(plan.section_hints, {})

    def test_missing_student_profile_fails_with_legacy_error_message(self):
        """If only CVGenerationRequest fields are present but no profile, we still emit the legacy error message."""
        from schemas.input_schema import CVGenerationRequest, Language

        processor = GuardrailsProcessor()

        # Build minimal request with template_id + sections but NO student_profile
        req = CVGenerationRequest.model_construct(  # bypass pydantic required fields
            user_id="test-user-123",
            language=Language.EN,
            template_id="T_EMPLOYER_STD_V3",
            sections=["profile_summary"],
            student_profile=None,  # simulate missing profile
            target_role_taxonomy=None,
            target_jd_taxonomy=None,
        )

        result = processor.validate_and_sanitize(req)
        self.assertFalse(result.is_valid)
        self.assertIn(
            "Profile info is required (missing `profile_info`).",
            result.errors,
        )

    def test_new_api_shape_without_jd_info_warns_not_error(self):
        """No target_role_taxonomy / target_jd_taxonomy should produce a warning, not a hard error."""
        from schemas.input_schema import (
            CVGenerationRequest,
            StudentProfile,
            PersonalInfo,
            Education,
            Skill,
            SkillLevel,
            Language,
        )
        from datetime import date

        processor = GuardrailsProcessor()

        student_profile = StudentProfile(
            personal_info=PersonalInfo(
                name="Test User",
                email="test@example.com",
            ),
            education=[
                Education(
                    id="edu#1",
                    degree="B.Sc. Computer Science",
                    institution="Test University",
                    gpa=3.5,
                    start_date=date(2020, 1, 1),
                    graduation_date=date(2024, 1, 1),
                    major="CS",
                )
            ],
            skills=[
                Skill(
                    id="skill#python",
                    name="Python",
                    description="Python programming",
                    level=SkillLevel.L2_INTERMEDIATE,
                ),
                Skill(
                    id="skill#sql",
                    name="SQL",
                    description="SQL queries",
                    level=SkillLevel.L2_INTERMEDIATE,
                ),
                Skill(
                    id="skill#comm",
                    name="Communication",
                    description="Communication",
                    level=SkillLevel.L2_INTERMEDIATE,
                ),
            ],
            experience=[],
            awards=[],
            extracurriculars=[],
        )

        # Use model_construct to bypass strict validation and simulate a "no JD info" request
        req = CVGenerationRequest.model_construct(
            user_id="test-user-123",
            language=Language.EN,
            template_id="T_EMPLOYER_STD_V3",
            sections=["profile_summary", "skills", "education"],
            student_profile=student_profile,
            target_role_taxonomy=None,
            target_jd_taxonomy=None,
        )

        result = processor.validate_and_sanitize(req)
        self.assertTrue(result.is_valid)

        missing_jd_warning = (
            "No job role / job position information provided "
            "(expected `job_role_info` and/or `job_position_info`). "
            "Targeting quality may be reduced."
        )
        self.assertIn(missing_jd_warning, result.warnings or [])

    # ------------------------------------------------------------------
    # New tests_utils for resolve_template_info (legacy + new API shapes)
    # ------------------------------------------------------------------

    def test_resolve_template_info_legacy_request_uses_template_info_field(self):
        """resolve_template_info should return the legacy request.template_info unchanged."""
        request = _build_request_from_yaml(
            include_job_role=True,
            include_job_position=True,
            include_company=True,
            include_user_text=True,
        )

        tmpl = self.processor.resolve_template_info(request)
        self.assertIsNotNone(tmpl)
        # template_info.yaml has template_id set
        self.assertTrue(
            hasattr(tmpl, "template_id"),
            msg="Resolved template_info should expose template_id attribute for legacy shape.",
        )
        # sanity: it should match the YAML's template_id
        yaml_tmpl = _load_yaml("template_info.yaml")
        self.assertEqual(
            getattr(tmpl, "template_id", None),
            yaml_tmpl.get("template_id"),
        )

    def test_resolve_template_info_new_api_builds_namespace_from_template_id_and_sections(self):
        """resolve_template_info should construct a SimpleNamespace for new CVGenerationRequest shape."""
        from schemas.input_schema import CVGenerationRequest, Language

        processor = GuardrailsProcessor()

        # Minimal new-shape request: template_id + sections + language, no template_info
        req = CVGenerationRequest.model_construct(
            user_id="test-user-123",
            language=Language.EN,
            template_id="T_EMPLOYER_STD_V3",
            sections=["profile_summary", "skills"],
            student_profile=None,
            target_role_taxonomy=None,
            target_jd_taxonomy=None,
        )

        tmpl = processor.resolve_template_info(req)
        self.assertIsNotNone(tmpl, "resolve_template_info should not return None for new API shape.")

        # Should behave like template_info: have template_id, sections, language, max_chars_per_section
        self.assertEqual(getattr(tmpl, "template_id", None), "T_EMPLOYER_STD_V3")
        self.assertEqual(getattr(tmpl, "sections", None), ["profile_summary", "skills"])

        # language should be the underlying value ("en"), not the enum object
        self.assertEqual(getattr(tmpl, "language", None), "en")

        # For bare CVGenerationRequest, we expect max_chars_per_section to be None
        self.assertTrue(
            hasattr(tmpl, "max_chars_per_section"),
            msg="Resolved template_info should expose max_chars_per_section attribute.",
        )
        self.assertIsNone(
            getattr(tmpl, "max_chars_per_section"),
            msg="max_chars_per_section should be None when not provided by new API shape.",
        )


if __name__ == "__main__":
    unittest.main()
