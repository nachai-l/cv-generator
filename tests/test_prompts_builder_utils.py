# tests_utils/test_prompts_builder_utils.py
"""
Unittest suite for prompt-building utilities.

This module tests the pure helper functions in:

    functions.utils.prompts_builder

Core responsibilities covered:

- load_section_prompts_config / _load_prompts_from_file:
    - Loads section-level prompts from YAML via load_yaml_dict.
    - Handles non-mapping / error cases by returning {}.
    - Ensures keys are normalised to strings.

- _normalize_section_id_for_evidence:
    - Maps "*_structured" section ids to their canonical base section id.

- _collect_education_facts_from_request:
    - Builds concise education fact strings from student_profile / profile_info.

- build_section_prompt:
    - Injects language, tone, profile info, job context, evidence facts.
    - Adds user draft + user_draft_rewrite_suffix when provided.
    - Appends section-specific or default output requirements.
    - Optionally injects justification instructions when enabled.

- build_skills_selection_prompt:
    - Builds JSON-first prompt for structured skills selection.
    - Includes canonical skills, evidence facts, and (optional) justification
      suffix for skills_structured.

- build_experience_justification_prompt:
    - Builds a justification-only prompt for the final experience section text.
    - Includes profile info, evidence facts, and justification instructions
      from prompts.yaml.
"""

from __future__ import annotations

import json
import sys
import unittest
from typing import Any, Dict, List
from unittest.mock import patch

from functions.utils.prompts_builder import (
    load_section_prompts_config,
    build_section_prompt,
    build_skills_selection_prompt,
    build_experience_justification_prompt,
    _normalize_section_id_for_evidence,
    _collect_education_facts_from_request,
    _load_prompts_from_file,
)


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
# Simple dummy models for requests / plans (no dependency on real schemas)
# ---------------------------------------------------------------------------


class DummyTemplateInfo:
    def __init__(self, tone: str | None = None) -> None:
        self.tone = tone


class DummyRequest:
    def __init__(
        self,
        language: str = "en",
        tone: str | None = None,
        profile_info: Any | None = None,
        student_profile: Any | None = None,
        job_role_info: Any | None = None,
        job_position_info: Any | None = None,
        company_info: Any | None = None,
        drafts: Dict[str, str] | None = None,
        template_tone: str | None = None,
    ) -> None:
        self.language = language
        self.tone = tone
        self.style_tone = None
        self.template_info = DummyTemplateInfo(tone=template_tone)
        self.profile_info = profile_info
        self.student_profile = student_profile
        self.job_role_info = job_role_info
        self.job_position_info = job_position_info
        self.company_info = company_info
        self.user_input_cv_text_by_section = drafts or {}


class DummyEducationEntry:
    def __init__(
        self,
        degree: str | None = None,
        institution: str | None = None,
        major: str | None = None,
        gpa: float | None = None,
        honors: str | None = None,
        start_date: str | None = None,
        graduation_date: str | None = None,
    ) -> None:
        self.degree = degree
        self.institution = institution
        self.major = major
        self.gpa = gpa
        self.honors = honors
        self.start_date = start_date
        self.graduation_date = graduation_date


class DummyStudentProfile:
    def __init__(self, education: List[Any] | None = None) -> None:
        self.education = education or []


class DummySkill:
    def __init__(self, name: str, level: str | None = None) -> None:
        self.name = name
        self.level = level


class DummySkillsPlan:
    def __init__(self, canonical_skills: List[DummySkill]) -> None:
        self.canonical_skills = canonical_skills


class DummyEvidence:
    def __init__(self, evidence_id: str, fact: str) -> None:
        self.evidence_id = evidence_id
        self.fact = fact


class DummyEvidencePlan:
    """
    Minimal EvidencePlan-like object that exposes get_evidence_for_section.
    """

    def __init__(self, mapping: Dict[str, List[DummyEvidence]]) -> None:
        self._mapping = mapping
        self.section_hints = {sec: [e.evidence_id for e in evids] for sec, evids in mapping.items()}
        self.evidences = [e for evids in mapping.values() for e in evids]

    def get_evidence_for_section(self, section_id: str) -> List[DummyEvidence]:
        return self._mapping.get(section_id, [])


# ---------------------------------------------------------------------------
# Tests for load_section_prompts_config / _load_prompts_from_file
# ---------------------------------------------------------------------------


class TestLoadSectionPromptsConfig(LoggingTestCase):
    """Tests for loading prompts configuration from YAML."""

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_loads_mapping_and_normalises_keys_to_strings(self, mock_load_yaml) -> None:
        """Valid mapping from YAML is returned unchanged with string keys."""
        mock_load_yaml.return_value = {
            "default": "DEFAULT_PROMPT",
            "profile_summary": "PROFILE_PROMPT",
        }
        _load_prompts_from_file.cache_clear()

        cfg = load_section_prompts_config()

        self.assertIsInstance(cfg, dict)
        self.assertIn("default", cfg)
        self.assertEqual(cfg["profile_summary"], "PROFILE_PROMPT")

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_non_mapping_yaml_results_in_empty_dict(self, mock_load_yaml) -> None:
        """If the YAML root is not a mapping, we should fail safely with {}."""
        mock_load_yaml.return_value = ["not", "a", "mapping"]
        _load_prompts_from_file.cache_clear()

        cfg = load_section_prompts_config()
        self.assertEqual(cfg, {})

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_yaml_loader_error_results_in_empty_dict(self, mock_load_yaml) -> None:
        """Any exception from load_yaml_dict should produce a safe empty mapping."""
        mock_load_yaml.side_effect = RuntimeError("file not found")
        _load_prompts_from_file.cache_clear()

        cfg = load_section_prompts_config()
        self.assertEqual(cfg, {})


# ---------------------------------------------------------------------------
# Tests for _normalize_section_id_for_evidence
# ---------------------------------------------------------------------------


class TestNormalizeSectionIdForEvidence(LoggingTestCase):
    """Tests for mapping structured section IDs to their canonical names."""

    def test_skills_structured_maps_to_skills(self) -> None:
        self.assertEqual(
            _normalize_section_id_for_evidence("skills_structured"),
            "skills",
        )

    def test_non_structured_ids_are_unchanged(self) -> None:
        for sec in ["skills", "experience", "education", "profile_summary"]:
            with self.subTest(section=sec):
                self.assertEqual(
                    _normalize_section_id_for_evidence(sec),
                    sec,
                )


# ---------------------------------------------------------------------------
# Tests for _collect_education_facts_from_request
# ---------------------------------------------------------------------------


class TestCollectEducationFactsFromRequest(LoggingTestCase):
    """Tests for building education facts from request/student_profile."""

    def test_collects_structured_education_from_student_profile(self) -> None:
        """Education facts should contain key fields like degree, major, institution, GPA, years."""
        edu_entry = DummyEducationEntry(
            degree="BSc in Computer Science",
            institution="Kasetsart University",
            major="Computer Science",
            gpa=3.8,
            honors="First Class Honours",
            start_date="2015-08-01",
            graduation_date="2019-05-01",
        )
        student_profile = DummyStudentProfile(education=[edu_entry])
        req = DummyRequest(student_profile=student_profile)

        facts = _collect_education_facts_from_request(req)

        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertIn("BSc in Computer Science", fact)
        self.assertIn("Major: Computer Science", fact)
        self.assertIn("Kasetsart University", fact)
        self.assertIn("GPA: 3.8", fact)
        self.assertIn("Years: 2015–2019", fact)

    def test_falls_back_to_profile_info_when_no_student_profile(self) -> None:
        """If student_profile is absent, it should use profile_info['education']."""
        edu_entry = {
            "degree": "PhD in Biological Science",
            "institution": "NAIST",
            "major": "Biotechnology",
        }
        profile_info = {"education": [edu_entry]}
        req = DummyRequest(profile_info=profile_info, student_profile=None)

        facts = _collect_education_facts_from_request(req)

        self.assertEqual(len(facts), 1)
        fact = facts[0]
        self.assertIn("PhD in Biological Science", fact)
        self.assertIn("NAIST", fact)


# ---------------------------------------------------------------------------
# Tests for build_section_prompt
# ---------------------------------------------------------------------------


class TestBuildSectionPrompt(LoggingTestCase):
    """Tests for generic section prompt construction."""

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_builds_profile_summary_prompt_with_user_draft_and_suffix(
        self,
        mock_load_yaml,
    ) -> None:
        """
        build_section_prompt should:
        - include language + tone
        - include profile info + job context JSON
        - include evidence facts header
        - include user draft and user_draft_rewrite_suffix
        - append section-specific Output Requirements
        """
        mock_load_yaml.return_value = {
            "default": "DEFAULT_PROMPT",
            "profile_summary": "PROFILE_SUMMARY_PROMPT",
            "user_draft_rewrite_suffix": "Please rewrite this draft in a professional tone.",
        }
        _load_prompts_from_file.cache_clear()

        profile_info = {"summary": "Existing summary"}
        job_role_info = {"title": "Data Scientist"}
        drafts = {"profile_summary": "This is my raw draft."}

        req = DummyRequest(
            language="en",
            tone="formal",
            profile_info=profile_info,
            job_role_info=job_role_info,
            drafts=drafts,
        )

        prompt = build_section_prompt(
            request=req,
            evidence_plan=None,
            section_id="profile_summary",
        )

        # Core blocks
        self.assertIn("You are an expert CV writer.", prompt)
        self.assertIn("The CV language is", prompt)
        self.assertIn("Generate a strong 'profile_summary' section", prompt)

        # Profile / job context
        self.assertIn("=== Profile Info (JSON) ===", prompt)
        self.assertIn('"summary": "Existing summary"', prompt)
        self.assertIn("=== Target Job Role / Position / Company (JSON) ===", prompt)
        self.assertIn('"title": "Data Scientist"', prompt)

        # Evidence header (no evidence → default line)
        self.assertIn("=== Evidence Facts for profile_summary ===", prompt)
        self.assertIn("- (No specific evidence provided)", prompt)

        # User draft + suffix
        self.assertIn("=== User Draft for profile_summary ===", prompt)
        self.assertIn("This is my raw draft.", prompt)
        self.assertIn("Please rewrite this draft in a professional tone.", prompt)

        # Section-specific output requirements
        self.assertIn("=== Output Requirements ===", prompt)
        self.assertIn("PROFILE_SUMMARY_PROMPT", prompt)
        self.assertNotIn("DEFAULT_PROMPT", prompt)  # section-specific overrides default

    @patch("functions.utils.prompts_builder.should_require_justification", return_value=True)
    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_injects_justification_instructions_when_enabled(
        self,
        mock_load_yaml,
        mock_should_req,
    ) -> None:
        """When justification is enabled, prompt must include a Justification Instructions block."""
        mock_load_yaml.return_value = {
            "default": "DEFAULT_PROMPT",
            "profile_summary": "PROFILE_SUMMARY_PROMPT",
            "justification": "Return JUSTIFICATION_JSON object containing claims.",
        }
        _load_prompts_from_file.cache_clear()

        req = DummyRequest(profile_info={"summary": "x"})
        prompt = build_section_prompt(
            request=req,
            evidence_plan=None,
            section_id="profile_summary",
        )

        self.assertIn("=== Justification Instructions ===", prompt)
        self.assertIn("Return JUSTIFICATION_JSON object containing claims.", prompt)


# ---------------------------------------------------------------------------
# Tests for build_skills_selection_prompt
# ---------------------------------------------------------------------------


class TestBuildSkillsSelectionPrompt(LoggingTestCase):
    """Tests for structured skills selection prompt."""

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_builds_skills_prompt_with_canonical_skills(self, mock_load_yaml) -> None:
        """
        build_skills_selection_prompt should:
        - include language and tone
        - include student/profile JSON
        - include skills_structured instructions
        - list canonical skills
        """
        mock_load_yaml.return_value = {
            "skills_structured": "SKILLS_JSON_SPEC",
        }
        _load_prompts_from_file.cache_clear()

        skills_plan = DummySkillsPlan(
            canonical_skills=[
                DummySkill("Python", "advanced"),
                DummySkill("Machine Learning", "intermediate"),
            ]
        )

        req = DummyRequest(
            language="en",
            profile_info={"skills": ["Python", "Machine Learning"]},
        )

        prompt = build_skills_selection_prompt(
            request=req,
            evidence_plan=None,
            skills_plan=skills_plan,
            language="en",
            require_justification=False,
        )

        self.assertIn("The CV language is", prompt)
        self.assertIn("Student profile", prompt)
        self.assertIn("structured JSON", prompt)
        self.assertIn("Skills selection instructions", prompt)
        self.assertIn("SKILLS_JSON_SPEC", prompt)
        self.assertIn("Python", prompt)
        self.assertIn("Machine Learning", prompt)

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_appends_skills_justification_suffix_when_required(
        self,
        mock_load_yaml,
    ) -> None:
        """When require_justification=True, skills_structured_justification_suffix is appended."""
        mock_load_yaml.return_value = {
            "skills_structured": "SKILLS_JSON_SPEC",
            "skills_structured_justification_suffix": "Explain why each skill is supported.",
        }
        _load_prompts_from_file.cache_clear()

        skills_plan = DummySkillsPlan(canonical_skills=[DummySkill("SQL", "advanced")])
        req = DummyRequest(profile_info={"skills": ["SQL"]})

        prompt = build_skills_selection_prompt(
            request=req,
            evidence_plan=None,
            skills_plan=skills_plan,
            language="en",
            require_justification=True,
        )

        self.assertIn("=== Skills selection instructions ===", prompt)
        self.assertIn("SKILLS_JSON_SPEC", prompt)
        self.assertIn("=== Justification Instructions ===", prompt)
        self.assertIn("Explain why each skill is supported.", prompt)

    @patch("functions.utils.prompts_builder._collect_evidence_facts_for_section", autospec=True)
    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_includes_evidence_facts_when_present(
            self,
            mock_load_yaml,
            mock_collect_evidence,
    ) -> None:
        """
        If _collect_evidence_facts_for_section returns facts, they should be listed
        under '=== Evidence facts for skills (cross-section) ==='.
        """
        mock_load_yaml.return_value = {"skills_structured": "SKILLS_JSON_SPEC"}
        _load_prompts_from_file.cache_clear()

        mock_collect_evidence.return_value = ["Fact A", "Fact B"]

        skills_plan = DummySkillsPlan(canonical_skills=[DummySkill("Python")])
        req = DummyRequest(profile_info={"skills": ["Python"]})

        prompt = build_skills_selection_prompt(
            request=req,
            evidence_plan=DummyEvidencePlan({}),
            skills_plan=skills_plan,
            language="en",
            require_justification=False,
        )

        self.assertIn("=== Evidence facts for skills (cross-section) ===", prompt)
        self.assertIn("- Fact A", prompt)
        self.assertIn("- Fact B", prompt)


# ---------------------------------------------------------------------------
# Tests for build_experience_justification_prompt
# ---------------------------------------------------------------------------


class TestBuildExperienceJustificationPrompt(LoggingTestCase):
    """Tests for justification prompt for the experience section."""

    @patch("functions.utils.prompts_builder.load_yaml_dict")
    def test_builds_experience_justification_with_all_blocks(
            self,
            mock_load_yaml,
    ) -> None:
        """
        build_experience_justification_prompt should:
        - include language
        - include profile info JSON
        - include evidence facts
        - include final rendered experience text
        - include justification instructions from prompts.yaml
        """
        mock_load_yaml.return_value = {
            # Use the key actually read by build_experience_justification_prompt
            "experience_justification_json_only": "Return JUSTIFICATION_JSON with evidence_map.",
        }
        _load_prompts_from_file.cache_clear()

        ev_plan = DummyEvidencePlan(
            mapping={
                "experience": [
                    DummyEvidence("ev1", "Led cross-functional project X."),
                    DummyEvidence("ev2", "Achieved revenue growth of 20%."),
                ]
            }
        )

        profile_info = {"experience": ["role-1", "role-2"]}
        job_role_info = {"title": "Senior Data Scientist"}

        req = DummyRequest(
            language="en",
            profile_info=profile_info,
            job_role_info=job_role_info,
        )

        section_text = "- Led X\n- Delivered Y"
        prompt = build_experience_justification_prompt(
            request=req,
            evidence_plan=ev_plan,
            section_text=section_text,
        )

        # Language + general instructions
        self.assertIn("You are an assistant that produces JSON justifications", prompt)
        self.assertIn("The CV language is", prompt)

        # Profile info (no longer job context block)
        self.assertIn("=== Profile Info (JSON) ===", prompt)
        self.assertIn('"experience": [', prompt)
        self.assertIn('"role-1"', prompt)
        self.assertIn('"role-2"', prompt)

        # Evidence facts
        self.assertIn("=== Evidence Facts for experience ===", prompt)
        self.assertIn("Led cross-functional project X.", prompt)
        self.assertIn("Achieved revenue growth of 20%.", prompt)

        # Final rendered section text
        self.assertIn("=== Final rendered 'experience' section text ===", prompt)
        self.assertIn("- Led X", prompt)

        # Justification instructions (from prompts.yaml key we mocked)
        self.assertIn("=== Justification Instructions ===", prompt)
        self.assertIn("Return JUSTIFICATION_JSON with evidence_map.", prompt)


if __name__ == "__main__":
    unittest.main()
