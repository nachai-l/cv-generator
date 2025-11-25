# tests/test_stage_b_claims_and_justifications.py
"""
Additional Stage B tests focusing on claims / justification wiring.

These are intentionally IO-free (no real LLM calls or CSV logging).
They complement:

    - test_stage_b_generation.py
    - test_stage_b_parameters_and_io.py

Covered here:

- section prompt building (now via prompts_builder.build_section_prompt):
    - When GENERATION_CFG.sections_req_justification includes a section,
      the prompt should include a "Justification Instructions" block
      (if a justification prompt is configured).

- CVGenerationEngine.generate_cv:
    - When justification is required, and split_section_and_justification
      returns a JSON tail, the final CVGenerationResponse.justification
      is populated with a Justification model derived from that JSON.
"""

from __future__ import annotations

import unittest
from typing import Any, Dict

import functions.stage_b_generation as stage_b_generation
import functions.utils.prompts_builder as prompts_builder
from functions.stage_b_generation import CVGenerationEngine
from .test_stage_b_generation import LoggingTestCase  # reuse logging style
from schemas.output_schema import Justification


class _SimpleTemplateInfo:
    """Minimal stub for request.template_info."""

    def __init__(self) -> None:
        self.template_id = "T_EMPLOYER_STD_V3"
        self.sections_order = ["profile_summary"]
        self.max_chars_per_section: Dict[str, int] = {}


class _SimpleRequest:
    """Minimal request object used to trigger Stage B flows.

    We intentionally do NOT use the full CVGenerationRequest model here
    to keep these tests IO-free and focused on wiring.
    """

    def __init__(self) -> None:
        self.language = "en"
        self.sections = ["profile_summary"]
        # profile_info.summary → makes 'profile_summary' available
        self.profile_info = {"summary": "Some basic summary text from profile_info."}
        self.user_input_cv_text_by_section: Dict[str, str] = {}
        self.student_profile = None
        self.template_info = _SimpleTemplateInfo()
        self.user_id = "U-TEST"
        self.job_id = "JOB_TEST"
        self.request_id = "REQ-TEST"


# ---------------------------------------------------------------------------
# Tests for section prompt justification behaviour
# ---------------------------------------------------------------------------


class TestBuildSectionPromptJustificationFlag(LoggingTestCase):
    """Tests that build_section_prompt honors sections_req_justification."""

    def setUp(self) -> None:
        super().setUp()

        # Snapshot originals to restore later (PROMPTS NOW LIVE IN prompts_builder)
        self._orig_generation_cfg = prompts_builder.GENERATION_CFG
        self._orig_load_prompts = prompts_builder._load_prompts_from_file

        # Use a very small in-memory prompts config to avoid reading files
        def _fake_load_prompts() -> Dict[str, str]:
            return {
                "default": "DEFAULT OUTPUT REQUIREMENTS",
                "justification": "JUSTIFY_JSON_INSTRUCTIONS",
            }

        # Drive justification behaviour through prompts_builder config
        prompts_builder.GENERATION_CFG = {
            "sections_req_justification": ["profile_summary"]
        }
        prompts_builder._load_prompts_from_file = _fake_load_prompts  # type: ignore[assignment]

    def tearDown(self) -> None:
        # Restore globals on prompts_builder
        prompts_builder.GENERATION_CFG = self._orig_generation_cfg
        prompts_builder._load_prompts_from_file = self._orig_load_prompts  # type: ignore[assignment]
        super().tearDown()

    def test_prompt_includes_justification_block_when_enabled(self) -> None:
        """profile_summary should include a justification block when configured."""
        req = _SimpleRequest()

        prompt = prompts_builder.build_section_prompt(
            request=req,
            evidence_plan=None,
            section_id="profile_summary",
        )

        self.assertIn("=== Output Requirements ===", prompt)
        self.assertIn("DEFAULT OUTPUT REQUIREMENTS", prompt)

        # Justification instructions must appear for profile_summary
        self.assertIn("=== Justification Instructions ===", prompt)
        self.assertIn("JUSTIFY_JSON_INSTRUCTIONS", prompt)

    def test_prompt_omits_justification_block_for_other_sections(self) -> None:
        """Sections not listed in sections_req_justification should be untouched."""
        req = _SimpleRequest()

        prompt = prompts_builder.build_section_prompt(
            request=req,
            evidence_plan=None,
            section_id="experience",  # not in sections_req_justification
        )

        self.assertIn("=== Output Requirements ===", prompt)
        self.assertIn("DEFAULT OUTPUT REQUIREMENTS", prompt)
        # No justification block for 'experience'
        self.assertNotIn("=== Justification Instructions ===", prompt)
        self.assertNotIn("JUSTIFY_JSON_INSTRUCTIONS", prompt)


# ---------------------------------------------------------------------------
# Tests for generate_cv justification wiring end-to-end
# ---------------------------------------------------------------------------


class TestGenerateCVJustificationFlow(LoggingTestCase):
    """Tests that generate_cv propagates justification into the response."""

    def setUp(self) -> None:
        super().setUp()

        # Snapshot originals we will patch
        self._orig_generation_cfg = stage_b_generation.GENERATION_CFG
        self._orig_load_prompts = prompts_builder._load_prompts_from_file
        self._orig_call_llm = stage_b_generation.CVGenerationEngine._call_llm_with_retries
        self._orig_split = stage_b_generation.split_section_and_justification
        self._orig_should = stage_b_generation.should_require_justification

        # Minimal prompts configuration to avoid file IO
        def _fake_load_prompts() -> Dict[str, str]:
            return {
                "default": "DEFAULT OUTPUT REQUIREMENTS",
                "justification": "JUSTIFY_JSON_INSTRUCTIONS",
            }

        stage_b_generation.GENERATION_CFG = {
            "sections_req_justification": ["profile_summary"]
        }
        # Patch prompts_builder loader (Stage B delegates prompt construction there)
        prompts_builder._load_prompts_from_file = _fake_load_prompts  # type: ignore[assignment]

        # Stub: do NOT touch real LLM client; this bypasses metrics helper.
        def _fake_call_llm(self, prompt: str, section_id: str) -> str:  # type: ignore[override]
            return "RAW_LLM_TEXT_IGNORED_IN_SPLIT"

        stage_b_generation.CVGenerationEngine._call_llm_with_retries = _fake_call_llm  # type: ignore[assignment]

        # Stub: justification is required only for profile_summary
        def _fake_should_require(section_id: str, gen_cfg: Dict[str, Any] | None) -> bool:
            return section_id == "profile_summary"

        stage_b_generation.should_require_justification = _fake_should_require  # type: ignore[assignment]

        # Stub: split_section_and_justification always returns a fixed section text
        # and a valid JSON justification tail, regardless of raw text.
        def _fake_split(raw: str) -> tuple[str, str]:
            section_text = (
                "This is a generated profile summary used for justification tests."
            )
            justification_json = (
                '{"evidence_map": [], '
                '"unsupported_claims": [], '
                '"coverage_score": 0.8, '
                '"total_claims_analyzed": 2}'
            )
            return section_text, justification_json

        stage_b_generation.split_section_and_justification = _fake_split  # type: ignore[assignment]

    def tearDown(self) -> None:
        # Restore all patched globals
        stage_b_generation.GENERATION_CFG = self._orig_generation_cfg
        prompts_builder._load_prompts_from_file = self._orig_load_prompts  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._call_llm_with_retries = self._orig_call_llm  # type: ignore[assignment]
        stage_b_generation.split_section_and_justification = self._orig_split  # type: ignore[assignment]
        stage_b_generation.should_require_justification = self._orig_should  # type: ignore[assignment]

        super().tearDown()

    def test_generate_cv_populates_justification_model(self) -> None:
        """
        When justification is required and split_section_and_justification
        yields a JSON tail, the final CVGenerationResponse.justification
        should be a Justification instance derived from that JSON.
        """
        req = _SimpleRequest()

        engine = CVGenerationEngine(llm_client=None)
        resp = engine.generate_cv(request=req, evidence_plan=None, skills_plan=None)

        # 1) The profile_summary section should be present and use our stub text
        self.assertIn("profile_summary", resp.sections)
        section_obj = resp.sections["profile_summary"]
        self.assertIn("generated profile summary", section_obj.text)

        # 2) Justification object should be populated from the stubbed JSON
        self.assertIsInstance(resp.justification, Justification)

        # Coverage score should be a sane probability-like value in [0, 1].
        if hasattr(resp.justification, "coverage_score"):
            score = resp.justification.coverage_score
            # We don't rely on the exact value because validate_justification_against_text
            # may normalize or override it.
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        if hasattr(resp.justification, "total_claims_analyzed"):
            analyzed = resp.justification.total_claims_analyzed

            # Must be an integer ≥ 0
            self.assertIsInstance(analyzed, int)
            self.assertGreaterEqual(analyzed, 0)


# ---------------------------------------------------------------------------
# New tests: justification disabled, malformed JSON, multi-section behaviour
# ---------------------------------------------------------------------------


class TestGenerateCVNoJustification(LoggingTestCase):
    """If sections_req_justification is empty, justification should be effectively disabled."""

    def setUp(self) -> None:
        super().setUp()
        self._orig_generation_cfg = stage_b_generation.GENERATION_CFG
        self._orig_split = stage_b_generation.split_section_and_justification
        self._orig_call_llm = stage_b_generation.CVGenerationEngine._call_llm_with_retries

        # Stub LLM calls so we never hit real clients
        def _fake_call_llm(self, prompt: str, section_id: str) -> str:  # type: ignore[override]
            return "DUMMY SECTION TEXT"

        stage_b_generation.CVGenerationEngine._call_llm_with_retries = _fake_call_llm  # type: ignore[assignment]

    def tearDown(self) -> None:
        stage_b_generation.GENERATION_CFG = self._orig_generation_cfg
        stage_b_generation.split_section_and_justification = self._orig_split  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._call_llm_with_retries = self._orig_call_llm  # type: ignore[assignment]
        super().tearDown()

    def test_generate_cv_omits_justification_when_disabled(self) -> None:
        req = _SimpleRequest()

        # Patch config: justification OFF
        stage_b_generation.GENERATION_CFG = {"sections_req_justification": []}

        # Patch splitting to prove it *would* produce JSON, but config disables it.
        def _fake_split(raw: str) -> tuple[str, str]:
            return "TEXT", '{"coverage_score": 1.0}'

        stage_b_generation.split_section_and_justification = _fake_split  # type: ignore[assignment]

        engine = CVGenerationEngine(llm_client=None)
        resp = engine.generate_cv(request=req, evidence_plan=None, skills_plan=None)

        # Should still return a Justification instance, but effectively "empty"
        self.assertIsInstance(resp.justification, Justification)
        # In your model this may be evidence_map/unsupported_claims
        if hasattr(resp.justification, "evidence_map"):
            self.assertEqual(resp.justification.evidence_map, [])
        if hasattr(resp.justification, "unsupported_claims"):
            self.assertEqual(resp.justification.unsupported_claims, [])


class TestMalformedJustification(LoggingTestCase):
    """Malformed justification JSON must not break Stage B; it should fall back to an empty model."""

    def setUp(self) -> None:
        super().setUp()
        self._orig_generation_cfg = stage_b_generation.GENERATION_CFG
        self._orig_split = stage_b_generation.split_section_and_justification
        self._orig_call_llm = stage_b_generation.CVGenerationEngine._call_llm_with_retries

        def _fake_call_llm(self, prompt: str, section_id: str) -> str:  # type: ignore[override]
            return "DUMMY SECTION TEXT"

        stage_b_generation.CVGenerationEngine._call_llm_with_retries = _fake_call_llm  # type: ignore[assignment]

    def tearDown(self) -> None:
        stage_b_generation.GENERATION_CFG = self._orig_generation_cfg
        stage_b_generation.split_section_and_justification = self._orig_split  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._call_llm_with_retries = self._orig_call_llm  # type: ignore[assignment]
        super().tearDown()

    def test_malformed_justification_json_falls_back_to_empty(self) -> None:
        req = _SimpleRequest()
        stage_b_generation.GENERATION_CFG = {
            "sections_req_justification": ["profile_summary"]
        }

        # Return invalid JSON tail
        def _fake_split(raw: str) -> tuple[str, str]:
            return "Some text", "{not valid json"

        stage_b_generation.split_section_and_justification = _fake_split  # type: ignore[assignment]

        engine = CVGenerationEngine(llm_client=None)
        resp = engine.generate_cv(request=req, evidence_plan=None, skills_plan=None)

        # Must not raise & must return some normalized Justification model
        j = resp.justification
        self.assertIsInstance(j, Justification)


class TestMultipleSectionsJustification(LoggingTestCase):
    """Ensure multi-section requests behave correctly with justification flags."""

    def setUp(self) -> None:
        super().setUp()
        self._orig_generation_cfg = stage_b_generation.GENERATION_CFG
        self._orig_split = stage_b_generation.split_section_and_justification
        self._orig_call_llm = stage_b_generation.CVGenerationEngine._call_llm_with_retries

        def _fake_call_llm(self, prompt: str, section_id: str) -> str:  # type: ignore[override]
            return "DUMMY SECTION TEXT"

        stage_b_generation.CVGenerationEngine._call_llm_with_retries = _fake_call_llm  # type: ignore[assignment]

    def tearDown(self) -> None:
        stage_b_generation.GENERATION_CFG = self._orig_generation_cfg
        stage_b_generation.split_section_and_justification = self._orig_split  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._call_llm_with_retries = self._orig_call_llm  # type: ignore[assignment]
        super().tearDown()

    def test_only_selected_sections_receive_justification(self) -> None:
        req = _SimpleRequest()
        req.sections = ["profile_summary", "experience"]

        stage_b_generation.GENERATION_CFG = {
            "sections_req_justification": ["profile_summary"]
        }

        def _fake_split(raw: str) -> tuple[str, str]:
            # Every LLM output returns justification, but config says: only profile_summary needs it.
            return "Section text", '{"coverage_score": 0.9}'

        stage_b_generation.split_section_and_justification = _fake_split  # type: ignore[assignment]

        engine = CVGenerationEngine(llm_client=None)
        resp = engine.generate_cv(request=req, evidence_plan=None, skills_plan=None)

        # We only have a single Justification object in the top-level response.
        self.assertIsInstance(resp.justification, Justification)
        # Sanity: still in [0, 1]
        if hasattr(resp.justification, "coverage_score"):
            score = resp.justification.coverage_score
            self.assertIsInstance(score, (int, float))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ---------------------------------------------------------------------------
# Tests for skills_structured justification suffix wiring
# ---------------------------------------------------------------------------


class _SimpleSkillsPlan:
    """Minimal stub for SkillsSectionPlan used by skills selection prompt tests."""

    class _Skill:
        def __init__(self, name: str, level: str | None) -> None:
            self.name = name
            self.level = level

    def __init__(self) -> None:
        # One canonical skill is enough to exercise the canonical-skills loop
        self.canonical_skills = [self._Skill(name="Python", level="Advanced")]


class TestBuildSkillsSelectionPromptJustificationSuffix(LoggingTestCase):
    """Tests that skills_structured justification instructions are appended, not replaced."""

    def setUp(self) -> None:
        super().setUp()

        # Snapshot originals to restore later
        self._orig_load_prompts = prompts_builder._load_prompts_from_file

        # In-memory prompts config so we don't touch the filesystem
        def _fake_load_prompts() -> Dict[str, str]:
            return {
                "skills_structured": "BASE_SKILLS_STRUCTURED_PROMPT",
                "skills_structured_justification_suffix": "SKILLS_JUST_SUFFIX",
            }

        prompts_builder._load_prompts_from_file = _fake_load_prompts  # type: ignore[assignment]

    def tearDown(self) -> None:
        prompts_builder._load_prompts_from_file = self._orig_load_prompts  # type: ignore[assignment]
        super().tearDown()

    def test_skills_prompt_without_justification_suffix(self) -> None:
        """When require_justification=False, base skills_structured prompt is used without suffix."""
        req = _SimpleRequest()
        skills_plan = _SimpleSkillsPlan()

        prompt = prompts_builder.build_skills_selection_prompt(
            request=req,
            evidence_plan=None,
            skills_plan=skills_plan,
            language="en",
            require_justification=False,
        )

        # Base skills_structured instructions must be present
        self.assertIn("BASE_SKILLS_STRUCTURED_PROMPT", prompt)
        self.assertIn("=== Skills selection instructions ===", prompt)

        # No justification block or suffix in this mode
        self.assertNotIn("=== Justification Instructions ===", prompt)
        self.assertNotIn("SKILLS_JUST_SUFFIX", prompt)

        # Canonical skills must still be listed
        self.assertIn('name: "Python", level: "Advanced"', prompt)

    def test_skills_prompt_with_justification_suffix(self) -> None:
        """When require_justification=True, suffix is appended after base instructions."""
        req = _SimpleRequest()
        skills_plan = _SimpleSkillsPlan()

        prompt = prompts_builder.build_skills_selection_prompt(
            request=req,
            evidence_plan=None,
            skills_plan=skills_plan,
            language="en",
            require_justification=True,
        )

        # Base skills_structured instructions must still be there
        self.assertIn("BASE_SKILLS_STRUCTURED_PROMPT", prompt)
        self.assertIn("=== Skills selection instructions ===", prompt)

        # Justification instructions are appended, not replacing base prompt
        self.assertIn("=== Justification Instructions ===", prompt)
        self.assertIn("SKILLS_JUST_SUFFIX", prompt)

        # Canonical skills are still listed correctly
        self.assertIn('name: "Python", level: "Advanced"', prompt)


class TestJustificationAggregationAcrossSections(LoggingTestCase):
    """
    Ensure that:
    - profile_summary justification (normal section flow), and
    - skills justification (structured-first skills flow)
    are both aggregated into the single top-level Justification object,
    instead of one overwriting the other.
    """

    def setUp(self) -> None:
        super().setUp()

        # Snapshot originals we patch
        self._orig_generation_cfg = stage_b_generation.GENERATION_CFG
        self._orig_load_prompts = prompts_builder._load_prompts_from_file
        self._orig_call_llm = stage_b_generation.CVGenerationEngine._call_llm_with_retries
        self._orig_split = stage_b_generation.split_section_and_justification
        self._orig_should = stage_b_generation.should_require_justification
        self._orig_load_params = stage_b_generation.load_parameters
        self._orig_gen_skills = stage_b_generation.CVGenerationEngine._generate_structured_skills
        # 🔹 NEW: snapshot parse/validate so we can stub them
        self._orig_parse_just = stage_b_generation.parse_justification_json
        self._orig_validate_just = stage_b_generation.validate_justification_against_text

        # --- Config: require justification for profile_summary + skills ---
        stage_b_generation.GENERATION_CFG = {
            "sections_req_justification": ["profile_summary", "skills"]
        }

        # --- Use in-memory prompts; no FS access ---
        def _fake_load_prompts() -> Dict[str, str]:
            return {
                "default": "DEFAULT OUTPUT REQUIREMENTS",
                "justification": "JUSTIFY_JSON_INSTRUCTIONS",
                "skills_structured": "BASE_SKILLS_STRUCTURED_PROMPT",
            }

        prompts_builder._load_prompts_from_file = _fake_load_prompts  # type: ignore[assignment]

        # --- Force enable_structured_skills via load_parameters() path ---
        def _fake_load_parameters() -> Dict[str, Any]:
            return {
                "generation": {
                    "enable_structured_skills": True,
                    "prompts_file": "prompts.yaml",
                    "sections_req_justification": ["profile_summary", "skills"],
                },
                "core_sections": [],
            }

        stage_b_generation.load_parameters = _fake_load_parameters  # type: ignore[assignment]

        # --- Stub LLM: output is irrelevant, split() + parse() drive justification ---
        def _fake_call_llm(self, prompt: str, section_id: str) -> str:  # type: ignore[override]
            return "PROFILE_PLUS_J"

        stage_b_generation.CVGenerationEngine._call_llm_with_retries = _fake_call_llm  # type: ignore[assignment]

        # --- Only profile_summary is actually split here ---
        def _fake_split(raw: str) -> tuple[str, str]:
            section_text = "This is the profile summary text."
            # Tail JSON is ignored by our fake parse_justification_json,
            # but we keep it syntactically valid anyway.
            justification_json = """
            {
              "evidence_map": [],
              "unsupported_claims": [],
              "coverage_score": 0.8,
              "total_claims_analyzed": 1
            }
            """
            return section_text, justification_json

        stage_b_generation.split_section_and_justification = _fake_split  # type: ignore[assignment]

        # --- Justification is required for both profile_summary and skills ---
        def _fake_should_require(section_id: str, gen_cfg: Dict[str, Any] | None) -> bool:
            return section_id in ("profile_summary", "skills", "skills_structured")

        stage_b_generation.should_require_justification = _fake_should_require  # type: ignore[assignment]

        # 🔹 NEW: force profile_summary justification to contain one evidence_map item
        def _fake_parse_justification_json(raw: str) -> Justification:
            return Justification(
                evidence_map=[
                    {
                        "section": "profile_summary",
                        "sentence": "Profile justification sentence.",
                        "evidence_ids": ["ev_profile"],
                        "match_score": 0.9,
                    }
                ],
                unsupported_claims=[],
                coverage_score=0.85,
                total_claims_analyzed=1,
            )

        def _fake_validate_justification_against_text(
            j: Justification, text: str
        ) -> Justification:
            # No-op: keep what parse_justification_json produced
            return j

        stage_b_generation.parse_justification_json = _fake_parse_justification_json  # type: ignore[assignment]
        stage_b_generation.validate_justification_against_text = _fake_validate_justification_against_text  # type: ignore[assignment]

        # --- Stub structured skills generation to inject skills justification ---
        def _fake_generate_structured_skills(
            self, request, skills_plan, skills_section_text=None
        ):
            # Inject a skills-specific justification; skills_output itself is irrelevant here.
            self._skills_justification = Justification(
                evidence_map=[
                    {
                        "section": "skills",
                        "sentence": "Skills justification sentence.",
                        "evidence_ids": ["ev_skills"],
                        "match_score": 0.95,
                    }
                ],
                unsupported_claims=[],
                coverage_score=0.9,
                total_claims_analyzed=1,
            )
            # Return an empty list; we only care about justification wiring
            return []

        stage_b_generation.CVGenerationEngine._generate_structured_skills = _fake_generate_structured_skills  # type: ignore[assignment]

    def tearDown(self) -> None:
        # Restore all patched globals
        stage_b_generation.GENERATION_CFG = self._orig_generation_cfg
        prompts_builder._load_prompts_from_file = self._orig_load_prompts  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._call_llm_with_retries = self._orig_call_llm  # type: ignore[assignment]
        stage_b_generation.split_section_and_justification = self._orig_split  # type: ignore[assignment]
        stage_b_generation.should_require_justification = self._orig_should  # type: ignore[assignment]
        stage_b_generation.load_parameters = self._orig_load_params  # type: ignore[assignment]
        stage_b_generation.CVGenerationEngine._generate_structured_skills = self._orig_gen_skills  # type: ignore[assignment]
        # 🔹 NEW: restore parse/validate
        stage_b_generation.parse_justification_json = self._orig_parse_just  # type: ignore[assignment]
        stage_b_generation.validate_justification_against_text = self._orig_validate_just  # type: ignore[assignment]
        super().tearDown()

    def test_justification_evidence_map_aggregates_profile_and_skills(self) -> None:
        # Request that asks for both profile_summary and skills
        req = _SimpleRequest()
        req.sections = ["profile_summary", "skills"]
        # Make skills available: profile_info.skills non-empty
        req.profile_info["skills"] = ["Python"]

        # Also make template order include both for predictability
        req.template_info.sections_order = ["profile_summary", "skills"]

        engine = CVGenerationEngine(llm_client=None)
        resp = engine.generate_cv(request=req, evidence_plan=None, skills_plan=None)

        self.assertIsInstance(resp.justification, Justification)

        # Collect sections from evidence_map
        sections_in_evidence = []
        if hasattr(resp.justification, "evidence_map"):
            for ev in resp.justification.evidence_map:
                # ev may be a model or dict; normalize
                sec = getattr(ev, "section", None) or getattr(ev, "section_name", None)
                if sec is None and isinstance(ev, dict):
                    sec = ev.get("section") or ev.get("section_name")
                if sec:
                    sections_in_evidence.append(sec)

        # We expect at least one entry for profile_summary and one for skills
        self.assertIn("profile_summary", sections_in_evidence)
        self.assertIn("skills", sections_in_evidence)
        # And we expect at least 2 entries total (no overwrite)
        self.assertGreaterEqual(len(sections_in_evidence), 2)


if __name__ == "__main__":
    unittest.main()
