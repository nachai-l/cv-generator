"""
Additional Stage B tests focusing on small, IO-free helpers.

These are intentionally minimal so they don't depend on extra hooks or
non-existent fields. They complement test_stage_b_generation.py without
touching LLM clients or CSV logging.

Covered here:

- _build_taxonomy_only_fallback
- _summarize_skills_telemetry
"""

from __future__ import annotations

import unittest
from typing import Any, Dict, cast

import functions.stage_b_generation as stage_b_generation
from functions.stage_b_generation import (
    _build_taxonomy_only_fallback,
    _summarize_skills_telemetry,

)
from functions.stage_b_generation import CVGenerationEngine
from .test_stage_b_generation import LoggingTestCase, DummyRequest
from schemas.input_schema import CVGenerationRequest
from schemas.internal_schema import SkillsSectionPlan, CanonicalSkill
from schemas.output_schema import OutputSkillItem

class TestTaxonomyOnlyFallbackMinimal(unittest.TestCase):
    """Minimal tests for _build_taxonomy_only_fallback."""

    def test_taxonomy_only_fallback_preserves_name_level_and_source(self) -> None:
        sp = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Python", level="L3_Advanced", taxonomy_id=None),
                CanonicalSkill(name="Machine Learning", level="L2_Intermediate", taxonomy_id=None),
            ],
            allowed_additional_skills=[],
        )

        result = _build_taxonomy_only_fallback(sp)

        # Should mirror canonical list 1:1
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(s, OutputSkillItem) for s in result))

        names = [s.name for s in result]
        levels = {s.name: s.level for s in result}
        sources = {s.name: s.source for s in result}

        self.assertIn("Python", names)
        self.assertIn("Machine Learning", names)

        self.assertEqual(levels["Python"], "L3_Advanced")
        self.assertEqual(levels["Machine Learning"], "L2_Intermediate")

        # All items must be marked as taxonomy
        self.assertEqual(sources["Python"], "taxonomy")
        self.assertEqual(sources["Machine Learning"], "taxonomy")

    def test_taxonomy_only_fallback_empty_plan(self) -> None:
        sp = SkillsSectionPlan(
            canonical_skills=[],
            allowed_additional_skills=[],
        )
        result = _build_taxonomy_only_fallback(sp)
        self.assertEqual(result, [])


class TestSummarizeSkillsTelemetry(unittest.TestCase):
    """Unit tests for _summarize_skills_telemetry helper."""

    def test_telemetry_empty_none(self) -> None:
        metrics_none = _summarize_skills_telemetry(None)
        self.assertEqual(metrics_none["total_skills"], 0)
        self.assertEqual(metrics_none["taxonomy_count"], 0)
        self.assertEqual(metrics_none["inferred_count"], 0)

        metrics_empty = _summarize_skills_telemetry([])
        self.assertEqual(metrics_empty["total_skills"], 0)
        self.assertEqual(metrics_empty["taxonomy_count"], 0)
        self.assertEqual(metrics_empty["inferred_count"], 0)

    def test_telemetry_taxonomy_only(self) -> None:
        items: List[OutputSkillItem] = [
            OutputSkillItem(name="Python", level="L3", source="taxonomy"),
            OutputSkillItem(name="SQL", level="L2", source="taxonomy"),
        ]

        metrics = _summarize_skills_telemetry(items)
        self.assertEqual(metrics["total_skills"], 2)
        self.assertEqual(metrics["taxonomy_count"], 2)
        self.assertEqual(metrics["inferred_count"], 0)

    def test_telemetry_mixed_sources(self) -> None:
        items: List[OutputSkillItem] = [
            OutputSkillItem(name="Python", level="L3", source="taxonomy"),
            OutputSkillItem(name="Docker", level="L2", source="inferred"),
            OutputSkillItem(name="Kubernetes", level="L2", source="inferred"),
        ]

        metrics = _summarize_skills_telemetry(items)
        self.assertEqual(metrics["total_skills"], 3)
        self.assertEqual(metrics["taxonomy_count"], 1)
        self.assertEqual(metrics["inferred_count"], 2)

class TestConfigValidation(LoggingTestCase):
    """Tests for configuration parameter validation and clamping."""

    def test_min_skill_coverage_clamped_to_valid_range(self) -> None:
        """MIN_SKILL_COVERAGE should clamp min_coverage > 1.0 down to 1.0."""
        import importlib
        import functions.utils.llm_client as llm_client

        original_load_parameters = llm_client.load_parameters

        def fake_load_parameters() -> Dict[str, Any]:
            return {
                "generation": {
                    "skills_matching": {
                        "min_coverage": 1.5,  # invalid (>1.0)
                    }
                }
            }

        try:
            llm_client.load_parameters = fake_load_parameters
            importlib.reload(stage_b_generation)

            # After reload with fake config, MIN_SKILL_COVERAGE should be clamped
            self.assertEqual(stage_b_generation.MIN_SKILL_COVERAGE, 1.0)
            self.assertGreaterEqual(stage_b_generation.MIN_SKILL_COVERAGE, 0.0)
        finally:
            # Restore real loader + module state
            llm_client.load_parameters = original_load_parameters
            importlib.reload(stage_b_generation)

    def test_fuzzy_threshold_handles_none_value(self) -> None:
        """FUZZY_THRESHOLD should be None (and not crash) when config is None."""
        import importlib
        import functions.utils.llm_client as llm_client

        original_load_parameters = llm_client.load_parameters

        def fake_load_parameters() -> Dict[str, Any]:
            return {
                "generation": {
                    "skills_matching": {
                        "fuzzy_threshold": None,
                    }
                }
            }

        try:
            llm_client.load_parameters = fake_load_parameters
            importlib.reload(stage_b_generation)

            self.assertIsNone(stage_b_generation.FUZZY_THRESHOLD)
        finally:
            llm_client.load_parameters = original_load_parameters
            importlib.reload(stage_b_generation)

class TestDroppingIrrelevantSkillsFlag(LoggingTestCase):
    """Tests for dropping_irrelevant_skills configuration flag."""

    def test_dropping_disabled_preserves_all_canonical_skills(self) -> None:
        """When dropping_irrelevant_skills=False, all canonical skills should be preserved."""
        skills_plan = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Python", level="L3", taxonomy_id=None),
                CanonicalSkill(name="SQL", level="L2", taxonomy_id=None),
                CanonicalSkill(name="Java", level="L2", taxonomy_id=None),
            ],
            allowed_additional_skills=[],
        )

        # LLM only keeps Python, drops SQL and Java
        fake_json = """
        {
          "items": [
            { "name": "Python", "level": "L3", "keep": true, "source": "taxonomy" }
          ]
        }
        """

        original_dropping = stage_b_generation.dropping_irrelevant_skills
        stage_b_generation.dropping_irrelevant_skills = False

        try:
            engine = CVGenerationEngine(
                llm_client=lambda *_a, **_k: fake_json,
                generation_params={"max_retries": 1},
            )
            engine._call_llm_with_retries = (  # type: ignore[assignment]
                lambda prompt, section_id: fake_json
            )

            req = DummyRequest(
                sections=["skills"],
                language="en",
                profile_info={"name": "Test User"},
            )
            req_typed = cast(CVGenerationRequest, cast(object, req))

            result = engine._generate_structured_skills(req_typed, skills_plan)

            names = [s.name for s in result]
            self.assertIn("Python", names)
            self.assertIn("SQL", names)
            self.assertIn("Java", names)
            self.assertEqual(len(result), 3)
        finally:
            stage_b_generation.dropping_irrelevant_skills = original_dropping

    def test_dropping_enabled_allows_llm_to_drop_skills(self) -> None:
        """When dropping_irrelevant_skills=True, LLM can drop canonical skills."""
        skills_plan = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Python", level="L3", taxonomy_id=None),
                CanonicalSkill(name="SQL", level="L2", taxonomy_id=None),
            ],
            allowed_additional_skills=[],
        )

        fake_json = """
        {
          "items": [
            { "name": "Python", "level": "L3", "keep": true, "source": "taxonomy" }
          ]
        }
        """

        original_dropping = stage_b_generation.dropping_irrelevant_skills
        stage_b_generation.dropping_irrelevant_skills = True

        try:
            engine = CVGenerationEngine(
                llm_client=lambda *_a, **_k: fake_json,
                generation_params={"max_retries": 1},
            )
            engine._call_llm_with_retries = (  # type: ignore[assignment]
                lambda prompt, section_id: fake_json
            )

            req = DummyRequest(
                sections=["skills"],
                language="en",
                profile_info={"name": "Test User"},
            )
            req_typed = cast(CVGenerationRequest, cast(object, req))

            result = engine._generate_structured_skills(req_typed, skills_plan)

            names = [s.name for s in result]
            self.assertIn("Python", names)
            self.assertNotIn("SQL", names)
            self.assertEqual(len(result), 1)
        finally:
            stage_b_generation.dropping_irrelevant_skills = original_dropping

class TestSkillsAliasMapLoading(LoggingTestCase):
    """Tests for _load_skills_alias_map file loading logic."""

    def test_load_alias_map_returns_empty_dict_when_file_missing(self) -> None:
        """When alias map file is missing, should return empty dict."""
        original_file = stage_b_generation.ALIAS_MAP_FILE
        stage_b_generation.ALIAS_MAP_FILE = "definitely_missing_alias_map.yaml"

        stage_b_generation._load_skills_alias_map.cache_clear()
        try:
            result = stage_b_generation._load_skills_alias_map()
            self.assertEqual(result, {})
        finally:
            stage_b_generation.ALIAS_MAP_FILE = original_file
            stage_b_generation._load_skills_alias_map.cache_clear()

    def test_load_alias_map_handles_loader_error(self) -> None:
        """When load_yaml_dict raises, _load_skills_alias_map should return {}."""
        original_loader = stage_b_generation.load_yaml_dict

        def broken_loader(_path: Any) -> Any:
            raise RuntimeError("boom")

        stage_b_generation.load_yaml_dict = broken_loader  # type: ignore[assignment]
        stage_b_generation._load_skills_alias_map.cache_clear()

        try:
            result = stage_b_generation._load_skills_alias_map()
            self.assertEqual(result, {})
        finally:
            stage_b_generation.load_yaml_dict = original_loader  # type: ignore[assignment]
            stage_b_generation._load_skills_alias_map.cache_clear()

class TestSkillsMatchingEdgeCases(LoggingTestCase):
    """Tests for edge cases in canonical skill matching logic."""

    def test_fuzzy_matching_catches_typos(self) -> None:
        """Skills with minor typos should match canonical skills via fuzzy matching."""
        skills_plan = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Machine Learning", level="L4", taxonomy_id=None),
            ],
            allowed_additional_skills=[],
        )

        fake_json = """
        {
          "items": [
            { "name": "Machne Learning", "level": "L3", "keep": true, "source": "inferred" }
          ]
        }
        """

        original_threshold = stage_b_generation.FUZZY_THRESHOLD
        original_cov = stage_b_generation.MIN_SKILL_COVERAGE

        try:
            # Very permissive thresholds so matching is encouraged
            stage_b_generation.FUZZY_THRESHOLD = 0.0
            stage_b_generation.MIN_SKILL_COVERAGE = 0.0
            stage_b_generation._load_skills_alias_map.cache_clear()

            engine = CVGenerationEngine(
                llm_client=lambda *_a, **_k: fake_json,
                generation_params={"max_retries": 1},
            )
            engine._call_llm_with_retries = (  # type: ignore[assignment]
                lambda prompt, section_id: fake_json
            )

            req = DummyRequest(
                sections=["skills"],
                language="en",
                profile_info={"name": "Test User"},
            )
            req_typed = cast(CVGenerationRequest, cast(object, req))

            result = engine._generate_structured_skills(req_typed, skills_plan)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "Machine Learning")
            self.assertEqual(result[0].level, "L4")  # restored to canonical
            self.assertEqual(result[0].source, "taxonomy")
        finally:
            stage_b_generation.FUZZY_THRESHOLD = original_threshold
            stage_b_generation.MIN_SKILL_COVERAGE = original_cov

    def test_word_boundary_prevents_false_positives(self) -> None:
        """'JavaScript' should NOT match 'Java' due to word-boundary behaviour."""
        skills_plan = SkillsSectionPlan(
            canonical_skills=[
                CanonicalSkill(name="Java", level="L3", taxonomy_id=None),
            ],
            allowed_additional_skills=[],
        )

        fake_json = """
        {
          "items": [
            { "name": "JavaScript", "level": "L2", "keep": true, "source": "inferred" }
          ]
        }
        """

        original_threshold = stage_b_generation.FUZZY_THRESHOLD
        original_cov = stage_b_generation.MIN_SKILL_COVERAGE

        try:
            # Reasonable thresholds: similar but not overly permissive
            stage_b_generation.FUZZY_THRESHOLD = 0.8
            stage_b_generation.MIN_SKILL_COVERAGE = 0.8
            stage_b_generation._load_skills_alias_map.cache_clear()

            engine = CVGenerationEngine(
                llm_client=lambda *_a, **_k: fake_json,
                generation_params={"max_retries": 1},
            )
            engine._call_llm_with_retries = (  # type: ignore[assignment]
                lambda prompt, section_id: fake_json
            )

            req = DummyRequest(
                sections=["skills"],
                language="en",
                profile_info={"name": "Test User"},
            )
            req_typed = cast(CVGenerationRequest, cast(object, req))

            result = engine._generate_structured_skills(req_typed, skills_plan)

            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].name, "JavaScript")
            self.assertEqual(result[0].level, "L2")
            self.assertEqual(result[0].source, "inferred")
        finally:
            stage_b_generation.FUZZY_THRESHOLD = original_threshold
            stage_b_generation.MIN_SKILL_COVERAGE = original_cov

if __name__ == "__main__":
    unittest.main()
