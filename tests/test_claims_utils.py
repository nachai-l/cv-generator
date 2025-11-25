# tests_utils/test_claims_utils.py
"""
Unittest suite for justification/claims helper utilities.

This module tests the pure helper functions in:

    functions.utils.claims

Core responsibilities covered:

- should_require_justification:
    - Reads `generation.sections_req_justification` semantics.
    - Handles "all", explicit lists, empty lists, and missing config.

- split_section_and_justification:
    - Splits a raw LLM response into:
        (section_text, justification_text)
      where justification_text is the JSON (or JSON-like) tail.

- parse_justification_json:
    - Parses JSON text into a Justification model.
    - Raises on invalid JSON so callers can fall back to an empty object.

- build_empty_justification:
    - Produces a structurally valid Justification instance with
      sensible empty defaults.

- validate_justification_against_text:
    - Best-effort sanity check that returns a Justification object,
      even if it has to clean/normalize the input.
"""

from __future__ import annotations

import json
import sys
import unittest
from typing import Any, Dict, Tuple

from functions.utils.claims import (
    should_require_justification,
    split_section_and_justification,
    parse_justification_json,
    validate_justification_against_text,
    build_empty_justification,
)
from schemas.output_schema import Justification


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
# Tests for should_require_justification
# ---------------------------------------------------------------------------


class TestShouldRequireJustification(LoggingTestCase):
    """Tests for should_require_justification decision logic."""

    def test_missing_config_defaults_to_disabled(self) -> None:
        """No generation config or missing key → justification disabled."""
        gen_cfg: Dict[str, Any] = {}
        self.assertFalse(should_require_justification("profile_summary", gen_cfg))
        self.assertFalse(should_require_justification("skills", gen_cfg))

    def test_empty_list_disables_all_sections(self) -> None:
        """Explicit [] means justification off for every section."""
        gen_cfg: Dict[str, Any] = {"sections_req_justification": []}
        self.assertFalse(should_require_justification("profile_summary", gen_cfg))
        self.assertFalse(should_require_justification("skills", gen_cfg))
        self.assertFalse(should_require_justification("experience", gen_cfg))

    def test_all_enables_every_section(self) -> None:
        """The special value 'all' enables justification for any section."""
        gen_cfg: Dict[str, Any] = {"sections_req_justification": "all"}
        for sec in ["profile_summary", "skills", "experience", "education"]:
            with self.subTest(section=sec):
                self.assertTrue(should_require_justification(sec, gen_cfg))

    def test_list_enables_only_specific_sections(self) -> None:
        """A list of section_ids enables justification only for those sections."""
        gen_cfg: Dict[str, Any] = {
            "sections_req_justification": ["profile_summary", "skills"]
        }
        self.assertTrue(should_require_justification("profile_summary", gen_cfg))
        self.assertTrue(should_require_justification("skills", gen_cfg))
        self.assertFalse(should_require_justification("experience", gen_cfg))
        self.assertFalse(should_require_justification("education", gen_cfg))

    def test_robust_to_non_list_non_all_values(self) -> None:
        """Unexpected types should fail closed (treat as disabled)."""
        for bad_value in [123, True, None, {"foo": "bar"}]:
            gen_cfg: Dict[str, Any] = {"sections_req_justification": bad_value}
            with self.subTest(bad_value=bad_value):
                self.assertFalse(
                    should_require_justification("profile_summary", gen_cfg)
                )
                self.assertFalse(
                    should_require_justification("skills", gen_cfg)
                )


# ---------------------------------------------------------------------------
# Tests for split_section_and_justification
# ---------------------------------------------------------------------------


class TestSplitSectionAndJustification(LoggingTestCase):
    """Tests for split_section_and_justification safe splitting."""

    def test_no_json_like_tail_returns_full_text_and_empty_justification(self) -> None:
        """If no JSON-like part is found, function should return (text, '')."""
        raw = "This is a simple section body without any justification JSON."
        section, j_raw = split_section_and_justification(raw)
        self.assertEqual(section.strip(), raw.strip())
        self.assertEqual(j_raw.strip(), "")

    def test_text_followed_by_json_block_is_treated_as_plain_text(self) -> None:
        """
        Current behaviour: even if the output ends with something that *looks*
        like JSON, split_section_and_justification is conservative and treats
        the whole block as section text (no split).
        """
        section_body = "Generated profile summary goes here."
        justification_obj = {
            "claims": [],
            "unsupported_claims": [],
            "coverage_score": 0.0,
        }
        justification_json = json.dumps(justification_obj)

        raw = f"""
        {section_body}

        {justification_json}
        """.strip()

        section, j_raw = split_section_and_justification(raw)

        # Entire content remains in the section text
        self.assertIn("Generated profile summary", section)
        self.assertIn(justification_json, section)
        # No justification tail extracted
        self.assertEqual(j_raw.strip(), "")

# ---------------------------------------------------------------------------
# Tests for parse_justification_json + build_empty_justification
# ---------------------------------------------------------------------------

class TestParseAndBuildJustification(LoggingTestCase):
    """Tests for parsing and building Justification objects."""
    def test_build_empty_justification_returns_valid_model(self) -> None:
        """build_empty_justification should always return a Justification instance."""
        j = build_empty_justification()
        self.assertIsInstance(j, Justification)

    def test_parse_invalid_json_returns_empty_justification(self) -> None:
        """Clearly invalid JSON should return a safe empty Justification model."""
        bad_raw = "this is not valid JSON at all"

        j = parse_justification_json(bad_raw)

        self.assertIsInstance(j, Justification)
        self.assertEqual(j.evidence_map, [])
        self.assertEqual(j.unsupported_claims, [])
        self.assertEqual(j.unsupported_claims, [])
        self.assertEqual(j.total_claims_analyzed, 0)
        # Coverage may be 0.0 or None depending on your Justification default;
        # here we only assert it doesn't explode and stays within [0, 1].
        if j.coverage_score is not None:
            self.assertGreaterEqual(j.coverage_score, 0.0)
            self.assertLessEqual(j.coverage_score, 1.0)


# ---------------------------------------------------------------------------
# Tests for validate_justification_against_text
#---------------------------------------------------------------------------

class TestValidateJustificationAgainstText(LoggingTestCase):
    """Tests for validate_justification_against_text sanity wrapper."""

    def test_validate_keeps_justification_type(self) -> None:
        """
        validate_justification_against_text should always return a Justification
        instance, even when the text is arbitrary.
        """
        j = build_empty_justification()
        text = "This is an example section body used for sanity checking."
        validated = validate_justification_against_text(j, text)

        self.assertIsInstance(validated, Justification)

    def test_validate_with_non_empty_claims_is_still_safe(self) -> None:
        """
        If the justification already has some fields filled, validation should not
        break the object type, even if it normalizes internal fields.
        """
        # Start from a safe baseline
        j = build_empty_justification()
        # Mutate a couple of fields in a way that the model allows
        # (we don't rely on exact schema keys here).
        if hasattr(j, "coverage_score"):
            setattr(j, "coverage_score", 0.75)
        if hasattr(j, "raw_text_snippet"):
            setattr(j, "raw_text_snippet", "original snippet")

        validated = validate_justification_against_text(
            j,
            "Another synthetic section body text for validation.",
        )

        self.assertIsInstance(validated, Justification)

if __name__ == "__main__":
    unittest.main()
