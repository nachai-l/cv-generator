"""
Unittest suite for skills formatting utilities.

This module tests the pure helper functions in:

    functions.utils.skills_formatting

Core responsibilities covered:

- pretty_skill_level:
    - Maps taxonomy-style level labels (e.g. "L3_Advanced") into
      human-readable strings ("Advanced").
    - Handles None / plain labels gracefully.

- format_plain_skill_bullets:
    - Converts structured OutputSkillItem objects into a bullet-list
      string suitable for the free-text 'skills' section.

- parse_skills_from_bullets:
    - Parses a simple bullet-list skills block (one skill per line)
      back into a list of skill phrases, used as a fallback when
      structured JSON is missing.

These tests intentionally do **not** depend on Stage B, LLM clients,
or parameters.yaml — they ensure the formatting behaviour remains
stable and predictable across the pipeline.
"""

from __future__ import annotations

import sys
import unittest
from typing import List

from functions.utils.skills_formatting import (
    pretty_skill_level,
    format_plain_skill_bullets,
    parse_skills_from_bullets,
)
from schemas.output_schema import OutputSkillItem


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
# Tests for pretty_skill_level
# ---------------------------------------------------------------------------


class TestPrettySkillLevel(LoggingTestCase):
    """Tests for pretty_skill_level normalisation behaviour."""

    def test_none_returns_none(self) -> None:
        """None input should produce None output."""
        self.assertIsNone(pretty_skill_level(None))

    def test_taxonomy_style_label_is_cleaned(self) -> None:
        """Labels like 'L3_Advanced' should drop the prefix and normalize casing."""
        self.assertEqual(pretty_skill_level("L3_Advanced"), "Advanced")
        self.assertEqual(pretty_skill_level("L4_EXPERT"), "Expert")
        self.assertEqual(pretty_skill_level("L2_intermediate"), "Intermediate")

    def test_plain_label_is_title_cased(self) -> None:
        """Plain labels without prefix should just be title-cased."""
        self.assertEqual(pretty_skill_level("advanced"), "Advanced")
        self.assertEqual(pretty_skill_level("Intermediate"), "Intermediate")

    def test_non_string_levels_are_coerced(self) -> None:
        """Non-string level values (e.g. Enums/int-like) should be stringified safely."""
        class DummyEnum:
            def __str__(self) -> str:
                return "L3_Advanced"

        self.assertEqual(pretty_skill_level(DummyEnum()), "Advanced")


# ---------------------------------------------------------------------------
# Tests for format_plain_skill_bullets
# ---------------------------------------------------------------------------


class TestFormatPlainSkillBullets(LoggingTestCase):
    """Tests for format_plain_skill_bullets bullet rendering."""

    def test_empty_list_yields_empty_string(self) -> None:
        """No skills should produce an empty string."""
        self.assertEqual(format_plain_skill_bullets([]), "")

    def test_skills_with_and_without_levels(self) -> None:
        """Skills should be rendered with '(Level)' when present, or just the name otherwise."""
        skills: List[OutputSkillItem] = [
            OutputSkillItem(name="Python", level="L3_Advanced", source="taxonomy"),
            OutputSkillItem(name="Machine Learning", level=None, source="inferred"),
        ]

        out = format_plain_skill_bullets(skills)
        lines = [line.strip() for line in out.splitlines() if line.strip()]

        self.assertEqual(len(lines), 2)
        self.assertIn("- Python (Advanced)", lines)
        self.assertIn("- Machine Learning", lines)
        # Ensure we did not accidentally append "(None)"
        self.assertNotIn("None", out)

    def test_skips_empty_or_whitespace_only_names(self) -> None:
        """Entries with empty/whitespace-only names should be skipped."""
        skills: List[OutputSkillItem] = [
            OutputSkillItem(name="  ", level="L2_Intermediate", source="taxonomy"),
            OutputSkillItem(name="Data Analysis", level="L3_Advanced", source="taxonomy"),
        ]

        out = format_plain_skill_bullets(skills)
        lines = [line.strip() for line in out.splitlines() if line.strip()]

        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], "- Data Analysis (Advanced)")


# ---------------------------------------------------------------------------
# Tests for parse_skills_from_bullets
# ---------------------------------------------------------------------------


class TestParseSkillsFromBullets(LoggingTestCase):
    """Tests for parse_skills_from_bullets line-oriented parsing."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty or None-like text should return an empty list."""
        self.assertEqual(parse_skills_from_bullets(""), [])
        self.assertEqual(parse_skills_from_bullets("   "), [])

    def test_simple_dash_bullets(self) -> None:
        """Lines starting with '-' should be parsed as skills with the dash stripped."""
        text = """
        - Python
        - Machine Learning
        - SQL.
        """
        skills = parse_skills_from_bullets(text)
        self.assertEqual(skills, ["Python", "Machine Learning", "SQL"])

    def test_star_and_dot_bullets(self) -> None:
        """Lines starting with '*', '•' should also be handled."""
        text = """
        * Data Analysis
        • Communication Skills.
        """
        skills = parse_skills_from_bullets(text)
        self.assertEqual(skills, ["Data Analysis", "Communication Skills"])

    def test_ignores_blank_lines(self) -> None:
        """Blank lines between bullets should be ignored."""
        text = """
        - Python

        - Machine Learning

        """
        skills = parse_skills_from_bullets(text)
        self.assertEqual(skills, ["Python", "Machine Learning"])

    def test_preserves_internal_punctuation(self) -> None:
        """Internal punctuation (commas, slashes) should be preserved."""
        text = """
        - Natural Language Processing / NLP
        - Data cleaning, preprocessing & feature engineering.
        """
        skills = parse_skills_from_bullets(text)
        self.assertEqual(
            skills,
            [
                "Natural Language Processing / NLP",
                "Data cleaning, preprocessing & feature engineering",
            ],
        )


if __name__ == "__main__":
    unittest.main()
