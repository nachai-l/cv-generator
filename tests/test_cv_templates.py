# tests_utils/test_cv_templates.py
"""
Unit Tests for CV Template Rendering and File Helpers
=====================================================

This suite verifies:
- Template registry integrity (EMPLOYER_STANDARD_V3, MINIMALIST_V1, CREATIVE_V1)
- HTML and Markdown rendering consistency
- Section ordering, formatting, and missing-section handling
- File saving helpers (HTML/Markdown)
- Integration test to ensure all example CVs generate correctly

Each test prints readable start/end banners for easier PyCharm log scanning.
"""

import re
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys

# Ensure project root is on sys.path when running this file directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_cv_templates.cv_templates import (
    EMPLOYER_STANDARD_V3,
    MINIMALIST_V1,
    CREATIVE_V1,
    get_template,
    render_cv_html,
    render_cv_markdown,
    save_cv_html,
    save_cv_markdown,
)
from schemas.output_schema import CVGenerationResponse, SectionContent


# ---------------------------------------------------------------------------
# Pretty test case for consistent header/divider output
# ---------------------------------------------------------------------------

class PrettyTestCase(unittest.TestCase):
    """Base test case adding consistent headers and dividers."""

    def setUp(self):
        test_name = self._testMethodName
        print("\n" + "=" * 90, file=sys.stderr)
        print(f"🧪 STARTING TEST: {test_name}", file=sys.stderr)
        print("=" * 90, file=sys.stderr)

    def tearDown(self):
        print("-" * 90 + "\n", file=sys.stderr)


# ---------------------------------------------------------------------------
# Helpers to construct schema objects safely
# ---------------------------------------------------------------------------

def _build_section_content(text: str) -> SectionContent:
    """Construct SectionContent safely across Pydantic versions."""
    if hasattr(SectionContent, "model_construct"):
        return SectionContent.model_construct(text=text)
    if hasattr(SectionContent, "construct"):
        return SectionContent.construct(text=text)
    raise TypeError("Unsupported Pydantic version for SectionContent")


def _build_response(template_id: str, sections: dict[str, str]) -> CVGenerationResponse:
    """Construct a CVGenerationResponse for testing."""
    section_dict = {name: _build_section_content(text) for name, text in sections.items()}
    metadata_stub = type("Meta", (), {"generated_at": datetime(2025, 1, 1)})

    base_kwargs = {
        "template_id": template_id,
        "sections": section_dict,
        "job_id": "JOB-123",
        "metadata": metadata_stub,
    }

    if hasattr(CVGenerationResponse, "model_construct"):
        return CVGenerationResponse.model_construct(**base_kwargs)
    if hasattr(CVGenerationResponse, "construct"):
        return CVGenerationResponse.construct(**base_kwargs)
    raise TypeError("Unsupported Pydantic version for CVGenerationResponse")


# ---------------------------------------------------------------------------
# Shared helper to generate example CVs
# ---------------------------------------------------------------------------

def generate_example_cvs(output_dir: Path) -> None:
    """Generate example CVs (HTML + Markdown) for all dev templates."""
    output_dir.mkdir(exist_ok=True)
    templates = (EMPLOYER_STANDARD_V3, MINIMALIST_V1, CREATIVE_V1)

    for tpl in templates:
        response = _build_response(
            template_id=tpl.template_id,
            sections={
                "profile_summary": f"Sample summary for {tpl.name}.",
                "skills": "Python, Data Analysis, Machine Learning",
                "experience": "Worked at ExampleCorp as Data Scientist.",
                "education": "PhD in Biological Science, Japan",
            },
        )
        save_cv_html(response, (output_dir / f"{tpl.template_id}.html").as_posix())
        save_cv_markdown(response, (output_dir / f"{tpl.template_id}.md").as_posix())


# ---------------------------------------------------------------------------
# Main test class
# ---------------------------------------------------------------------------

class TestCVTemplates(PrettyTestCase):
    """Tests for template registry, rendering, and file helpers."""

    # --- Template registry ---

    def test_templates_loaded_and_constants_present(self):
        self.assertEqual(EMPLOYER_STANDARD_V3.template_id, "T_EMPLOYER_STD_V3")
        self.assertEqual(MINIMALIST_V1.template_id, "T_MINIMALIST_V1")
        self.assertEqual(CREATIVE_V1.template_id, "T_CREATIVE_V1")

    def test_get_template_known_and_unknown_id(self):
        tpl_known = get_template("T_MINIMALIST_V1")
        self.assertIs(tpl_known, MINIMALIST_V1)

        tpl_fallback = get_template("NON_EXISTING_TEMPLATE")
        self.assertIs(tpl_fallback, EMPLOYER_STANDARD_V3)

    # --- HTML rendering ---

    def test_render_cv_html_basic_structure(self):
        response = _build_response(
            template_id="T_MINIMALIST_V1",
            sections={
                "profile_summary": "Experienced data scientist.",
                "experience": "Worked at Company A.",
            },
        )
        html = render_cv_html(response)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Experienced data scientist.", html)
        self.assertIn("Job ID: JOB-123", html)

    def test_render_cv_html_respects_section_order(self):
        template = MINIMALIST_V1
        sections = template.sections_order[:4]
        response = _build_response(
            template_id=template.template_id,
            sections={s: f"Content for {s}" for s in sections},
        )
        html = render_cv_html(response)

        def pos(section):
            title = section.replace("_", " ").title()
            match = re.search(rf"<h2 class=\"section-title\">{re.escape(title)}</h2>", html)
            self.assertIsNotNone(match, f"Missing heading for {section}")
            return match.start()

        positions = [pos(s) for s in sections]
        self.assertEqual(positions, sorted(positions))

    def test_render_cv_html_skips_missing_sections(self):
        template = MINIMALIST_V1
        first = template.sections_order[0]
        response = _build_response(template.template_id, {first: "Only this section"})
        html = render_cv_html(response)
        self.assertIn(first.replace("_", " ").title(), html)
        for s in template.sections_order[1:]:
            self.assertNotIn(s.replace("_", " ").title(), html)

    def test_render_cv_html_experience_block_formatting(self):
        response = _build_response(
            template_id="T_EMPLOYER_STD_V3",
            sections={"experience": "Role A at X.\n\nRole B at Y."},
        )
        html = render_cv_html(response)
        self.assertIn("Role A at X.", html)
        self.assertIn("Role B at Y.", html)
        self.assertGreaterEqual(html.count("experience-item"), 2)

    # --- Markdown rendering ---

    def test_render_cv_markdown_basic(self):
        response = _build_response(
            template_id="T_CREATIVE_V1",
            sections={
                "profile_summary": "Creative technologist.",
                "skills": "- Python\n- ML",
            },
        )
        md = render_cv_markdown(response)
        self.assertTrue(md.startswith("# Candidate Name"))
        self.assertIn("Creative technologist.", md)
        self.assertIn("Generated: January 01, 2025", md)

    def test_render_cv_markdown_skips_missing_sections(self):
        template = MINIMALIST_V1
        last = template.sections_order[-1]
        response = _build_response(template.template_id, {last: "Only last"})
        md = render_cv_markdown(response)
        self.assertIn(last.replace("_", " ").title(), md)
        for s in template.sections_order[:-1]:
            self.assertNotIn(s.replace("_", " ").title(), md)

    # --- File saving ---

    def test_save_cv_html_writes_file(self):
        response = _build_response("T_MINIMALIST_V1", {"profile_summary": "HTML save test"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.html"
            save_cv_html(response, path.as_posix())
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertIn("HTML save test", content)

    def test_save_cv_markdown_writes_file(self):
        response = _build_response("T_MINIMALIST_V1", {"profile_summary": "MD save test"})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cv.md"
            save_cv_markdown(response, path.as_posix())
            self.assertTrue(path.exists())
            content = path.read_text()
            self.assertIn("MD save test", content)
            self.assertIn("# Candidate Name", content)

    # --- Integration test ---

    def test_generate_example_cvs_files_exist(self):
        out_dir = Path(__file__).parent / "generated_test_cvs"
        generate_example_cvs(out_dir)
        for tpl in (EMPLOYER_STANDARD_V3, MINIMALIST_V1, CREATIVE_V1):
            self.assertTrue((out_dir / f"{tpl.template_id}.html").exists())
            self.assertTrue((out_dir / f"{tpl.template_id}.md").exists())


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent / "generated_test_cvs"
    print(f"--- Generating example CVs into: {out_dir} ---")
    generate_example_cvs(out_dir)
    print("Done.")


if __name__ == "__main__":
    unittest.main(exit=False)
    main()
