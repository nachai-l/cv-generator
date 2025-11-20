# tests/utils_test_support.py
"""
Shared helpers for tests and mock CV generation scripts.

This module is intentionally test-only and is NOT part of the public
service API. It provides:

- Loading the legacy YAML fixtures under tests/json_test_inputs/
- Computing a section order from legacy template_info.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from functions.utils.common import load_yaml_file  # type: ignore


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]
JSON_INPUT_DIR = THIS_FILE.parent / "json_test_inputs"


# ---------------------------------------------------------------------------
# Legacy YAML fixture loader
# ---------------------------------------------------------------------------

def load_legacy_yaml_payload() -> Dict[str, Any]:
    """
    Load the legacy YAML inputs from tests/json_test_inputs/ and
    combine them into a single dict.

    This is essentially the "Stage 0" mock payload used by:
      - tests/mock_api_generate_cv.py
      - tests/mock_api_generate_cv_v2.py

    Returned structure:
    {
        "company_info": {...},
        "job_position_info": {...},
        "job_role_info": {...},
        "profile_info": {...},
        "template_info": {...},
        "user_input_cv_text_by_section": {...},
        "language": "en",     # <-- now added (if present in template_info.yaml)
        "tone": "funny",      # <-- now added (if present in template_info.yaml)
    }
    """
    company_info = load_yaml_file(JSON_INPUT_DIR / "company_info.yaml")
    job_position_info = load_yaml_file(JSON_INPUT_DIR / "job_position_info.yaml")
    job_role_info = load_yaml_file(JSON_INPUT_DIR / "job_role_info.yaml")
    profile_info = load_yaml_file(JSON_INPUT_DIR / "profile_info.yaml")
    template_info = load_yaml_file(JSON_INPUT_DIR / "template_info.yaml")
    user_sections = load_yaml_file(JSON_INPUT_DIR / "user_input_cv_text_by_section.yaml")

    # 🔹 Lift language/tone out of template_info into top-level payload
    language = template_info.pop("language", None)
    tone = template_info.pop("tone", None)

    payload: Dict[str, Any] = {
        "company_info": company_info,
        "job_position_info": job_position_info,
        "job_role_info": job_role_info,
        "profile_info": profile_info,
        "template_info": template_info,
        "user_input_cv_text_by_section": user_sections,
    }

    if language is not None:
        payload["language"] = language
    if tone is not None:
        payload["tone"] = tone

    return payload


# ---------------------------------------------------------------------------
# Legacy template → sections helper
# ---------------------------------------------------------------------------

def compute_sections_from_legacy_template(template_info: Dict[str, Any]) -> List[str]:
    """
    Determine the ordered list of sections from a legacy template_info dict.

    Rules:
    - Use ONLY the order defined in template_info["sections_order"]
      (or fallback to template_info["sections"]).
    - Ignore any extra user-provided section IDs; user content is mapped
      into these template-defined sections instead of creating new ones.

    This is used by mock_api_generate_cv_v2.py to simulate a Stage 0
    orchestrator deciding which sections to generate.
    """
    result: List[str] = []
    seen: set[str] = set()

    raw_order = (
        template_info.get("sections_order")
        or template_info.get("sections")
        or []
    )

    if isinstance(raw_order, list):
        for item in raw_order:
            section_id: str | None
            if isinstance(item, str):
                section_id = item
            elif isinstance(item, dict):
                section_id = item.get("id")
            else:
                section_id = None

            if not section_id or section_id in seen:
                continue

            seen.add(section_id)
            result.append(section_id)

    return result


__all__ = [
    "ROOT",
    "JSON_INPUT_DIR",
    "load_legacy_yaml_payload",
    "compute_sections_from_legacy_template",
]
