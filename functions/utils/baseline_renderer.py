# functions/utils/baseline_renderer.py
"""
Utilities to build a deterministic NO-LLM baseline CVGenerationResponse.

Used for:
- regression testing
- visual diffing against LLM output
- template preview without running Stage B
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from schemas.output_schema import CVGenerationResponse, SectionContent, OutputSkillItem


# ---------------------------------------------------------------------------
# Section formatting utilities
# ---------------------------------------------------------------------------

def format_section_value(section_key: str, value: Any) -> str:
    """Convert structured dict/list data into Markdown-like text."""
    if isinstance(value, str):
        return value

    if section_key == "skills" and isinstance(value, list):
        return "\n".join(
            f"- {item.get('name')} ({item.get('level')})"
            if isinstance(item, dict) and item.get("level")
            else f"- {item.get('name')}" if isinstance(item, dict)
            else f"- {item}"
            for item in value
        )

    if section_key == "experience" and isinstance(value, list):
        blocks = []
        for job in value:
            if not isinstance(job, dict):
                continue
            title = job.get("title")
            company = job.get("company")
            period = job.get("period")
            highlights = job.get("highlights") or []

            header = []
            if title:
                header.append(f"**{title}**")
            if company or period:
                cp = ", ".join(p for p in [company, period] if p)
                header.append(f"*{cp}*")

            block = []
            if header:
                block.extend(header)
            block.extend(f"- {h}" for h in highlights)
            blocks.append("\n".join(block))

        return "\n\n".join(blocks)

    if section_key == "education" and isinstance(value, list):
        return "\n".join(
            "- " + ", ".join(p for p in [
                edu.get("degree"),
                edu.get("institution"),
                edu.get("location"),
            ] if p)
            for edu in value if isinstance(edu, dict)
        )

    if section_key in ("certifications", "awards") and isinstance(value, list):
        return "\n".join(
            f"- {item.get('name')}, {item.get('organization')}"
            if isinstance(item, dict) and item.get("organization")
            else f"- {item.get('name')}"
            if isinstance(item, dict)
            else f"- {item}"
            for item in value
        )

    if section_key == "interests" and isinstance(value, list):
        return "\n".join(f"- {i}" for i in value)

    return str(value)


def build_section_content(section_key: str, value: Any) -> SectionContent:
    """Convert raw YAML → SectionContent(dataclass)."""
    txt = format_section_value(section_key, value)
    return SectionContent.model_construct(text=txt)


# ---------------------------------------------------------------------------
# NO-LLM baseline renderer
# ---------------------------------------------------------------------------

def build_baseline_response(mock_payload: Dict[str, Any]) -> CVGenerationResponse:
    """
    Deterministically build a CVGenerationResponse WITHOUT LLM.

    This is used for:
    - baseline preview
    - regression testing
    - visual comparison with the full A–D pipeline
    """
    template_info = mock_payload.get("template_info", {}) or {}
    user_sections = mock_payload.get("user_input_cv_text_by_section", {}) or {}
    profile_info = mock_payload.get("profile_info", {}) or {}

    template_id = template_info.get("template_id", "T_EMPLOYER_STD_V3")

    # Map section → profile_info key
    profile_map = {
        "profile_summary": "summary",
        "skills": "skills",
        "experience": "experience",
        "education": "education",
        "projects": "projects",
        "certifications": "certifications",
        "awards": "awards",
        "extracurricular": "extracurricular",
        "volunteering": "volunteering",
        "interests": "interests",
    }

    # Use template-defined order only
    ordered_section_ids = (
        template_info.get("sections_order")
        or template_info.get("sections")
        or []
    )

    def non_empty(val: Any) -> bool:
        if val is None:
            return False
        if isinstance(val, str):
            return bool(val.strip())
        return bool(val)

    sections = {}
    for sid in ordered_section_ids:
        if sid in user_sections and non_empty(user_sections[sid]):
            value = user_sections[sid]
        else:
            key = profile_map.get(sid)
            value = profile_info.get(key)
            if not non_empty(value):
                continue

        sections[sid] = build_section_content(sid, value)

    # Minimal metadata object
    meta_cls = type("Meta", (), {})
    meta = meta_cls()
    meta.generated_at = datetime.now()
    meta.profile_info = profile_info
    meta.template_info = template_info

    # Structured skills
    raw_skills = profile_info.get("skills") or []
    skills_structured = []
    for item in raw_skills:
        if isinstance(item, dict) and item.get("name"):
            skills_structured.append(
                OutputSkillItem(
                    name=item["name"],
                    level=item.get("level"),
                    source="profile",
                )
            )

    return CVGenerationResponse.model_construct(
        template_id=template_id,
        sections=sections,
        metadata=meta,
        job_id="JOB_MOCK_001",
        skills=skills_structured or None,
    )
