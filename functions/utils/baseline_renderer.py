# functions/utils/baseline_renderer.py
"""
Utilities to build a deterministic NO-LLM baseline CVGenerationResponse.

This renderer:
- Uses only the structured data provided in the mock payload (YAML/JSON fixtures).
- Does NOT call any LLMs (Stage B is skipped).
- Produces a CVGenerationResponse that can be used for:
  - regression testing,
  - visual diffing against LLM output,
  - template preview without running the full A–D pipeline.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, List

from schemas.output_schema import CVGenerationResponse, SectionContent, OutputSkillItem

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

# Mapping from section_id -> key in profile_info
PROFILE_SECTION_MAP: Mapping[str, str] = {
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
    "publications": "publications",
    "training": "training",
    "references": "references",
    "additional_info": "additional_info",
}


def _non_empty(val: Any) -> bool:
    """Return True if a value should be treated as 'present' (non-empty) for baseline rendering."""
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val.strip())
    return bool(val)


# ---------------------------------------------------------------------------
# Section formatting utilities
# ---------------------------------------------------------------------------

def _format_skills_list(value: Any) -> str:
    """
    Format skills from a list of items into bullet list markdown.

    Supports:
    - list[dict]: expects items with at least "name" and optional "level".
    - list[str]: direct skill names.
    """
    if not isinstance(value, list):
        return str(value)

    lines: List[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            level = item.get("level")
            if name and level:
                lines.append(f"- {name} ({level})")
            elif name:
                lines.append(f"- {name}")
        else:
            # Fallback for simple values (e.g., strings)
            lines.append(f"- {item}")
    return "\n".join(lines)


def _format_experience_list(value: Any) -> str:
    """
    Format experience entries as markdown blocks.

    Expects a list of dicts with:
    - title: str
    - company: str
    - period: str
    - highlights: list[str]
    """
    if not isinstance(value, list):
        return str(value)

    blocks: List[str] = []
    for job in value:
        if not isinstance(job, dict):
            continue

        title = job.get("title")
        company = job.get("company")
        period = job.get("period")
        highlights = job.get("highlights") or []

        header_lines: List[str] = []

        if title:
            header_lines.append(f"**{title}**")

        if company or period:
            cp = ", ".join(p for p in (company, period) if p)
            header_lines.append(f"*{cp}*")

        block_lines: List[str] = []
        if header_lines:
            block_lines.extend(header_lines)

        # Bullet points for achievements / responsibilities
        block_lines.extend(f"- {h}" for h in highlights)

        if block_lines:
            blocks.append("\n".join(block_lines))

    return "\n\n".join(blocks)


def _format_education_list(value: Any) -> str:
    """
    Format education entries as simple list items.

    Expects a list of dicts with keys:
    - degree
    - institution
    - location
    """
    if not isinstance(value, list):
        return str(value)

    lines: List[str] = []
    for edu in value:
        if not isinstance(edu, dict):
            continue
        parts = [
            edu.get("degree"),
            edu.get("institution"),
            edu.get("location"),
        ]
        text = ", ".join(p for p in parts if p)
        if text:
            lines.append(f"- {text}")
    return "\n".join(lines)


def _format_named_items_list(value: Any) -> str:
    """
    Format certifications / awards.

    Expects a list of dicts with:
    - name: str
    - organization: str (optional)

    Falls back to stringification for non-dict items.
    """
    if not isinstance(value, list):
        return str(value)

    lines: List[str] = []
    for item in value:
        if isinstance(item, dict):
            name = item.get("name")
            org = item.get("organization")
            if name and org:
                lines.append(f"- {name}, {org}")
            elif name:
                lines.append(f"- {name}")
        else:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _format_interests_list(value: Any) -> str:
    """Format interests as a simple bullet list."""
    if not isinstance(value, list):
        return str(value)
    return "\n".join(f"- {i}" for i in value)


def format_section_value(section_key: str, value: Any) -> str:
    """
    Convert structured dict/list data into Markdown-like text for a given section.

    This function acts as a dispatcher to section-specific formatters while
    preserving a simple fallback for plain strings and unknown structures.
    """
    # If it's already a string, return as-is
    if isinstance(value, str):
        return value

    # ---------------------------
    # Existing section handlers
    # ---------------------------
    if section_key == "skills":
        return _format_skills_list(value)

    if section_key == "experience":
        return _format_experience_list(value)

    if section_key == "education":
        return _format_education_list(value)

    if section_key in ("certifications", "awards"):
        return _format_named_items_list(value)

    if section_key == "interests":
        return _format_interests_list(value)

    # ---------------------------
    # NEW: publications
    # ---------------------------
    if section_key == "publications":
        if not isinstance(value, list):
            return str(value)

        lines: List[str] = []
        for pub in value:
            if isinstance(pub, dict):
                # Support both our schema ("title", "venue", "year")
                # and more generic keys ("name", "journal").
                title = pub.get("title") or pub.get("name")
                venue = pub.get("venue") or pub.get("journal")
                year = pub.get("year")

                pieces = [p for p in (title, venue) if p]
                main = ", ".join(pieces) if pieces else None

                if main and year:
                    lines.append(f"- {main} ({year})")
                elif main:
                    lines.append(f"- {main}")
                elif year is not None:
                    lines.append(f"- {year}")
            else:
                lines.append(f"- {pub}")
        return "\n".join(lines)

    # ---------------------------
    # NEW: training
    # ---------------------------
    if section_key == "training":
        if not isinstance(value, list):
            return str(value)

        lines: List[str] = []
        for t in value:
            if isinstance(t, dict):
                title = t.get("title")
                provider = t.get("provider")
                date = t.get("date")

                # date may be a date object or string
                if hasattr(date, "strftime"):
                    date_str = date.strftime("%Y-%m")
                else:
                    date_str = date

                pieces = [p for p in (title, provider, date_str) if p]
                if pieces:
                    lines.append(f"- {', '.join(pieces)}")
            else:
                lines.append(f"- {t}")
        return "\n".join(lines)

    # ---------------------------
    # NEW: references
    # ---------------------------
    if section_key == "references":
        if not isinstance(value, list):
            return str(value)

        lines: List[str] = []
        for r in value:
            if isinstance(r, dict):
                name = r.get("name")
                title = r.get("title")
                company = r.get("company")
                email = r.get("email")
                phone = r.get("phone")

                header_parts = [p for p in (name, title, company) if p]
                header = " — ".join(header_parts) if header_parts else None

                contact_parts = [p for p in (email, phone) if p]
                contact = " | ".join(contact_parts) if contact_parts else None

                if header and contact:
                    lines.append(f"- {header} ({contact})")
                elif header:
                    lines.append(f"- {header}")
                elif contact:
                    lines.append(f"- {contact}")
            else:
                lines.append(f"- {r}")
        return "\n".join(lines)

    # ---------------------------
    # NEW: additional_info
    # ---------------------------
    if section_key == "additional_info":
        if not isinstance(value, list):
            return str(value)

        lines: List[str] = []
        for item in value:
            if isinstance(item, dict):
                label = item.get("label")
                val = item.get("value")
                if label and val:
                    lines.append(f"- **{label}:** {val}")
                elif label:
                    lines.append(f"- **{label}:**")
                elif val:
                    lines.append(f"- {val}")
            else:
                lines.append(f"- {item}")
        return "\n".join(lines)

    # Fallback: stringify any other structure
    return str(value)


def build_section_content(section_key: str, value: Any) -> SectionContent:
    """
    Convert a raw section value (from profile_info or user_input) into SectionContent.

    - Applies section-specific formatting (e.g., bullet lists, markdown blocks).
    - Only populates the `text` field; other SectionContent fields are left to
      the full LLM pipeline.
    """
    text = format_section_value(section_key, value)
    return SectionContent.model_construct(text=text)


# ---------------------------------------------------------------------------
# NO-LLM baseline renderer
# ---------------------------------------------------------------------------

def _build_metadata(profile_info: Dict[str, Any], template_info: Dict[str, Any]) -> Any:
    """
    Build a minimal metadata object for the baseline response.

    This uses a lightweight dynamic object instead of the full metadata schema,
    because the baseline is primarily used for tests and visual comparison.
    """
    meta_cls = type("Meta", (), {})
    meta = meta_cls()
    meta.generated_at = datetime.now()
    meta.profile_info = profile_info
    meta.template_info = template_info
    return meta


def _build_structured_skills(profile_info: Dict[str, Any]) -> List[OutputSkillItem]:
    """
    Build a list of OutputSkillItem from profile_info["skills"].

    Baseline behavior:
    - Only uses the skills list from profile_info.
    - Does not perform cross-section inference.
    """
    raw_skills = profile_info.get("skills") or []
    skills_structured: List[OutputSkillItem] = []

    for item in raw_skills:
        if isinstance(item, dict) and item.get("name"):
            skills_structured.append(
                OutputSkillItem(
                    name=item["name"],
                    level=item.get("level"),
                    source="profile",
                )
            )
    return skills_structured


def build_baseline_response(mock_payload: Dict[str, Any]) -> CVGenerationResponse:
    """
    Deterministically build a CVGenerationResponse WITHOUT any LLM calls.

    Expected mock_payload structure (typically from YAML fixtures):
    - template_info: dict
      - template_id: str
      - sections_order | sections: list[str] (section IDs in display order)
    - profile_info: dict
      - Contains keys described in PROFILE_SECTION_MAP (e.g., "summary", "skills").
    - user_input_cv_text_by_section: dict (optional)
      - If present and non-empty for a section, it overrides profile_info for that section.

    Behavior:
    - Uses template-defined section order only.
    - For each section ID:
      - Use user-provided text if present and non-empty.
      - Otherwise, fall back to profile_info via PROFILE_SECTION_MAP.
      - Skip sections where no non-empty value is available.
    - Builds minimal metadata and structured skills.
    """
    template_info = mock_payload.get("template_info", {}) or {}
    user_sections = mock_payload.get("user_input_cv_text_by_section", {}) or {}
    profile_info = mock_payload.get("profile_info", {}) or {}

    template_id = template_info.get("template_id", "T_EMPLOYER_STD_V3")

    # Use template-defined section order only; do not infer new sections here.
    ordered_section_ids = (
        template_info.get("sections_order")
        or template_info.get("sections")
        or []
    )

    sections: Dict[str, SectionContent] = {}

    for sid in ordered_section_ids:
        # 1) Prefer user-provided section content if present and non-empty.
        if sid in user_sections and _non_empty(user_sections[sid]):
            raw_value = user_sections[sid]
        else:
            # 2) Fall back to profile_info using PROFILE_SECTION_MAP.
            profile_key = PROFILE_SECTION_MAP.get(sid)
            raw_value = profile_info.get(profile_key) if profile_key else None
            if not _non_empty(raw_value):
                # Nothing to render for this section.
                continue

        sections[sid] = build_section_content(sid, raw_value)

    metadata = _build_metadata(profile_info=profile_info, template_info=template_info)
    skills_structured = _build_structured_skills(profile_info)

    return CVGenerationResponse.model_construct(
        template_id=template_id,
        sections=sections,
        metadata=metadata,
        job_id="JOB_MOCK_001",         # Fixed ID for deterministic baseline
        skills=skills_structured or None,
    )
