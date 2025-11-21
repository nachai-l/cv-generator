# local_cv_templates/cv_templates.py
"""Local-only CV templates and rendering utilities for development/testing.

These templates are *not* used in production. In actual usage, the API request
will contain the template info (see `schemas.cv_template_schema.TemplateSchema`).

This module is handy for:
- local development
- unit/integration tests_utils
- manual HTML/Markdown preview of generated CVs
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Tuple, Mapping, Sequence

import re
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape

from schemas.output_schema import CVGenerationResponse, SectionContent
from schemas.cv_template_schema import TemplateSchema


# ---------------------------------------------------------------------------
# Internal template config model (dev-only)
# ---------------------------------------------------------------------------


@dataclass
class TemplateConfig:
    """Internal configuration for a CV template (dev-only).

    This mirrors the API-facing TemplateSchema closely, but is a simple dataclass
    used only for local YAML-backed templates.
    """

    template_id: str
    name: str
    style: Literal["modern", "classic", "minimalist", "creative"]
    sections_order: list[str]
    color_scheme: dict[str, str]
    font_family: str
    max_pages: int
    max_chars_per_section: dict[str, int]


# ---------------------------------------------------------------------------
# YAML loading (local_cv_templates/cv_templates.yaml)
# ---------------------------------------------------------------------------


def _load_templates_from_yaml() -> dict[str, TemplateConfig]:
    """Load local dev templates from cv_templates.yaml."""
    yaml_path = Path(__file__).with_name("cv_templates.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"Template YAML not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    templates: dict[str, TemplateConfig] = {}

    for template_id, cfg in raw.items():
        if not isinstance(cfg, dict):
            raise ValueError(
                f"Template '{template_id}' must be a mapping, got {type(cfg)}"
            )

        templates[template_id] = TemplateConfig(
            template_id=template_id,
            name=cfg["name"],
            style=cfg["style"],
            sections_order=list(cfg["sections_order"]),
            color_scheme=dict(cfg["color_scheme"]),
            font_family=cfg["font_family"],
            max_pages=int(cfg["max_pages"]),
            max_chars_per_section=dict(cfg.get("max_chars_per_section", {})),
        )

    return templates


# Local dev registry
TEMPLATES: dict[str, TemplateConfig] = _load_templates_from_yaml()

# Convenience constants for tests_utils/dev
EMPLOYER_STANDARD_V3: TemplateConfig = TEMPLATES["T_EMPLOYER_STD_V3"]
MINIMALIST_V1: TemplateConfig = TEMPLATES["T_MINIMALIST_V1"]
CREATIVE_V1: TemplateConfig = TEMPLATES["T_CREATIVE_V1"]


# ---------------------------------------------------------------------------
# Jinja2 environment (for HTML templates)
# ---------------------------------------------------------------------------

_TEMPLATES_DIR = Path(__file__).with_name("templates")

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(_TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml", "jinja2"]),
)

# ---- Skill level label helper & filter ------------------------------------

_LEVEL_LABELS: dict[str, str] = {
    "L1_Basic": "Basic",
    "L2_Intermediate": "Intermediate",
    "L3_Advanced": "Advanced",
    "L4_Expert": "Expert",
}

def _parse_bullet_lines(content: str) -> list[str]:
    """
    Extract a clean list of text items from a multiline or bullet-style section.

    Handles lines beginning with '-', '•', or '*', ignores empty lines, and strips whitespace.
    """
    if not content:
        return []
    items: list[str] = []
    for ln in content.replace("\r\n", "\n").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(("-", "•", "*")):
            ln = ln.lstrip("-•*").strip()
        items.append(ln)
    return items


def skill_level_label(level: str | None) -> str:
    """Map internal level codes like 'L4_Expert' to nice display labels."""
    if not level:
        return ""
    return _LEVEL_LABELS.get(level, level)


_TEMPLATE_ENV.filters["skill_level_label"] = skill_level_label


def _get_template_filename(style: str) -> str:
    # If caller passes a raw filename, just use it
    if "." in style:
        return style

    mapping = {
        "modern": "modern.jinja2",
        "minimalist": "minimalist.jinja2",
        "double": "double.jinja2",
        "classic": "modern.jinja2",
        "creative": "modern.jinja2",
    }
    return mapping.get(style, "modern.jinja2")


# ---------------------------------------------------------------------------
# Access helpers (dev-only)
# ---------------------------------------------------------------------------


def get_template(template_id: str) -> TemplateConfig:
    """Get local dev template by ID, falling back to EMPLOYER_STANDARD_V3."""
    return TEMPLATES.get(template_id, EMPLOYER_STANDARD_V3)


def list_templates_for_api() -> list[TemplateSchema]:
    """Return all local dev templates as API-facing TemplateSchema models.

    Useful only for dev/testing; production should not rely on these templates.
    """
    return [
        TemplateSchema(
            template_id=t.template_id,
            name=t.name,
            style=t.style,
            sections_order=t.sections_order,
            color_scheme=t.color_scheme,
            font_family=t.font_family,
            max_pages=t.max_pages,
            max_chars_per_section=t.max_chars_per_section,
        )
        for t in TEMPLATES.values()
    ]


# ---------------------------------------------------------------------------
# Rendering helpers (dev-only HTML/Markdown preview)
# ---------------------------------------------------------------------------


def _extract_personal_info(response: CVGenerationResponse) -> Tuple[str, str, str, str]:
    """
    Extract personal info (name, email, phone, linkedin) from response / metadata.

    Order of precedence:
    1) response.profile_info
    2) metadata.profile_info
    3) metadata top-level fields
    4) fallbacks (dummy dev values)
    """
    meta: Any = getattr(response, "metadata", None)

    # 1) Try top-level profile_info on the response
    profile: Any = getattr(response, "profile_info", None)

    # 2) If absent, fall back to metadata.profile_info
    if profile is None and meta is not None:
        if hasattr(meta, "profile_info"):
            profile = getattr(meta, "profile_info")
        elif isinstance(meta, dict) and "profile_info" in meta:
            profile = meta.get("profile_info")

    def get_field(field: str, default: str) -> str:
        # First: profile_info (attr or dict)
        if profile is not None:
            if hasattr(profile, field):
                value = getattr(profile, field)
                if value:
                    return str(value)
            if isinstance(profile, dict) and profile.get(field):
                return str(profile[field])

        # Second: metadata top-level (attr or dict)
        if meta is not None:
            if hasattr(meta, field):
                value = getattr(meta, field)
                if value:
                    return str(value)
            if isinstance(meta, dict) and meta.get(field):
                return str(meta[field])

        # Fallback: dummy dev values
        return default

    name = get_field("name", "Candidate Name")
    email = get_field("email", "email@example.com")
    phone = get_field("phone", "+66-XX-XXX-XXXX")
    linkedin = get_field("linkedin", "linkedin.com/in/candidate")

    return name, email, phone, linkedin


def render_cv_html(
    response: CVGenerationResponse,
    style_override: str | None = None,
) -> str:
    """
    Render CV response as HTML (dev/testing).

    Uses Jinja2 templates under local_cv_templates/templates.
    Template selection is based on:

      1. style_override (if provided)
      2. metadata.template_info.style (if provided)
      3. TemplateConfig.style (from cv_templates.yaml)
    """
    # Base template from local registry (dev fallback)
    base_template = get_template(response.template_id)

    # Try to pull template_info from metadata (API-style config)
    meta = getattr(response, "metadata", None)
    template_info: dict[str, Any] | None = None
    if meta is not None:
        if hasattr(meta, "template_info") and isinstance(meta.template_info, dict):
            template_info = meta.template_info  # type: ignore[assignment]
        elif isinstance(meta, dict) and isinstance(meta.get("template_info"), dict):
            template_info = meta["template_info"]  # type: ignore[index]

    # ---------- sections_order ----------
    if template_info is not None:
        so = template_info.get("sections_order")
        if isinstance(so, Sequence) and not isinstance(so, (str, bytes)):
            sections_order: list[str] = [str(x) for x in so]
        else:
            sections_order = list(base_template.sections_order)
    else:
        sections_order = list(base_template.sections_order)

    # ---------- color_scheme ----------
    colors: dict[str, str] = dict(base_template.color_scheme)
    if template_info is not None:
        cs = template_info.get("color_scheme")
        if isinstance(cs, Mapping):
            override = {str(k): str(v) for k, v in cs.items()}
            colors.update(override)

    # ---------- font_family ----------
    if template_info is not None and isinstance(template_info.get("font_family"), str):
        font_family: str = template_info["font_family"]  # type: ignore[index]
    else:
        font_family = base_template.font_family

    # ---------- personal info for header ----------
    personal_name, personal_email, personal_phone, personal_linkedin = _extract_personal_info(
        response
    )

    # ---------- sections payload for template ----------
    sections: list[dict[str, str]] = []
    for section_name in sections_order:
        if section_name not in response.sections:
            continue

        section: SectionContent = response.sections[section_name]
        section_title = section_name.replace("_", " ").title()
        section_html = _format_section_content(section_name, section.text)

        sections.append(
            {
                "id": section_name,
                "title": section_title,
                "html": section_html,
            }
        )

    # ---------- date / metadata ----------
    try:
        generated_at_str = response.metadata.generated_at.strftime("%B %d, %Y")  # type: ignore[union-attr]
    except Exception:
        generated_at_str = ""

    # ---------- choose template file based on style ----------
    if style_override is not None:
        style = style_override
    else:
        style = base_template.style
        if template_info is not None and isinstance(template_info.get("style"), str):
            style = str(template_info["style"])

    template_file = _get_template_filename(style)
    jinja_template = _TEMPLATE_ENV.get_template(template_file)

    # ---------- structured skills list for templates ----------
    skills_structured = getattr(response, "skills", None) or []

    context = {
        "name": personal_name,
        "email": personal_email,
        "phone": personal_phone,
        "linkedin": personal_linkedin,
        "colors": colors,
        "font_family": font_family,
        "sections": sections,
        "generated_at_str": generated_at_str,
        "job_id": response.job_id,
        # 🔹 New: structured skills (list[OutputSkillItem]) for template usage
        "skills": skills_structured,
    }

    return jinja_template.render(**context)


def _strip_md(s: str) -> str:
    """Strip **bold**, *italic*, and extra trailing punctuation like '**,'."""
    s = s.strip()

    # Handle '**something**,' case
    if s.startswith("**") and s.endswith("**,"):
        return s[2:-3].strip()

    # Handle '**something**' case
    if s.startswith("**") and s.endswith("**"):
        return s[2:-2].strip()

    # Handle '*something*'
    if s.startswith("*") and s.endswith("*"):
        return s[1:-1].strip()

    return s



def _parse_bullet_lines(content: str) -> list[str]:
    """
    Extract a clean list of text items from a multiline or bullet-style section.

    Handles lines beginning with '-' or '•', ignores empty lines, and strips whitespace.
    """
    if not content:
        return []
    items: list[str] = []
    for ln in content.replace("\r\n", "\n").split("\n"):
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith(("-", "•")):
            ln = ln.lstrip("-•").strip()
        items.append(ln)
    return items


def _format_section_content(section_name: str, content: str) -> str:
    """
    Format section content based on section type.

    Input is plain text (often Markdown-like) and we turn it into nicer HTML.
    """
    if not content:
        return ""

    # Normalize line endings and split
    lines = [ln for ln in content.replace("\r\n", "\n").split("\n")]

    # ---------------- SKILLS ----------------
    # NOTE: HTML templates are encouraged to use the structured `skills` list
    # when available. This branch remains as a fallback when only text exists.
    if section_name == "skills":
        # Expect lines like: "- Molecular Biology (Expert)"
        items = []
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith(("-", "•")):
                ln = ln.lstrip("-•").strip()
            items.append(ln)

        if not items:
            return ""

        pills = "".join(
            f'<li class="inline-pill">{item}</li>'
            for item in items
        )
        return f'<ul class="inline-list">{pills}</ul>'

    # ---------------- EXPERIENCE ----------------
    if section_name == "experience":
        # State machine parser over lines:
        #   **Job Title**
        #   *Company, 2020–2024*
        #   - Bullet 1
        #   - Bullet 2
        #
        # Also supports older shape:
        #   *Company, 2020–2022*
        #   - Bullet 1
        #   - Bullet 2
        raw_lines = [
            ln.strip()
            for ln in content.replace("\r\n", "\n").split("\n")
        ]

        jobs: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        def flush_current() -> None:
            nonlocal current
            if current and (current["title"] or current["company"] or current["bullets"]):
                jobs.append(current)
            current = None

        for ln in raw_lines:
            if not ln:
                continue

            # New job title
            if ln.startswith("**") and ln.endswith("**"):
                flush_current()
                current = {
                    "title": _strip_md(ln),
                    "company": "",
                    "bullets": [],
                }
                continue

            # Company + dates (italic)
            if ln.startswith("*") and ln.endswith("*"):
                text = _strip_md(ln)
                if current is None:
                    # Company-only block (no explicit title)
                    current = {
                        "title": "",
                        "company": text,
                        "bullets": [],
                    }
                elif not current["company"]:
                    current["company"] = text
                else:
                    current["bullets"].append(text)
                continue

            # Bullet / responsibility line
            if ln.startswith(("-", "•")):
                bullet = ln.lstrip("-•").strip()
                if current is None:
                    current = {
                        "title": "",
                        "company": "",
                        "bullets": [],
                    }
                current["bullets"].append(bullet)
                continue

            # Fallback: plain text → treat as extra bullet or title-only job
            if current is None:
                current = {
                    "title": ln,
                    "company": "",
                    "bullets": [],
                }
            else:
                current["bullets"].append(ln)

        flush_current()

        html_blocks: list[str] = []
        for job in jobs:
            title_html = ""
            meta_html = ""

            if job["title"]:
                title_html = f'<div class="item-title">{job["title"]}</div>'
                if job["company"]:
                    meta_html = f'<div class="item-meta">{job["company"]}</div>'
            elif job["company"]:
                # No explicit title → show company+years as title line
                title_html = f'<div class="item-title">{job["company"]}</div>'

            bullets_html = ""
            if job["bullets"]:
                bullets_html = (
                    '<ul class="item-bullets">'
                    + "".join(f"<li>{b}</li>" for b in job["bullets"])
                    + "</ul>"
                )

            html_blocks.append(
                f"""
                <div class="experience-item">
                    <div class="item-header">
                        {title_html}
                        {meta_html}
                    </div>
                    {bullets_html}
                </div>
                """
            )

        return "".join(html_blocks)


    # ---------------- EDUCATION ----------------
    if section_name == "education":
        items_html: list[str] = []

        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue

            # Remove bullet markers if any
            if ln.startswith(("-", "•", "*")):
                ln = ln.lstrip("-•*").strip()

            # Extract **degree**
            degree = ""
            rest = ln

            if ln.startswith("**"):
                end = ln.find("**", 2)
                if end != -1:
                    degree = _strip_md(ln[:end+2])
                    rest = ln[end+2:].lstrip(" ,")
                else:
                    rest = ln  # fallback

            # Parse rest: Field | Institution | Years | GPA
            parts = [p.strip() for p in rest.split("|")]

            field        = parts[0] if len(parts) > 0 else ""
            institution  = parts[1] if len(parts) > 1 else ""
            year_range   = parts[2] if len(parts) > 2 else ""
            gpa_raw      = parts[3] if len(parts) > 3 else ""

            gpa = gpa_raw.split(":", 1)[-1].strip() if gpa_raw.lower().startswith("gpa") else gpa_raw

            # Build meta line
            meta_parts = []
            if field:       meta_parts.append(field)
            if institution: meta_parts.append(institution)
            if year_range:  meta_parts.append(year_range)
            if gpa:         meta_parts.append(f"GPA: {gpa}")

            meta = " | ".join(meta_parts)

            # Fallback if degree failed
            if not degree:
                degree = field or ln

            items_html.append(f"""
                <div class="education-item">
                    <div class="degree"><strong>{degree}</strong></div>
                    <div class="institution">{meta}</div>
                </div>
            """)

        return "".join(items_html)


    # ---------------- CERTIFICATIONS ----------------
    if section_name == "certifications":
        # Treat like a vertical list
        items = _parse_bullet_lines(content)
        if not items:
            return ""
        return (
            '<ul class="stacked-list">'
            + "".join(f"<li>{item}</li>" for item in items)
            + "</ul>"
        )

    # ---------------- AWARDS ----------------
    if section_name == "awards":
        items = _parse_bullet_lines(content)
        if not items:
            return ""
        return (
            '<ul class="stacked-list">'
            + "".join(f"<li>{item}</li>" for item in items)
            + "</ul>"
        )

    # ---------------- PUBLICATIONS / TRAINING / REFERENCES / ADDITIONAL INFO ----------------
    if section_name in {"publications", "training", "references", "additional_info"}:
        # Treat as simple stacked bullet list.
        # Any heading lines like "### ..." will be ignored because they
        # don't start with '-' or '•'.
        items = _parse_bullet_lines(content)
        if not items:
            return ""
        return (
            '<ul class="stacked-list">'
            + "".join(f"<li>{item}</li>" for item in items)
            + "</ul>"
        )

    # ---------------- INTERESTS ----------------
    if section_name == "interests":
        # Build inline pills, similar to skills
        items = _parse_bullet_lines(content)

        if not items:
            return ""

        pills = "".join(
            f'<li class="inline-pill">{item}</li>'
            for item in items
        )
        return f'<ul class="inline-list">{pills}</ul>'

    # ---------------- DEFAULT (any other section) ----------------
    # Just split double newlines into paragraphs
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if paragraphs:
        return "".join(f"<p>{p}</p>" for p in paragraphs)

    return f"<p>{content}</p>"


def render_cv_markdown(response: CVGenerationResponse) -> str:
    """
    Render CV response as Markdown (dev/testing).
    """
    template = get_template(response.template_id)

    personal_name, personal_email, personal_phone, personal_linkedin = _extract_personal_info(
        response
    )

    md = f"""# {personal_name}

**Email:** {personal_email} | **Phone:** {personal_phone} | **LinkedIn:** {personal_linkedin}

---

"""

    structured_skills = getattr(response, "skills", None) or []

    for section_name in template.sections_order:
        if section_name not in response.sections:
            continue

        section: SectionContent = response.sections[section_name]
        section_title = section_name.replace("_", " ").title()

        md += f"## {section_title}\n\n"

        # Prefer structured skills list if available
        if section_name == "skills" and structured_skills:
            for skill in structured_skills:
                label = skill_level_label(getattr(skill, "level", None))
                if label:
                    md += f"- {skill.name} ({label})\n"
                else:
                    md += f"- {skill.name}\n"
            md += "\n"
        else:
            md += f"{section.text}\n\n"

    try:
        generated_str = response.metadata.generated_at.strftime("%B %d, %Y")  # type: ignore[union-attr]
    except Exception:
        generated_str = ""

    md += (
        f"---\n\n"
        f"*Generated: {generated_str}*  \n"
        f"*Job ID: {response.job_id}*\n"
    )

    return md


def save_cv_html(
    response: CVGenerationResponse,
    output_path: str,
    style: str | None = None,
) -> None:
    """Save dev HTML CV to a file."""
    html_content = render_cv_html(response, style_override=style)
    Path(output_path).write_text(html_content, encoding="utf-8")


def save_cv_markdown(response: CVGenerationResponse, output_path: str) -> None:
    """Save dev Markdown CV to a file."""
    md_content = render_cv_markdown(response)
    Path(output_path).write_text(md_content, encoding="utf-8")
