# functions/utils/experience_functions.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional

from pydantic import BaseModel
import datetime
import re
import json

from schemas.input_schema import CVGenerationRequest


@dataclass
class ExperienceItem:
    """Normalized representation of a single experience entry."""

    title: Optional[str]
    company: Optional[str]
    start_year: Optional[str]
    end_year: Optional[str]

    # Normalized bullets we will render
    bullets: List[str]

    # Backwards-compat: Stage B still expects `.responsibilities`
    # for bullet generation / normalization.
    responsibilities: List[str] | None = None

    # Simple provenance tag (optional)
    source: Optional[str] = "profile"


class ExperienceAugmentationResponse(BaseModel):
    """Pydantic model for LLM-augmented experience items."""
    new_items: list[ExperienceItem] = []


def normalize_experience_bullets(
    responsibilities: list[str],
    max_bullets: int = 6,
) -> list[str]:
    """
    Normalize a list of responsibility lines into clean '- ' bullets.

    - Strips leading bullet marks (-, •, *).
    - Drops empty / whitespace-only entries.
    - Caps at `max_bullets`.
    """
    cleaned: list[str] = []
    for raw in responsibilities or []:
        if not raw:
            continue
        text = str(raw).strip()
        if not text:
            continue

        # Strip any leading bullet marks / symbols
        text = text.lstrip("-•* ").strip()
        if not text:
            continue

        cleaned.append(f"- {text}")
        if len(cleaned) >= max_bullets:
            break

    return cleaned

def merge_llm_experience_augmentation(
    base_items: list[ExperienceItem],
    raw_json: str,
) -> list[ExperienceItem]:
    """
    Parse LLM augmentation JSON and merge into base_items.

    - Expects JSON: { "new_items": [ {title, company, start_date, end_date, ...}, ... ] }
    - Drops entries missing both title and company.
    - Deduplicates by (title, company, start_year, end_year).
    - Marks new items as source='llm' (if you decide to add that field later).
    """
    if not raw_json:
        return base_items

    try:
        cleaned = raw_json.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        data = json.loads(cleaned)
        resp = ExperienceAugmentationResponse.model_validate(data)
    except Exception:
        # On any parsing error, just return the originals
        return base_items

    if not resp.new_items:
        return base_items

    def _key(it: ExperienceItem) -> tuple[str, str, str | None, str | None]:
        title = (it.title or "").strip().lower()
        company = (it.company or "").strip().lower()
        # For now we only have start_year/end_year in ExperienceItem
        start_year = getattr(it, "start_year", None)
        end_year = getattr(it, "end_year", None)
        return (title, company, start_year, end_year)

    existing_keys = {_key(it) for it in base_items}

    for new in resp.new_items:
        if not (new.title or new.company):
            continue

        # Normalize bullets
        if not new.bullets:
            if getattr(new, "responsibilities", None):
                new.bullets = normalize_experience_bullets(new.responsibilities)
            else:
                new.bullets = []

        # Mark provenance
        if new.source is None:
            new.source = "llm"

        k = _key(new)
        if k in existing_keys:
            continue

        base_items.append(new)
        existing_keys.add(k)

    return base_items

def parse_experience_bullets_response(
    raw_text: str,
    existing_responsibilities: list[str] | None = None,
    max_bullets: int = 6,
) -> list[str]:
    """
    Turn raw LLM text into a cleaned list of '- ' bullets.

    - Prefers LLM bullets if any valid lines are present.
    - Falls back to existing_responsibilities via normalize_experience_bullets.
    - Returns [] if absolutely nothing usable.
    """
    cleaned = (raw_text or "").strip()
    bullets: list[str] = []

    if cleaned:
        lines = cleaned.splitlines()
        for ln in lines:
            core = ln.strip()
            if not core:
                continue

            # Strip existing bullet symbols
            if core.startswith("-") or core.startswith("•") or core.startswith("*"):
                core = core.lstrip("-•* ").strip()

            if not core:
                continue

            bullets.append(f"- {core}")
            if len(bullets) >= max_bullets:
                break

    if bullets:
        return bullets

    # fallback to existing responsibilities, if given
    if existing_responsibilities:
        return normalize_experience_bullets(
            existing_responsibilities, max_bullets=max_bullets
        )

    return []


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _get_attr_or_key(obj: Any, name: str) -> Any:
    """Safely read attribute or dict key."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _first_non_empty(*values: Any) -> Any:
    """Return the first truthy / non-empty value."""
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def extract_year_from_date(value: Any) -> Optional[str]:
    """
    Best-effort extraction of a 4-digit year from various date-like inputs.

    Supports:
    - datetime.date / datetime.datetime
    - strings containing YYYY or YYYY-MM-DD
    - anything else is converted to str() and scanned for a year pattern.
    """
    if not value:
        return None

    # datetime/date
    if isinstance(value, (datetime.date, datetime.datetime)):
        return str(value.year)

    s = str(value).strip()
    if not s:
        return None

    # Simple YYYY or YYYY-MM-DD
    m = re.search(r"(19|20)\d{2}", s)
    if m:
        return m.group(0)

    return None


# ---------------------------------------------------------------------------
# Experience header + rendering
# ---------------------------------------------------------------------------

def render_experience_header(entry: Any) -> str:
    """
    Render a deterministic two-line header for an experience entry.

    Rules:
    - Prefer `position` over `title` over `job_title` as the role name.
    - Company is taken from `company` / `organization` / `employer`.
    - Years are derived from start/end date fields when available.
    - If no role/title exists, only the italic company line is returned.
    - If no dates are available, we don't fabricate any years.

    Output examples:
        "**Assistant Manager, Research Division**\n"
        "*Mitsui Chemicals Singapore R&D Centre, 2017–2023*"

        "*Mojia Biotech Pte. Ltd., 2023–Present*"
    """
    title = _first_non_empty(
        _get_attr_or_key(entry, "position"),
        _get_attr_or_key(entry, "title"),
        _get_attr_or_key(entry, "job_title"),
        _get_attr_or_key(entry, "role"),
    )
    company = _first_non_empty(
        _get_attr_or_key(entry, "company"),
        _get_attr_or_key(entry, "organization"),
        _get_attr_or_key(entry, "employer"),
    )

    start_raw = _first_non_empty(
        _get_attr_or_key(entry, "year"),          # some schemas
        _get_attr_or_key(entry, "start_year"),
        _get_attr_or_key(entry, "startDate"),
        _get_attr_or_key(entry, "start_date"),
    )
    end_raw = _first_non_empty(
        _get_attr_or_key(entry, "end_year"),
        _get_attr_or_key(entry, "endDate"),
        _get_attr_or_key(entry, "end_date"),
    )

    start_year = extract_year_from_date(start_raw)
    end_year = extract_year_from_date(end_raw)

    years_part: Optional[str] = None
    if start_year and end_year:
        years_part = f"{start_year}–{end_year}"
    elif start_year and not end_year:
        years_part = f"{start_year}–Present"
    elif not start_year and end_year:
        years_part = end_year

    lines: List[str] = []

    if title:
        lines.append(f"**{title}**")

    if company:
        if years_part:
            lines.append(f"*{company}, {years_part}*")
        else:
            lines.append(f"*{company}*")

    if not lines and company:
        if years_part:
            return f"*{company}, {years_part}*"
        return f"*{company}*"

    return "\n".join(lines)

def render_experience_section_from_structured(
    items: Iterable[ExperienceItem],
) -> str:
    """
    Render a full 'experience' section from normalized items.

    For ExperienceItem, we now prefer a single header line that includes:
        "Junior Data Scientist, True Digital Group, 2020–2022"
    followed by bullets.

    Items are separated by a blank line.
    """
    blocks: List[str] = []

    for item in items:
        lines: List[str] = []

        if isinstance(item, ExperienceItem):
            # --- Build years part ---
            years_part: Optional[str] = None
            if item.start_year and item.end_year:
                years_part = f"{item.start_year}–{item.end_year}"
            elif item.start_year and not item.end_year:
                years_part = f"{item.start_year}–Present"
            elif not item.start_year and item.end_year:
                years_part = item.end_year

            # --- Build a single header line: title, company, years ---
            header_parts: List[str] = []

            if item.title:
                header_parts.append(item.title.strip())

            if item.company:
                header_parts.append(item.company.strip())

            if years_part:
                header_parts.append(years_part)

            if header_parts:
                # Example: "Junior Data Scientist, True Digital Group, 2020–2022"
                lines.append(", ".join(header_parts))
        else:
            # Legacy / fallback path (non-ExperienceItem shapes)
            header = render_experience_header(item)
            if header:
                lines.append(header)

        # --- Bullets ---
        for b in item.bullets:
            if not b:
                continue
            text = str(b).strip()
            if not text:
                continue
            if text.startswith("-"):
                lines.append(text)
            else:
                lines.append(f"- {text}")

        if lines:
            blocks.append("\n".join(lines))

    return "\n\n".join(blocks).strip()



def _normalize_bullets(raw: Any) -> List[str]:
    """Normalize various bullet/description fields into a clean list of strings."""
    if raw is None:
        return []

    if isinstance(raw, str):
        # treat as multi-line description
        return [ln.strip() for ln in raw.splitlines() if ln.strip()]

    if isinstance(raw, (list, tuple)):
        out: List[str] = []
        for item in raw:
            if item is None:
                continue
            txt = str(item).strip()
            if txt:
                out.append(txt)
        return out

    # Fallback: best-effort str()
    s = str(raw).strip()
    return [s] if s else []


def build_structured_experience(request: CVGenerationRequest | Any) -> List[ExperienceItem]:
    """
    Construct a list of ExperienceItem from the request.

    Sources (in order of preference):
    - request.student_profile.experience
    - request.profile_info["experience"] (legacy dict shape)

    We don't touch LLM or templates here; this is a pure normalization layer.
    """
    items: List[ExperienceItem] = []

    # NEW API: student_profile.experience (Pydantic models)
    student_profile = getattr(request, "student_profile", None)
    if student_profile is not None:
        sp_exps = getattr(student_profile, "experience", None) or []
        for entry in sp_exps:
            title = _first_non_empty(
                _get_attr_or_key(entry, "position"),
                _get_attr_or_key(entry, "title"),
                _get_attr_or_key(entry, "job_title"),
                _get_attr_or_key(entry, "role"),
            )
            company = _first_non_empty(
                _get_attr_or_key(entry, "company"),
                _get_attr_or_key(entry, "organization"),
                _get_attr_or_key(entry, "employer"),
            )
            start_year = extract_year_from_date(
                _first_non_empty(
                    _get_attr_or_key(entry, "year"),
                    _get_attr_or_key(entry, "start_year"),
                    _get_attr_or_key(entry, "startDate"),
                    _get_attr_or_key(entry, "start_date"),
                )
            )
            end_year = extract_year_from_date(
                _first_non_empty(
                    _get_attr_or_key(entry, "end_year"),
                    _get_attr_or_key(entry, "endDate"),
                    _get_attr_or_key(entry, "end_date"),
                )
            )
            bullets = _normalize_bullets(
                _first_non_empty(
                    _get_attr_or_key(entry, "bullets"),
                    _get_attr_or_key(entry, "highlights"),
                    _get_attr_or_key(entry, "responsibilities"),
                    _get_attr_or_key(entry, "description"),
                )
            )

            if not any([title, company, bullets]):
                continue

            items.append(
                ExperienceItem(
                    title=title,
                    company=company,
                    start_year=start_year,
                    end_year=end_year,
                    bullets=bullets,
                    responsibilities=bullets.copy(),  # keep both populated
                    source="profile",
                )
            )

    # Legacy shape: request.profile_info["experience"]
    if not items:
        profile_info = getattr(request, "profile_info", None)
        if isinstance(profile_info, dict):
            raw_exps = profile_info.get("experience") or []
            for entry in raw_exps:
                title = _first_non_empty(
                    _get_attr_or_key(entry, "position"),
                    _get_attr_or_key(entry, "title"),
                    _get_attr_or_key(entry, "job_title"),
                    _get_attr_or_key(entry, "role"),
                )
                company = _first_non_empty(
                    _get_attr_or_key(entry, "company"),
                    _get_attr_or_key(entry, "organization"),
                    _get_attr_or_key(entry, "employer"),
                )
                start_year = extract_year_from_date(
                    _first_non_empty(
                        _get_attr_or_key(entry, "year"),
                        _get_attr_or_key(entry, "start_year"),
                        _get_attr_or_key(entry, "startDate"),
                        _get_attr_or_key(entry, "start_date"),
                    )
                )
                end_year = extract_year_from_date(
                    _first_non_empty(
                        _get_attr_or_key(entry, "end_year"),
                        _get_attr_or_key(entry, "endDate"),
                        _get_attr_or_key(entry, "end_date"),
                    )
                )
                bullets = _normalize_bullets(
                    _first_non_empty(
                        _get_attr_or_key(entry, "bullets"),
                        _get_attr_or_key(entry, "highlights"),
                        _get_attr_or_key(entry, "responsibilities"),
                        _get_attr_or_key(entry, "description"),
                    )
                )

                if not any([title, company, bullets]):
                    continue

                items.append(
                    ExperienceItem(
                        title=title,
                        company=company,
                        start_year=start_year,
                        end_year=end_year,
                        bullets=bullets,
                        responsibilities=bullets.copy(),
                        source="profile",
                    )
                )

    return items


