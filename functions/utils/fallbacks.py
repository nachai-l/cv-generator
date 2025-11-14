# functions/utils/fallbacks.py

"""
Fallback helpers for CV section generation.

All fallback outputs are prefixed with the configurable prefix from:
    generation.fallback_text
"""

from __future__ import annotations

from typing import Any, Optional, Dict, List

import structlog
from pydantic import BaseModel

from schemas.input_schema import CVGenerationRequest
from functions.utils.llm_client import load_parameters

logger = structlog.get_logger(__name__).bind(module="fallbacks")


def _get_fallback_prefix() -> str:
    """
    Read the fallback text prefix from parameters.yaml:

        generation:
          fallback_text: "[Fallback]"

    Defaults to "[Fallback]" if missing.
    """
    try:
        params = load_parameters() or {}
    except Exception as exc:  # pragma: no cover
        logger.warning("fallback_prefix_load_failed", error=str(exc))
        return "[Fallback]"

    gen_cfg = params.get("generation", {}) or {}
    return str(gen_cfg.get("fallback_text", "[Fallback]"))


# ---------------------------------------------------------------------------
# Section-specific helpers
# ---------------------------------------------------------------------------

def _build_education_fallback_from_profile(
        request: CVGenerationRequest,
) -> Optional[str]:
    """
    Deterministic education fallback built from profile data.
    Returns bullet-list text OR None if no usable data.
    """
    raw_profile_info = getattr(request, "profile_info", None)
    student_profile = getattr(request, "student_profile", None)

    edu_items: List[Dict[str, Any]] = []

    # Legacy API
    if isinstance(raw_profile_info, dict):
        raw_edu = raw_profile_info.get("education") or []
        if isinstance(raw_edu, list):
            edu_items.extend(e for e in raw_edu if isinstance(e, dict))

    # New API (Pydantic)
    if student_profile is not None:
        sp_edu_list = getattr(student_profile, "education", []) or []
        for edu in sp_edu_list:
            if isinstance(edu, BaseModel):
                edu_items.append(edu.model_dump(mode="python"))
            elif isinstance(edu, dict):
                edu_items.append(edu)

    if not edu_items:
        return None

    lines: List[str] = []
    for e in edu_items:
        degree = (e.get("degree") or e.get("title") or "").strip()
        institution = (e.get("institution") or e.get("school") or "").strip()
        location = (e.get("location") or e.get("country") or "").strip()
        years = (e.get("years") or e.get("period") or "").strip()

        parts = [p for p in (degree, institution, location) if p]

        if not parts:
            continue

        line = "- " + ", ".join(parts)
        if years:
            line += f" ({years})"

        lines.append(line)

    if not lines:
        return None

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API — the ONLY function Stage B calls
# ---------------------------------------------------------------------------

def build_section_fallback_text(
    request: CVGenerationRequest,
    section_id: str,
    *,
    reason: str | None = None,
) -> str:
    """
    Returns a clean fallback text for any failed section.

    Priority:
    1) If user provided a draft for this section (and it's long enough), use that.
    2) For 'education', try to build a structured fallback from profile data.
    3) Otherwise, use a generic fallback message.

    ALL outputs are prefixed with generation.fallback_text (e.g. "[Fallback]").
    """
    prefix = _get_fallback_prefix()

    # ---------------------------------------------------
    # 1) USER DRAFT FALLBACK (highest priority)
    # ---------------------------------------------------
    drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}
    raw_draft = drafts.get(section_id)

    if raw_draft is not None:
        if isinstance(raw_draft, str):
            draft_text = raw_draft.strip()
        else:
            draft_text = str(raw_draft).strip()

        # "Long enough" – reusing the same minimum length heuristic (10 chars)
        if len(draft_text) >= 10:
            logger.error(
                "section_fallback_used",
                section_id=section_id,
                fallback_type="user_draft",
                reason=reason or "LLM output empty/too short; using user draft",
                preview=draft_text[:100],
            )
            return f"{prefix} {draft_text}"

    # ---------------------------------------------------
    # 2) SECTION-SPECIFIC: EDUCATION FROM PROFILE
    # ---------------------------------------------------
    if section_id == "education":
        edu_text = _build_education_fallback_from_profile(request)
        if edu_text:
            logger.error(
                "section_fallback_used",
                section_id=section_id,
                fallback_type="education_from_profile",
                reason=reason or "LLM output empty/too short; using profile education",
                preview=edu_text[:100],
            )
            return f"{prefix} {edu_text}"

        # Fall through to generic if no education data

    # ---------------------------------------------------
    # 3) GENERIC FALLBACK FOR ALL SECTIONS
    # ---------------------------------------------------
    generic_msg = "(Not enough information to generate this section.)"

    logger.error(
        "section_fallback_used",
        section_id=section_id,
        fallback_type="generic",
        reason=reason or "LLM returned empty/failed output; no better fallback",
    )

    return f"{prefix} {generic_msg}"
