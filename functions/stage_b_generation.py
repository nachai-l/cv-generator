# functions/stage_b_generation.py

"""
Stage B: CV generation.

This module handles:
- Building prompts per CV section from validated request + evidence plan
- Calling the LLM with retry logic
- Applying character limits
- Returning a CVGenerationResponse compliant with schemas/output_schema.py


STRUCTURED-FIRST vs LEGACY skills modes
---------------------------------------

Stage B supports two behaviours for the "skills" family of sections:

1) STRUCTURED-FIRST mode  (recommended)
   - Trigger: "skills_structured" is present in `effective_sections`.
   - Flow:
       a) Build a SkillsSectionPlan from the request profile/student_profile.
       b) Call the LLM once for "skills_structured" to generate JSON skills
          (taxonomy + inferred), with full evidence and job-context prompts.
       c) Render the human-readable "skills" section text from the structured
          skills output (bullet list), then apply per-section char limits.
   - Guarantees:
       - Canonical/taxonomy skills always keep their original levels.
       - Any canonical skill that disappears from the LLM output is re-added.
       - A final reconciliation step restores original levels for any
         canonical skill names present in the output.

2) LEGACY mode  (backwards-compatible)
   - Trigger: "skills_structured" is NOT in `effective_sections`.
   - Flow:
       a) Generate the free-text "skills" section like any other section.
       b) Optionally derive structured skills by parsing the generated bullets.
       c) If parsing fails, fall back to taxonomy-only structured skills.
   - Guarantees:
       - The same reconciliation rules apply: any skill whose name matches a
         canonical skill gets its level forced back to the canonical value.
       - Taxonomy-only fallback is used as a last resort if the LLM output is
         unusable.

Shared invariants across both modes:
- Structured skills are always the source of truth for levels.
- The free-text "skills" section is either:
    - rendered from structured skills (STRUCTURED-FIRST), or
    - used as a hint to construct structured skills (LEGACY).
- In all cases, canonical skills from the incoming request must never be
  silently downgraded or lose their original level.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Union
from pathlib import Path
from functools import lru_cache

import structlog
from dataclasses import dataclass
from structlog.contextvars import bind_contextvars, clear_contextvars
from pydantic import BaseModel

from schemas.input_schema import CVGenerationRequest
from schemas.internal_schema import (
    EvidencePlan,
    SkillsSectionPlan,
    SkillSelectionItem,
    CanonicalSkill,
)
from schemas.output_schema import (
    CVGenerationResponse,
    SectionContent,
    Metadata,
    Justification,
    GenerationStatus,
    OutputSkillItem,
)
from functions.utils.llm_client import load_parameters
from functions.utils.llm_metrics import call_llm_section_with_metrics
from functions.utils.fallbacks import build_section_fallback_text
from functions.utils.language_tone import describe_language, describe_tone
from functions.utils.skills_formatting import (
    format_plain_skill_bullets,
    parse_skills_from_bullets,
    is_combined_canonical_name,
    match_canonical_skill,
    compute_section_matched_jd_skills,
)
from functions.utils.experience_functions import (
    ExperienceItem,
    extract_year_from_date,
    render_experience_header,
    build_structured_experience,
    render_experience_section_from_structured,
    normalize_experience_bullets,
    merge_llm_experience_augmentation,
    parse_experience_bullets_response,
)
from functions.utils.claims import (
    should_require_justification,
    split_section_and_justification,
    parse_justification_json,
    validate_justification_against_text,
    build_empty_justification,
    adjust_section_token_budget,
)
from functions.utils.common import (
    resolve_token_budget,
    load_yaml_dict,
    get_pricing_for_model,
    get_thb_per_usd_from_params,
    strip_redundant_section_heading,
)
from functions.utils.prompts_builder import (
    build_section_prompt,
    build_skills_selection_prompt,
    build_experience_justification_prompt,
    load_section_prompts_config,
    _collect_evidence_facts_for_section,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level configuration (read parameters.yaml exactly once here)
# ---------------------------------------------------------------------------

PARAMS = load_parameters() or {}

GENERATION_CFG: Dict[str, Any] = PARAMS.get("generation", {}) or {}
RETRY_CFG: Dict[str, Any] = GENERATION_CFG.get("retry_thresholds", {}) or {}
SKILLS_CFG: Dict[str, Any] = GENERATION_CFG.get("skills_matching", {}) or {}
CORE_SECTIONS: list[str] = PARAMS.get("core_sections", []) or []
CROSS_SECTION_CFG: Dict[str, Any] = PARAMS.get(
    "cross_section_evidence_sharing", {}
) or {}

# ---- Skills / prompts config (from parameters/) ----
PROMPTS_FILE = GENERATION_CFG.get("prompts_file", "prompts.yaml")
ALIAS_MAP_FILE = SKILLS_CFG.get("alias_map_file", "alias_mapping.yaml")

dropping_irrelevant_skills = bool(GENERATION_CFG.get("dropping_irrelevant_skills", True))
enable_llm_experience = bool(GENERATION_CFG.get("enable_llm_experience", False))

MIN_SECTION_LENGTH = int(RETRY_CFG.get("min_section_length", 10) or 10)
MIN_RETRY_LENGTH = int(RETRY_CFG.get("retry_short_length", 10) or 10)
RETRY_SLEEP_MULTIPLIER = float(
    RETRY_CFG.get("retry_backoff_multiplier", 1.2) or 1.2
)

# ---- Skills matching numeric thresholds ----
_min_cov_raw = SKILLS_CFG.get("min_coverage", 0.6)
try:
    _min_cov = float(_min_cov_raw)
except (TypeError, ValueError):
    _min_cov = 0.6

MIN_SKILL_COVERAGE = max(0.0, min(1.0, _min_cov))

_raw_fuzzy = SKILLS_CFG.get("fuzzy_threshold")
try:
    _fuzzy_val = float(_raw_fuzzy)
    FUZZY_THRESHOLD = max(0.0, min(1.0, _fuzzy_val))
except (TypeError, ValueError):
    FUZZY_THRESHOLD = None

# ---------------------------------------------------------------------------
# Helper classes / functions
# ---------------------------------------------------------------------------


@dataclass
class StageBTelemetry:
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    calls: int = 0

    def add_usage_snapshot(self, snapshot: Dict[str, Any]) -> None:
        """
        Accepts the usage_snapshot dict from llm_metrics.call_llm_section_with_metrics.
        Expected keys: input_tokens, output_tokens, total_cost_usd (others ignored).
        """
        if not snapshot:
            return

        try:
            self.total_input_tokens += int(snapshot.get("input_tokens", 0) or 0)
            self.total_output_tokens += int(snapshot.get("output_tokens", 0) or 0)
            self.total_cost_usd += float(snapshot.get("total_cost_usd", 0.0) or 0.0)
            self.calls += 1
        except Exception:
            # best-effort; don't break pipeline if metrics are weird
            return

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def cost_estimate_thb(self) -> float:
        rate = get_thb_per_usd_from_params()
        return float(self.total_cost_usd) * float(rate)


class _CallableToClientAdapter:
    """
    Wrap a callable(prompt, **kwargs) into a `.generate(model=..., messages=[...])` client
    that `llm_metrics.call_llm_section_with_metrics` expects.

    Crucially, this adapter forwards *engine defaults* (temperature, top_p, max_tokens,
    timeout_seconds, max_retries) so the underlying client never receives None values.
    """

    class _Resp:
        def __init__(self, llm_text_response):
            """Wrap an LLMText response, preserving all metadata for metrics."""
            # Keep the original SDK object
            self._original_response_object = llm_text_response
            self.text = str(llm_text_response)

            # Forward all critical Gemini attributes
            self.usage_metadata = getattr(llm_text_response, "usage_metadata", None)
            self.usage = {
                "prompt_tokens": getattr(
                    getattr(llm_text_response, "usage_metadata", None),
                    "prompt_token_count",
                    0,
                ),
                "completion_tokens": getattr(
                    getattr(llm_text_response, "usage_metadata", None),
                    "candidates_token_count",
                    0,
                ),
            }
            # Preserve deeper SDK layers (if present)
            self.raw = getattr(llm_text_response, "raw", None)
            self.snapshot_wo_text = getattr(
                llm_text_response, "snapshot_wo_text", None
            )

        def __getattr__(self, name):
            """Proxy any missing attribute to the original LLMText object."""
            return getattr(self._original_response_object, name, None)

    def __init__(self, call_fn, defaults: Dict[str, Any]):
        self._call = call_fn
        self._defaults = defaults or {}

    def generate(self, *, model: str, messages: list[dict], **_kwargs):
        # Convert messages → single user prompt string
        prompt_parts: list[str] = []
        for m in messages or []:
            content = m.get("content", "")
            if content:
                prompt_parts.append(str(content))
        prompt = "\n".join(prompt_parts)

        # Section-specific budget from Stage B (highest priority)
        section_budget = self._defaults.get("max_output_tokens")
        # Any max_output_tokens the metrics helper might have passed
        dynamic_max_output = _kwargs.get("max_output_tokens")

        if section_budget is not None:
            # 🔹 Always prefer the section token budget when present
            effective_max_output = int(section_budget)
        elif dynamic_max_output is not None:
            effective_max_output = int(dynamic_max_output)
        else:
            effective_max_output = self._defaults.get("max_output_tokens")

        llm_text_response = self._call(
            prompt,
            model=model,
            temperature=self._defaults.get("temperature"),
            top_p=self._defaults.get("top_p"),
            max_output_tokens=effective_max_output,
            timeout=self._defaults.get("timeout_seconds"),
            max_retries=self._defaults.get("max_retries"),
        )

        return self._Resp(llm_text_response)


def _extract_training_year(entry):
    return (
        extract_year_from_date(getattr(entry, "year", None))
        or extract_year_from_date(getattr(entry, "start_date", None))
        or extract_year_from_date(getattr(entry, "end_date", None))
        or extract_year_from_date(getattr(entry, "training_date", None))
    )


def _render_experience_header(entry: Any) -> str:
    """
    Backwards-compatible wrapper for tests_utils and legacy callers.

    The actual implementation lives in functions.utils.experience_functions
    as `render_experience_header`.
    """
    return render_experience_header(entry)


def _extract_jd_target_skills(request: CVGenerationRequest) -> set[str]:
    """
    Collect target JD skills from job_role_info / job_position_info, if present.

    Expected shapes (examples only):
      job_role_info.required_skills: list[str] or list[{"name": str, ...}]
      job_position_info.required_skills: same idea.

    Returns:
        set[str]: normalized (lowercased) skill names.
    """

    def _names_from_list(raw: Any) -> set[str]:
        out: set[str] = set()
        if not raw:
            return out
        for item in raw:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                name = str(item.get("name", "")).strip()
            else:
                name = str(item).strip()
            if name:
                out.add(name.lower())
        return out

    jd_skills: set[str] = set()

    job_role_info = getattr(request, "job_role_info", None) or {}
    job_position_info = getattr(request, "job_position_info", None) or {}

    for src in (job_role_info, job_position_info):
        if not isinstance(src, dict):
            continue
        for key in ("required_skills", "preferred_skills", "skills"):
            if key in src:
                jd_skills |= _names_from_list(src.get(key))

    logger.info("jd_target_skills_extracted", count=len(jd_skills))
    return jd_skills


BAD_SECTION_HEADINGS = (
    "training",
    "training section",
    "publications",
    "publication",
    "publications section",
    "ผลงานตีพิมพ์",
    "การฝึกอบรม",
)


def _get_available_sections(request: CVGenerationRequest) -> set[str]:
    """
    Infer which sections are allowed to be generated, based on the presence
    of source data in profile_info, student_profile, or user_input drafts.

    Legacy rules:
      - If profile_info.<field> is non-empty → that section is available.
      - If user_input_cv_text_by_section[section_id] is non-empty → available.

    New API rules (student_profile):
      - Non-empty student_profile.education → 'education' + 'profile_summary'
      - Non-empty student_profile.experience → 'experience' + 'profile_summary'
      - Non-empty student_profile.skills → 'skills' + 'profile_summary'
      - Non-empty student_profile.awards → 'awards'
      - Non-empty student_profile.extracurriculars → 'extracurricular'
    """
    profile_info = getattr(request, "profile_info", {}) or {}
    drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}
    student_profile = getattr(request, "student_profile", None)

    available: set[str] = set()

    def _non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    # --- Legacy: profile_info + drafts ---
    profile_key_map = {
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

    for section_id, key in profile_key_map.items():
        if key in profile_info and _non_empty(profile_info.get(key)):
            available.add(section_id)

    for section_id, draft in drafts.items():
        if _non_empty(draft):
            available.add(section_id)

    # --- NEW: infer availability from student_profile (new API shape) ---
    if student_profile is not None:
        sp = student_profile

        if _non_empty(getattr(sp, "education", [])):
            available.add("education")
            available.add("profile_summary")

        if _non_empty(getattr(sp, "experience", [])):
            available.add("experience")
            available.add("profile_summary")

        if _non_empty(getattr(sp, "skills", [])):
            available.add("skills")
            available.add("profile_summary")

        if _non_empty(getattr(sp, "awards", [])):
            available.add("awards")

        if _non_empty(getattr(sp, "extracurriculars", [])):
            available.add("extracurricular")

        if _non_empty(getattr(sp, "publications", [])):
            available.add("publications")

        if _non_empty(getattr(sp, "training", [])):
            available.add("training")

        if _non_empty(getattr(sp, "references", [])):
            available.add("references")

        if _non_empty(getattr(sp, "additional_info", [])):
            available.add("additional_info")

        if _non_empty(getattr(sp, "projects", [])):
            available.add("projects")

        if _non_empty(getattr(sp, "certifications", [])):
            available.add("certifications")

    # Structured skills only if we have any skills
    if "skills" in available:
        available.add("skills_structured")

    logger.info("resolved_available_sections", available_sections=list(available))
    return available


def _resolve_effective_sections(
    request: CVGenerationRequest,
    available_sections: set[str],
) -> list[str]:
    """
    Decide *which* sections to generate, **strictly honoring template_info.sections_order**
    when configured.

    Precedence:

    1. If generation.honor_template_sections_only is True AND template_info.sections_order
       is non-empty → use template_info.sections_order as the base order.
    2. Else, if request.sections is provided and non-empty → use request.sections.
    3. Else, if template_info.sections_order exists → use that.
    4. Else → fall back to CORE_SECTIONS.

    Then:

    - If generation.expand_to_core is True → union with CORE_SECTIONS (preserving order).
    - If generation.enable_structured_skills is False → drop "skills_structured".
    - Finally, filter to sections that are actually in available_sections.
    """
    # Pull fresh parameters each time so test patches to load_parameters() take effect
    params = load_parameters()
    gen_cfg = params.get("generation", {}) or {}

    honor_template = bool(gen_cfg.get("honor_template_sections_only", False))
    expand_core = bool(gen_cfg.get("expand_to_core", False))
    enable_structured = bool(gen_cfg.get("enable_structured_skills", False))

    tmpl = getattr(request, "template_info", None)
    template_order: list[str] = list(getattr(tmpl, "sections_order", []) or [])

    # Sections explicitly requested on the API/request
    explicit_sections: list[str] = list(getattr(request, "sections", None) or [])

    # ------------------------------------------------------------
    # Step 1: decide the base "requested" order
    # ------------------------------------------------------------
    if honor_template and template_order:
        # STRICT: use template sections_order and ignore request.sections
        requested: list[str] = template_order[:]
    else:
        if explicit_sections:
            requested = explicit_sections
        elif template_order:
            requested = template_order[:]
        else:
            # Last-resort fallback: core default sections
            requested = CORE_SECTIONS[:]

    # ------------------------------------------------------------
    # Step 2: optional expansion to core sections
    # ------------------------------------------------------------
    if expand_core:
        # Preserve existing order, then append any missing core sections
        requested = list(dict.fromkeys([*requested, *CORE_SECTIONS]))

    # ------------------------------------------------------------
    # Step 3: handle structured skills flag
    # ------------------------------------------------------------
    if not enable_structured:
        # When structured skills are disabled, drop skills_structured entirely
        requested = [s for s in requested if s != "skills_structured"]
    if (
        enable_structured
        and "skills" in requested
        and "skills" in available_sections
        and "skills_structured" in available_sections
        and "skills_structured" not in requested
    ):
        # Order here doesn't matter much because we never render a text section
        # for skills_structured, but appending is simple and safe.
        requested.append("skills_structured")

    # ------------------------------------------------------------
    # Step 4: filter by available_sections
    # ------------------------------------------------------------
    effective = [s for s in requested if s in available_sections]

    return effective


def _strip_markdown_fence(text: str) -> str:
    """Remove ``` / ```json fences if present, otherwise return text unchanged."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # drop first line (``` or ```json)
        if lines:
            lines = lines[1:]
        # drop last line if it's a closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    return stripped

def _get_section_char_limits(request: CVGenerationRequest) -> Dict[str, int]:
    """Extract per-section character limits from template_info."""
    tmpl = getattr(request, "template_info", None)
    if not tmpl or not getattr(tmpl, "max_chars_per_section", None):
        return {}
    return dict(tmpl.max_chars_per_section)


# ---------------------------------------------------------------------------
# Education section local helper
# ---------------------------------------------------------------------------


# def _format_education_fact_entry(entry: Any) -> str | None:
#     """Format a single education entry into a concise fact string.
#
#     Supports:
#     - object-like entries (attrs: degree, institution, major, gpa, start_date, graduation_date)
#     - dict-like entries with the same keys
#     - plain strings (returned as-is)
#     """
#     if isinstance(entry, str):
#         val = entry.strip()
#         return val or None
#
#     # Attribute or dict access helpers
#     def _get(attr: str) -> Any:
#         if hasattr(entry, attr):
#             return getattr(entry, attr)
#         if isinstance(entry, dict):
#             return entry.get(attr)
#         return None
#
#     degree = _get("degree")
#     institution = _get("institution")
#     major = _get("major")
#     gpa = _get("gpa")
#     honors = _get("honors")
#     start_date = _get("start_date")
#     graduation_date = _get("graduation_date")
#
#     parts: List[str] = []
#
#     # Degree + major
#     if degree and major:
#         parts.append(str(degree))
#         # Major might already be in degree name; keep it explicit for robustness
#         parts.append(f"Major: {major}")
#     elif degree:
#         parts.append(str(degree))
#     elif major:
#         parts.append(f"Major: {major}")
#
#     # Institution
#     if institution:
#         parts.append(str(institution))
#
#     # GPA
#     if gpa is not None:
#         parts.append(f"GPA: {gpa}")
#
#     # Honors / distinctions
#     if honors:
#         parts.append(f"Honors: {honors}")
#
#     # Years
#     start_year = extract_year_from_date(start_date)
#     end_year = extract_year_from_date(graduation_date)
#     if start_year or end_year:
#         if start_year and end_year:
#             parts.append(f"Years: {start_year}–{end_year}")
#         elif start_year:
#             parts.append(f"Years: {start_year}–")
#         else:
#             parts.append(f"Years: –{end_year}")
#
#     if not parts:
#         return None
#
#     return " | ".join(parts)

#
# def _collect_education_facts_from_request(request: CVGenerationRequest) -> List[str]:
#     """Collect education facts from student_profile or legacy profile_info.
#
#     This is used to enrich the LLM prompt for the 'education' section.
#     """
#     facts: List[str] = []
#
#     # Preferred path: student_profile.education
#     student_profile = getattr(request, "student_profile", None)
#     if student_profile is not None:
#         edu_list = getattr(student_profile, "education", None)
#         if edu_list:
#             for entry in edu_list:
#                 fact = _format_education_fact_entry(entry)
#                 if fact:
#                     facts.append(fact)
#
#     # Legacy / fallback: profile_info["education"]
#     if not facts:
#         profile_info = getattr(request, "profile_info", {}) or {}
#         edu_list = profile_info.get("education")
#         if edu_list:
#             for entry in edu_list:
#                 fact = _format_education_fact_entry(entry)
#                 if fact:
#                     facts.append(fact)
#
#     return facts


@lru_cache(maxsize=1)
def _load_skills_alias_map() -> dict[str, str]:
    """Load skills alias mapping from parameters/alias_mapping.yaml (or override)."""
    try:
        root = Path(__file__).resolve().parents[1]
        alias_path = root / "parameters" / ALIAS_MAP_FILE

        if not alias_path.exists():
            logger.warning(
                "skills_alias_map_file_not_found",
                path=str(alias_path),
                message="Using empty alias map",
            )
            return {}

        data = load_yaml_dict(alias_path) or {}

        if not isinstance(data, dict):
            logger.warning(
                "skills_alias_map_root_not_dict",
                path=str(alias_path),
                type=str(type(data)),
            )
            return {}

        raw = data.get("alias_map", {})
        if not isinstance(raw, dict):
            logger.warning(
                "skills_alias_map_not_dict",
                path=str(alias_path),
                type=str(type(raw)),
            )
            return {}

        cleaned: dict[str, str] = {}
        for k, v in raw.items():
            k_norm = str(k).strip().lower()
            v_norm = str(v).strip().lower()
            if not k_norm or not v_norm:
                logger.debug("skills_alias_empty_entry", key=k, value=v)
                continue
            cleaned[k_norm] = v_norm

        logger.info(
            "skills_alias_map_loaded",
            path=str(alias_path),
            alias_count=len(cleaned),
        )
        return cleaned

    except Exception as exc:
        logger.exception("skills_alias_map_load_failed", error=str(exc))
        return {}


def _build_skills_plan_from_profile(request: CVGenerationRequest) -> SkillsSectionPlan | None:
    """Build a SkillsSectionPlan from available profile skills.

    Supports:
    - Legacy shape: request.profile_info["skills"] (list of dicts/strings)
    - New API shape: request.student_profile.skills (list[Skill] models)
    """
    # -------- Legacy path: profile_info.skills (dict/list) --------
    profile_info = getattr(request, "profile_info", None)
    if isinstance(profile_info, dict):
        raw_skills = profile_info.get("skills") or []

        if isinstance(raw_skills, list) and raw_skills:
            canonical: List[CanonicalSkill] = []
            for item in raw_skills:
                if isinstance(item, dict):
                    name = item.get("name")
                    if not name:
                        continue
                    level = item.get("level")
                else:
                    # Allow simple string skills as a fallback
                    name = str(item)
                    level = None
                try:
                    canonical.append(
                        CanonicalSkill(
                            name=name,
                            level=level,
                            taxonomy_id=None,
                        )
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning(
                        "build_skills_plan_invalid_skill",
                        error=str(exc),
                        raw=item,
                    )

            if canonical:
                return SkillsSectionPlan(
                    canonical_skills=canonical,
                    allowed_additional_skills=[],
                )

    # -------- New API path: student_profile.skills (Pydantic models) --------
    student_profile = getattr(request, "student_profile", None)
    if student_profile is not None:
        raw_sp_skills = getattr(student_profile, "skills", []) or []
        canonical: List[CanonicalSkill] = []

        for sp_skill in raw_sp_skills:
            # Pydantic Skill model: has .name, .level, etc.
            name = getattr(sp_skill, "name", None)
            if not name:
                continue
            level = getattr(sp_skill, "level", None)
            # If level is an Enum, use .value; otherwise keep as-is
            if hasattr(level, "value"):
                level = level.value

            try:
                canonical.append(
                    CanonicalSkill(
                        name=name,
                        level=level,
                        taxonomy_id=None,
                    )
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "build_skills_plan_invalid_skill_new_api",
                    error=str(exc),
                    raw=str(sp_skill),
                )

        if canonical:
            return SkillsSectionPlan(
                canonical_skills=canonical,
                allowed_additional_skills=[],
            )

    # Nothing usable
    return None


def _build_taxonomy_only_fallback(skills_plan: SkillsSectionPlan) -> list[OutputSkillItem]:
    """
    Last-resort fallback: return taxonomy skills unchanged.

    Used when:
    - The LLM returns invalid/empty JSON for skills_structured
    - The bullet-list fallback also fails
    - Structured skills generation errors out

    Always preserves the original canonical levels.
    """
    return [
        OutputSkillItem(
            name=sk.name,
            level=sk.level,
            source="taxonomy",
        )
        for sk in skills_plan.canonical_skills
    ]


def _extract_original_skill_levels(request: CVGenerationRequest) -> dict[str, str]:
    """
    Collect original skill levels from the incoming request.

    We normalise all skills into a dict keyed by lowercased skill name so that
    later stages (_reconcile_skill_levels_with_request) can restore the original
    levels and prevent the LLM from silently downgrading or changing them.

    Returns:
        dict[str, str]: mapping { skill_name_lower: level_string }
    """
    levels: dict[str, str] = {}

    # --- Legacy shape: request.profile_info["skills"] (dicts/strings) ---
    profile_info = getattr(request, "profile_info", None)
    if isinstance(profile_info, dict):
        raw_skills = profile_info.get("skills") or []
        if isinstance(raw_skills, list):
            for item in raw_skills:
                if isinstance(item, dict):
                    name = item.get("name")
                    level = item.get("level")
                else:
                    # Simple string skill: we cannot infer a level, so skip.
                    name = str(item)
                    level = None
                if not name or not level:
                    continue
                key = str(name).strip().lower()
                if not key:
                    continue
                levels[key] = str(level)

    # --- New API shape: request.student_profile.skills (Pydantic models) ---
    student_profile = getattr(request, "student_profile", None)
    if student_profile is not None:
        raw_sp_skills = getattr(student_profile, "skills", []) or []
        for sp_skill in raw_sp_skills:
            name = getattr(sp_skill, "name", None)
            level = getattr(sp_skill, "level", None)
            if not name or level is None:
                continue
            # Enum levels → use `.value`
            if hasattr(level, "value"):
                level = level.value
            key = str(name).strip().lower()
            if not key:
                continue
            levels[key] = str(level)

    logger.info("original_skill_levels_extracted", original_levels=levels)
    return levels


def _reconcile_skill_levels_with_request(
    request: CVGenerationRequest,
    skills_output: list[OutputSkillItem] | None,
) -> list[OutputSkillItem] | None:
    """
    Restore canonical levels for any skills that already existed in the request.

    This is the *last line of defence* to ensure that:
      - Existing skills NEVER have their levels changed by the LLM.
      - Only truly new inferred skills keep their LLM-proposed levels (or None).

    Args:
        request: Original CVGenerationRequest containing profile/student skills.
        skills_output: Final structured skills list about to be returned.

    Returns:
        The same list, but with levels corrected for any matching skills.
    """
    if not skills_output:
        return skills_output

    original_levels = _extract_original_skill_levels(request)
    if not original_levels:
        # Nothing to reconcile against.
        return skills_output

    for item in skills_output:
        name = getattr(item, "name", None)
        if not name:
            continue
        key = str(name).strip().lower()
        orig_level = original_levels.get(key)
        if orig_level:
            old_level = item.level
            # Force level back to original profile/taxonomy value.
            item.level = orig_level

            # Log whenever we correct an LLM-proposed level.
            if old_level != orig_level:
                logger.info(
                    "skill_level_reconciled",
                    skill=name,
                    llm_level=old_level,
                    corrected_to=orig_level,
                )

    return skills_output


def _summarize_skills_telemetry(
    skills_output: list[OutputSkillItem] | None,
) -> dict[str, Any]:
    """
    Build a lightweight telemetry snapshot for structured skills.

    Metrics:
    - total_skills: total number of skills in the final list
    - taxonomy_count: skills with source == "taxonomy"
    - inferred_count: skills with source != "taxonomy"
    """
    if not skills_output:
        return {
            "total_skills": 0,
            "taxonomy_count": 0,
            "inferred_count": 0,
        }

    total = len(skills_output)
    taxonomy = sum(1 for s in skills_output if getattr(s, "source", None) == "taxonomy")
    inferred = total - taxonomy

    return {
        "total_skills": total,
        "taxonomy_count": taxonomy,
        "inferred_count": inferred,
    }


def _normalize_section_id_for_evidence(section_id: str) -> str:
    """Map internal/derived section IDs to their canonical evidence section name.

    This lets us treat `skills_structured` as `skills` when looking up:
    - EvidencePlan.get_evidence_for_section
    - cross_section_evidence_sharing config
    """
    if section_id.endswith("_structured"):
        return section_id[: -len("_structured")]
    return section_id


def _to_serializable(value: Any) -> Any:
    """
    Best-effort conversion of arbitrary objects (including Pydantic models and
    HttpUrl / AnyUrl) into JSON-serializable structures.

    - BaseModel        → model_dump(mode="python")
    - dict / list / set / tuple → recurse
    - primitives       → unchanged
    - everything else  → str(value)
    """
    # Pydantic models
    if isinstance(value, BaseModel):
        return value.model_dump(mode="python")

    # Mappings
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}

    # Sequences / sets
    if isinstance(value, (list, tuple, set)):
        t = type(value)
        return t(_to_serializable(v) for v in value)

    # JSON-safe primitives
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    # Fallback: HttpUrl / AnyUrl / SimpleNamespace / anything else
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


# def _collect_evidence_facts_for_section(
#     evidence_plan: EvidencePlan | None,
#     section_id: str,
# ) -> List[str]:
#     """
#     Collect evidence facts for a section, *including* any cross-section
#     sharing rules from CROSS_SECTION_CFG (from parameters.yaml).
#     """
#     if evidence_plan is None:
#         return []
#
#     canonical_section_id = _normalize_section_id_for_evidence(section_id)
#
#     cross_cfg: Dict[str, Any] = CROSS_SECTION_CFG or {}
#
#     # Prefer explicit rule for the actual section_id; fall back to canonical
#     cfg_key = section_id if section_id in cross_cfg else canonical_section_id
#     share_from = cross_cfg.get(cfg_key, cross_cfg.get("default", [])) or []
#
#     sections_to_pull: set[str] = {canonical_section_id}
#
#     # If "all" is present, we will expand to all known sections below
#     use_all = "all" in share_from
#     if not use_all:
#         for src in share_from:
#             if isinstance(src, str) and src:
#                 sections_to_pull.add(src)
#
#     evidence_facts: List[str] = []
#     seen_ids: set[str] = set()
#
#     # ---------------------------
#     # Preferred API path
#     # ---------------------------
#     if hasattr(evidence_plan, "get_evidence_for_section"):
#         # If "all" → include every section we know about
#         if use_all:
#             hints = getattr(evidence_plan, "section_hints", {}) or {}
#             for sec in hints.keys():
#                 sections_to_pull.add(sec)
#
#         for sec in sections_to_pull:
#             for ev in evidence_plan.get_evidence_for_section(sec) or []:
#                 ev_id = getattr(ev, "evidence_id", None)
#                 if ev_id and ev_id in seen_ids:
#                     continue
#                 if ev_id:
#                     seen_ids.add(ev_id)
#                 fact = getattr(ev, "fact", None)
#                 if fact:
#                     evidence_facts.append(fact)
#
#         return evidence_facts
#
#     # ---------------------------
#     # Fallback: manual wiring via section_hints + evidences
#     # ---------------------------
#     hints = getattr(evidence_plan, "section_hints", {}) or {}
#     evidences = getattr(evidence_plan, "evidences", []) or []
#
#     # Start with canonical; hints may already contain it
#     ids_for_section = set(hints.get(canonical_section_id, []))
#
#     if use_all:
#         for ev in evidences:
#             ev_id = getattr(ev, "evidence_id", None)
#             if ev_id:
#                 ids_for_section.add(ev_id)
#     else:
#         for src_section in share_from:
#             ids_for_section.update(hints.get(src_section, []))
#
#     for ev in evidences:
#         ev_id = getattr(ev, "evidence_id", None)
#         fact = getattr(ev, "fact", None)
#         if ev_id in ids_for_section and fact:
#             evidence_facts.append(fact)
#
#     return evidence_facts


# ---------------------------------------------------------------------------
# CV Generation Engine
# ---------------------------------------------------------------------------


class CVGenerationEngine:
    """Encapsulates Stage B logic: prompt building, LLM calls, and response assembly."""

    generation_params: Dict[str, Any]

    def __init__(self, llm_client, generation_params: Dict[str, Any] | None = None):
        self.llm_client = llm_client

        file_params = GENERATION_CFG

        defaults: Dict[str, Any] = {
            "model": file_params.get("model_name", "gemini-2.5-flash"),
            "temperature": file_params.get("temperature", 0.3),
            "top_p": file_params.get("top_p", 0.9),
            "max_output_tokens": file_params.get("max_tokens", 2048),
            "max_retries": file_params.get("max_retries", 2),
            "timeout_seconds": file_params.get("timeout_seconds", 30),
            "target_word_count": file_params.get("target_word_count", 100),
            "word_count_tolerance": file_params.get("word_count_tolerance", 10),
            "log_prompts": True,
            "retry_on_zero_tokens": file_params.get("retry_on_zero_tokens", True),
        }

        if generation_params:
            defaults.update(generation_params)

        self.generation_params = defaults
        self._evidence_plan: EvidencePlan | None = None
        self._metrics_client: _CallableToClientAdapter | None = None
        self._current_user_id: str = "unknown"
        self._last_retry_count: int = 0
        self._llm_usage_records: list[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # LLM call with retries (Option C: feature flag for zero-token retry)
    # ------------------------------------------------------------------
    @staticmethod
    def _get_section_token_budget_for_attempt(
        section_id: str,
        attempt: int,
    ) -> int | None:
        """
        Compatibility wrapper around the pure resolve_token_budget().

        Uses load_parameters() so that tests_utils can monkeypatch it.
        Falls back to module-level PARAMS if load_parameters() fails.
        """
        try:
            all_params = load_parameters() or {}
        except Exception:
            # Fallback to module-level snapshot
            try:
                all_params = PARAMS or {}
            except Exception:
                return None

        # Base budget from section_token_budgets
        base_budget = resolve_token_budget(section_id, attempt, all_params)

        # If resolve_token_budget() has no idea, just return as-is
        if base_budget is None:
            return None

        # Generation config (where justification flags & extra tokens live)
        generation_cfg = all_params.get("generation", {}) or {}

        # Let claims.py decide whether and how to bump the budget
        adjusted_budget = adjust_section_token_budget(
            section_id=section_id,
            base_budget=base_budget,
            generation_cfg=generation_cfg,
        )

        return adjusted_budget


    def _aggregate_usage_and_cost(self) -> tuple[int, float, float, dict[str, Any]]:
        """
        Aggregate total tokens and estimated cost in THB for this request.

        Uses:
        - self._llm_usage_records collected during _call_llm_with_retries
        - pricing config from parameters.yaml (via get_pricing_for_model)
        - THB conversion rate from parameters.yaml (via get_thb_per_usd_from_params)
        """
        if not self._llm_usage_records:
            return 0, 0.0, 0.0, {
                "total_input_tokens": 0,
                "total_output_tokens": 0,
                "usage_by_section": {},
            }

        model_name = self.generation_params.get("model", "gemini-2.5-flash")
        pricing = get_pricing_for_model(model_name)
        usd_per_input = pricing.get("usd_per_input_token", 0.0)
        usd_per_output = pricing.get("usd_per_output_token", 0.0)

        total_input_tokens = 0
        total_output_tokens = 0
        usage_by_section: dict[str, dict[str, Any]] = {}

        for rec in self._llm_usage_records:
            section_id = str(rec.get("section_id", "unknown"))

            inp = int(
                rec.get("input_tokens")
                or rec.get("prompt_tokens")
                or 0
            )
            out = int(
                rec.get("output_tokens")
                or rec.get("completion_tokens")
                or 0
            )

            total_input_tokens += inp
            total_output_tokens += out

            sec = usage_by_section.setdefault(
                section_id,
                {"input_tokens": 0, "output_tokens": 0, "calls": 0},
            )
            sec["input_tokens"] += inp
            sec["output_tokens"] += out
            sec["calls"] += 1

        total_tokens = total_input_tokens + total_output_tokens
        total_cost_usd = (
            total_input_tokens * usd_per_input
            + total_output_tokens * usd_per_output
        )

        thb_per_usd = get_thb_per_usd_from_params()
        total_cost_thb = total_cost_usd * thb_per_usd

        extra = {
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "thb_per_usd": thb_per_usd,
            "usage_by_section": usage_by_section,
        }
        return total_tokens, total_cost_thb, total_cost_usd, extra

    # ------------------------------------------------------------------
    # LLM call with retries (Option C: feature flag for zero-token retry)
    # ------------------------------------------------------------------
    def _call_llm_with_retries(self, prompt: str, section_id: str) -> str:
        """
        LLM call with retry logic + per-section token budgets.

        Behaviour:

        - If section_token_budgets are configured, we look up the budget for
          (section_id, attempt) via _get_section_token_budget_for_attempt.
          When a budget is found, we create a fresh _CallableToClientAdapter
          for that attempt with max_output_tokens set to that budget.

        - If no budget is found, we fall back to a shared adapter built from
          self.generation_params.

        - "Real" LLM (non-mock) calls:
            * may retry on zero-token outputs (if retry_on_zero_tokens=True)
            * may retry on very short outputs (< MIN_RETRY_LENGTH)

        - "Mock" / unit-test LLM calls (usage._source in {None, "merged"}):
            * NEVER retry on zero tokens
            * NEVER retry on short outputs
        """

        params = self.generation_params
        max_retries = int(params.get("max_retries", 2))
        retry_on_zero_tokens = bool(params.get("retry_on_zero_tokens", True))
        min_length_threshold = MIN_RETRY_LENGTH

        attempt = 0
        last_result: str = ""
        self._last_retry_count = 0

        while attempt < max_retries:
            attempt += 1

            logger.info(
                "llm_call_start",
                attempt=attempt,
                model=params.get("model"),
                section_id=section_id,
                stage="B_generation",
            )

            # --------------------------------------------------------------
            # 1) Per-attempt section token budget
            # --------------------------------------------------------------
            max_tokens_for_attempt: int | None = None
            try:
                max_tokens_for_attempt = self._get_section_token_budget_for_attempt(
                    section_id=section_id,
                    attempt=attempt,
                )
            except Exception as e:  # ultra defensive; shouldn't normally happen
                logger.warning(
                    "section_token_budget_lookup_failed",
                    section_id=section_id,
                    attempt=attempt,
                    error=str(e),
                )
                max_tokens_for_attempt = None

            if max_tokens_for_attempt is not None:
                # We have a budget → override max_output_tokens for THIS attempt
                local_defaults = dict(params)
                local_defaults["max_output_tokens"] = int(max_tokens_for_attempt)
                logger.info(
                    "section_token_budget_applied",
                    section_id=section_id,
                    attempt=attempt,
                    max_output_tokens=int(max_tokens_for_attempt),
                )
                metrics_client = _CallableToClientAdapter(self.llm_client, local_defaults)
            else:
                # No budget → reuse or create a shared adapter with base params
                if self._metrics_client is None:
                    self._metrics_client = _CallableToClientAdapter(
                        self.llm_client, params
                    )
                metrics_client = self._metrics_client

            try:
                # ---- define usage callback so we can aggregate tokens/cost ----
                def _record_usage(snapshot: Dict[str, Any] | None) -> None:
                    if not snapshot:
                        return
                    try:
                        record = dict(snapshot)
                    except Exception:
                        record = {"_raw": str(snapshot)}
                    # keep section info for later breakdown
                    record.setdefault("section_id", section_id)
                    self._llm_usage_records.append(record)

                resp = call_llm_section_with_metrics(
                    llm_client=metrics_client,
                    model=params.get("model"),
                    prompt=prompt,
                    section_id=section_id,
                    purpose="stage_b_generation",
                    user_id=getattr(self, "_current_user_id", "unknown"),
                    messages=None,
                    usage_callback=_record_usage,
                )

                raw_text = str(resp).strip()

                # ----------------------------------------------------------
                # 2) Handle stub / API error sentinel text
                # ----------------------------------------------------------
                is_stub_error = (
                    raw_text.startswith("[STUB_ERROR:")
                    or raw_text.startswith("STUB_ERROR:")
                )
                is_api_error = raw_text.startswith("[API_ERROR]")

                if is_stub_error or is_api_error:
                    logger.error(
                        "llm_stub_like_error_received",
                        section_id=section_id,
                        stub_type="api_error" if is_api_error else "stub_error",
                        preview=raw_text[:200],
                        attempt=attempt,
                    )

                    if attempt < max_retries:
                        time.sleep(RETRY_SLEEP_MULTIPLIER * attempt)
                        continue
                    else:
                        self._last_retry_count = attempt - 1
                        return ""

                usage = self._llm_usage_records[-1] if self._llm_usage_records else None
                total_tokens = int(usage.get("total_tokens", 0) or 0) if usage else 0

                is_mock_llm = (
                    usage is None
                    or (
                        isinstance(usage, dict)
                        and usage.get("_source") in (None, "merged")
                    )
                )

                # Track last non-empty text
                if raw_text:
                    last_result = raw_text

                # ----------------------------------------------------------
                # 4) REAL LLM: retry on zero-token output (if enabled)
                # ----------------------------------------------------------
                if (
                    not is_mock_llm
                    and retry_on_zero_tokens
                    and total_tokens == 0
                ):
                    logger.warning(
                        "llm_retry_due_to_zero_tokens",
                        attempt=attempt,
                        section_id=section_id,
                    )
                    if attempt < max_retries:
                        time.sleep(RETRY_SLEEP_MULTIPLIER * attempt)
                        continue
                    else:
                        logger.error(
                            "llm_all_retries_exhausted_zero_token_output",
                            section_id=section_id,
                        )
                        self._last_retry_count = attempt - 1
                        return last_result or "[ERROR: LLM returned zero tokens]"

                # ----------------------------------------------------------
                # 5) REAL LLM: retry on very short output
                #    (mock LLM never retries on length)
                # ----------------------------------------------------------
                if not is_mock_llm and len(raw_text) < min_length_threshold:
                    logger.warning(
                        "llm_result_too_short_retry",
                        attempt=attempt,
                        section_id=section_id,
                        length=len(raw_text),
                        content_preview=raw_text[:80],
                    )
                    if attempt < max_retries:
                        time.sleep(RETRY_SLEEP_MULTIPLIER * attempt)
                        continue
                    else:
                        logger.error(
                            "llm_all_retries_exhausted_short_output",
                            section_id=section_id,
                            final_length=len(raw_text),
                        )
                        self._last_retry_count = attempt - 1
                        return last_result

                # ----------------------------------------------------------
                # 6) SUCCESS
                # ----------------------------------------------------------
                logger.info(
                    "llm_call_success",
                    attempt=attempt,
                    result_preview=raw_text[:100],
                    section_id=section_id,
                    stage="B_generation",
                )
                self._last_retry_count = attempt - 1
                return raw_text

            except (RuntimeError, TimeoutError, ValueError) as exc:
                # Network/transport/runtime failure; may retry
                logger.warning(
                    "llm_call_failed_exception",
                    attempt=attempt,
                    error=str(exc),
                    section_id=section_id,
                )
                if attempt >= max_retries:
                    logger.error(
                        "llm_all_retries_failed_exception",
                        error=str(exc),
                        section_id=section_id,
                    )
                    self._last_retry_count = attempt - 1
                    if last_result:
                        return last_result
                    raise RuntimeError(
                        f"LLM generation failed for section '{section_id}': {exc}"
                    )

                time.sleep(RETRY_SLEEP_MULTIPLIER * attempt)

        # --------------------------------------------------------------
        # 7) All attempts exhausted without hard error
        # --------------------------------------------------------------
        self._last_retry_count = max(attempt - 1, 0)
        return last_result

    # ---------------------------------------------------------------------------
    # Experience section local helper
    # ---------------------------------------------------------------------------
    def _augment_experience_with_llm(
        self,
        request: CVGenerationRequest,
        items: list[ExperienceItem],
    ) -> list[ExperienceItem]:
        if not items:
            return items

        prompts_cfg: dict[str, str] = load_section_prompts_config()
        base_style_prompt = prompts_cfg.get("experience", "")

        raw_language = getattr(request, "language", "en") or "en"
        language_name = describe_language(raw_language)

        tone = (
            getattr(request, "tone", None)
            or getattr(request, "style_tone", None)
            or getattr(getattr(request, "template_info", None), "tone", None)
        )
        tone_desc = describe_tone(tone)

        evidence_facts = _collect_evidence_facts_for_section(
            self._evidence_plan, "experience"
        )

        raw_profile_info = getattr(request, "profile_info", None)
        if not raw_profile_info:
            raw_profile_info = getattr(request, "student_profile", None) or {}
        profile_info = _to_serializable(raw_profile_info)

        drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}
        draft_text = drafts.get("experience")

        current_items_payload = [
            i.__dict__ for i in items  # ExperienceItem is a dataclass
        ]

        lines: list[str] = [
            "You are helping to refine the work experience section of a CV.",
            f"The CV language is {language_name} (language_code='{raw_language}').",
            f"The overall tone of the CV is {tone_desc}.",
            "",
            "=== Profile Info (JSON) ===",
            json.dumps(profile_info, ensure_ascii=False, indent=2),
            "",
            "=== Current experience entries (JSON) ===",
            json.dumps(current_items_payload, ensure_ascii=False, indent=2),
        ]

        if draft_text:
            lines.extend(
                [
                    "",
                    "=== User-provided experience draft (raw text) ===",
                    str(draft_text),
                ]
            )

        if evidence_facts:
            lines.append("\n=== Evidence facts for experience ===")
            lines.extend([f"- {fact}" for fact in evidence_facts])

        if base_style_prompt:
            lines.extend(
                [
                    "",
                    "=== High-level style guide for experience ===",
                    base_style_prompt,
                ]
            )

        lines.extend(
            [
                "",
                "Now, based on all of the information above, suggest any additional",
                "roles or positions that clearly belong in the candidate's work history",
                "but are missing from the current list.",
                "",
                "Return ONLY valid JSON with this exact structure:",
                '{ "new_items": [ { ... } ] }',
                "",
                'If you have nothing to add, respond with: { "new_items": [] }',
                "Do NOT include any extra text or markdown.",
            ]
        )

        prompt = "\n".join(lines)

        raw_json = self._call_llm_with_retries(
            prompt,
            section_id="experience_augment",
        )

        # 🔸 delegate parsing + dedup to utils
        return merge_llm_experience_augmentation(items, raw_json)

    def _generate_experience_bullets_for_item(
        self,
        request: CVGenerationRequest,
        item: ExperienceItem,
        max_bullets: int = 6,
    ) -> list[str]:

        # Heuristic: treat existing bullets as a fallback (not a shortcut)
        existing = [b for b in (item.responsibilities or []) if (b or "").strip()]
        long_enough = [
            r for r in existing
            if len(str(r).lstrip("-•* ").strip()) >= 20
        ]
        prefer_existing = len(long_enough) >= 2  # just a fallback signal

        # -----------------------------
        # LLM path (always attempt)
        # -----------------------------

        prompts_cfg: dict[str, str] = load_section_prompts_config()
        bullets_prompt_template = prompts_cfg.get("experience_bullets_only", "")

        raw_language = getattr(request, "language", "en")
        if hasattr(raw_language, "value"):
            raw_language = raw_language.value  # <-- convert enum → "en"
        language_name = describe_language(raw_language)

        tone = (
            getattr(request, "tone", None)
            or getattr(request, "style_tone", None)
            or getattr(getattr(request, "template_info", None), "tone", None)
        )
        tone_desc = describe_tone(tone)

        raw_profile_info = getattr(request, "profile_info", None)
        if not raw_profile_info:
            raw_profile_info = getattr(request, "student_profile", None) or {}
        profile_info = _to_serializable(raw_profile_info)

        evidence_facts = _collect_evidence_facts_for_section(
            self._evidence_plan, "experience"
        )

        drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}
        draft_text = drafts.get("experience")

        item_payload = item.__dict__

        lines = [
            "You are generating bullet points for a single work experience entry in a CV.",
            f"The CV language is {language_name}.",
            f"The overall tone of the CV is {tone_desc}.",
            "",
            "=== Profile Info (JSON) ===",
            json.dumps(profile_info, ensure_ascii=False, indent=2),
            "",
            "=== Experience item (JSON) ===",
            json.dumps(item_payload, ensure_ascii=False, indent=2),
        ]

        if draft_text:
            lines.extend([
                "",
                "=== User-provided experience draft (raw text) ===",
                str(draft_text),
            ])

        if evidence_facts:
            lines.append("\n=== Evidence facts for experience ===")
            lines.extend([f"- {fact}" for fact in evidence_facts])

        if bullets_prompt_template:
            lines.extend([
                "",
                "=== Instructions for responsibility bullets ===",
                bullets_prompt_template,
            ])

        prompt = "\n".join(lines)

        try:
            raw_text = self._call_llm_with_retries(
                prompt,
                section_id="experience_bullets",
            )

            # Try parser first
            bullets = parse_experience_bullets_response(
                raw_text,
                existing_responsibilities=existing,
                max_bullets=max_bullets,
            )

            # 🔹 If parser failed but LLM clearly returned bullet lines, extract them manually
            if not bullets and raw_text:
                candidate_lines: list[str] = []
                for ln in raw_text.splitlines():
                    stripped = ln.strip()
                    if not stripped:
                        continue
                    if stripped.startswith(("-", "•", "*")):
                        # Normalize to "- "
                        candidate_lines.append(
                            "- " + stripped.lstrip("-•* ").strip()
                        )

                if candidate_lines:
                    bullets = candidate_lines[:max_bullets]

            if bullets:
                full_bullet_text = "\n".join(bullets)

                if should_require_justification("experience_bullets_only", GENERATION_CFG):
                    just_prompt = build_experience_justification_prompt(
                        request=request,
                        evidence_plan=self._evidence_plan,
                        section_text=full_bullet_text,
                        section_id="experience_bullets_only",
                    )

                    just_raw = self._call_llm_with_retries(
                        just_prompt,
                        section_id="experience_bullets_justification",
                    )

                    justification = parse_justification_json(just_raw)
                    justification = validate_justification_against_text(
                        justification,
                        full_bullet_text,
                    )

                    # Store the result so the parent experience section can use it
                    existing = getattr(self, "_experience_bullets_justification", None)
                    if existing is None:
                        self._experience_bullets_justification = justification
                    else:
                        existing.evidence_map.extend(justification.evidence_map)
                        existing.unsupported_claims.extend(justification.unsupported_claims)
                return bullets

        except Exception as exc:
            logger.warning(
                "experience_bullets_llm_failed",
                error=str(exc),
                title=item.title,
                company=item.company,
            )

        # Fallback to existing bullets if they were "good"
        if prefer_existing:
            return normalize_experience_bullets(existing, max_bullets=max_bullets)

        # Last-resort stub
        return [
            "- Role responsibilities were not fully provided; details are available upon request."
        ]

    # ------------------------------------------------------------------
    # Structured skills generation
    # ------------------------------------------------------------------
    def _generate_structured_skills(
        self,
        request: CVGenerationRequest,
        skills_plan: SkillsSectionPlan,
        skills_section_text: str | None = None,
    ) -> list[OutputSkillItem]:
        """
        Generate structured skills via LLM, with robust multi-step fallbacks.

        Flow:
        1. Ask the LLM to produce JSON-only `items` (SkillSelectionItem) based on:
           - Canonical taxonomy skills
           - Evidence facts (cross-section)
           - Profile + job context
        2. Deduplicate by name (case-insensitive) and filter out invalid items.
        3. If JSON is unusable:
           - Try to parse skills from the free-text "skills" section bullets.
           - If still empty → fall back to taxonomy-only skills.
        4. For each kept skill:
           - If its name matches a canonical skill (exact or close match), snap the
             name and level back to the canonical values and mark `source="taxonomy"`.
           - Otherwise, treat it as a new inferred skill using the LLM-proposed
             name/level/source.
        5. Optionally re-add missing canonical skills (depending on the
           `dropping_irrelevant_skills` flag), ensuring they appear with their
           original levels when enforced.
        6. Drop obvious “combined” skills that merely join multiple canonicals
           (e.g. "SQL & Python").
        7. Finally, reconcile against the original request once more so that any
           skill whose name matches a canonical entry always keeps the canonical level.

        Returns:
            List[OutputSkillItem] representing the final structured skills.
        """

        language = getattr(request, "language", "en")
        evidence_plan: EvidencePlan | None = self._evidence_plan

        # NEW: decide if this run should include justification for skills
        want_justification = (
            should_require_justification("skills_structured", GENERATION_CFG)
            or should_require_justification("skills", GENERATION_CFG)
        )

        # Build the JSON-only selection prompt and call the LLM.
        prompt = build_skills_selection_prompt(
            request=request,
            evidence_plan=evidence_plan,
            skills_plan=skills_plan,
            language=language,
            require_justification=want_justification,
        )
        raw_json = self._call_llm_with_retries(prompt, section_id="skills_structured")

        # NEW: split SKILLS_JSON vs JUSTIFICATION_JSON when requested
        skills_raw = raw_json or ""
        justification_raw = ""

        if want_justification and raw_json:
            try:
                skills_raw, justification_raw = split_section_and_justification(raw_json)
            except Exception as exc:
                logger.warning(
                    "skills_justification_split_failed",
                    error=str(exc),
                )
                skills_raw = raw_json or ""
                justification_raw = ""

        # ---------- Canonical lookup (case-insensitive) ----------
        canonical_by_name: dict[str, CanonicalSkill] = {
            (s.name or "").strip().lower(): s for s in skills_plan.canonical_skills
        }

        candidate_items: list[SkillSelectionItem] = []

        # ---------- Parse JSON from LLM ----------
        try:
            cleaned = _strip_markdown_fence(skills_raw)
            if cleaned:
                data = json.loads(cleaned)
                if isinstance(data, list):
                    items_data = data
                else:
                    items_data = data.get("items", []) or []

                for item in items_data:
                    try:
                        candidate_items.append(SkillSelectionItem.model_validate(item))
                    except Exception as exc:  # pragma: no cover
                        logger.warning(
                            "skills_structured_invalid_item",
                            error=str(exc),
                            raw=item,
                        )
        except Exception as exc:
            logger.warning(
                "skills_structured_json_parse_failed",
                error=str(exc),
                raw_preview=(raw_json or "")[:200],
            )
            candidate_items = []

        # ---------- Dedup by name (case-insensitive) ----------
        filtered: list[SkillSelectionItem] = []
        seen_names_lower: set[str] = set()

        for sel in candidate_items:
            name = (sel.name or "").strip()
            if not name:
                continue
            key = name.lower()
            if key in seen_names_lower:
                # Drop duplicates, keep the first occurrence.
                continue
            seen_names_lower.add(key)
            sel.name = name  # normalise whitespace
            filtered.append(sel)

        # ---------- Fallback if no usable JSON ----------
        if not filtered:
            inferred_from_bullets: list[SkillSelectionItem] = []

            # Try extracting skills from the free-text "skills" section.
            if skills_section_text:
                bullet_skills = parse_skills_from_bullets(skills_section_text)
                seen_bullet: set[str] = set()
                for name in bullet_skills:
                    norm = name.strip()
                    if not norm:
                        continue
                    key = norm.lower()
                    if key in seen_bullet:
                        continue
                    seen_bullet.add(key)
                    inferred_from_bullets.append(
                        SkillSelectionItem(
                            name=norm,
                            level=None,
                            keep=True,
                            source="inferred",
                        )
                    )

            if inferred_from_bullets:
                logger.info(
                    "skills_structured_fallback_from_bullets",
                    count=len(inferred_from_bullets),
                )
                filtered = inferred_from_bullets
            else:
                # Hard fallback: mirror canonical skills exactly as taxonomy items.
                logger.warning(
                    "skills_structured_fallback_taxonomy_only",
                    reason="no_json_and_no_bullet_skills",
                )
                # Pure taxonomy fallback → just mirror canonical skills
                return [
                    OutputSkillItem(
                        name=s.name,
                        level=s.level,
                        source="taxonomy",
                    )
                    for s in skills_plan.canonical_skills
                ]

        # ---------- Build result, PRESERVING canonical levels ----------
        result: list[OutputSkillItem] = []
        result_names_lower: set[str] = set()

        alias_map = _load_skills_alias_map()

        for sel in filtered:
            if not sel.keep:
                continue

            name_clean = (sel.name or "").strip()
            if not name_clean:
                continue

            canon = match_canonical_skill(
                name_clean,
                canonical_by_name,
                alias_map=alias_map,
                min_coverage=MIN_SKILL_COVERAGE,
                fuzzy_threshold=FUZZY_THRESHOLD,
            )

            if canon is not None:
                # Treat as canonical: snap name + level back to taxonomy
                final_name = canon.name
                level = canon.level
                source = "taxonomy"
            else:
                # Genuine new inferred skill
                final_name = name_clean
                level = sel.level
                source = sel.source or "inferred"

            key = (final_name or "").strip().lower()
            result.append(
                OutputSkillItem(
                    name=final_name,
                    level=level,
                    source=source,
                )
            )
            result_names_lower.add(key)

        # If we are NOT allowing dropping, re-add missing canonical skills
        if not dropping_irrelevant_skills:
            # ---------- Ensure taxonomy skills are NOT lost ----------
            for canon in skills_plan.canonical_skills:
                key = (canon.name or "").strip().lower()
                if not key:
                    continue
                if key in result_names_lower:
                    continue

                # Append any canonical skill that never appeared in the LLM output.
                result.append(
                    OutputSkillItem(
                        name=canon.name,
                        level=canon.level,
                        source="taxonomy",
                    )
                )
                result_names_lower.add(key)

        # Drop obvious combined canonical skills like "SQL & Python"
        filtered_result: list[OutputSkillItem] = []
        for item in result:
            if is_combined_canonical_name(getattr(item, "name", "") or "", canonical_by_name):
                logger.info(
                    "skills_combined_item_dropped",
                    name=item.name,
                )
                continue
            filtered_result.append(item)

        result = filtered_result

        # ---------- Final safety: reconcile levels with canonical ----------
        for item in result:
            key = (item.name or "").strip().lower()
            canon = canonical_by_name.get(key)
            if canon is not None:
                item.level = canon.level

        # NEW: parse + validate justification against rendered skills text
        if want_justification:
            try:
                if justification_raw:
                    j = parse_justification_json(justification_raw)
                else:
                    j = build_empty_justification()

                # Render a temporary skills text just for validation
                skills_text_for_justif = format_plain_skill_bullets(result) or ""
                j = validate_justification_against_text(
                    j,
                    skills_text_for_justif,
                )
            except Exception as exc:
                logger.warning(
                    "skills_justification_parse_or_validate_failed",
                    error=str(exc),
                )
                j = build_empty_justification()

            # Store for later attachment when we render the 'skills' section
            self._skills_justification = j

        return result

    # ------------------------------------------------------------------
    # Build CV response
    # ------------------------------------------------------------------
    def _build_cv_response(
        self,
        request: CVGenerationRequest,
        generated_sections: Union[Dict[str, SectionContent], List[SectionContent]],
        *,
        job_id: str | None = None,
        status: GenerationStatus = GenerationStatus.COMPLETED,
        generation_time_ms: int | None = None,
        retry_count: int = 0,
        cache_hit: bool = False,
        tokens_used: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
        section_breakdown: list[dict[str, Any]] | None = None,
        cost_estimate_thb: float = 0.0,
        cost_estimate_usd: float = 0.0,
        skills_output: list[OutputSkillItem] | None = None,
        justification: Justification | None = None,
    ) -> CVGenerationResponse:
        """Normalize generated sections and build CVGenerationResponse."""
        # Normalize sections into a dict
        if isinstance(generated_sections, list):
            requested_ids = getattr(request, "sections", []) or []
            sections_dict = {
                requested_ids[i] if i < len(requested_ids) else f"section_{i}": sec
                for i, sec in enumerate(generated_sections)
            }
        else:
            sections_dict = generated_sections

        profile_info = getattr(request, "profile_info", None)

        # Clamp generation time
        gen_cfg = GENERATION_CFG
        max_time_ms = int(gen_cfg.get("max_generation_time_ms", 60_000))

        raw_time_ms = generation_time_ms or 0
        safe_time_ms = min(max(raw_time_ms, 0), max_time_ms)

        metadata = Metadata(
            generation_time_ms=safe_time_ms,
            model_version=self.generation_params.get("model", "gemini-2.5-flash"),
            retry_count=retry_count,
            cache_hit=cache_hit,
            sections_requested=len(getattr(request, "sections", []) or []),
            sections_generated=len(sections_dict),
            tokens_used=tokens_used,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            section_breakdown=section_breakdown or [],
            cost_estimate_thb=cost_estimate_thb,
            cost_estimate_usd=cost_estimate_usd,
            profile_info=profile_info,
        )

        justification_obj = justification or Justification()

        # Final job_id
        user_part = getattr(request, "user_id", "unknown")
        safe_user_part = "".join(
            ch for ch in str(user_part) if ch.isalnum() or ch in "-_"
        )
        final_job_id = job_id or getattr(request, "job_id", f"JOB_{safe_user_part}")

        # Template id (legacy → new → fallback)
        template_info = getattr(request, "template_info", None)
        template_id = (
            getattr(template_info, "template_id", None)
            or getattr(request, "template_id", None)
            or "UNKNOWN_TEMPLATE"
        )

        payload = {
            "job_id": final_job_id,
            "template_id": template_id,
            "language": getattr(request, "language", "en"),
            "status": status,
            "sections": sections_dict,
            "skills": skills_output,
            "metadata": metadata,
            "justification": justification_obj,
            "quality_metrics": None,
            "warnings": [],
            "error": None,
            "error_details": None,
        }

        return CVGenerationResponse.model_validate(payload)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate_cv(
        self,
        request: CVGenerationRequest,
        evidence_plan: EvidencePlan | None = None,
        skills_plan: SkillsSectionPlan | None = None,
    ) -> CVGenerationResponse:
        """
        Generate all CV sections end-to-end for Stage B.

        Responsibilities:
        - Bind request-scoped logging context (request_id, user_id, template_id, etc.)
        - Resolve which sections are effectively generated based on template + request
        - Generate structured skills first (when enabled) and render text skills from them
        - Call the LLM for all remaining sections with retry + truncation
        - Apply section-specific post-processing:
            * experience: deterministic header + bullet list per role
            * training: append year hint when missing
            * training/publications: strip meaningless heading bullets
        - Emit telemetry for skills and overall Stage B performance
        - Build a CVGenerationResponse with all generated sections and metadata.
        """
        # ------------------------------------------------------------
        # Bind request-scoped structured logging context
        # ------------------------------------------------------------
        bind_contextvars(
            request_id=getattr(request, "request_id", None) or "N/A",
            job_id=getattr(request, "job_id", None) or "N/A",
            user_id=getattr(request, "user_id", "unknown"),
            template_id=(
                getattr(getattr(request, "template_info", None), "template_id", None)
                or getattr(request, "template_id", None)
                or "UNKNOWN_TEMPLATE"
            ),
            language=getattr(request, "language", "en"),
            sections_count=len(getattr(request, "sections", []) or []),
        )

        try:
            sections = getattr(request, "sections", []) or []
            self._current_user_id = getattr(request, "user_id", "unknown")
            self._last_retry_count = 0
            self._llm_usage_records = []  # reset per-request
            self.telemetry = StageBTelemetry()

            # NEW: per-request justification object (may remain empty if disabled)
            justification = build_empty_justification()
            self._skills_justification: Justification | None = None
            self._experience_bullets_justification = None

            char_limits = _get_section_char_limits(request)
            available_sections = _get_available_sections(request)

            # Resolve strictly based on template order
            effective_sections = _resolve_effective_sections(
                request, available_sections
            )

            logger.info(
                "stage_b_resolved_sections",
                requested_sections=sections,
                available_sections=list(available_sections),
                effective_sections=effective_sections,
            )
            structured_first = "skills_structured" in effective_sections

            logger.info(
                "stage_b_telemetry_sections",
                mode="structured_first" if structured_first else "legacy_skills",
                sections_requested=len(sections),
                sections_available=len(available_sections),
                sections_effective=len(effective_sections),
                has_skills_section=("skills" in effective_sections),
                has_skills_structured=("skills_structured" in effective_sections),
            )

            generated_sections: dict[str, SectionContent] = {}
            skills_output: list[OutputSkillItem] | None = None

            start_time = time.monotonic()
            self._evidence_plan = evidence_plan

            # Build skills plan
            if (structured_first or "skills" in effective_sections) and skills_plan is None:
                skills_plan = _build_skills_plan_from_profile(request)

            # 1) Structured skills FIRST if enabled
            if structured_first and skills_plan is not None:
                try:
                    skills_output = self._generate_structured_skills(
                        request,
                        skills_plan,
                        skills_section_text=None,
                    )
                except Exception as exc:
                    logger.error(
                        "skills_structured_generation_failed_prepass",
                        error=str(exc),
                    )
                    skills_output = _build_taxonomy_only_fallback(skills_plan)

            # 2) Generate all text sections (skills text is special)
            for section_id in effective_sections:
                if section_id == "skills_structured":
                    continue

                # ------------------------------------------------------------------
                # NEW: Experience uses structured-first path with NO LLM call
                # ------------------------------------------------------------------
                if section_id == "experience":
                    # 1) Build base structured experience from profile / student_profile / drafts
                    structured_items = build_structured_experience(request)

                    # If we have nothing structured, fall back to LLM section generation
                    if not structured_items:
                        prompt = build_section_prompt(
                            request,
                            evidence_plan,
                            section_id,
                        )
                        raw_text = self._call_llm_with_retries(
                            prompt,
                            section_id,
                        )

                        text_for_section = raw_text
                        if len(text_for_section.strip()) < MIN_SECTION_LENGTH:
                            text_for_section = build_section_fallback_text(
                                request,
                                section_id,
                                reason="LLM output empty for experience",
                            )

                        generated_sections[section_id] = SectionContent(
                            text=text_for_section,
                            word_count=len(text_for_section.split()),
                            matched_jd_skills=[],
                            confidence_score=1.0,
                        )
                        continue

                    items: list[ExperienceItem] = structured_items

                    # 2) Optionally augment with LLM (new roles) if feature flag enabled
                    if enable_llm_experience:
                        try:
                            items = self._augment_experience_with_llm(request, items)
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                "experience_augmentation_failed",
                                error=str(exc),
                            )

                    # 3) Ensure bullets for each item (LLM per-role when needed)
                    for item in items:
                        try:
                            new_bullets = self._generate_experience_bullets_for_item(
                                request,
                                item,
                                max_bullets=6,
                            )
                            item.responsibilities = new_bullets
                            item.bullets = new_bullets
                        except Exception as exc:  # pragma: no cover
                            logger.warning(
                                "experience_bullets_generation_failed",
                                error=str(exc),
                                title=item.title,
                                company=item.company,
                            )
                            # Fallback using normalized bullets from utils
                            item.responsibilities = normalize_experience_bullets(
                                item.responsibilities or []
                            ) or [
                                "- Role responsibilities were not fully provided; details are available upon request."
                            ]

                    # 4) Deterministic markdown rendering
                    section_text = render_experience_section_from_structured(items)

                    if len((section_text or "").strip()) < MIN_SECTION_LENGTH:
                        section_text = build_section_fallback_text(
                            request,
                            section_id,
                            reason="experience text too short",
                        )

                    # NEW: experience justification (separate LLM call)
                    experience_requires_justif = should_require_justification("experience", GENERATION_CFG)
                    if experience_requires_justif:
                        try:
                            just_prompt = build_experience_justification_prompt(
                                request,
                                evidence_plan,
                                section_text,
                            )
                            just_raw = self._call_llm_with_retries(
                                just_prompt,
                                section_id="experience_justification",
                            )

                            if just_raw:
                                try:
                                    j = parse_justification_json(just_raw)
                                    j = validate_justification_against_text(
                                        j,
                                        section_text,
                                    )
                                except Exception as exc:
                                    logger.warning(
                                        "experience_justification_parse_or_validate_failed",
                                        error=str(exc),
                                    )
                                    j = build_empty_justification()
                            else:
                                j = build_empty_justification()

                            # 🔹 Merge main experience justification
                            if j:
                                if j.evidence_map:
                                    justification.evidence_map.extend(j.evidence_map)
                                if j.unsupported_claims:
                                    justification.unsupported_claims.extend(j.unsupported_claims)

                        except Exception as exc:
                            logger.warning(
                                "experience_justification_generation_failed",
                                error=str(exc),
                            )

                    # 🔹 Always merge bullet-level justification (if any), even if "experience" is not configured
                    bullet_j = getattr(self, "_experience_bullets_justification", None)
                    if bullet_j:
                        if bullet_j.evidence_map:
                            justification.evidence_map.extend(bullet_j.evidence_map)
                        if bullet_j.unsupported_claims:
                            justification.unsupported_claims.extend(bullet_j.unsupported_claims)

                    generated_sections[section_id] = SectionContent(
                        text=section_text,
                        word_count=len(section_text.split()),
                        matched_jd_skills=[],
                        confidence_score=1.0,
                    )
                    continue

                # ------------------------------------------------------------------
                # Special handling: skills text rendered from structured-first output
                # ------------------------------------------------------------------
                if section_id == "skills" and structured_first and skills_output is not None:
                    text_for_section = format_plain_skill_bullets(skills_output)

                    if len(text_for_section.strip()) < MIN_SECTION_LENGTH:
                        logger.warning(
                            "section_text_too_short",
                            section_id=section_id,
                            length=len(text_for_section),
                            reason="structured_skills_rendered_empty_or_short",
                        )
                        text_for_section = build_section_fallback_text(
                            request,
                            section_id,
                            reason="no structured skills available",
                        )

                    # NEW: if we computed a skills justification earlier, attach it
                    if (
                        hasattr(self, "_skills_justification")
                        and self._skills_justification
                        and self._skills_justification.evidence_map
                    ):
                        justification.evidence_map.extend(self._skills_justification.evidence_map)

                else:
                    # --------------------------------------------------------------
                    # Default generation path for all other sections (LLM-backed)
                    # --------------------------------------------------------------
                    want_justification = should_require_justification(
                        section_id,
                        GENERATION_CFG,
                    )

                    # Build the full prompt ONCE so we can also reuse it as
                    # the "source text" for justification validation.
                    prompt_for_section = build_section_prompt(
                        request,
                        evidence_plan,
                        section_id,
                    )

                    raw_text = self._call_llm_with_retries(
                        prompt_for_section,
                        section_id,
                    )

                    # --- Split section text vs justification (if enabled) ---
                    section_text_raw = raw_text or ""
                    justification_raw = ""

                    if want_justification and raw_text:
                        try:
                            section_text_raw, justification_raw = (
                                split_section_and_justification(raw_text)
                            )
                        except Exception as exc:
                            logger.warning(
                                "justification_split_failed",
                                section_id=section_id,
                                error=str(exc),
                            )
                            section_text_raw = raw_text or ""
                            justification_raw = ""

                    # 🔹 Strip Markdown bold markers from LLM text only in this path
                    if section_text_raw:
                        # Keep it simple and conservative: only remove "**"
                        section_text_raw = section_text_raw.replace("**", "")
                        section_text_raw = strip_redundant_section_heading(
                            section_text_raw,
                            section_id,
                            removal_map={
                                "references": "references",
                                "additional_info": "additional information",
                            },
                        )

                    text_for_section = section_text_raw

                    if len(text_for_section) < MIN_SECTION_LENGTH:
                        logger.warning(
                            "section_text_too_short",
                            section_id=section_id,
                            length=len(text_for_section),
                            reason="LLM returned empty/failed output",
                        )
                        text_for_section = build_section_fallback_text(
                            request,
                            section_id,
                            reason="LLM output empty",
                        )

                    # NEW: parse + validate justification JSON for this section (if any)
                    if want_justification:
                        try:
                            if justification_raw:
                                j = parse_justification_json(justification_raw)
                                # IMPORTANT: validate against *input prompt + context*,
                                # not the generated section text.
                                j = validate_justification_against_text(
                                    j,
                                    prompt_for_section,
                                )
                            else:
                                j = build_empty_justification()
                        except Exception as exc:
                            logger.warning(
                                "justification_parse_or_validate_failed",
                                section_id=section_id,
                                error=str(exc),
                            )
                            j = build_empty_justification()

                        if j and j.evidence_map:
                            justification.evidence_map.extend(j.evidence_map)

                    # Legacy mode: derive structured skills from free-text "skills"
                    if (
                        section_id == "skills"
                        and not structured_first
                        and skills_plan is not None
                        and skills_output is None
                    ):
                        try:
                            skills_output = self._generate_structured_skills(
                                request,
                                skills_plan,
                                skills_section_text=text_for_section,
                            )
                        except Exception as exc:
                            logger.error(
                                "skills_structured_generation_failed_legacy",
                                error=str(exc),
                            )
                            skills_output = _build_taxonomy_only_fallback(skills_plan)

                # ------------------------------------------------------------------
                # Post-processing hooks for specific sections (non-experience)
                # ------------------------------------------------------------------

                # Training: append year hint from first training entry if missing
                if section_id == "training":
                    sp = getattr(request, "student_profile", None)
                    trainings = getattr(sp, "training", []) if sp else []
                    if trainings:
                        yr = _extract_training_year(trainings[0])
                        if yr and yr not in text_for_section:
                            lines = text_for_section.splitlines()
                            if lines:
                                lines[-1] = lines[-1].rstrip() + f" ({yr})"
                                text_for_section = "\n".join(lines)

                # Remove meaningless heading bullets for training/publications
                if section_id in ("training", "publications"):
                    lines = text_for_section.splitlines()
                    cleaned: list[str] = []
                    for ln in lines:
                        core = (
                            ln.strip()
                            .lstrip("-•* ")
                            .rstrip(":")
                            .strip()
                            .lower()
                        )
                        if core in BAD_SECTION_HEADINGS:
                            continue
                        cleaned.append(ln)
                    text_for_section = "\n".join(cleaned).strip()

                generated_sections[section_id] = SectionContent(
                    text=text_for_section,
                    word_count=len(text_for_section.split()),
                    matched_jd_skills=[],
                    confidence_score=1.0,
                )

            elapsed_ms = int((time.monotonic() - start_time) * 1000)

            if skills_output:
                logger.info(
                    "skills_before_reconcile",
                    skills=[(s.name, s.level, s.source) for s in skills_output],
                )
                skills_output = _reconcile_skill_levels_with_request(
                    request, skills_output
                )
                logger.info(
                    "skills_after_reconcile",
                    skills=[(s.name, s.level, s.source) for s in skills_output],
                )

            # 🔹 Aggregate LLM usage and cost (USD → THB) from collected records
            total_tokens_used, total_cost_thb, total_cost_usd, usage_extra = (
                self._aggregate_usage_and_cost()
            )
            usage_by_section = usage_extra.get("usage_by_section", {}) or {}
            section_breakdown = [
                {
                    "section_name": section_id,
                    "section_input_tokens": int(stats.get("input_tokens", 0) or 0),
                    "section_output_tokens": int(stats.get("output_tokens", 0) or 0),
                }
                for section_id, stats in usage_by_section.items()
            ]

            # 🔹 Telemetry snapshot for skills + overall Stage B
            skills_metrics = _summarize_skills_telemetry(skills_output)

            # NEW: expose usage for Stage D
            self._model_name = (
                getattr(self, "_model_name", None)
                or self.generation_params.get("model")
                or "gemini-2.5-flash"
            )
            self._stage_b_total_input_tokens = int(
                usage_extra.get("total_input_tokens", 0)
            )
            self._stage_b_total_output_tokens = int(
                usage_extra.get("total_output_tokens", 0)
            )
            self._stage_b_total_tokens = int(total_tokens_used)
            self._stage_b_total_cost_usd = float(round(total_cost_usd, 4))
            self._stage_b_total_cost_thb = float(round(total_cost_thb, 4))
            self._stage_b_section_breakdown = section_breakdown

            logger.info(
                "stage_b_telemetry_summary",
                generation_time_ms=elapsed_ms,
                retry_count=self._last_retry_count,
                sections_generated=len(generated_sections),
                cache_hit=False,
                tokens_used=total_tokens_used,
                cost_estimate_usd=round(total_cost_usd, 4),
                cost_estimate_thb=round(total_cost_thb, 4),
                total_input_tokens=usage_extra.get("total_input_tokens", 0),
                total_output_tokens=usage_extra.get("total_output_tokens", 0),
                section_breakdown=section_breakdown,
                **skills_metrics,
            )

            # Compute matched_jd_skills per section using structured skills + JD info
            try:
                jd_target_skills = _extract_jd_target_skills(request)
            except Exception as exc:
                logger.warning(
                    "jd_target_skills_extraction_failed",
                    error=str(exc),
                )
                jd_target_skills = set()

            if jd_target_skills and generated_sections:
                for section_id, section in generated_sections.items():
                    matched = compute_section_matched_jd_skills(
                        section_id=section_id,
                        section_text=section.text or "",
                        skills_output=skills_output,
                        jd_target_skill_names=jd_target_skills,
                    )
                    section.matched_jd_skills = matched or []

            return self._build_cv_response(
                request=request,
                generated_sections=generated_sections,
                generation_time_ms=elapsed_ms,
                retry_count=self._last_retry_count,
                cache_hit=False,
                tokens_used=total_tokens_used,
                input_tokens=usage_extra.get("total_input_tokens", 0),
                output_tokens=usage_extra.get("total_output_tokens", 0),
                section_breakdown=section_breakdown,
                cost_estimate_thb=round(total_cost_thb, 4),
                cost_estimate_usd=round(total_cost_usd, 4),
                skills_output=skills_output,
                justification=justification,
            )

        finally:
            clear_contextvars()
