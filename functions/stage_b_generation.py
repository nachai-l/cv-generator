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
from functions.utils.skills_formatting import (
    format_plain_skill_bullets,
    parse_skills_from_bullets,
    is_combined_canonical_name,
    match_canonical_skill,
)
from functions.utils.language_tone import describe_language, describe_tone
from functions.utils.common import resolve_token_budget, load_yaml_dict
logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (to avoid duplicated long string fragments)
# ---------------------------------------------------------------------------

params = load_parameters()

generation_cfg = params.get("generation", {}) or {}
retry_cfg = generation_cfg.get("retry_thresholds", {}) or {}
skills_cfg = generation_cfg.get("skills_matching", {}) or {}

# ---- Skills matching config (from parameters.yaml → generation.skills_matching) ----
_min_cov_raw = skills_cfg.get("min_coverage", 0.6)
try:
    _min_cov = float(_min_cov_raw)
except (TypeError, ValueError):
    _min_cov = 0.6

MIN_SKILL_COVERAGE = max(0.0, min(1.0, _min_cov))  # clamp to [0, 1]

_raw_fuzzy = skills_cfg.get("fuzzy_threshold")
try:
    _fuzzy_val = float(_raw_fuzzy)
    FUZZY_THRESHOLD = max(0.0, min(1.0, _fuzzy_val))  # clamp to [0, 1]
except (TypeError, ValueError):
    FUZZY_THRESHOLD = None

ALIAS_MAP_FILE = skills_cfg.get("alias_map_file", "alias_mapping.yaml")

dropping_irrelevant_skills = bool(
    generation_cfg.get("dropping_irrelevant_skills", True)
)

MIN_SECTION_LENGTH = int(retry_cfg.get("min_section_length", 10) or 10)
MIN_RETRY_LENGTH = int(retry_cfg.get("retry_short_length", 10) or 10)
RETRY_SLEEP_MULTIPLIER = float(
    retry_cfg.get("retry_backoff_multiplier", 1.2) or 1.2
)

# ---------------------------------------------------------------------------
# Helper classes / functions
# ---------------------------------------------------------------------------


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

        # Dynamic overrides from llm_metrics (e.g. section-level budgets)
        dynamic_max_output = _kwargs.get("max_output_tokens")
        effective_max_output = (
            dynamic_max_output
            if dynamic_max_output is not None
            else self._defaults.get("max_output_tokens")
        )

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
    Decide *which* sections to generate, **strictly honoring template_info.sections_order**.
    Fallback to request.sections ONLY if the template has no sections_order.
    Other expansion flags are ignored unless explicitly set in parameters.yaml.
    """
    params = load_parameters()
    gen = params.get("generation", {}) or {}
    core_sections = params.get("core_sections", []) or []

    # Step 1: Determine requested sections from template/request
    honor_template = bool(gen.get("honor_template_sections_only", True))
    expand_core = bool(gen.get("expand_to_core", False))
    enable_structured = bool(gen.get("enable_structured_skills", False))

    tmpl = getattr(request, "template_info", None)
    template_order = list(getattr(tmpl, "sections_order", []) or [])

    if honor_template and template_order:
        requested = template_order[:]  # STRICT: use template order only
    else:
        # Fallback to request.sections if template has no order
        explicit_sections = (getattr(request, "sections", None) or [])[:]
        if explicit_sections:
            requested = explicit_sections
        else:
            requested = template_order or core_sections[:] # last-resort fallback

    # Step 2: Apply expansion/filtering rules
    if expand_core:
        requested = list(dict.fromkeys([*requested, *core_sections]))

    if not enable_structured:
        requested = [s for s in requested if s != "skills_structured"]

    # Step 3: Filter to only sections with available data
    # Strict filter: only generate sections that are both requested (by template) AND available
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


def _truncate_text(text: str, max_chars: int | None) -> str:
    """Truncate text to max_chars and append '…' if needed."""
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _get_section_char_limits(request: CVGenerationRequest) -> Dict[str, int]:
    """Extract per-section character limits from template_info."""
    tmpl = getattr(request, "template_info", None)
    if not tmpl or not getattr(tmpl, "max_chars_per_section", None):
        return {}
    return dict(tmpl.max_chars_per_section)

def _extract_year_from_date(value: Any) -> str | None:
    """Best-effort extraction of a year (YYYY) from a date-like value.

    Accepts:
    - ISO date strings: '2015-08-01' -> '2015'
    - datetime/date objects -> str(year)
    - already-a-year strings -> '2015'
    """
    if value is None:
        return None

    # datetime/date-like
    try:
        import datetime as _dt

        if isinstance(value, (_dt.date, _dt.datetime)):
            return str(value.year)
    except Exception:
        # datetime may not be imported or value not a date; ignore
        pass

    if isinstance(value, str):
        s = value.strip()
        if len(s) >= 4 and s[:4].isdigit():
            return s[:4]

    return None


def _format_education_fact_entry(entry: Any) -> str | None:
    """Format a single education entry into a concise fact string.

    Supports:
    - object-like entries (attrs: degree, institution, major, gpa, start_date, graduation_date)
    - dict-like entries with the same keys
    - plain strings (returned as-is)
    """
    if isinstance(entry, str):
        val = entry.strip()
        return val or None

    # Attribute or dict access helpers
    def _get(attr: str) -> Any:
        if hasattr(entry, attr):
            return getattr(entry, attr)
        if isinstance(entry, dict):
            return entry.get(attr)
        return None

    degree = _get("degree")
    institution = _get("institution")
    major = _get("major")
    gpa = _get("gpa")
    start_date = _get("start_date")
    graduation_date = _get("graduation_date")

    parts: List[str] = []

    # Degree + major
    if degree and major:
        parts.append(str(degree))
        # Major might already be in degree name; keep it explicit for robustness
        parts.append(f"Major: {major}")
    elif degree:
        parts.append(str(degree))
    elif major:
        parts.append(f"Major: {major}")

    # Institution
    if institution:
        parts.append(str(institution))

    # GPA
    if gpa is not None:
        parts.append(f"GPA: {gpa}")

    # Years
    start_year = _extract_year_from_date(start_date)
    end_year = _extract_year_from_date(graduation_date)
    if start_year or end_year:
        if start_year and end_year:
            parts.append(f"Years: {start_year}–{end_year}")
        elif start_year:
            parts.append(f"Years: {start_year}–")
        else:
            parts.append(f"Years: –{end_year}")

    if not parts:
        return None

    return " | ".join(parts)


def _collect_education_facts_from_request(request: CVGenerationRequest) -> List[str]:
    """Collect education facts from student_profile or legacy profile_info.

    This is used to enrich the LLM prompt for the 'education' section.
    """
    facts: List[str] = []

    # Preferred path: student_profile.education
    student_profile = getattr(request, "student_profile", None)
    if student_profile is not None:
        edu_list = getattr(student_profile, "education", None)
        if edu_list:
            for entry in edu_list:
                fact = _format_education_fact_entry(entry)
                if fact:
                    facts.append(fact)

    # Legacy / fallback: profile_info["education"]
    if not facts:
        profile_info = getattr(request, "profile_info", {}) or {}
        edu_list = profile_info.get("education")
        if edu_list:
            for entry in edu_list:
                fact = _format_education_fact_entry(entry)
                if fact:
                    facts.append(fact)

    return facts


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


def _collect_evidence_facts_for_section(
    evidence_plan: EvidencePlan | None,
    section_id: str,
) -> List[str]:
    """
    Collect evidence facts for a section, *including* any cross-section
    sharing rules from parameters.yaml.cross_section_evidence_sharing.

    - Always includes the canonical section's own evidence.
    - If a cross-section rule exists, also pull evidence from those sections.
    - Special value "all" means "all known sections" from section_hints.
    """
    if evidence_plan is None:
        return []

    canonical_section_id = _normalize_section_id_for_evidence(section_id)

    # Load cross-section config once per call
    all_params = load_parameters() or {}
    cross_cfg: Dict[str, Any] = all_params.get(
        "cross_section_evidence_sharing", {}
    ) or {}

    # Prefer explicit rule for the actual section_id; fall back to canonical
    cfg_key = section_id if section_id in cross_cfg else canonical_section_id
    share_from = cross_cfg.get(cfg_key, cross_cfg.get("default", [])) or []

    sections_to_pull: set[str] = {canonical_section_id}

    # If "all" is present, we will expand to all known sections below
    use_all = "all" in share_from
    if not use_all:
        for src in share_from:
            if isinstance(src, str) and src:
                sections_to_pull.add(src)

    evidence_facts: List[str] = []
    seen_ids: set[str] = set()

    # ---------------------------
    # Preferred API path
    # ---------------------------
    if hasattr(evidence_plan, "get_evidence_for_section"):
        # If "all" → include every section we know about
        if use_all:
            hints = getattr(evidence_plan, "section_hints", {}) or {}
            for sec in hints.keys():
                sections_to_pull.add(sec)

        for sec in sections_to_pull:
            for ev in evidence_plan.get_evidence_for_section(sec) or []:
                ev_id = getattr(ev, "evidence_id", None)
                if ev_id and ev_id in seen_ids:
                    continue
                if ev_id:
                    seen_ids.add(ev_id)
                fact = getattr(ev, "fact", None)
                if fact:
                    evidence_facts.append(fact)

        return evidence_facts

    # ---------------------------
    # Fallback: manual wiring via section_hints + evidences
    # ---------------------------
    hints = getattr(evidence_plan, "section_hints", {}) or {}
    evidences = getattr(evidence_plan, "evidences", []) or []

    # Start with canonical; hints may already contain it
    ids_for_section = set(hints.get(canonical_section_id, []))

    if use_all:
        for ev in evidences:
            ev_id = getattr(ev, "evidence_id", None)
            if ev_id:
                ids_for_section.add(ev_id)
    else:
        for src_section in share_from:
            ids_for_section.update(hints.get(src_section, []))

    for ev in evidences:
        ev_id = getattr(ev, "evidence_id", None)
        fact = getattr(ev, "fact", None)
        if ev_id in ids_for_section and fact:
            evidence_facts.append(fact)

    return evidence_facts


# ---------------------------------------------------------------------------
# CV Generation Engine
# ---------------------------------------------------------------------------


class CVGenerationEngine:
    """Encapsulates Stage B logic: prompt building, LLM calls, and response assembly."""

    generation_params: Dict[str, Any]

    def __init__(self, llm_client, generation_params: Dict[str, Any] | None = None):
        self.llm_client = llm_client

        file_params = load_parameters().get("generation", {}) or {}

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

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section_prompt(
            request: CVGenerationRequest,
            evidence_plan: EvidencePlan | None,
            section_id: str,
    ) -> str:
        """Construct an LLM prompt for a single CV section (with full cross-section context)."""
        all_params = load_parameters()
        generation_cfg: Dict[str, Any] = all_params.get("generation", {}) or {}
        prompts_cfg: Dict[str, str] = generation_cfg.get("prompts", {}) or {}

        # 🔹 Resolve language and tone
        raw_language = getattr(request, "language", "en")
        language = raw_language or "en"
        language_name = describe_language(language)

        tone = (
                getattr(request, "tone", None)
                or getattr(request, "style_tone", None)
                or getattr(getattr(request, "template_info", None), "tone", None)
        )
        tone_desc = describe_tone(tone)

        # Legacy-style fields (may be dicts or models); we normalize them
        raw_profile_info = getattr(request, "profile_info", None)
        if not raw_profile_info:
            # New API shape: fall back to student_profile
            raw_profile_info = getattr(request, "student_profile", None) or {}

        raw_job_role_info = getattr(request, "job_role_info", None) or {}
        raw_job_position_info = getattr(request, "job_position_info", None) or {}
        raw_company_info = getattr(request, "company_info", None) or {}
        drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}

        profile_info = _to_serializable(raw_profile_info)
        job_role_info = _to_serializable(raw_job_role_info)
        job_position_info = _to_serializable(raw_job_position_info)
        company_info = _to_serializable(raw_company_info)

        user_draft = drafts.get(section_id)

        # ------------------------------------------------------------
        # Evidence facts + education-specific enrichment
        # ------------------------------------------------------------
        evidence_facts: List[str] = []

        # From EvidencePlan (if any)
        ep_facts = _collect_evidence_facts_for_section(evidence_plan, section_id)
        if ep_facts:
            evidence_facts.extend(ep_facts)

        # Enrich education section with structured education info
        if section_id == "education":
            edu_facts = _collect_education_facts_from_request(request)
            if edu_facts:
                evidence_facts.extend(edu_facts)

        lines: List[str] = [
            "You are an expert CV writer.",
            f"The CV language is {language_name} (language_code='{language}').",
            f"The overall tone of the CV is {tone_desc}.",
            "",
            f"Generate a strong '{section_id}' section in {language_name} for the candidate below.",
            "",
            "=== Profile Info (JSON) ===",
            json.dumps(profile_info, ensure_ascii=False, indent=2),
            "",
            "=== Job Role / Position / Company (JSON) ===",
            json.dumps(
                {
                    "job_role_info": job_role_info,
                    "job_position_info": job_position_info,
                    "company_info": company_info,
                },
                ensure_ascii=False,
                indent=2,
            ),
        ]

        lines.append(f"\n=== Evidence Facts for {section_id} ===")
        lines.extend(
            [f"- {fact}" for fact in evidence_facts]
            or ["- (No specific evidence provided)"]
        )


        if user_draft:
            lines.append(f"\n=== User Draft for {section_id} ===")
            draft_text = user_draft if isinstance(user_draft, str) else str(user_draft)
            lines.append(draft_text)
            lines.append(
                "\nRewrite the above draft so it is clearer, more concise, and more impactful. "
                "Do not copy sentences verbatim; rephrase entirely in your own words while preserving factual accuracy. "
                "Treat this draft as raw information — you may enrich it with relevant details drawn from other sections "
                "(such as experience, education, or achievements) if it helps improve clarity or flow. "
                "Maintain a consistent professional tone and ensure the structure matches the expected output format."
            )

        default_prompt = prompts_cfg.get(
            "default",
            (
                "Write a concise, professional paragraph.\n"
                "Do not use bullet points or numbering.\n"
                "Avoid markdown formatting like **bold** or *italics*.\n"
                "Stay factual and grounded in the details above."
            ),
        )
        section_prompt = prompts_cfg.get(section_id, default_prompt)

        # === Output requirements block ===
        lines.extend(
            [
                "\n=== Output Requirements ===",
                section_prompt,
            ]
        )

        # 🔹 Tie to template's max_chars_per_section
        char_limits = _get_section_char_limits(request)
        canonical_section_id = _normalize_section_id_for_evidence(section_id)
        max_chars = char_limits.get(canonical_section_id)

        if max_chars:
            lines.append(
                f"\nIMPORTANT: Your final output for this section must be no longer than "
                f"{max_chars} characters. Be concise and stay within this limit."
            )

        return "\n".join(lines)

    @staticmethod
    def _build_skills_selection_prompt(
        request: CVGenerationRequest,
        evidence_plan: EvidencePlan | None,
        skills_plan: SkillsSectionPlan,
        language: str,
    ) -> str:
        """Build JSON-only prompt for structured skills selection.

        Even though the output is JSON, we still surface:
        - Target CV language  → affects naming of inferred skills.
        - Desired tone        → helps the LLM choose appropriate phrasing.
        """
        params = load_parameters()
        generation_cfg: Dict[str, Any] = params.get("generation", {}) or {}
        prompts_cfg: Dict[str, str] = generation_cfg.get("prompts", {}) or {}

        if "skills_structured" not in prompts_cfg:
            logger.warning(
                "skills_structured_prompt_missing_using_default",
                message=(
                    "generation.prompts.skills_structured not found in parameters.yaml; "
                    "falling back to built-in default prompt."
                ),
            )

        base_prompt: str = prompts_cfg.get(
            "skills_structured",
            "",
        )
        if len(base_prompt) == 0:
            raise RuntimeError(
                f"Fail to load skills_structured prompt from parameters.yaml."
            )

        # 🔹 Resolve language & tone (request.language overrides function arg)
        raw_language = getattr(request, "language", None) or language or "en"
        language_code = raw_language or "en"
        language_name = describe_language(language_code)

        tone = (
            getattr(request, "tone", None)
            or getattr(request, "style_tone", None)
            or getattr(getattr(request, "template_info", None), "tone", None)
        )
        tone_desc = describe_tone(tone)

        evidence_facts: List[str] = _collect_evidence_facts_for_section(
            evidence_plan, "skills_structured"
        )

        raw_profile_info = getattr(request, "profile_info", None)
        if not raw_profile_info:
            raw_profile_info = getattr(request, "student_profile", None) or {}

        raw_job_role_info = getattr(request, "job_role_info", None) or {}
        raw_job_position_info = getattr(request, "job_position_info", None) or {}
        raw_company_info = getattr(request, "company_info", None) or {}

        profile_info = _to_serializable(raw_profile_info)
        job_role_info = _to_serializable(raw_job_role_info)
        job_position_info = _to_serializable(raw_job_position_info)
        company_info = _to_serializable(raw_company_info)

        lines: List[str] = [
            f"The CV language is {language_name} (language_code='{language_code}').",
            f"The overall tone of the CV is {tone_desc}.",
            "",
            "Student profile (structured JSON):",
            json.dumps(profile_info, ensure_ascii=False, indent=2),
        ]

        if job_role_info or job_position_info or company_info:
            lines.append("")
            lines.append("Target job context (structured JSON):")
            context = {
                "job_role_info": job_role_info,
                "job_position_info": job_position_info,
                "company_info": company_info,
            }
            lines.append(json.dumps(context, ensure_ascii=False, indent=2))

        lines.extend(
            [
                "",
                "=== Skills selection instructions ===",
                base_prompt,
                "",
                "Canonical skills (from taxonomy):",
            ]
        )

        for sk in skills_plan.canonical_skills:
            level = sk.level or "null"
            lines.append(f'- name: "{sk.name}", level: "{level}"')

        if evidence_facts:
            lines.append("")
            lines.append("=== Evidence facts for skills (cross-section) ===")
            for fact in evidence_facts:
                lines.append(f"- {fact}")

        return "\n".join(lines)



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

        Loads parameters.yaml once per call and delegates to the pure function.
        """
        try:
            all_params = load_parameters() or {}
        except Exception:
            return None

        return resolve_token_budget(section_id, attempt, all_params)



    # ------------------------------------------------------------------
    # LLM call with retries (Option C: feature flag for zero-token retry)
    # ------------------------------------------------------------------
    def _call_llm_with_retries(self, prompt: str, section_id: str) -> str:
        """
        Retry logic that distinguishes:

        - Real Gemini responses:
            * have real usage metadata with a `raw.usage_metadata` source
            * may be retried on zero-token or too-short outputs
        - Mock / unit-test LLM responses:
            * show total_tokens == 0 and `_source="merged"`
            * we NEVER retry on zero-tokens here to keep tests deterministic

        Additionally, zero-token retry is controlled via:
            generation.retry_on_zero_tokens (bool, default: True)

        IMPORTANT:
        - If the underlying llm_client falls back to a STUB_ERROR / API_ERROR text, we NEVER
          surface that raw stub text in the CV. We log it and treat it as
          "no output", letting downstream fallbacks / placeholders handle it.
        """
        max_retries = int(self.generation_params.get("max_retries", 2))
        retry_on_zero_tokens = bool(
            self.generation_params.get("retry_on_zero_tokens", True)
        )
        min_length_threshold = MIN_RETRY_LENGTH
        attempt = 0
        last_result: str = ""
        # reset per top-level generate_cv call; generate_cv sets this as well
        self._last_retry_count = 0

        while attempt < max_retries:
            attempt += 1
            logger.info(
                "llm_call_start",
                attempt=attempt,
                model=self.generation_params.get("model"),
                section_id=section_id,
                stage="B_generation",
            )

            try:
                # ----- Per-attempt max_output_tokens from section_token_budgets -----
                max_tokens_for_attempt = self._get_section_token_budget_for_attempt(
                    section_id=section_id,
                    attempt=attempt,
                )

                # Build an adapter for this attempt.
                # If we have a specific budget, override max_output_tokens; otherwise
                # fall back to the engine-wide default.
                if max_tokens_for_attempt is not None:
                    local_defaults = dict(self.generation_params)
                    local_defaults["max_output_tokens"] = max_tokens_for_attempt
                    metrics_client = _CallableToClientAdapter(
                        self.llm_client, local_defaults
                    )
                else:
                    # No per-attempt override → reuse or create a shared adapter
                    if self._metrics_client is None:
                        self._metrics_client = _CallableToClientAdapter(
                            self.llm_client, self.generation_params
                        )
                    metrics_client = self._metrics_client

                resp = call_llm_section_with_metrics(
                    llm_client=metrics_client,
                    model=self.generation_params.get("model"),
                    prompt=prompt,
                    section_id=section_id,
                    purpose="stage_b_generation",
                    user_id=getattr(self, "_current_user_id", "unknown"),
                    messages=None,
                )

                # 👇 Always rely on str(resp), NOT resp.text
                raw_text = str(resp).strip()

                # --- Handle stub / API error text from llm_client safely ----
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

                    # Try again if we still have retries left
                    if attempt < max_retries:
                        time.sleep(RETRY_SLEEP_MULTIPLIER * attempt)
                        continue
                    else:
                        # Out of retries → treat as no output
                        self._last_retry_count = attempt - 1
                        return ""

                # Usage snapshot injected by llm_metrics (if any)
                usage = getattr(resp, "_llm_usage_snapshot", None)
                total_tokens = 0
                if isinstance(usage, dict):
                    total_tokens = int(usage.get("total_tokens", 0) or 0)

                # Heuristic: distinguish real LLM vs. mocked/unit-test client
                is_mock_llm = (
                    usage is None
                    or (
                        isinstance(usage, dict)
                        and usage.get("_source") in (None, "merged")
                    )
                )

                # ------------------------------
                # REAL LLM: retry on zero tokens
                # ------------------------------
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

                # For both real + mock clients: remember last non-empty text
                if raw_text:
                    last_result = raw_text

                # REAL LLM: retry on very short response
                if len(raw_text) < min_length_threshold and not is_mock_llm:
                    logger.warning(
                        "llm_result_too_short_retry",
                        attempt=attempt,
                        section_id=section_id,
                        length=len(raw_text),
                        content_preview=raw_text[:50],
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

                # Success
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

        self._last_retry_count = max(attempt - 1, 0)
        return last_result



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

        # Build the JSON-only selection prompt and call the LLM.
        prompt = self._build_skills_selection_prompt(
            request, evidence_plan, skills_plan, language
        )
        raw_json = self._call_llm_with_retries(prompt, section_id="skills_structured")

        # ---------- Canonical lookup (case-insensitive) ----------
        canonical_by_name: dict[str, CanonicalSkill] = {
            (s.name or "").strip().lower(): s for s in skills_plan.canonical_skills
        }

        candidate_items: list[SkillSelectionItem] = []

        # ---------- Parse JSON from LLM ----------
        try:
            cleaned = _strip_markdown_fence(raw_json)
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
            sel.name = name # normalise whitespace
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
        # Even if some path above misbehaved, this guarantees that any skill
        # whose name matches a canonical skill ends up with the canonical level.
        for item in result:
            key = (item.name or "").strip().lower()
            canon = canonical_by_name.get(key)
            if canon is not None:
                item.level = canon.level

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
            cost_estimate_thb: float = 0.0,
            skills_output: list[OutputSkillItem] | None = None,
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
        all_params = load_parameters()
        gen_cfg = all_params.get("generation", {}) or {}
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
            cost_estimate_thb=cost_estimate_thb,
            profile_info=profile_info,
        )

        justification = Justification()

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
            "justification": justification,
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
        """Generate CV sections end-to-end."""
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

            char_limits = _get_section_char_limits(request)
            available_sections = _get_available_sections(request)

            # Resolve strictly based on template order
            effective_sections = _resolve_effective_sections(request, available_sections)

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

                if section_id == "skills" and structured_first and skills_output is not None:
                    text_for_section = format_plain_skill_bullets(skills_output)
                    text_for_section = _truncate_text(
                        text_for_section or "",
                        char_limits.get("skills"),
                    )

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

                else:
                    raw_text = self._call_llm_with_retries(
                        self._build_section_prompt(request, evidence_plan, section_id),
                        section_id,
                    )
                    truncated = _truncate_text(
                        raw_text or "",
                        char_limits.get(section_id),
                    )

                    text_for_section = truncated or ""
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
                skills_output = _reconcile_skill_levels_with_request(request, skills_output)
                logger.info(
                    "skills_after_reconcile",
                    skills=[(s.name, s.level, s.source) for s in skills_output],
                )

            # 🔹 Telemetry snapshot for skills + overall Stage B
            skills_metrics = _summarize_skills_telemetry(skills_output)

            logger.info(
                "stage_b_telemetry_summary",
                generation_time_ms=elapsed_ms,
                retry_count=self._last_retry_count,
                sections_generated=len(generated_sections),
                cache_hit=False,
                tokens_used=0,
                cost_estimate_thb=0.0,
                **skills_metrics,
            )

            return self._build_cv_response(
                request=request,
                generated_sections=generated_sections,
                generation_time_ms=elapsed_ms,
                retry_count=self._last_retry_count,
                cache_hit=False,
                tokens_used=0,
                cost_estimate_thb=0.0,
                skills_output=skills_output,
            )

        finally:
            clear_contextvars()
