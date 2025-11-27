from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Tuple
from types import SimpleNamespace

import re

import structlog

from schemas.output_schema import CVGenerationResponse, SectionContent, OutputSkillItem
from functions.utils.llm_client import load_parameters
from functions.utils.claims import (
    build_empty_justification,
    validate_justification_against_text,
)
from functions.utils.safe_truncate import smart_truncate_markdown

logger = structlog.get_logger(__name__).bind(module="stage_c_validation")


class StageCValidationError(Exception):
    """Fatal validation error for Stage C validation."""


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_validation_params() -> Dict[str, Any]:
    """
    Load validation-related config from parameters.yaml via load_parameters().

    Expected structure in parameters.yaml (optional):

    validation:
      min_skills_required: 3
      min_education_required: 1
      require_email: true
      require_name: true
      max_section_chars_default: 1500
      drop_empty_sections: true
      enable_safety_cleaning: true
      max_skills: 50
      strict_mode: false

    security:
      critical_patterns: [...]
      suspicious_patterns: [...]
    """
    params = load_parameters() or {}

    cfg = params.get("validation") or {}
    if not isinstance(cfg, dict):
        cfg = {}

    # Defaults for validation behaviour
    cfg.setdefault("min_skills_required", 0)
    cfg.setdefault("min_education_required", 0)
    cfg.setdefault("require_email", False)
    cfg.setdefault("require_name", False)
    cfg.setdefault("max_section_chars_default", 1500)
    cfg.setdefault("drop_empty_sections", True)
    cfg.setdefault("enable_safety_cleaning", True)
    cfg.setdefault("max_skills", 50)
    cfg.setdefault("strict_mode", False)

    # Attach security config (if present) so Stage C can use it as a backstop
    security_cfg = params.get("security") or {}
    if isinstance(security_cfg, dict):
        cfg.setdefault("_security_cfg", security_cfg)

    # Attach truncation config (if present) for Stage C length control
    trunc_cfg = params.get("truncation_config") or {}
    if isinstance(trunc_cfg, dict):
        cfg.setdefault("_truncation_cfg", trunc_cfg)

    return cfg

def _fallback_skills_from_request(original_request: Any) -> List[OutputSkillItem]:
    """
    Last-resort fallback: reconstruct skills list from the original request
    if Stage B returned zero skills.

    Supports both:
    - Legacy shape: original_request.profile_info["skills"]
    - New API shape: original_request.student_profile.skills
    """
    skills_out: List[OutputSkillItem] = []

    # 1) Legacy profile_info dict
    profile_info = getattr(original_request, "profile_info", None)
    if isinstance(profile_info, dict):
        raw_skills = profile_info.get("skills") or []
        if isinstance(raw_skills, list):
            for item in raw_skills:
                if isinstance(item, dict):
                    name = item.get("name")
                    level = item.get("level")
                else:
                    name = str(item)
                    level = None
                if not name:
                    continue
                skills_out.append(
                    OutputSkillItem(
                        name=name[:100],
                        level=str(level) if level is not None else None,
                        source="profile_info",
                    )
                )

    # 2) New API shape: student_profile.skills
    student_profile = getattr(original_request, "student_profile", None)
    if student_profile is not None:
        raw_sp_skills = getattr(student_profile, "skills", []) or []
        for sp_skill in raw_sp_skills:
            name = getattr(sp_skill, "name", None)
            if not name:
                continue
            level = getattr(sp_skill, "level", None)
            # Enum → value
            if hasattr(level, "value"):
                level = level.value
            skills_out.append(
                OutputSkillItem(
                    name=str(name)[:100],
                    level=str(level) if level is not None else None,
                    source="student_profile",
                )
            )

    # Deduplicate by name (case-insensitive)
    deduped: List[OutputSkillItem] = []
    seen: set[str] = set()
    for sk in skills_out:
        key = (sk.name or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(sk)

    return deduped


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_stage_c_validation(
    response: CVGenerationResponse,
    *,
    template_info: Dict[str, Any] | Any | None = None,
    original_request: Any | None = None,
) -> CVGenerationResponse:
    """
    Stage C: post-LLM validation & sanitization.

    - Enforces section set & order according to template_info
    - Cleans each section's text (whitespace, injection backstop, markdown artifacts)
    - Validates justification against FULL (pre-truncation) section text
    - Applies smart truncation AFTER justification (JSON-safe, markdown-aware)
    - Normalizes and deduplicates skills (aligned with Stage B canonical logic)
    - Runs global consistency checks:
        * name/email present (optional)
        * min skills / education thresholds
        * cross-checks vs original_request (template_id, job_id, language)
    - Optionally enforces strict_mode via StageCValidationError
    """
    cfg = _load_validation_params()
    strict_mode = bool(cfg.get("strict_mode", False))

    # 1) Sanity-check / ensure we have a CVGenerationResponse instance
    resp = _ensure_cv_response(response)

    # 2) Resolve + normalize template_info (supports legacy + new request shapes)
    resolved_template_info = _resolve_template_info(
        explicit_template_info=template_info,
        original_request=original_request,
    )
    tmpl_dict = _normalize_template_info(resolved_template_info)
    allowed_sections = _compute_allowed_sections(tmpl_dict)

    # 3a) CLEAN sections (injection, markdown cleanup, drop-empty) – NO truncation
    sections = getattr(resp, "sections", None) or {}
    cleaned_sections = _clean_sections_no_truncate(
        sections=sections,
        allowed_sections=allowed_sections,
        template_info=tmpl_dict,
        cfg=cfg,
        resp=resp,
    )
    resp.sections = cleaned_sections

    # 4) Sanitize skills (normalize + dedup + cap)
    skills = getattr(resp, "skills", None)
    resp.skills = _sanitize_skills(skills, cfg, resp)

    # 5) Normalize metadata (best-effort, non-breaking)
    _normalize_metadata(resp)

    # 6) Global consistency checks (name/email, min skills/education, cross-checks)
    _run_global_consistency_checks(
        resp=resp,
        template_info=tmpl_dict,
        cfg=cfg,
        original_request=original_request,
    )

    # 7) Ensure justification object is present, migrated, and VALIDATED
    try:
        from schemas.output_schema import Justification
        from functions.utils.claims import (
            migrate_legacy_justification_schema,
            should_require_justification,
        )

        # Load generation config so we can see which sections actually require justification
        params_full = load_parameters() or {}
        generation_cfg = params_full.get("generation") or {}

        justification = getattr(resp, "justification", None)
        if justification is None:
            justification = build_empty_justification()

        # Detect "empty" justification → candidate for reconstruction
        is_empty = (
            len(getattr(justification, "evidence_map", []) or []) == 0
            and len(getattr(justification, "unsupported_claims", []) or []) == 0
            and getattr(justification, "total_claims_analyzed", 0) == 0
        )

        # Use FULL, pre-truncation section text for coverage validation
        all_section_text = "\n\n".join(
            (sec.text or "") for sec in (resp.sections or {}).values()
        )

        # ------------------------------------------------------------------
        # Legacy → new schema migration (ONLY if Stage B gave us nothing)
        # ------------------------------------------------------------------
        if is_empty:
            legacy_union: dict[str, Any] = {
                "evidence_map": [],
                "unsupported_claims": [],
            }

            # Reconstruct per-section, but ONLY for sections that require justification.
            for section_id, sec in (resp.sections or {}).items():
                if not should_require_justification(section_id, generation_cfg):
                    continue

                sec_text = (sec.text or "").strip()
                if not sec_text:
                    continue

                # Build a simple legacy-style dict for THIS section only
                legacy_base = {
                    "evidence_map": [
                        {
                            "claim": sent.strip(),
                            "source": "profile_info",
                        }
                        for sent in re.split(r"[.!?]\s+", sec_text)
                        if sent.strip()
                    ],
                    "unsupported_claims": [],
                }

                migrated = migrate_legacy_justification_schema(
                    legacy_base,
                    section_id=section_id,
                    section_text=sec_text,
                )

                for ev in migrated.get("evidence_map") or []:
                    legacy_union["evidence_map"].append(ev)

            if legacy_union["evidence_map"]:
                justification = Justification.model_validate(legacy_union)
            else:
                justification = build_empty_justification()

        # ---------------------------------------------------------------
        # Real validation against FULL pre-truncation section text
        # ---------------------------------------------------------------
        justification = validate_justification_against_text(
            justification,
            all_section_text,
        )
        resp.justification = justification

    except Exception as exc:
        logger.warning("stage_c_justification_failed", error=str(exc))

    # 8) NOW apply smart truncation (JSON-safe, markdown-aware)
    resp.sections = _truncate_sections_post_justification(
        sections=resp.sections or {},
        template_info=tmpl_dict,
        cfg=cfg,
        resp=resp,
    )

    # 9) Strict mode: if we ended with no sections at all, consider this fatal
    if strict_mode and not resp.sections:
        msg = "Stage C removed all sections; failing in strict_mode."
        logger.error("stage_c_all_sections_removed_strict", message=msg)
        raise StageCValidationError(msg)

    logger.info(
        "stage_c_validation_completed",
        num_sections=len(resp.sections or {}),
        num_skills=len(resp.skills or []) if getattr(resp, "skills", None) else 0,
    )
    # ----------------------------------------------------------------------
    # JD Skill Matching (deterministic, non-mutating to text or skills list)
    # ----------------------------------------------------------------------
    try:
        from functions.utils.jd_matching import (
            extract_canonical_jd_required_skills,
            annotate_matched_jd_skills,
        )
        jd_required = extract_canonical_jd_required_skills(original_request)
        resp = annotate_matched_jd_skills(resp, jd_required_skills=jd_required)
    except Exception as e:
        logger.warning("stage_c_jd_matching_failed", error=str(e))

    return resp



# ---------------------------------------------------------------------------
# Internal helpers: response / template handling
# ---------------------------------------------------------------------------


def _ensure_cv_response(response: Any) -> CVGenerationResponse:
    """
    Ensure we are working with a CVGenerationResponse instance.

    We expect Stage B to already return this type; if not, we attempt a
    best-effort coercion via model_construct / from_orm / direct init.
    """
    if isinstance(response, CVGenerationResponse):
        return response

    # Try Pydantic v2-style model_construct
    if hasattr(CVGenerationResponse, "model_construct"):
        logger.warning("stage_c_coercing_response_via_model_construct")
        return CVGenerationResponse.model_construct(**dict(response))  # type: ignore[arg-type]

    # Fallback: assume dict-like and pass to constructor
    try:
        logger.warning("stage_c_coercing_response_via_init")
        return CVGenerationResponse(**dict(response))  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - hard failure path
        logger.error("stage_c_response_coercion_failed", error=str(exc))
        raise StageCValidationError("Cannot coerce Stage B output into CVGenerationResponse") from exc


def _resolve_template_info(
    *,
    explicit_template_info: Any | None,
    original_request: Any | None,
) -> Any | None:
    """
    Resolve template_info in a way that supports both legacy and new request shapes.

    Priority:
    1) If explicit_template_info is passed to Stage C, use it as-is.
    2) Else, if original_request has .template_info (legacy shape), use that.
    3) Else, if original_request looks like the new CVGenerationRequest shape,
       synthesize a SimpleNamespace(template_id, sections, language, max_chars_per_section).

    If nothing is resolvable, return None and Stage C will behave with an empty template.
    """
    # 1) Explicit override wins
    if explicit_template_info is not None:
        return explicit_template_info

    # 2) Legacy shape: request.template_info
    if original_request is not None and hasattr(original_request, "template_info"):
        try:
            return getattr(original_request, "template_info")
        except AttributeError:
            # Very defensive; should not normally happen
            pass

    # 3) New shape (CVGenerationRequest) – synthesize from top-level fields
    if original_request is not None:
        template_id = getattr(original_request, "template_id", None)
        sections = getattr(original_request, "sections", None)
        language = getattr(original_request, "language", None)
        max_chars_per_section = getattr(original_request, "max_chars_per_section", None)

        # If we don't even have template_id or sections, nothing useful to synthesize
        if template_id is not None or sections:
            # If language is an Enum (e.g. LanguageEnum), prefer its .value
            lang_value = getattr(language, "value", language)
            return SimpleNamespace(
                template_id=template_id,
                sections=sections,
                language=lang_value,
                max_chars_per_section=max_chars_per_section,
            )

    # 4) Nothing to resolve
    return None


def _normalize_template_info(template_info: Any | None) -> Dict[str, Any]:
    """
    Normalize template_info to a plain dict for ease of use.

    Accepts:
    - dict
    - Pydantic models with model_dump() / dict()
    - arbitrary objects with attributes
    """
    if template_info is None:
        return {}

    if isinstance(template_info, dict):
        return template_info

    # Pydantic v2
    if hasattr(template_info, "model_dump"):
        try:
            return template_info.model_dump()
        except Exception:
            pass

    # Pydantic v1
    if hasattr(template_info, "dict"):
        try:
            return template_info.dict()
        except Exception:
            pass

    # Fallback: shallow attribute extraction
    result: Dict[str, Any] = {}
    for key in ("template_id", "sections_order", "sections", "max_chars_per_section", "language"):
        if hasattr(template_info, key):
            result[key] = getattr(template_info, key)
    return result


def _compute_allowed_sections(template_info: Dict[str, Any]) -> List[str]:
    """
    Determine allowed section IDs, following the same rule as the
    test helper _compute_sections_from_template_and_user:

    - Use ONLY the order defined in template_info["sections_order"]
      (or fallback to template_info["sections"]).
    - Do NOT append extra user-only sections here.
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
            section_id: Optional[str]
            if isinstance(item, str):
                section_id = item
            elif isinstance(item, dict):
                section_id = item.get("id")
            else:
                section_id = None

            if not section_id:
                continue

            if section_id not in seen:
                seen.add(section_id)
                result.append(section_id)

    return result


# ---------------------------------------------------------------------------
# Internal helpers: section sanitization
# ---------------------------------------------------------------------------


def _matches_any_pattern(text: str, patterns: Iterable[str]) -> bool:
    """Return True if `text` matches any regex pattern (case-insensitive)."""
    for pattern in patterns:
        try:
            if re.search(pattern, text, flags=re.IGNORECASE):
                return True
        except re.error:
            # Ignore invalid regex patterns
            continue
    return False


def _strip_leading_markdown_headers(text: str) -> str:
    """
    Remove leading markdown headings like '#', '##', '###' and pure bold
    heading lines such as '**References**' or '**บุคคลอ้างอิง**' that
    the LLM sometimes adds at the very top of a section.
    """
    if not text:
        return text

    lines = text.splitlines()
    idx = 0

    while idx < len(lines):
        line = lines[idx].strip()
        if not line:
            idx += 1
            continue

        # '# Heading', '## Heading', etc.
        if line.startswith("#"):
            idx += 1
            continue

        # Pure bold heading line: '**something**'
        if line.startswith("**") and line.endswith("**") and len(line) > 4:
            idx += 1
            continue

        break

    return "\n".join(lines[idx:]).lstrip("\n")


def _strip_markdown_bullet_prefixes(text: str) -> str:
    """
    For sections where the Jinja template already renders bullets,
    strip markdown-style bullet prefixes ('* ' or '- ') to avoid
    nested bullets like '* หลักสูตร ...'.
    """
    if not text:
        return text

    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.lstrip()
        if line.startswith("* ") or line.startswith("- "):
            line = line[2:]
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _clean_section_markdown(section_id: str, text: str) -> str:
    """
    Apply markdown cleanup rules per section.

    - Always remove LLM-added leading headings / bold headings.
    - For sections where the template already provides bullets,
      remove explicit bullet prefixes.
    """
    if not text:
        return text

    text = _strip_leading_markdown_headers(text)

    # These sections are rendered as bullet lists by the template;
    # we don't want extra '* ' / '- ' from the LLM.
    if section_id in {"publications", "training", "references", "additional_info"}:
        text = _strip_markdown_bullet_prefixes(text)

    return text

def _clean_sections_no_truncate(
    *,
    sections: Dict[str, Any],
    allowed_sections: List[str],
    template_info: Dict[str, Any],
    cfg: Dict[str, Any],
    resp: CVGenerationResponse,
) -> Dict[str, SectionContent]:
    """
    Clean sections WITHOUT applying any length truncation.

    Responsibilities:
    - Keep only sections that are allowed by template_info (for ordering),
      but NEVER silently drop sections that Stage B already generated.
    - Enforce deterministic ordering (template-first, then any extra sections).
    - Run injection backstop using security patterns and drop critical sections.
    - Normalize markdown artifacts (headings, extra bullets).
    - Basic text cleanup.
    - Drop sections that are truly empty after cleaning (if configured).
    """
    cleaned: Dict[str, SectionContent] = {}

    default_max = int(cfg.get("max_section_chars_default", 1500))
    drop_empty = bool(cfg.get("drop_empty_sections", True))
    enable_cleaning = bool(cfg.get("enable_safety_cleaning", True))
    strict_mode = bool(cfg.get("strict_mode", False))

    # Optional security backstop
    security_cfg = cfg.get("_security_cfg") or {}
    if not isinstance(security_cfg, dict):
        security_cfg = {}
    critical_patterns = security_cfg.get("critical_patterns") or []
    suspicious_patterns = security_cfg.get("suspicious_patterns") or []

    # Decide which section IDs to process:
    if allowed_sections:
        section_ids: List[str] = [sid for sid in allowed_sections if sid in sections]
        for sid in sections.keys():
            if sid not in section_ids:
                section_ids.append(sid)
    else:
        section_ids = list(sections.keys())

    for section_id in section_ids:
        if section_id not in sections:
            continue

        raw_val = sections[section_id]
        sec_obj = _ensure_section_content(raw_val)

        text = getattr(sec_obj, "text", "")
        if not isinstance(text, str):
            text = str(text)

        orig_text = text

        # Injection backstop BEFORE further cleaning
        if text and critical_patterns:
            if _matches_any_pattern(text, critical_patterns):
                msg = (
                    f"Critical injection pattern detected in section '{section_id}'. "
                    "Section removed by Stage C."
                )
                _append_metadata_warning(resp, msg)
                logger.warning("stage_c_section_dropped_injection", section_id=section_id)
                if strict_mode:
                    raise StageCValidationError(msg)
                # Drop this section in non-strict mode
                continue

        if text and suspicious_patterns:
            if _matches_any_pattern(text, suspicious_patterns):
                msg = f"Suspicious pattern detected in section '{section_id}'. Content kept but flagged."
                _append_metadata_warning(resp, msg)
                logger.warning("stage_c_section_suspicious", section_id=section_id)

        # Normalize markdown artifacts (headings, extra bullets)
        text = _clean_section_markdown(section_id, text)

        # Basic cleaning
        text = _clean_text(text, enable_cleaning)

        # Section-specific cleanup: drop bullets that are just headings
        if section_id in ("training", "publications"):
            lines = text.splitlines()
            cleaned_lines: list[str] = []
            for line in lines:
                stripped = line.strip()
                core = stripped.lstrip("-•").strip().lower()
                if core in {
                    "training",
                    "publications",
                    "ผลงานตีพิมพ์",
                    "การฝึกอบรม",
                }:
                    continue
                cleaned_lines.append(line)
            text = "\n".join(cleaned_lines).strip()
            text = _strip_leading_section_heading(section_id, text)

        # Only drop sections that are TRULY empty after cleaning
        if not text.strip() and drop_empty:
            _append_metadata_warning(
                resp,
                f"Section '{section_id}' removed as empty after cleaning.",
            )
            logger.info("stage_c_section_dropped_empty", section_id=section_id)
            continue

        if text != orig_text:
            sec_obj.text = text
            if hasattr(sec_obj, "word_count"):
                try:
                    sec_obj.word_count = len(text.split())
                except Exception:
                    pass

        cleaned[section_id] = sec_obj

    return cleaned

def _truncate_sections_post_justification(
    *,
    sections: Dict[str, SectionContent],
    template_info: Dict[str, Any],
    cfg: Dict[str, Any],
    resp: CVGenerationResponse,
) -> Dict[str, SectionContent]:
    """
    Apply smart, JSON-safe truncation AFTER justification has been validated.

    - Respects per-section max_chars_per_section from template_info.
    - Uses smart_truncate_markdown (sentence-aware, markdown-aware, JSON-protected).
    - Logs truncation and attaches warnings when truncation is applied.
    - If truncation is disabled in truncation_config, falls back to naive cut.
    """
    if not sections:
        return sections

    # Per-section max limits as in Stage B
    per_section_limits: Dict[str, int] = {}
    tmpl_limits = template_info.get("max_chars_per_section") or {}
    if isinstance(tmpl_limits, dict):
        for k, v in tmpl_limits.items():
            try:
                per_section_limits[str(k)] = int(v)
            except Exception:
                continue

    default_max = int(cfg.get("max_section_chars_default", 1500))

    # Truncation config
    trunc_cfg = cfg.get("_truncation_cfg") or {}
    if not isinstance(trunc_cfg, dict):
        trunc_cfg = {}
    truncation_enable = bool(trunc_cfg.get("truncation_enable", True))
    overflow_limit = int(trunc_cfg.get("overflow_limit", 64))
    reduction_limit = float(trunc_cfg.get("reduction_limit", 0.15))

    # Language hint for sentence boundaries
    lang_norm = _normalize_language(getattr(resp, "language", None)) or "en"
    lang_short: str = "th" if lang_norm.startswith("th") else "en"

    out: Dict[str, SectionContent] = {}

    for section_id, sec_obj in sections.items():
        text = getattr(sec_obj, "text", "")
        if not isinstance(text, str):
            text = str(text)

        original_text = text
        max_len = per_section_limits.get(section_id, default_max)

        if max_len > 0 and len(text) > max_len:
            if truncation_enable:
                trunc_result = smart_truncate_markdown(
                    text=text,
                    max_len=max_len,
                    lang=lang_short,  # type: ignore[arg-type]
                    overflow_limit=overflow_limit,
                    reduction_limit=reduction_limit,
                    section_id=section_id,
                )

                truncated_text = trunc_result.get("safe_text", text)
                trunc_applied = bool(trunc_result.get("truncation_applied", False))
                strategy_used = trunc_result.get("strategy_used", "none")

                # Only log when real truncation happened (JSON path will return trunc_applied=False)
                if trunc_applied and truncated_text != text:
                    _append_metadata_warning(
                        resp,
                        (
                            f"Section '{section_id}' truncated from "
                            f"{len(text)} to {len(truncated_text)} characters "
                            f"(strategy={strategy_used})."
                        ),
                    )
                    logger.info(
                        "stage_c_section_truncated",
                        section_id=section_id,
                        old_len=len(text),
                        new_len=len(truncated_text),
                        max_len=max_len,
                        strategy=strategy_used,
                    )

                text = truncated_text
            else:
                # Legacy naive truncation
                truncated_text = text[:max_len].rstrip()
                _append_metadata_warning(
                    resp,
                    f"Section '{section_id}' truncated from {len(text)} to {len(truncated_text)} characters (strategy=naive).",
                )
                logger.info(
                    "stage_c_section_truncated",
                    section_id=section_id,
                    old_len=len(text),
                    new_len=len(truncated_text),
                    max_len=max_len,
                    strategy="naive",
                )
                text = truncated_text

        if text != original_text:
            sec_obj.text = text
            if hasattr(sec_obj, "word_count"):
                try:
                    sec_obj.word_count = len(text.split())
                except Exception:
                    pass

        out[section_id] = sec_obj

    return out

_SECTION_MULTI_BLANK_RE = re.compile(r"\n{3,}")
def _strip_leading_section_heading(section_id: str, text: str) -> str:
    """
    Remove a redundant first line like 'Publications', 'Training',
    'ผลงานตีพิมพ์', 'การฝึกอบรม', etc., which Gemini sometimes adds
    inside the section body.
    """
    if not text:
        return text

    lines = text.splitlines()
    if not lines:
        return text

    first = lines[0].strip(" -*•\t")

    # English headings
    en_map = {
        "publications": ["Publications"],
        "training": ["Training"],
        "references": ["References"],
    }

    # Thai headings we already see in your examples
    th_map = {
        "publications": ["ผลงานตีพิมพ์"],
        "training": ["การฝึกอบรม"],
        "references": ["บุคคลอ้างอิง", "**บุคคลอ้างอิง**"],
    }

    candidates = (en_map.get(section_id, []) +
                  th_map.get(section_id, []))

    if any(first.lower() == c.lower().strip("*") for c in candidates):
        # Drop the first line and trim leading whitespace
        return "\n".join(lines[1:]).lstrip()

    return text


def _clean_text(text: str, enable_cleaning: bool) -> str:
    """
    Basic, conservative text cleanup:
    - Normalize newlines
    - Strip outer whitespace
    - Collapse 3+ blank lines → 2
    - Remove stray markdown code fences (```), but keep their content
    """
    if not enable_cleaning:
        return text

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove markdown code fence markers but keep content
    text = text.replace("```", "")

    # Collapse multiple blank lines
    text = _SECTION_MULTI_BLANK_RE.sub("\n\n", text)

    # Strip leading/trailing whitespace & quotes artifacts
    text = text.strip(" \t\n\r\"'")


    return text


def _ensure_section_content(value: Any) -> SectionContent:
    """
    Ensure a value is a SectionContent.

    Accepts:
    - SectionContent instances
    - dicts with "text"
    - arbitrary values (converted to str)
    """
    if isinstance(value, SectionContent):
        return value

    if isinstance(value, dict):
        base_text = value.get("text", "")
    else:
        base_text = value

    if not isinstance(base_text, str):
        base_text = str(base_text)

    # Prefer Pydantic v2-style model_construct
    if hasattr(SectionContent, "model_construct"):
        return SectionContent.model_construct(text=base_text)  # type: ignore[call-arg]

    # Fallback to classic init
    try:
        return SectionContent(text=base_text)  # type: ignore[call-arg]
    except Exception:
        # Last resort: create instance without __init__ and set .text
        obj = SectionContent.__new__(SectionContent)  # type: ignore[call-arg]
        setattr(obj, "text", base_text)
        return obj  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Internal helpers: skills sanitization
# ---------------------------------------------------------------------------


def _sanitize_skills(
    skills: Any,
    cfg: Dict[str, Any],
    resp: CVGenerationResponse,
) -> Optional[List[OutputSkillItem]]:
    """
    Normalize and deduplicate skills.

    Aligned with Stage B canonical skill behaviour:

    - Do NOT change canonical levels decided in Stage B.
    - Deduplicate by skill *name* only (case-insensitive).
    - If the same skill name appears multiple times and an existing
      entry has no level but a new one does, upgrade the existing entry.
    - Cap list size via cfg["max_skills"].
    """
    if not skills:
        return None

    max_skills = int(cfg.get("max_skills", 50))
    cleaned: List[OutputSkillItem] = []
    # Align with Stage B canonical reconciliation → dedupe by name only
    seen_names: set[str] = set()

    def _mk_item(name: str, level: Any, source: Any) -> OutputSkillItem:
        if hasattr(OutputSkillItem, "model_construct"):
            return OutputSkillItem.model_construct(  # type: ignore[call-arg]
                name=name,
                level=level,
                source=source,
            )
        return OutputSkillItem(name=name, level=level, source=source)  # type: ignore[call-arg]

    for idx, item in enumerate(skills):
        name: Optional[str] = None
        level: Any = None
        source: Any = None

        if isinstance(item, OutputSkillItem):
            name = getattr(item, "name", None)
            level = getattr(item, "level", None)
            source = getattr(item, "source", None)
        elif isinstance(item, dict):
            name = item.get("name")
            level = item.get("level")
            source = item.get("source")
        else:
            # fallback: treat as a raw name
            name = str(item)

        if not name:
            continue

        name_clean = str(name).strip()
        if not name_clean:
            continue

        # Normalize level for simple checks / upgrades
        level_norm = "" if level is None else str(level).strip()
        name_key = name_clean.lower()

        if name_key in seen_names:
            # If we already have this skill by name, we *never* add a new entry
            # (to preserve Stage B's single canonical record per skill).
            # However, if the existing record has no level and this one does,
            # we can safely upgrade it.
            if level_norm:
                for existing in cleaned:
                    if existing.name and existing.name.strip().lower() == name_key:
                        existing_level = getattr(existing, "level", None)
                        if not existing_level or not str(existing_level).strip():
                            try:
                                existing.level = level  # type: ignore[attr-defined]
                            except Exception:
                                # non-fatal; just keep the original
                                pass
                        break
            continue

        seen_names.add(name_key)

        if isinstance(item, OutputSkillItem):
            # Reuse existing instance but normalize name
            try:
                item.name = name_clean  # type: ignore[attr-defined]
                cleaned.append(item)
            except Exception:
                cleaned.append(_mk_item(name_clean, level, source or "stage_b"))
        else:
            cleaned.append(_mk_item(name_clean, level, source or "stage_b"))

        if len(cleaned) >= max_skills:
            _append_metadata_warning(
                resp,
                f"Skills list truncated to max_skills={max_skills}.",
            )
            logger.info(
                "stage_c_skills_truncated",
                max_skills=max_skills,
                original_count=len(skills),
            )
            break

    return cleaned or None


# ---------------------------------------------------------------------------
# Internal helpers: metadata & global checks
# ---------------------------------------------------------------------------


def _normalize_metadata(resp: CVGenerationResponse) -> None:
    """
    Best-effort metadata normalization.

    - Optionally mark validation as passed
    - Accumulate warnings list (already handled in _append_metadata_warning)
    """
    meta = getattr(resp, "metadata", None)
    if meta is None:
        return

    # Attach a simple "stage_c_validated" flag if possible
    try:
        setattr(meta, "stage_c_validated", True)
    except Exception:
        # non-fatal
        pass


def _append_metadata_warning(resp: CVGenerationResponse, message: str) -> None:
    """
    Attach a warning string to resp.metadata.validation_warnings if possible.
    """
    meta = getattr(resp, "metadata", None)
    if meta is None:
        return

    try:
        existing = getattr(meta, "validation_warnings", None)
        if not isinstance(existing, list):
            existing = []
        existing.append(message)
        setattr(meta, "validation_warnings", existing)
    except Exception:
        # metadata might be a plain object / stub; ignore failures
        pass


def _get_meta_field(meta: Any, field_name: str) -> Any:
    """
    Helper to retrieve fields like name/email from metadata.

    Tries:
    - meta.<field_name>
    - meta.profile_info[field_name] if profile_info is a dict
    """
    if meta is None:
        return None

    if hasattr(meta, field_name):
        value = getattr(meta, field_name)
        if value:
            return value

    profile_info = getattr(meta, "profile_info", None)
    if isinstance(profile_info, dict):
        return profile_info.get(field_name)

    return None


def _run_global_consistency_checks(
    *,
    resp: CVGenerationResponse,
    template_info: Dict[str, Any],
    cfg: Dict[str, Any],
    original_request: Any | None,
) -> None:
    """
    Global consistency checks that can emit warnings or raise StageCValidationError
    depending on strict_mode.

    - require_name / require_email
    - min_skills_required / min_education_required
    - cross-check template_id, job_id, language vs original_request
    """
    strict_mode = bool(cfg.get("strict_mode", False))
    meta = getattr(resp, "metadata", None)

    # --- Required metadata fields (name/email) ---
    missing_fields: List[str] = []
    if cfg.get("require_name"):
        name = _get_meta_field(meta, "name")
        if not name or not str(name).strip():
            missing_fields.append("name")

    if cfg.get("require_email"):
        email = _get_meta_field(meta, "email")
        if not email or not str(email).strip():
            missing_fields.append("email")

    if missing_fields:
        msg = f"Missing required metadata fields: {', '.join(missing_fields)}."
        _append_metadata_warning(resp, msg)
        logger.warning("stage_c_missing_required_metadata", missing_fields=missing_fields)
        if strict_mode:
            raise StageCValidationError(msg)

    # --- Minimum skills / education checks ---
    min_skills_required = int(cfg.get("min_skills_required", 0))
    min_education_required = int(cfg.get("min_education_required", 0))

    # Current skills list
    skills_list: List[OutputSkillItem] = list(resp.skills or []) if getattr(resp, "skills", None) else []
    num_skills = len(skills_list)

    if min_skills_required > 0 and num_skills < min_skills_required:
        # Try to recover skills from the original request
        if original_request is not None:
            fallback_skills = _fallback_skills_from_request(original_request)
            if fallback_skills:
                existing_lower = {s.name.lower() for s in skills_list if getattr(s, "name", None)}
                for sk in fallback_skills:
                    if sk.name and sk.name.lower() not in existing_lower:
                        skills_list.append(sk)
                        existing_lower.add(sk.name.lower())

                resp.skills = skills_list
                num_skills = len(skills_list)

                logger.info(
                    "stage_c_filled_skills_from_request",
                    num_skills=num_skills,
                    min_skills_required=min_skills_required,
                )

        # Re-check after fallback
        if num_skills < min_skills_required:
            msg = (
                f"Only {num_skills} skills present; "
                f"min_skills_required={min_skills_required}."
            )
            _append_metadata_warning(resp, msg)
            logger.warning(
                "stage_c_insufficient_skills",
                min_skills_required=min_skills_required,
                num_skills=num_skills,
            )
            if strict_mode:
                raise StageCValidationError(msg)

    if min_education_required > 0:
        has_education_section = "education" in (resp.sections or {})
        if not has_education_section:
            msg = (
                f"No 'education' section present but "
                f"min_education_required={min_education_required}."
            )
            _append_metadata_warning(resp, msg)
            logger.warning(
                "stage_c_missing_education_section",
                min_education_required=min_education_required,
            )
            if strict_mode:
                raise StageCValidationError(msg)

    # --- Cross-checks vs original_request (template_id, job_id, language) ---
    if original_request is not None:
        _cross_check_response_vs_request(resp, original_request, strict_mode)


def _normalize_language(value: Any) -> Optional[str]:
    """
    Normalize language values for comparison.

    Supports:
    - Plain strings ("en", "EN", "th-TH")
    - Enums with .value (LanguageEnum.EN)
    """
    if value is None:
        return None

    lang = getattr(value, "value", value)
    try:
        norm = str(lang).strip().lower()
    except Exception:
        return None

    return norm or None


def _cross_check_response_vs_request(
    resp: CVGenerationResponse,
    original_request: Any,
    strict_mode: bool,
) -> None:
    """
    Cross-check response metadata against the original request.

    - template_id consistency
    - job_id consistency (if both sides have it)
    - language consistency (aligned with Stage A/B enum-or-string usage)
    """
    # template_id
    req_template_id = None

    # Legacy shape: nested template_info
    if hasattr(original_request, "template_info"):
        tmpl = getattr(original_request, "template_info")
        if tmpl is not None:
            if isinstance(tmpl, dict):
                req_template_id = tmpl.get("template_id")
            elif hasattr(tmpl, "template_id"):
                req_template_id = getattr(tmpl, "template_id")

    # New shape: top-level template_id on the request
    if req_template_id is None:
        req_template_id = getattr(original_request, "template_id", None)

    resp_template_id = getattr(resp, "template_id", None)

    if req_template_id and not resp_template_id:
        # Fill in missing template_id in response
        try:
            setattr(resp, "template_id", req_template_id)
        except Exception:
            pass
    elif req_template_id and resp_template_id and resp_template_id != req_template_id:
        msg = (
            f"template_id mismatch between request ({req_template_id}) "
            f"and response ({resp_template_id})."
        )
        _append_metadata_warning(resp, msg)
        logger.warning(
            "stage_c_template_id_mismatch",
            req_template_id=req_template_id,
            resp_template_id=resp_template_id,
        )
        if strict_mode:
            raise StageCValidationError(msg)

    # job_id
    req_job_id = getattr(original_request, "job_id", None)
    resp_job_id = getattr(resp, "job_id", None)

    if req_job_id and not resp_job_id:
        try:
            setattr(resp, "job_id", req_job_id)
        except Exception:
            pass
    elif req_job_id and resp_job_id and resp_job_id != req_job_id:
        msg = f"job_id mismatch between request ({req_job_id}) and response ({resp_job_id})."
        _append_metadata_warning(resp, msg)
        logger.warning(
            "stage_c_job_id_mismatch",
            req_job_id=req_job_id,
            resp_job_id=resp_job_id,
        )
        if strict_mode:
            raise StageCValidationError(msg)

    # language (align enum/string with Stage A/B prompt injection)
    raw_req_lang = getattr(original_request, "language", None)
    raw_resp_lang = getattr(resp, "language", None)

    req_lang_norm = _normalize_language(raw_req_lang)
    resp_lang_norm = _normalize_language(raw_resp_lang)

    if req_lang_norm and not resp_lang_norm:
        # Fill missing language in response; prefer the original type
        try:
            setattr(resp, "language", raw_req_lang if raw_req_lang is not None else req_lang_norm)
        except Exception:
            # non-fatal
            pass
    elif req_lang_norm and resp_lang_norm and req_lang_norm != resp_lang_norm:
        msg = f"language mismatch between request ({req_lang_norm}) and response ({resp_lang_norm})."
        _append_metadata_warning(resp, msg)
        logger.warning(
            "stage_c_language_mismatch",
            req_language=req_lang_norm,
            resp_language=resp_lang_norm,
        )
        if strict_mode:
            raise StageCValidationError(msg)
