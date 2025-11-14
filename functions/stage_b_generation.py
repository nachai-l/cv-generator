# functions/stage_b_generation.py

"""
Stage B: CV generation.

This module handles:
- Building prompts per CV section from validated request + evidence plan
- Calling the LLM with retry logic
- Applying character limits
- Returning a CVGenerationResponse compliant with schemas/output_schema.py
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Union

import structlog
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

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants (to avoid duplicated long string fragments)
# ---------------------------------------------------------------------------

DEFAULT_SKILLS_STRUCTURED_PROMPT = (
    'You are helping to prepare the "skills" section of a CV.\n\n'
    "You are given:\n"
    "- A list of existing skills from our controlled taxonomy (canonical skills).\n"
    "- You may also add new skills freely, based on the provided evidence context.\n\n"
    "IMPORTANT RULES:\n"
    "1. You MUST NOT rename, paraphrase, or otherwise change any existing skill name.\n"
    "2. You MUST preserve the exact level value (e.g., L4_Expert, L3_Advanced, L2_Intermediate) for all existing taxonomy skills. Do NOT modify, downgrade, or change these level values.\n"  # ← ADD THIS
    "3. You may reorder skills to optimize relevance for the target role.\n"
    "4. You may remove skills that are less relevant by setting \"keep\": false.\n"
    "5. You may add new skills that are clearly supported by the candidate's experience, profile, or projects.\n"
    "6. Keep skill names concise (2–5 words). Avoid overly broad or vague terms.\n"
    "7. When in doubt, keep existing skills and add only meaningful new ones.\n\n"
    "Return ONLY valid JSON with this exact structure:\n"
    "{\n"
    "  \"items\": [\n"
    "    {\n"
    "      \"name\": \"string\",\n"
    "      \"level\": \"string or null\",\n"
    "      \"keep\": true,\n"
    "      \"source\": \"taxonomy|inferred\"\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}\n\n"
    "Do not include any explanatory text, comments, or markdown — output JSON only."
)

# Treat these as the canonical “core bundle” only when explicitly enabled
CORE_SECTIONS = [
    "profile_summary",
    "skills",
    "experience",
    "education",
    "certifications",
    "awards",
    "projects",
    "interests",
]

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

    # Defaults keep behavior conservative (no auto-expansion)
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
            requested = template_order or CORE_SECTIONS[:]  # last-resort fallback

    if expand_core:
        requested = list(dict.fromkeys([*requested, *CORE_SECTIONS]))

    if not enable_structured:
        requested = [s for s in requested if s != "skills_structured"]

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

def _extract_original_skill_levels(request: CVGenerationRequest) -> dict[str, str]:
    """
    Collect original skill levels from the incoming request so we can
    hard-override whatever the LLM suggests.
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
    Final safety net: for any skill whose name matches the original profile
    (case-insensitive), force the level to match the original profile level.

    This guarantees that:
      - Existing skills NEVER have their levels downgraded/changed by the LLM.
      - Only truly new inferred skills keep LLM-proposed levels (or None).
    """
    if not skills_output:
        return skills_output

    original_levels = _extract_original_skill_levels(request)
    if not original_levels:
        return skills_output

    for item in skills_output:
        name = getattr(item, "name", None)
        if not name:
            continue
        key = str(name).strip().lower()
        orig_level = original_levels.get(key)
        if orig_level:
            old_level = item.level  # Add this
            item.level = orig_level
            # Add this logging:
            if old_level != orig_level:
                logger.info(
                    "skill_level_reconciled",
                    skill=name,
                    llm_level=old_level,
                    corrected_to=orig_level,
                )

    return skills_output


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
    if evidence_plan is None:
        return []

    evidence_facts: List[str] = []
    canonical_section_id = _normalize_section_id_for_evidence(section_id)

    # Preferred API: EvidencePlan.get_evidence_for_section(section_id)
    if hasattr(evidence_plan, "get_evidence_for_section"):
        for ev in evidence_plan.get_evidence_for_section(canonical_section_id) or []:
            fact = getattr(ev, "fact", None)
            if fact:
                evidence_facts.append(fact)
        return evidence_facts

    # Fallback: manual wiring via section_hints + evidences
    if hasattr(evidence_plan, "section_hints") and hasattr(evidence_plan, "evidences"):
        params = load_parameters()
        cross_cfg: Dict[str, Any] = params.get("cross_section_evidence_sharing", {}) or {}

        hints = getattr(evidence_plan, "section_hints", {}) or {}
        evidences = getattr(evidence_plan, "evidences", []) or []

        ids_for_section = set(hints.get(canonical_section_id, []))

        # 🔧 NEW: prefer explicit config for the *actual* section_id if present
        cfg_key = section_id if section_id in cross_cfg else canonical_section_id
        share_from = cross_cfg.get(cfg_key, cross_cfg.get("default", [])) or []

        shared_ids: List[str] = []
        if "all" in share_from:
            for ev in evidences:
                ev_id = getattr(ev, "evidence_id", None)
                if ev_id:
                    shared_ids.append(ev_id)
        else:
            for src_section in share_from:
                shared_ids.extend(hints.get(src_section, []))

        if shared_ids:
            ids_for_section.update(shared_ids)

        for ev in evidences:
            ev_id = getattr(ev, "evidence_id", None)
            fact = getattr(ev, "fact", None)
            if ev_id in ids_for_section and fact:
                evidence_facts.append(fact)

    return evidence_facts


def _extract_skills_from_bullet_text(text: str) -> list[str]:
    """Simple parser: pull skill phrases from a bullet-list skills section."""
    if not text:
        return []
    skills: list[str] = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        # Strip common bullet markers
        if raw and raw[0] in "-*•":
            raw = raw[1:].strip()
        if not raw:
            continue
        # Drop trailing period if present
        if raw.endswith("."):
            raw = raw[:-1].rstrip()
        if raw:
            skills.append(raw)
    return skills


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
        params = load_parameters()
        generation_cfg: Dict[str, Any] = params.get("generation", {}) or {}
        prompts_cfg: Dict[str, str] = generation_cfg.get("prompts", {}) or {}

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

        evidence_facts: List[str] = _collect_evidence_facts_for_section(
            evidence_plan, section_id
        )

        language = getattr(request, "language", "en")

        lines: List[str] = [
            "You are an expert CV writer.",
            f"Generate a strong '{section_id}' section in {language} for the candidate below.",
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

        # 🔹 Tie to template's max_chars_per_section (already added earlier)
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
        """Build JSON-only prompt for structured skills selection."""
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
            DEFAULT_SKILLS_STRUCTURED_PROMPT,
        )

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
            f"The CV language is {language}.",
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

    def _get_section_token_budget_for_attempt(
        self,
        section_id: str,
        attempt: int,
    ) -> int | None:
        """
        Look up max_output_tokens for a given section and attempt, based on
        section_token_budgets in parameters.yaml.

        Examples in parameters.yaml:

            section_token_budgets:
              default: [1024, 2048]
              profile_summary: [3036, 4096]
              skills: [3036, 4096]
              skills_structured: [4096]      # same for all attempts
              education: 2048                # scalar also allowed

        Behaviour:
        - attempt=1 → index 0
        - attempt=2 → index 1, etc.
        - If attempts exceed the configured list length, we reuse the **last** value.
        - If config is a single int, we reuse that value for all attempts.
        """
        try:
            params = load_parameters() or {}
        except Exception:
            return None

        budgets_cfg = params.get("section_token_budgets", {}) or {}

        # Prefer section-specific, fallback to default
        raw = budgets_cfg.get(section_id, budgets_cfg.get("default"))

        if raw is None:
            return None

        # Case 1: scalar int → same for all attempts
        if isinstance(raw, int):
            return raw if raw > 0 else None

        # Case 2: list/tuple of ints → index by attempt, clamp to last
        if isinstance(raw, (list, tuple)) and raw:
            # Clamp attempt index into [0, len(raw)-1]
            idx = max(0, min(attempt - 1, len(raw) - 1))
            try:
                value = int(raw[idx])
            except Exception:
                return None
            return value if value > 0 else None

        # Anything else is invalid
        return None



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
        min_length_threshold = 10  # characters
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
                        time.sleep(1.2 * attempt)
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
                        time.sleep(1.2 * attempt)
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
                        time.sleep(1.2 * attempt)
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

                time.sleep(1.2 * attempt)

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
        """Generate structured skills via LLM, with robust fallbacks.

        - Preferred path: JSON from skills_structured prompt (can drop/keep/add skills).
        - Fallback path: parse bullet-list 'skills' section text to infer skills.
        - Hard fallback: taxonomy-only skills from profile_info.

        IMPORTANT:
        - For any skill that already exists in the canonical skills list,
          we ALWAYS preserve the original level from skills_plan.canonical_skills.
        - The LLM is only allowed to assign levels for new inferred skills.
        """
        language = getattr(request, "language", "en")
        evidence_plan: EvidencePlan | None = self._evidence_plan

        prompt = self._build_skills_selection_prompt(
            request, evidence_plan, skills_plan, language
        )
        raw_json = self._call_llm_with_retries(prompt, section_id="skills_structured")

        # ---------- Canonical lookup (case-insensitive) ----------
        canonical_by_name: dict[str, CanonicalSkill] = {
            (s.name or "").strip().lower(): s for s in skills_plan.canonical_skills
        }
        canonical_names_lower: set[str] = set(canonical_by_name.keys())

        candidate_items: list[SkillSelectionItem] = []
        had_json_items = False

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
                had_json_items = bool(candidate_items)
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
                continue
            seen_names_lower.add(key)
            sel.name = name
            filtered.append(sel)

        # ---------- Fallback if no usable JSON ----------
        if not filtered:
            inferred_from_bullets: list[SkillSelectionItem] = []
            if skills_section_text:
                bullet_skills = _extract_skills_from_bullet_text(skills_section_text)
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

        for sel in filtered:
            if not sel.keep:
                continue

            name_clean = (sel.name or "").strip()
            if not name_clean:
                continue

            key = name_clean.lower()
            canon = canonical_by_name.get(key)

            if canon is not None:
                # Existing skill → ALWAYS use canonical level from profile/taxonomy
                level = canon.level
                source = "taxonomy"
            else:
                # New inferred skill → use LLM-proposed level (can be None)
                level = sel.level
                source = sel.source or "inferred"

            result.append(
                OutputSkillItem(
                    name=name_clean,
                    level=level,
                    source=source,
                )
            )
            result_names_lower.add(key)

        # ---------- Ensure taxonomy skills are NOT lost ----------
        # If LLM JSON was valid, we DO NOT add duplicates, but we make sure
        # any canonical skill that never appeared at all is appended with its original level.
        for canon in skills_plan.canonical_skills:
            key = (canon.name or "").strip().lower()
            if not key:
                continue
            if key in result_names_lower:
                continue
            result.append(
                OutputSkillItem(
                    name=canon.name,
                    level=canon.level,
                    source="taxonomy",
                )
            )
            result_names_lower.add(key)

        # ---------- Final safety: reconcile levels with canonical ----------
        # Even if some code path above misbehaved, this guarantees:
        # - Any skill whose name matches a canonical skill (case-insensitive)
        #   will end up with the canonical level.
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
        if isinstance(generated_sections, list):
            requested_ids = getattr(request, "sections", []) or []
            sections_dict = {
                requested_ids[i] if i < len(requested_ids) else f"section_{i}": sec
                for i, sec in enumerate(generated_sections)
            }
        else:
            sections_dict = generated_sections

        profile_info = getattr(request, "profile_info", None)

        params = load_parameters()
        gen_cfg = params.get("generation", {}) or {}
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

        user_part = getattr(request, "user_id", "unknown")
        safe_user_part = "".join(
            ch for ch in str(user_part) if ch.isalnum() or ch in "-_"
        )
        final_job_id = job_id or getattr(request, "job_id", f"JOB_{safe_user_part}")

        template_info = getattr(request, "template_info", None)

        # Prefer legacy template_info.template_id if present,
        # otherwise fall back to the new public API's top-level template_id,
        # and only then use UNKNOWN_TEMPLATE.
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
        sections = getattr(request, "sections", []) or []
        self._current_user_id = getattr(request, "user_id", "unknown")
        self._last_retry_count = 0

        char_limits = _get_section_char_limits(request)
        available_sections = _get_available_sections(request)

        # Resolve strictly based on template order (fallbacks described above)
        effective_sections = _resolve_effective_sections(request, available_sections)

        logger.info(
            "stage_b_resolved_sections",
            requested_sections=sections,
            available_sections=list(available_sections),
            effective_sections=effective_sections,
        )

        generated_sections: dict[str, SectionContent] = {}
        skills_output: list[OutputSkillItem] | None = None

        # use monotonic to align with tests and llm_metrics
        start_time = time.monotonic()

        self._evidence_plan = evidence_plan

        if "skills" in effective_sections and skills_plan is None:
            skills_plan = _build_skills_plan_from_profile(request)

        for section_id in effective_sections:
            raw_text = self._call_llm_with_retries(
                self._build_section_prompt(request, evidence_plan, section_id),
                section_id,
            )
            truncated = _truncate_text(
                raw_text or "", char_limits.get(section_id)
            )

            text_for_section = truncated or ""
            if len(text_for_section) < 10:
                logger.warning(
                    "section_text_too_short",
                    section_id=section_id,
                    length=len(text_for_section),
                    reason="LLM returned empty/failed output",
                )

                text_for_section = build_section_fallback_text(
                    request,
                    section_id,
                    reason="LLM output empty/too short after retries",
                )

            word_count = len(text_for_section.split())
            generated_sections[section_id] = SectionContent(
                text=text_for_section,
                word_count=word_count,
                matched_jd_skills=[],
                confidence_score=1.0,
            )

            if (
                section_id == "skills"
                and skills_plan is not None
                and skills_output is None
            ):
                try:
                    skills_output = self._generate_structured_skills(
                        request, skills_plan, skills_section_text=text_for_section
                    )
                except Exception as exc:  # pragma: no cover - defensive
                    logger.error(
                        "skills_structured_generation_failed", error=str(exc)
                    )
                    skills_output = []
                    for sk in skills_plan.canonical_skills:
                        skills_output.append(
                            OutputSkillItem(
                                name=sk.name,
                                level=sk.level,
                                source="taxonomy",
                            )
                        )

        # elapsed time
        elapsed_ms = int((time.monotonic() - start_time) * 1000)

        # 🔒 FINAL SAFETY: restore original levels for matching skills
        if skills_output:
            logger.info(
                "skills_before_reconcile",
                skills=[(s.name, s.level, s.source) for s in skills_output],
            )
            skills_output = _reconcile_skill_levels_with_request(
                request,
                skills_output,
            )
            logger.info(
                "skills_after_reconcile",
                skills=[(s.name, s.level, s.source) for s in skills_output],
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

