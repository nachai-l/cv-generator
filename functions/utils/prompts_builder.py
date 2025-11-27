# functions/utils/prompts_builder.py

from __future__ import annotations

import json
from functools import lru_cache

from typing import Any, Dict, List

import structlog
from pydantic import BaseModel

from schemas.input_schema import CVGenerationRequest
from schemas.internal_schema import EvidencePlan, SkillsSectionPlan
from functions.utils.llm_client import load_parameters, _project_root
from functions.utils.language_tone import describe_language, describe_tone
from functions.utils.common import load_yaml_dict
from functions.utils.claims import should_require_justification
from functions.utils.experience_functions import extract_year_from_date

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level configuration (read parameters.yaml once here)
# ---------------------------------------------------------------------------

PARAMS = load_parameters() or {}

GENERATION_CFG: Dict[str, Any] = PARAMS.get("generation", {}) or {}
CROSS_SECTION_CFG: Dict[str, Any] = PARAMS.get(
    "cross_section_evidence_sharing", {}
) or {}

PROMPTS_FILE = GENERATION_CFG.get("prompts_file", "prompts.yaml")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_prompts_from_file() -> dict[str, str]:
    """
    Load section-level prompts from parameters/prompts.yaml.

    Expected structure (top-level mapping):

        default: |
          ...
        profile_summary: |
          ...
        skills_structured: |
          ...
        experience: |
          ...
        experience_bullets_only: |
          ...
        user_draft_rewrite_suffix: |
          ...

    Returns {} on any error.
    """
    try:
        root = _project_root()
        candidate = root / "parameters" / PROMPTS_FILE

        if not candidate.exists():
            # Fallback: PROMPTS_FILE might already contain a subpath
            candidate_alt = root / PROMPTS_FILE
            if candidate_alt.exists():
                candidate = candidate_alt

        data = load_yaml_dict(str(candidate)) or {}

    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("prompts_config_load_failed", error=str(exc))
        return {}

    if not isinstance(data, dict):
        logger.warning(
            "prompts_config_not_mapping",
            type=str(type(data)),
        )
        return {}

    # Ensure keys are strings
    return {str(k): v for k, v in data.items()}

def load_section_prompts_config() -> dict[str, str]:
    """
    Public, cached accessor for section prompts.

    Stage B and other modules should import and use this instead of touching
    the private `_load_prompts_from_file` directly.
    """
    return _load_prompts_from_file()

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

def _resolve_job_context_for_prompts(
    request: CVGenerationRequest | Any,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Resolve job/role/company context for prompts in a backward-compatible way.

    Priority:
    1) Legacy fields: job_role_info, job_position_info, company_info
    2) New API fields: target_role_taxonomy, target_jd_taxonomy
       - If company_info is missing, try to derive it from target_jd_taxonomy.
    """
    # Legacy-style fields
    raw_job_role_info = getattr(request, "job_role_info", None)
    raw_job_position_info = getattr(request, "job_position_info", None)
    raw_company_info = getattr(request, "company_info", None)

    # New API fields (CVGenerationRequest)
    target_role = getattr(request, "target_role_taxonomy", None)
    target_jd = getattr(request, "target_jd_taxonomy", None)

    # Prefer legacy if present; otherwise fall back to new API
    job_role_obj = raw_job_role_info or target_role or {}
    job_position_obj = raw_job_position_info or target_jd or {}

    # Company: prefer explicit company_info, otherwise try to derive from JD
    company_obj = raw_company_info
    if not company_obj and target_jd is not None:
        company_obj = (
            getattr(target_jd, "company_info", None)
            or getattr(target_jd, "company", None)
            or getattr(target_jd, "employer", None)
            or getattr(target_jd, "organization", None)
        )

    # Ensure JSON-serializable outputs
    job_role_info = _to_serializable(job_role_obj) or {}
    job_position_info = _to_serializable(job_position_obj) or {}
    company_info = _to_serializable(company_obj) or {}

    return job_role_info, job_position_info, company_info


def _normalize_section_id_for_evidence(section_id: str) -> str:
    """Map internal/derived section IDs to their canonical evidence section name.

    This lets us treat `skills_structured` as `skills` when looking up:
    - EvidencePlan.get_evidence_for_section
    - cross_section_evidence_sharing config
    """
    if section_id.endswith("_structured"):
        return section_id[: -len("_structured")]
    return section_id


def _collect_evidence_facts_for_section(
    evidence_plan: EvidencePlan | None,
    section_id: str,
) -> List[str]:
    """
    Collect evidence facts for a section, *including* any cross-section
    sharing rules from CROSS_SECTION_CFG (from parameters.yaml).
    """
    if evidence_plan is None:
        return []

    canonical_section_id = _normalize_section_id_for_evidence(section_id)

    cross_cfg: Dict[str, Any] = CROSS_SECTION_CFG or {}

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
# Education section helpers (moved from stage_b_generation)
# ---------------------------------------------------------------------------

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
    honors = _get("honors")
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

    # Honors / distinctions
    if honors:
        parts.append(f"Honors: {honors}")

    # Years
    start_year = extract_year_from_date(start_date)
    end_year = extract_year_from_date(graduation_date)
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


# ---------------------------------------------------------------------------
# Public prompt builders
# ---------------------------------------------------------------------------

def build_section_prompt(
    request: CVGenerationRequest,
    evidence_plan: EvidencePlan | None,
    section_id: str,
) -> str:
    """Construct an LLM prompt for a single CV section (with full cross-section context)."""
    prompts_cfg: Dict[str, str] = _load_prompts_from_file()

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

    drafts = getattr(request, "user_input_cv_text_by_section", {}) or {}

    profile_info = _to_serializable(raw_profile_info)
    job_role_info, job_position_info, company_info = _resolve_job_context_for_prompts(request)

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
        "=== Target Job Role / Position / Company (JSON) ===",
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

        # Optional suffix from prompts.yaml / parameters
        draft_suffix = prompts_cfg.get("user_draft_rewrite_suffix", "")
        if draft_suffix:
            lines.append("\n" + draft_suffix)

    # 🔹 Append section-specific or default prompt as "Output Requirements"
    default_prompt = prompts_cfg.get("default") or (
        "Write this CV section in a clear, concise, and professional style."
    )
    section_prompt = (prompts_cfg.get(section_id) or default_prompt).strip()

    lines.append("\n=== Output Requirements ===")
    lines.append(section_prompt)

    # --- Optional justification instructions (no extra LLM call) ---
    if should_require_justification(section_id, GENERATION_CFG):
        justification_prompt = prompts_cfg.get("justification")
        if justification_prompt:
            lines.append("\n=== Justification Instructions ===")
            lines.append(justification_prompt)

    if should_require_justification(section_id, GENERATION_CFG):
        logger.info(
            "debug_justification_prompt_injected",
            section_id=section_id,
            contains_instructions=("=== Justification Instructions ===" in "\n".join(lines)),
        )

    return "\n".join(lines)


def build_skills_selection_prompt(
    request: CVGenerationRequest,
    evidence_plan: EvidencePlan | None,
    skills_plan: SkillsSectionPlan,
    language: str,
    require_justification: bool = False,
) -> str:
    """
    Build the LLM prompt for generating structured skills JSON and,
    optionally, a justification block.

    Unlike normal section prompts, the skills flow is JSON-first:
      - The LLM must output SKILLS_JSON as the first and only content.
      - If justification is required, a second block follows:
            === JUSTIFICATION_JSON ===
            { justification JSON }

    Prompt structure (in order):
      1. CV metadata (language, tone)
      2. Student profile + optional job context (JSON)
      3. Skills instructions from `skills_structured`
      4. Canonical taxonomy skills (one per line)
      5. Evidence facts for grounding
      6. (Optional) justification suffix from
         `skills_structured_justification_suffix`

    This design keeps the core skills spec separate while allowing the
    justification block to be appended consistently with other sections.
    """
    prompts_cfg: dict[str, str] = _load_prompts_from_file()

    # Base = normal skills_structured prompt (JSON-only definition)
    base_prompt: str = prompts_cfg.get("skills_structured", "") or ""

    if not base_prompt:
        logger.warning(
            "skills_structured_prompt_missing_using_default",
            message=(
                "skills_structured not found in prompts.yaml; "
                "using a built-in JSON-only prompt instead."
            ),
        )
        base_prompt = (
            "You are an assistant that selects and structures the candidate's skills.\n"
            "Return ONLY valid JSON matching this schema:\n"
            "{\n"
            '  "items": [\n'
            '    { "name": string, "level": string | null, "keep": boolean, "source": "taxonomy" | "inferred" }\n'
            "  ]\n"
            "}\n"
        )

    # If justification is required, append the skills-specific justification
    # instructions instead of replacing the base prompt.
    if require_justification:
        just_suffix = prompts_cfg.get("skills_structured_justification_suffix", "")
        if just_suffix:
            base_prompt = (
                base_prompt.rstrip()
                + "\n\n=== Justification Instructions ===\n"
                + just_suffix.strip()
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

    profile_info = _to_serializable(raw_profile_info)
    job_role_info, job_position_info, company_info = _resolve_job_context_for_prompts(request)

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

def build_experience_justification_prompt(
    request: CVGenerationRequest,
    evidence_plan: EvidencePlan | None,
    section_text: str,
    section_id: str = "experience",
) -> str:
    """
    Build a justification-only prompt for experience sections.

    Used for:
      - final 'experience'
      - 'experience_bullets_only' justification

    Always outputs JSON-only (no section text, no separators).
    """
    prompts_cfg: dict[str, str] = _load_prompts_from_file()

    raw_language = getattr(request, "language", "en") or "en"
    language = raw_language
    language_name = describe_language(language)

    # Prefer profile_info; fall back to student_profile
    raw_profile_info = getattr(request, "profile_info", None)
    if not raw_profile_info:
        raw_profile_info = getattr(request, "student_profile", None) or {}
    profile_info = _to_serializable(raw_profile_info)

    # Use section_id so we can justify 'experience' or 'experience_bullets_only'
    evidence_facts = _collect_evidence_facts_for_section(evidence_plan, section_id)

    # Prefer JSON-only justification prompts from prompts.yaml
    justification_instructions = (
        prompts_cfg.get("experience_justification_json_only", "").strip()
        or prompts_cfg.get("justification_json_only", "").strip()
    )

    # If still empty, fall back to a built-in JSON-only specification
    if not justification_instructions:
        justification_instructions = """
You must output ONLY a single valid JSON object with this structure:

{
  "evidence_map": [
    {
      "section": "<SECTION_ID>",
      "sentence": "<full sentence copied exactly from the generated section text>",
      "evidence_ids": ["<evidence_ref_1>", "<evidence_ref_2>"],
      "match_score": <float between 0.0 and 1.0>
    }
  ],
  "unsupported_claims": [
    {
      "section": "<SECTION_ID>",
      "claim": "<claim that is unsupported>",
      "reason": "<explanation>",
      "severity": "info" | "warning" | "error"
    }
  ],
  "coverage_score": <float between 0.0 and 1.0>,
  "total_claims_analyzed": <integer >= 0>
}

STRICT RULES:
- Output ONLY JSON (no prose, no section text).
- Do NOT output any separator like "=== JUSTIFICATION_JSON ===".
- Do NOT wrap JSON in backticks or code fences.
- Allowed keys: evidence_map, unsupported_claims, coverage_score, total_claims_analyzed.
- If unsure about any field, use null.
""".strip()

    # Human-friendly label
    section_label = (
        "experience" if section_id == "experience" else section_id.replace("_", " ")
    )

    lines: list[str] = [
        "You are an assistant that produces JSON justifications for a CV section.",
        f"The CV language is {language_name} (language_code='{language}').",
        "",
        "Use the profile info and evidence facts to justify the claims",
        f"in the final '{section_label}' section text.",
        "",
        "=== Profile Info (JSON) ===",
        json.dumps(profile_info, ensure_ascii=False, indent=2),
        "",
        f"=== Evidence Facts for {section_label} ===",
    ]

    if evidence_facts:
        lines.extend([f"- {fact}" for fact in evidence_facts])
    else:
        lines.append("- (No specific evidence provided)")

    lines.extend(
        [
            "",
            f"=== Final rendered '{section_label}' section text ===",
            section_text,
            "",
            "=== Justification Instructions ===",
            justification_instructions,
        ]
    )

    return "\n".join(lines)


# def build_experience_justification_prompt(
#     request: CVGenerationRequest,
#     evidence_plan: EvidencePlan | None,
#     section_text: str,
#     section_id: str = "experience",
# ) -> str:
#     """
#     Build a justification-only prompt for experience sections.
#
#     Used for:
#       - final 'experience'
#       - 'experience_bullets_only' justification
#     Always outputs JSON-only (no section text, no separators).
#     """
#     prompts_cfg: dict[str, str] = _load_prompts_from_file()
#
#     raw_language = getattr(request, "language", "en") or "en"
#     language = raw_language
#     language_name = describe_language(language)
#
#     # Prefer profile_info; fall back to student_profile
#     raw_profile_info = getattr(request, "profile_info", None)
#     if not raw_profile_info:
#         raw_profile_info = getattr(request, "student_profile", None) or {}
#     profile_info = _to_serializable(raw_profile_info)
#
#     raw_job_role_info = getattr(request, "job_role_info", None) or {}
#     raw_job_position_info = getattr(request, "job_position_info", None) or {}
#     raw_company_info = getattr(request, "company_info", None) or {}
#
#     job_role_info = _to_serializable(raw_job_role_info)
#     job_position_info = _to_serializable(raw_job_position_info)
#     company_info = _to_serializable(raw_company_info)
#
#     # Use section_id so we can justify 'experience' or 'experience_bullets_only'
#     evidence_facts = _collect_evidence_facts_for_section(evidence_plan, section_id)
#
#     # Prefer JSON-only justification prompts from prompts.yaml
#     justification_instructions = (
#         prompts_cfg.get("experience_justification_json_only", "").strip()
#         or prompts_cfg.get("justification_json_only", "").strip()
#     )
#
#     # If still empty, fall back to a built-in JSON-only specification
#     if not justification_instructions:
#         justification_instructions = """
# You must output ONLY a single valid JSON object with this structure:
#
# {
#   "evidence_map": [
#     {
#       "section": "<SECTION_ID>",
#       "sentence": "<full sentence copied exactly from the generated section text>",
#       "evidence_ids": ["<evidence_ref_1>", "<evidence_ref_2>"],
#       "match_score": <float between 0.0 and 1.0>
#     }
#   ],
#   "unsupported_claims": [
#     {
#       "section": "<SECTION_ID>",
#       "claim": "<claim that is unsupported>",
#       "reason": "<explanation>",
#       "severity": "info" | "warning" | "error"
#     }
#   ],
#   "coverage_score": <float between 0.0 and 1.0>,
#   "total_claims_analyzed": <integer >= 0>
# }
#
# STRICT RULES:
# - Output ONLY JSON (no prose, no section text).
# - Do NOT output any separator like "=== JUSTIFICATION_JSON ===".
# - Do NOT wrap JSON in backticks or code fences.
# - Allowed keys: evidence_map, unsupported_claims, coverage_score, total_claims_analyzed.
# - If unsure about any field, use null.
# """.strip()
#
#     # Human-friendly label
#     section_label = (
#         "experience" if section_id == "experience" else section_id.replace("_", " ")
#     )
#
#     lines: list[str] = [
#         "You are an assistant that produces JSON justifications for a CV section.",
#         f"The CV language is {language_name} (language_code='{language}').",
#         "",
#         "Use the profile info, job context, and evidence facts to justify the claims",
#         f"in the final '{section_label}' section text.",
#         "",
#         "=== Profile Info (JSON) ===",
#         json.dumps(profile_info, ensure_ascii=False, indent=2),
#         "",
#         "=== Target Job Role / Position / Company (JSON) ===",
#         json.dumps(
#             {
#                 "job_role_info": job_role_info,
#                 "job_position_info": job_position_info,
#                 "company_info": company_info,
#             },
#             ensure_ascii=False,
#             indent=2,
#         ),
#         "",
#         f"=== Evidence Facts for {section_label} ===",
#     ]
#
#     if evidence_facts:
#         lines.extend([f"- {fact}" for fact in evidence_facts])
#     else:
#         lines.append("- (No specific evidence provided)")
#
#     lines.extend(
#         [
#             "",
#             f"=== Final rendered '{section_label}' section text ===",
#             section_text,
#             "",
#             "=== Justification Instructions ===",
#             justification_instructions,
#         ]
#     )
#
#     return "\n".join(lines)

