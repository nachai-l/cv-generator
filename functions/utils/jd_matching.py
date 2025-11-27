# functions/utils/jd_matching.py

"""
JD Skill Matching Utilities
---------------------------

This module provides **post-generation, deterministic JD-skills matching**
for CV sections. It is designed to run primarily in **Stage C/D**, *after*
Stage B has already generated all section texts.

Key design points:
- JD→text matching is done here, LLM-free, using simple heuristics.
- `section.matched_jd_skills` is overwritten with **canonical JD skills**
  that appear or are strongly implied in each section’s text.
- The same canonical JD skill list (`jd_required_skills`) should be used for:
      (1) matching
      (2) quality_metrics.jd_alignment_score

Expected integration pattern (Stage D):
---------------------------------------
    from functions.utils.jd_matching import (
        extract_canonical_jd_required_skills,
        annotate_matched_jd_skills,
        extract_and_annotate_jd_skills,
    )

    # Option A: explicit two-step usage
    jd_skills = extract_canonical_jd_required_skills(request)
    response = annotate_matched_jd_skills(response, jd_required_skills=jd_skills)

    # Option B: convenience helper
    response = extract_and_annotate_jd_skills(request, response)

Why this module exists:
-----------------------
Stage B previously tried to infer JD skills and match them, but behavior was
inconsistent and often produced empty matches.

This module fixes that by:
- Running matching **only once**, in Stage C/D.
- Using only canonical JD skills derived from the request.
- Ensuring deterministic, stable behavior for downstream scoring/UI.

Heuristics:
-----------
Skill is considered “matched” if:
1) The normalized skill appears as a substring in normalized text, OR
2) Token overlap is high enough:
     • For single-word skills → overlap ≥ 1 token
     • For multi-word skills → overlap ≥ 2 tokens

This avoids LLM cost, avoids hallucinations, and is easy to replace
later with a hybrid or LLM-tagging approach.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Dict, List

from schemas.output_schema import CVGenerationResponse
from schemas.input_schema import CVGenerationRequest

__all__ = [
    "_decode_job_skill_code",
    "extract_canonical_jd_required_skills",
    "annotate_matched_jd_skills",
    "extract_and_annotate_jd_skills",
]


def _decode_job_skill_code(code: str) -> str:
    """
    Convert job_required_skills codes like 'Python_L3' into a human skill name.

    Heuristic:
    - Split on '_'
    - If the last token looks like a level (e.g. L1/L2/L3/L4), drop it.
    - Re-join the remaining parts as the canonical name (spaces instead of '_').
    """
    if not code:
        return ""
    raw = str(code).strip()
    parts = raw.split("_")
    if len(parts) <= 1:
        return raw

    last = parts[-1].upper()
    # Very simple level-detection heuristic: 'L1', 'L2', 'L3', 'L4'
    if len(last) == 2 and last[0] == "L" and last[1].isdigit():
        base_parts = parts[:-1]
    else:
        base_parts = parts

    name = " ".join(base_parts).strip()
    return name or raw


def extract_canonical_jd_required_skills(
    request: CVGenerationRequest,
) -> list[str]:
    """
    Derive the canonical JD skills list (names only) for this request.

    Priority:
    1) If request.jd_required_skills is provided → use that (deduped).
    2) Else:
        - Use target_role_taxonomy.role_required_skills (if any)
        - Plus decoded target_jd_taxonomy.job_required_skills (if any)

    Returns:
        List of unique, cleaned skill names (e.g. ["Python", "Leadership"]).
    """
    # 1) Explicit canonical list wins
    explicit = getattr(request, "jd_required_skills", None) or []
    if explicit:
        seen = set()
        cleaned: list[str] = []
        for s in explicit:
            s_clean = (s or "").strip()
            if not s_clean or s_clean in seen:
                continue
            seen.add(s_clean)
            cleaned.append(s_clean)
        return cleaned

    # 2) Fallback: derive from role + JD taxonomy
    out: list[str] = []

    role = getattr(request, "target_role_taxonomy", None)
    if role and getattr(role, "role_required_skills", None):
        out.extend(role.role_required_skills)

    jd = getattr(request, "target_jd_taxonomy", None)
    if jd and getattr(jd, "job_required_skills", None):
        for code in jd.job_required_skills:
            name = _decode_job_skill_code(code)
            if name:
                out.append(name)

    # Deduplicate & clean
    seen = set()
    cleaned: list[str] = []
    for s in out:
        s_clean = (s or "").strip()
        if not s_clean or s_clean in seen:
            continue
        seen.add(s_clean)
        cleaned.append(s_clean)

    return cleaned


def _normalize_text(s: str) -> str:
    """
    Simple normalization for heuristic matching:
    - lowercase
    - replace non-word characters with spaces
    - collapse multiple spaces
    """
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[^\w]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _normalize_skill(skill: str) -> str:
    """Normalize a JD skill string for matching."""
    return _normalize_text(skill)


def _build_normalized_skill_map(
    jd_required_skills: Sequence[str] | None,
) -> Dict[str, str]:
    """
    Build a mapping:
        normalized_skill_text → original_canonical_skill

    So we do matching in normalized space but return canonical skill names.
    """
    norm_map: Dict[str, str] = {}
    if not jd_required_skills:
        return norm_map

    for raw in jd_required_skills:
        if not raw:
            continue
        original = str(raw).strip()
        if not original:
            continue
        norm = _normalize_skill(original)
        if not norm:
            continue
        # Preserve first occurrence
        norm_map.setdefault(norm, original)

    return norm_map


def _match_skills_in_text(
    section_text: str,
    norm_skill_map: Dict[str, str],
) -> List[str]:
    """
    Given section text + normalized skills, return canonical JD skills
    that clearly appear or are strongly implied.

    Heuristics:
    - direct normalized substring match
    - token overlap for multi-word skills
    """
    if not section_text or not norm_skill_map:
        return []

    norm_text = _normalize_text(section_text)
    if not norm_text:
        return []

    text_tokens = set(norm_text.split())
    matched_canonical: List[str] = []

    for norm_skill, canonical_name in norm_skill_map.items():
        if not norm_skill:
            continue

        # 1) direct substring in normalized text
        if norm_skill in norm_text:
            matched_canonical.append(canonical_name)
            continue

        # 2) token overlap rule
        skill_tokens = set(norm_skill.split())
        if not skill_tokens:
            continue

        overlap = skill_tokens & text_tokens

        # single-word skill → need ≥1 token overlap
        if len(skill_tokens) == 1 and overlap:
            matched_canonical.append(canonical_name)
        # multi-word skill → need ≥2 overlapping tokens
        elif len(skill_tokens) > 1 and len(overlap) >= 2:
            matched_canonical.append(canonical_name)

    # Deduplicate while preserving order
    seen = set()
    deduped: List[str] = []
    for name in matched_canonical:
        if name not in seen:
            seen.add(name)
            deduped.append(name)

    return deduped


def annotate_matched_jd_skills(
    response: CVGenerationResponse,
    jd_required_skills: Sequence[str] | None,
) -> CVGenerationResponse:
    """
    Overwrite `section.matched_jd_skills` for every section in the
    CVGenerationResponse using deterministic heuristic matching.

    - No LLM calls.
    - Runs after Stage B (typically in Stage C/D).
    - Should be called before computing quality metrics.

    If jd_required_skills is empty/None, the function is a no-op.
    """
    if not jd_required_skills:
        return response

    norm_skill_map = _build_normalized_skill_map(jd_required_skills)
    if not norm_skill_map:
        return response

    sections = getattr(response, "sections", {}) or {}

    for section_id, section in sections.items():
        # section is expected to be a SectionContent-like object
        text = getattr(section, "text", "") or ""
        matched = _match_skills_in_text(text, norm_skill_map)
        # Overwrite matched_jd_skills deterministically
        setattr(section, "matched_jd_skills", matched)

    return response


def extract_and_annotate_jd_skills(
    request: CVGenerationRequest,
    response: CVGenerationResponse,
) -> CVGenerationResponse:
    """
    Convenience helper for Stage C/D:

        response = extract_and_annotate_jd_skills(request, response)

    Equivalent to:
        jd_skills = extract_canonical_jd_required_skills(request)
        response = annotate_matched_jd_skills(response, jd_skills)
    """
    jd_skills = extract_canonical_jd_required_skills(request)
    return annotate_matched_jd_skills(response, jd_required_skills=jd_skills)
