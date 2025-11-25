from __future__ import annotations

"""
Utility functions for computing CV quality metrics.

This module provides heuristic scoring for:
- Clarity / readability
- JD alignment (if target skills are provided)
- Completeness (sections + content volume)
- Consistency (justification coverage vs. unsupported claims)
- Overall score + human-readable feedback

All metrics are returned as a `QualityMetrics` Pydantic model and are
intended to be populated in Stage D (packaging), after we have:

- Final `CVGenerationResponse` content (sections, skills, justification)
- Metadata (sections_requested, sections_generated, etc.)
- Optional job-description skill list (if available upstream)
"""

from typing import Iterable, Sequence, Tuple

from schemas.output_schema import (
    CVGenerationResponse,
    QualityMetrics,
    SectionContent,
)

# ---------------------------------------------------------------------------
# Tuning constants / magic numbers
# ---------------------------------------------------------------------------

# Score bounds
SCORE_MIN: float = 0.0
SCORE_MAX: float = 100.0

# Number of dimensions when averaging overall score
OVERALL_METRIC_DIMENSIONS: float = 4.0

# Language codes
LANG_TH: str = "th"

# Clarity: sentence length (words per sentence) thresholds
CLARITY_SENT_LEN_MIN: float = 8.0   # Below → too short
CLARITY_SENT_LEN_MAX: float = 35.0  # Above → too long
CLARITY_SENT_LEN_EDGE_LOW: float = 12.0
CLARITY_SENT_LEN_EDGE_HIGH: float = 28.0

# Clarity: base scores by sentence length band
CLARITY_SCORE_POOR: float = 65.0
CLARITY_SCORE_OK: float = 80.0
CLARITY_SCORE_GOOD: float = 90.0

# Clarity: section word-count thresholds (English / multi-section only)
CLARITY_SECTION_MIN_WORDS: int = 20
CLARITY_SECTION_MAX_WORDS: int = 220
CLARITY_SECTION_PENALTY: float = 5.0

# Completeness: total word-count thresholds
COMPLETENESS_SHORT_CV_WORDS: int = 120
COMPLETENESS_SHORT_CV_PENALTY: float = 10.0
COMPLETENESS_LONG_CV_WORDS: int = 800
COMPLETENESS_LONG_CV_PENALTY: float = 5.0

# Completeness: section presence (only if "full CV"-like)
COMPLETENESS_MIN_REQUESTED_FOR_SECTION_CHECK: int = 3
COMPLETENESS_MISSING_SECTIONS_PENALTY: float = 3.0

# Consistency: unsupported claims penalties
CONSISTENCY_UNSUPPORTED_PER_CLAIM_PENALTY: float = 5.0
CONSISTENCY_UNSUPPORTED_MAX_PENALTY: float = 20.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_quality_metrics(
    response: CVGenerationResponse,
    *,
    jd_required_skills: Sequence[str] | None = None,
) -> QualityMetrics:
    """
    Compute heuristic quality metrics for a generated CV.

    Parameters
    ----------
    response:
        Final CVGenerationResponse object with sections, skills, metadata,
        and justification already populated.
    jd_required_skills:
        Optional list of required JD skills (canonical names). If not
        provided or empty, jd_alignment_score will default to 0.0 and
        feedback will indicate that JD context is missing.

    Returns
    -------
    QualityMetrics
        Pydantic model with individual scores + overall score + feedback.
    """
    clarity_score, clarity_feedback = _compute_clarity_score(response)
    jd_score, jd_feedback = _compute_jd_alignment_score(response, jd_required_skills)
    completeness_score, completeness_feedback = _compute_completeness_score(response)
    consistency_score, consistency_feedback = _compute_consistency_score(response)

    # Simple average of the four dimensions
    overall_score = _clamp_score(
        (clarity_score + jd_score + completeness_score + consistency_score)
        / OVERALL_METRIC_DIMENSIONS
    )

    feedback_items: list[str] = []
    feedback_items.extend(clarity_feedback)
    feedback_items.extend(jd_feedback)
    feedback_items.extend(completeness_feedback)
    feedback_items.extend(consistency_feedback)

    # Deduplicate feedback while preserving order
    seen: set[str] = set()
    unique_feedback: list[str] = []
    for item in feedback_items:
        if item not in seen:
            seen.add(item)
            unique_feedback.append(item)

    return QualityMetrics(
        clarity_score=_round_score(clarity_score),
        jd_alignment_score=_round_score(jd_score),
        completeness_score=_round_score(completeness_score),
        consistency_score=_round_score(consistency_score),
        overall_score=_round_score(overall_score),
        feedback=unique_feedback,
    )


# ---------------------------------------------------------------------------
# Clarity score
# ---------------------------------------------------------------------------


def _compute_clarity_score(
    response: CVGenerationResponse,
) -> Tuple[float, list[str]]:
    """
    Heuristic clarity / readability score.

    Logic (coarse but stable):
    - Estimate average sentence length (words / sentence).
    - Penalize extremely short (< CLARITY_SENT_LEN_MIN) or very long
      (> CLARITY_SENT_LEN_MAX) sentences.
    - Penalize sections that are extremely short (< CLARITY_SECTION_MIN_WORDS)
      or very long (> CLARITY_SECTION_MAX_WORDS) but only for:
        * non-Thai outputs AND
        * multi-section / "full CV" style requests.
    """
    sections: list[SectionContent] = list(response.sections.values())
    feedback: list[str] = []

    if not sections:
        return 0.0, ["No sections generated – clarity cannot be assessed."]

    total_words = sum(s.word_count for s in sections)
    if total_words == 0:
        return 0.0, ["Generated content is empty – clarity score set to 0."]

    # Rough sentence counting based on punctuation
    total_sentences = 0
    for s in sections:
        text = s.text.strip()
        sentence_count = text.count(".") + text.count("?") + text.count("!")
        if sentence_count == 0:
            sentence_count = 1
        total_sentences += sentence_count

    avg_sentence_len = total_words / max(total_sentences, 1)

    # Base score determined by average sentence length
    if avg_sentence_len < CLARITY_SENT_LEN_MIN or avg_sentence_len > CLARITY_SENT_LEN_MAX:
        clarity = CLARITY_SCORE_POOR
        feedback.append(
            "Sentences are very short or very long on average – consider adjusting for readability."
        )
    elif (
        CLARITY_SENT_LEN_MIN
        <= avg_sentence_len
        <= CLARITY_SENT_LEN_EDGE_LOW
        or CLARITY_SENT_LEN_EDGE_HIGH
        <= avg_sentence_len
        <= CLARITY_SENT_LEN_MAX
    ):
        clarity = CLARITY_SCORE_OK
    else:
        # Rough "sweet spot" of ~12–28 words / sentence
        clarity = CLARITY_SCORE_GOOD
        feedback.append("Sentence length is within a good readability range.")

    # Section length penalties / nudges
    meta = response.metadata
    language = (response.language or "").lower()
    is_thai = language.startswith(LANG_TH)
    is_single_section_mode = meta.sections_requested <= 1 or len(sections) <= 1

    # Only apply "very short" / "very long" section heuristics for:
    # - non-Thai language, and
    # - multi-section or "full CV" style requests.
    if not is_thai and not is_single_section_mode:
        very_short = any(s.word_count < CLARITY_SECTION_MIN_WORDS for s in sections)
        very_long = any(s.word_count > CLARITY_SECTION_MAX_WORDS for s in sections)
    else:
        very_short = False
        very_long = False

    if very_short:
        clarity -= CLARITY_SECTION_PENALTY
        feedback.append(
            "Some sections are very short – consider adding more context or details."
        )
    if very_long:
        clarity -= CLARITY_SECTION_PENALTY
        feedback.append(
            "Some sections are quite long – consider splitting or tightening the text."
        )

    clarity = _clamp_score(clarity)
    if not feedback:
        feedback.append("Content is generally clear and well-structured.")

    return clarity, feedback


# ---------------------------------------------------------------------------
# JD alignment score
# ---------------------------------------------------------------------------


def _compute_jd_alignment_score(
    response: CVGenerationResponse,
    jd_required_skills: Sequence[str] | None = None,
) -> Tuple[float, list[str]]:
    """
    Heuristic JD alignment score.

    Logic:
    - If jd_required_skills is None / empty → 0.0 with 'no JD context' feedback.
    - Otherwise, compute overlap between:
        - required JD skill names (normalized)
        - generated CV skills (response.skills + matched_jd_skills in sections)
    """
    feedback: list[str] = []

    if not jd_required_skills:
        return 0.0, [
            "No job description skills provided – JD alignment score is not applicable (set to 0)."
        ]

    required_set = {s.strip().lower() for s in jd_required_skills if s.strip()}
    if not required_set:
        return 0.0, [
            "Job description skills list was empty after normalization – JD alignment score set to 0."
        ]

    cv_skill_names: set[str] = set()

    # From structured skills
    if response.skills:
        for item in response.skills:
            if item.name:
                cv_skill_names.add(item.name.strip().lower())

    # From section-level matched_jd_skills
    for section in response.sections.values():
        for skill in section.matched_jd_skills:
            if skill:
                cv_skill_names.add(skill.strip().lower())

    if not cv_skill_names:
        feedback.append(
            "No skills were identified in the generated CV – JD alignment score is low."
        )

    intersection = required_set.intersection(cv_skill_names)
    coverage_ratio = len(intersection) / len(required_set)
    score = coverage_ratio * SCORE_MAX

    if score == 0.0:
        feedback.append(
            "Generated CV does not explicitly mention any required JD skills – consider tailoring content."
        )
    elif score < 50.0:
        feedback.append(
            "Some required JD skills appear in the CV, but important skills are still missing."
        )
    elif score < 80.0:
        feedback.append(
            "Most key JD skills are covered – consider reinforcing a few remaining skills."
        )
    else:
        feedback.append("Generated CV aligns well with the job description skills.")

    score = _clamp_score(score)
    return score, feedback


# ---------------------------------------------------------------------------
# Completeness score
# ---------------------------------------------------------------------------


def _compute_completeness_score(
    response: CVGenerationResponse,
) -> Tuple[float, list[str]]:
    """
    Heuristic completeness score.

    Logic:
    - Use metadata.sections_requested vs. sections_generated as the base.
    - Adjust slightly based on total word count across sections.
      * For multi-section requests, very small total word count is penalized.
      * Extremely long CVs receive a small penalty.
    - Only treat "missing common sections" as an issue for full-CV style
      requests where at least COMPLETENESS_MIN_REQUESTED_FOR_SECTION_CHECK
      sections were requested.
    """
    feedback: list[str] = []
    meta = response.metadata

    # Section count based completeness
    if meta.sections_requested > 0:
        ratio = meta.sections_generated / float(meta.sections_requested)
        base = ratio * SCORE_MAX
    else:
        # If nothing explicitly requested, treat as binary:
        base = SCORE_MAX if response.sections else 0.0

    base = _clamp_score(base)

    total_words = response.get_total_word_count()

    # Very small CV → nudge down, but only when multiple sections were requested.
    if meta.sections_requested > 1 and total_words < COMPLETENESS_SHORT_CV_WORDS:
        base -= COMPLETENESS_SHORT_CV_PENALTY
        feedback.append(
            "Total content is quite short relative to the number of requested sections."
        )
    # Extremely long CV → slight penalty
    elif total_words > COMPLETENESS_LONG_CV_WORDS:
        base -= COMPLETENESS_LONG_CV_PENALTY
        feedback.append(
            "CV is very long – consider prioritizing the most relevant achievements."
        )

    # Section-level checks: only apply for "full CV" style requests.
    if meta.sections_requested >= COMPLETENESS_MIN_REQUESTED_FOR_SECTION_CHECK:
        missing_like_sections: list[str] = []
        for expected in ("profile_summary", "skills", "experience", "education"):
            if expected not in response.sections:
                missing_like_sections.append(expected)

        if missing_like_sections:
            base -= COMPLETENESS_MISSING_SECTIONS_PENALTY
            feedback.append(
                f"Some common sections are missing: {', '.join(missing_like_sections)}."
            )

    base = _clamp_score(base)

    if not feedback:
        feedback.append("Core sections and content length look reasonably complete.")

    return base, feedback


# ---------------------------------------------------------------------------
# Consistency score
# ---------------------------------------------------------------------------


def _compute_consistency_score(
    response: CVGenerationResponse,
) -> Tuple[float, list[str]]:
    """
    Heuristic consistency score.

    Logic:
    - Start from justification.coverage_score (0–1 → 0–100).
    - Penalize for unsupported claims.
    - DO NOT penalize inferred skills – Stage B is allowed to add skills.
      If additional skills are inferred vs. input, we only emit a neutral
      informational note.
    """
    feedback: list[str] = []

    justification = response.justification
    base = justification.coverage_score * SCORE_MAX

    # Unsupported claims penalty
    unsupported_count = len(justification.unsupported_claims)
    if unsupported_count > 0:
        penalty = min(
            CONSISTENCY_UNSUPPORTED_MAX_PENALTY,
            unsupported_count * CONSISTENCY_UNSUPPORTED_PER_CLAIM_PENALTY,
        )
        base -= penalty
        feedback.append(
            f"{unsupported_count} claims could not be fully justified – consider reviewing those statements."
        )

    # Skill count consistency:
    # Stage B intentionally infers NEW skills from content.
    # This is a feature, not a defect, so we DO NOT penalize.
    input_skill_count = None
    if response.metadata.profile_info and isinstance(response.metadata.profile_info, dict):
        input_skill_count = response.metadata.profile_info.get("skills_count")

    if input_skill_count is not None and response.skills is not None:
        try:
            requested = int(input_skill_count)
        except (TypeError, ValueError):
            requested = None

        if requested is not None:
            diff = len(response.skills) - requested
            if diff > 0:
                feedback.append(
                    f"{diff} additional skills were inferred from content – this is expected behavior."
                )

    base = _clamp_score(base)

    if base >= 90.0 and unsupported_count == 0:
        feedback.append(
            "Generated content is well supported by evidence with no unsupported claims."
        )
    elif not feedback:
        feedback.append(
            "Evidence coverage is acceptable, but there may be room to tighten consistency."
        )

    return base, feedback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clamp_score(value: float) -> float:
    """Clamp a score to the [SCORE_MIN, SCORE_MAX] range."""
    if value < SCORE_MIN:
        return SCORE_MIN
    if value > SCORE_MAX:
        return SCORE_MAX
    return value


def _round_score(value: float) -> float:
    """Round score to one decimal place after clamping."""
    return round(_clamp_score(value), 1)
