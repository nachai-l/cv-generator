# tests/test_utils_jd_matching.py

from datetime import date

import pytest

from functions.utils.jd_matching import (
    _decode_job_skill_code,
    extract_canonical_jd_required_skills,
    _build_normalized_skill_map,
    _match_skills_in_text,
    annotate_matched_jd_skills,
)
from schemas.input_schema import (
    CVGenerationRequest,
    StudentProfile,
    PersonalInfo,
    Education,
    Skill,
    SkillLevel,
    RoleTaxonomy,
    JobTaxonomy,
)
from schemas.output_schema import (
    CVGenerationResponse,
    SectionContent,
    Metadata,
    Justification,
    GenerationStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_student_profile() -> StudentProfile:
    """Build a minimal valid StudentProfile for constructing requests."""
    personal_info = PersonalInfo(
        name="Alice Example",
        email="alice@example.com",
    )
    edu = Education(
        id="edu#1",
        degree="BSc Computer Science",
        institution="Example University",
        gpa=3.5,
        start_date=date(2018, 1, 1),
        graduation_date=date(2022, 1, 1),
        major="Computer Science",
        honors=None,
    )
    skill = Skill(
        id="skill#python",
        name="Python",
        description="Python programming",
        level=SkillLevel.L2_INTERMEDIATE,
    )

    return StudentProfile(
        personal_info=personal_info,
        education=[edu],
        experience=[],  # optional
        skills=[skill],
        awards=[],
        extracurriculars=[],
        publications=[],
        training=[],
        references=[],
        additional_info=[],
    )


def _make_minimal_metadata(sections_requested: int, sections_generated: int) -> Metadata:
    """Minimal Metadata instance satisfying output_schema constraints."""
    return Metadata(
        generation_time_ms=0,
        model_version="test-model",
        retry_count=0,
        cache_hit=False,
        sections_requested=sections_requested,
        sections_generated=sections_generated,
        tokens_used=0,
        input_tokens=0,
        output_tokens=0,
        section_breakdown=[],
        cost_estimate_thb=0.0,
        cost_estimate_usd=0.0,
        profile_info=None,
    )


# ---------------------------------------------------------------------------
# _decode_job_skill_code
# ---------------------------------------------------------------------------


def test_decode_job_skill_code_drops_simple_level_suffix():
    assert _decode_job_skill_code("Python_L3") == "Python"
    assert _decode_job_skill_code("Machine_Learning_L4") == "Machine Learning"


def test_decode_job_skill_code_without_level_suffix_returns_raw():
    assert _decode_job_skill_code("Excel") == "Excel"
    assert _decode_job_skill_code("data_analysis") == "data analysis"


# ---------------------------------------------------------------------------
# extract_canonical_jd_required_skills
# ---------------------------------------------------------------------------


def test_extract_canonical_jd_required_skills_prefers_explicit_field():
    profile = _make_minimal_student_profile()

    # NOTE: relies on CVGenerationRequest having jd_required_skills field
    req = CVGenerationRequest(
        user_id="U-TEST",
        student_profile=profile,
        jd_required_skills=[
            "Python",
            "Leadership",
            "Python",  # duplicate – should be deduped
        ],
    )

    skills = extract_canonical_jd_required_skills(req)

    # Explicit list should be used as-is (deduped, order preserved)
    assert skills == ["Python", "Leadership"]


def test_extract_canonical_jd_required_skills_from_role_and_jd_taxonomy():
    profile = _make_minimal_student_profile()

    role = RoleTaxonomy(
        role_title="Data Analyst",
        role_description="Analyze data and build dashboards.",
        role_required_skills=["Excel", "SQL"],
    )

    jd = JobTaxonomy(
        job_title="Junior Data Analyst",
        job_required_skills=[
            "Python_L3",
            "Data_Analysis_L2",
        ],
        job_responsibilities=["Build dashboards", "Support decision making"],
        company_info=None,
    )

    # No explicit jd_required_skills → should fallback to role + JD
    req = CVGenerationRequest(
        user_id="U-TEST",
        student_profile=profile,
        target_role_taxonomy=role,
        target_jd_taxonomy=jd,
    )

    skills = extract_canonical_jd_required_skills(req)

    # Expect 4 unique skills, in order: role first, then decoded JD codes
    assert len(skills) == 4
    assert skills[:2] == ["Excel", "SQL"]
    assert "Python" in skills
    assert "Data Analysis" in skills


# ---------------------------------------------------------------------------
# _build_normalized_skill_map + _match_skills_in_text
# ---------------------------------------------------------------------------


def test_build_normalized_skill_map_and_match_skills_in_text():
    jd_skills = ["Python", "Data Analysis", "Teamwork"]
    norm_map = _build_normalized_skill_map(jd_skills)

    # Normalized keys exist and map back to canonical names
    assert "python" in norm_map
    assert "data analysis" in norm_map
    assert norm_map["python"] == "Python"

    section_text = (
        "I have strong Python programming skills and data analysis experience. "
        "I also enjoy working in teams and demonstrate good teamwork."
    )

    matched = _match_skills_in_text(section_text, norm_map)

    # All three skills should be matched
    assert set(matched) == {"Python", "Data Analysis", "Teamwork"}


def test_match_skills_in_text_handles_empty_inputs_gracefully():
    assert _match_skills_in_text("", {}) == []
    assert _match_skills_in_text("Some text", {}) == []
    assert _match_skills_in_text("", {"python": "Python"}) == []


# ---------------------------------------------------------------------------
# annotate_matched_jd_skills
# ---------------------------------------------------------------------------


def test_annotate_matched_jd_skills_populates_section_fields():
    # Build a simple response with 2 sections
    sections = {
        "profile_summary": SectionContent(
            text="Experienced Python developer with strong data analysis skills.",
            word_count=8,
            matched_jd_skills=[],
            confidence_score=1.0,
        ),
        "skills": SectionContent(
            text="- Python\n- SQL\n- Communication",
            word_count=4,
            matched_jd_skills=[],
            confidence_score=1.0,
        ),
    }

    metadata = _make_minimal_metadata(
        sections_requested=2,
        sections_generated=2,
    )

    response = CVGenerationResponse(
        job_id="JOB_TEST",
        template_id="T_EMPLOYER_STD_V3",
        language="en",
        status=GenerationStatus.COMPLETED,
        sections=sections,
        skills=None,
        metadata=metadata,
        justification=Justification(),
        quality_metrics=None,
        warnings=[],
        error=None,
        error_details=None,
    )

    jd_required_skills = ["Python", "Data Analysis", "Leadership"]

    updated = annotate_matched_jd_skills(response, jd_required_skills)

    # profile_summary should match Python + Data Analysis
    summary_matches = updated.sections["profile_summary"].matched_jd_skills
    assert "Python" in summary_matches
    assert "Data Analysis" in summary_matches
    assert "Leadership" not in summary_matches  # not present in text

    # skills section should at least match Python
    skills_matches = updated.sections["skills"].matched_jd_skills
    assert "Python" in skills_matches
    assert "Leadership" not in skills_matches


def test_annotate_matched_jd_skills_noop_when_no_jd_skills():
    sections = {
        "profile_summary": SectionContent(
            text="Some text that mentions Python.",
            word_count=5,
            matched_jd_skills=[],
            confidence_score=1.0,
        ),
    }

    metadata = _make_minimal_metadata(
        sections_requested=1,
        sections_generated=1,
    )

    response = CVGenerationResponse(
        job_id="JOB_TEST",
        template_id="T_EMPLOYER_STD_V3",
        language="en",
        status=GenerationStatus.COMPLETED,
        sections=sections,
        skills=None,
        metadata=metadata,
        justification=Justification(),
        quality_metrics=None,
        warnings=[],
        error=None,
        error_details=None,
    )

    updated = annotate_matched_jd_skills(response, jd_required_skills=[])

    # No JD skills → matched_jd_skills should remain unchanged (empty)
    assert updated.sections["profile_summary"].matched_jd_skills == []
