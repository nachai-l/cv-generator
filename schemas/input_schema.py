"""Input schema definitions for CV generation requests.

These models define the validated, sanitized input contract between the
frontend / data layer and the CV generation service. All fields are
designed to be:
- LLM-friendly (clear IDs and bounded lengths)
- ATS-friendly (normalized structure)
- Safe (no extra/unknown fields, basic guardrails in validators)
"""

from datetime import date
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, EmailStr, Field, HttpUrl, field_validator, model_validator

class SkillLevel(str, Enum):
    """Normalized skill proficiency levels used across the platform.

    Values follow a stable, machine-readable enum pattern (L1–L4) and are
    referenced by:
    - student profiles (self-reported skills)
    - job taxonomies (required skill levels)
    - downstream analytics (skill gap / progression)
    """

    L1_BEGINNER = "L1_Beginner"
    L2_INTERMEDIATE = "L2_Intermediate"
    L3_ADVANCED = "L3_Advanced"
    L4_EXPERT = "L4_Expert"


class Language(str, Enum):
    """Supported CV generation languages.

    Controls:
    - Output language of generated CV sections
    - Prompting and templates used by the LLM
    """

    EN = "en"
    TH = "th"


class LanguageTone(str, Enum):
    """Supported tone styles for generated CV text.

    Controls:
    - Wording style for all sections (profile summary, experience, etc.)
    - Prompt instructions given to the LLM
    """

    FORMAL = "formal"
    NEUTRAL = "neutral"
    ACADEMIC = "academic"
    FUNNY = "funny"
    CASUAL = "casual"


class PersonalInfo(BaseModel):
    """Minimal personal contact information for a student.

    This block is intentionally small to:
    - Avoid collecting unnecessary PII
    - Provide enough data for CV header and recruiter contact

    Validation ensures:
    - Name has no control characters
    - Email is syntactically valid
    - Phone/LinkedIn are optional and bounded in length
    """

    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: str | None = Field(None, max_length=20)
    linkedin: HttpUrl | None = None

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Remove control characters and trim whitespace from name."""
        import re

        cleaned = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", v)
        return cleaned.strip()

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: str | None) -> str | None:
        """Trim whitespace; treat empty strings as None for consistency."""
        if v is None:
            return None
        v = v.strip()
        return v or None


class Education(BaseModel):
    """Formal education entry for the student.

    Each record represents one program (e.g., Ph.D., Diploma):
    - `id` is a stable reference used in evidence mapping
    - `degree` and `institution` appear in the CV
    - `gpa` is optional but constrained to 0.0–4.0
    - `start_date` and `graduation_date` enable:
        * timeline reasoning (e.g., current vs completed)
        * experience ordering
    """

    id: str = Field(..., pattern=r"^edu#[a-zA-Z0-9_-]+$")
    degree: str = Field(..., min_length=1, max_length=200)
    institution: str = Field(..., min_length=1, max_length=200)
    gpa: float | None = Field(None, ge=0.0, le=4.0)
    start_date: date
    graduation_date: date | None
    major: str | None = Field(None, max_length=200)

    @model_validator(mode="after")
    def validate_dates(self) -> "Education":
        """Ensure graduation_date is not earlier than start_date, if provided."""
        if self.graduation_date and self.graduation_date < self.start_date:
            raise ValueError("graduation_date cannot be earlier than start_date")
        return self

class Experience(BaseModel):
    """Work or internship experience entry.

    Represents a single position (full-time, part-time, or internship):
    - `id` encodes type: work_exp#... or intern_exp#...
    - `title` and `company` appear as section headers
    - `start_date` and `end_date` enable:
        * timeline reasoning (e.g., current vs completed)
        * experience ordering
    - `responsibilities` are bullet-level facts limited in:
        * number of items (1–10)
        * per-item length (truncated to 500 chars)

    These constraints help keep LLM prompts compact and focused.
    """

    id: str = Field(..., pattern=r"^(work|intern)_exp#[a-zA-Z0-9_-]+$")
    title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(..., min_length=1, max_length=200)
    start_date: date
    end_date: date | None
    responsibilities: list[str] = Field(..., min_length=1, max_length=10) # Must include at least one responsibility

    @field_validator("responsibilities")
    @classmethod
    def truncate_responsibilities(cls, v: list[str]) -> list[str]:
        """Normalize and limit responsibility text length."""
        normalized: list[str] = []
        for r in v:
            text = (r or "").strip()
            if not text:
                continue
            normalized.append(text[:500])
        # Let pydantic enforce min_length=1 if all were empty
        return normalized

    @model_validator(mode="after")
    def validate_end_date(self) -> "Experience":
        """Ensure end_date is not earlier than start_date, if provided."""
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be earlier than start_date")
        return self


class Skill(BaseModel):
    """Structured representation of a student's skill.

    Each skill is:
    - Addressable via a stable `id` (e.g., 'skill#python')
    - Human-readable via `name` (e.g., 'Python', 'Data Analysis')
    - Quantified via `level` (SkillLevel)

    These are aligned with the job taxonomy to enable:
    - Skill matching
    - Gap analysis
    - Consistent LLM prompting across use cases
    """

    id: str = Field(..., pattern=r"^skill#[a-zA-Z0-9_-]+$")
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(..., min_length=1, max_length=200)
    level: SkillLevel


class Award(BaseModel):
    """Award, scholarship, or notable achievement.

    Typical examples:
    - Competition prizes
    - Dean's list / academic awards
    - Scholarships or recognition from organizations

    Used to enrich the CV with evidence of excellence without
    overloading the main education/experience sections.
    """

    id: str = Field(..., pattern=r"^award#[a-zA-Z0-9_-]+$")
    title: str = Field(..., max_length=200)
    issuer: str = Field(..., max_length=200)
    date: date
    description: str | None = Field(None, max_length=500)


class Extracurricular(BaseModel):
    """Extracurricular or co-curricular activity.

    Captures nonwork experiences that still demonstrate:
    - Leadership (club officer roles)
    - Teamwork (sports, projects)
    - Initiative (student-led activities)

    These help the LLM construct well-rounded profile summaries
    beyond purely academic/work achievements.
    """

    id: str = Field(..., pattern=r"^extra#[a-zA-Z0-9_-]+$")
    title: str = Field(..., max_length=200)
    organization: str = Field(..., max_length=200)
    role: str | None = Field(None, max_length=100)
    duration: str = Field(..., max_length=100)
    description: str | None = Field(None, max_length=500)

class Publication(BaseModel):
    """Academic or professional publication."""

    id: str = Field(..., pattern=r"^pub#[a-zA-Z0-9_-]+$")
    title: str = Field(..., max_length=300)
    venue: str | None = Field(None, max_length=200)  # journal / conference / media
    year: int | None = Field(None, ge=1900, le=2100)
    link: HttpUrl | None = None
    description: str | None = Field(None, max_length=500)


class Training(BaseModel):
    """Course, bootcamp, or professional training."""

    id: str = Field(..., pattern=r"^training#[a-zA-Z0-9_-]+$")
    title: str = Field(..., max_length=200)
    provider: str | None = Field(None, max_length=200)
    training_date: date | None = None
    description: str | None = Field(None, max_length=500)


class Reference(BaseModel):
    """Professional or academic reference contact."""
    id: str = Field(..., pattern=r"^ref#[a-zA-Z0-9_-]+$")
    name: str = Field(..., max_length=200)
    title: str | None = Field(None, max_length=200)
    company: str | None = Field(None, max_length=200)
    email: EmailStr | None = None
    phone: str | None = Field(None, max_length=50)
    relationship: str | None = Field(None, max_length=200)
    note: str | None = Field(None, max_length=300)


class AdditionalInfoItem(BaseModel):
    """Catch-all extra information as key-value items."""
    id: str = Field(..., pattern=r"^add#[a-zA-Z0-9_-]+$")
    label: str = Field(..., max_length=100)   # e.g. 'Nationality', 'Languages', 'Interests'
    value: str = Field(..., max_length=300)

class StudentProfile(BaseModel):
    """Aggregated, sanitized student profile used as LLM ground truth.

    This is the main data package passed (in structured form) to the LLM:
    - Enforces minimum evidence:
        * ≥1 education entry
        * ≥2 skills
    - Caps list sizes to keep prompts small and predictable:
        * education: up to 5 entries
        * experience: up to 10 entries
        * awards / extracurriculars: up to 10 each
    - Forbids extra fields to prevent accidental prompt injection
      via unexpected keys.
    """

    personal_info: PersonalInfo
    education: list[Education] = Field(..., min_length=1, max_length=5)
    experience: list[Experience] = Field(default_factory=list, max_length=10)
    skills: list[Skill] = Field(..., min_length=1, max_length=30)
    awards: list[Award] = Field(default_factory=list, max_length=10)
    extracurriculars: list[Extracurricular] = Field(default_factory=list, max_length=10)

    publications: list[Publication] = Field(default_factory=list, max_length=10)
    training: list[Training] = Field(default_factory=list, max_length=10)
    references: list[Reference] = Field(default_factory=list, max_length=5)
    additional_info: list[AdditionalInfoItem] = Field(default_factory=list, max_length=10)

    # Reject unexpected fields at this level (matches "no extra fields" design goal)
    model_config = {"extra": "forbid"}


class CompanyInfo(BaseModel):
    """Basic information about the target company.

    Used for:
    - Tailoring the tone and focus of the profile summary
    - Allowing the LLM to reference the company in a generic,
      non-hallucinated way (name + industry only)
    """

    name: str = Field(..., max_length=200)
    industry: str | None = Field(None, max_length=100)


class RoleTaxonomy(BaseModel):
    """Structured description of a target role (job family / generic role).

    This is not tied to a specific company or posting and can be reused
    across multiple JDs.

    Examples:
    - 'Data Analyst'
    - 'Frontend Engineer'
    """

    role_title: str = Field(..., min_length=1, max_length=200)
    role_description: str = Field(..., max_length=500)
    role_required_skills: list[str] = Field(..., min_length=1, max_length=20)


class JobTaxonomy(BaseModel):
    """Structured description of the target job / JD.

    Represents the normalized, system-internal view of a job:
    - `job_title`: concrete title for this job (can differ slightly from role)
    - `job_required_skills`: codes like 'Python_L3', aligned with SkillLevel
    - `job_responsibilities`: short bullet points of expected tasks
    - `company_info`: optional, used to personalize wording

    This schema is used as:
    - LLM input for role-specific generation
    - Anchor for evaluating JD alignment in validation.
    """

    job_title: str = Field(..., min_length=1, max_length=200)
    job_required_skills: list[str] = Field(..., min_length=1, max_length=20)
    job_responsibilities: list[str] = Field(default_factory=list, max_length=10)
    company_info: CompanyInfo | None = None

    model_config = {"extra": "forbid"}


class UserInputSkillItem(BaseModel):
    """User-provided skill override for the CV text."""
    name: str | None = Field(None, max_length=100)
    level: str | None = Field(None, max_length=50)
    model_config = {"extra": "forbid"}


class UserInputExperienceItem(BaseModel):
    """User-provided experience override for the CV text."""
    title: str | None = Field(None, max_length=200)
    company: str | None = Field(None, max_length=200)
    period: str | None = Field(None, max_length=100)
    highlights: list[str] | None = Field(None)

    @field_validator("highlights")
    @classmethod
    def normalize_highlights(cls, v: list[str] | None) -> list[str] | None:
        """Trim whitespace and cap highlight length."""
        if not v:
            return v
        normalized: list[str] = []
        for h in v:
            text = (h or "").strip()
            if not text:
                continue
            normalized.append(text[:500])
        return normalized

    model_config = {"extra": "forbid"}


class UserInputEducationItem(BaseModel):
    """User-provided education override for the CV text."""
    degree: str | None = Field(None, max_length=200)
    institution: str | None = Field(None, max_length=200)
    location: str | None = Field(None, max_length=200)
    model_config = {"extra": "forbid"}


class UserInputCVTextBySection(BaseModel):
    """Optional user-provided overrides for CV sections.

    Every field is optional. If a value is None or empty, the pipeline
    will fall back to structured profile data.
    """

    profile_summary: str | None = Field(None, max_length=2000)
    skills: list[UserInputSkillItem] | None = None
    experience: list[UserInputExperienceItem] | None = None
    education: list[UserInputEducationItem] | None = None

    # Allows only known keys; prevents user sending arbitrary dicts.
    model_config = {"extra": "forbid"}


class CVGenerationRequest(BaseModel):
    """Top-level CV generation request payload.

    This is the external API contract for the CV generation service.
    It bundles:
    - `user_id`: platform user identifier (no PII, just a stable key)
    - `language`: desired output language (EN/TH)
    - `language_tone`: desired tone style (formal, neutral, academic, funny)
    - `template_id`: which CV template style to use
    - `sections`: which CV sections to generate in this call
    - `student_profile`: ground-truth data for the student
    - `target_role_taxonomy`: optional generic role context
    - `target_jd_taxonomy`: normalized target job / JD information

    Design goals:
    - Safe by default: no extra fields, bounded list sizes
    - Compatible with caching: (user_id, template_id, sections, JD hash)
    - Directly mappable to LLM `structured_data` arguments.
    """

    user_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", max_length=50)
    language: Language = Language.EN
    language_tone: LanguageTone = LanguageTone.FORMAL
    template_id: str = Field(
        default="T_EMPLOYER_STD_V3", pattern=r"^T_[A-Z_]+_V\d+$"
    )
    sections: list[
        Literal[
            "profile_summary",
            "skills",
            "experience",
            "education",
            "projects",
            "certifications",
            "awards",
            "extracurricular",
            "volunteering",
            "interests",
            "publications",
            "training",
            "references",
            "additional_info",
        ]
    ] = Field(
        default=[
            "profile_summary",
            "skills",
            "experience",
            "education",
            "awards",
            "extracurricular",
        ],
        min_length=1,
        max_length=10,
    )
    student_profile: StudentProfile

    # Optional user-provided overrides for specific CV sections
    user_input_cv_text_by_section: dict[str, Any] | None = None
    # Optional generic role-level context (job family)
    target_role_taxonomy: RoleTaxonomy | None = None
    # Optional context for this generation request (normalized JD)
    target_jd_taxonomy: JobTaxonomy | None = None

    # Reject unexpected fields at the top-level request
    model_config = {"extra": "forbid"}

    @field_validator("sections")
    @classmethod
    def deduplicate_sections(cls, sections: list[str]) -> list[str]:
        """Ensure sections list has no duplicates while preserving order."""
        seen: set[str] = set()
        deduped: list[str] = []
        for s in sections:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        return deduped
