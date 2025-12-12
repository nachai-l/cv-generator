# schemas/input_schema.py
"""Input schema definitions for CV generation requests.

These models define the validated, sanitized input contract between the
frontend / data layer and the CV generation service.

Design intent (v1):
- Be tolerant to upstream Data API / Orchestrator variations.
- Keep bounded lengths and list caps for LLM friendliness.
- Forbid unknown fields at major boundaries for safety.
- Avoid enforcing strict ID patterns until upstream IDs are standardized.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import (
    AliasChoices,
    BaseModel,
    EmailStr,
    Field,
    HttpUrl,
    field_validator,
    model_validator,
)


class SkillLevel(str, Enum):
    """Normalized skill proficiency levels used across the platform."""
    L1_BEGINNER = "L1_Beginner"
    L2_INTERMEDIATE = "L2_Intermediate"
    L3_ADVANCED = "L3_Advanced"
    L4_EXPERT = "L4_Expert"


class Language(str, Enum):
    """Supported CV generation languages."""
    EN = "en"
    TH = "th"


class LanguageTone(str, Enum):
    """Supported tone styles for generated CV text."""
    FORMAL = "formal"
    NEUTRAL = "neutral"
    ACADEMIC = "academic"
    FUNNY = "funny"
    CASUAL = "casual"


class PersonalInfo(BaseModel):
    """Minimal personal contact information for a student."""
    name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: Optional[str] = Field(default=None, max_length=20)
    linkedin: Optional[HttpUrl] = None

    @field_validator("name")
    @classmethod
    def sanitize_name(cls, v: str) -> str:
        """Remove control characters and trim whitespace from name."""
        import re
        cleaned = re.sub(r"[\x00-\x1F\x7F-\x9F]", "", v)
        return cleaned.strip()

    @field_validator("phone")
    @classmethod
    def normalize_phone(cls, v: Optional[str]) -> Optional[str]:
        """Trim whitespace; treat empty strings as None for consistency."""
        if v is None:
            return None
        v = v.strip()
        return v or None


class Education(BaseModel):
    """Formal education entry for the student.

    NOTE: `id` is free-form for now (future enforcement can add patterns).
    """
    id: str = Field(
        ...,
        description=(
            "Free-form ID. Future strict format example: pattern='^edu#[a-zA-Z0-9_-]+$'"
        ),
    )
    degree: str = Field(..., min_length=1, max_length=200)
    institution: str = Field(..., min_length=1, max_length=200)
    gpa: Optional[float] = Field(default=None, ge=0.0, le=4.0)
    start_date: date
    graduation_date: Optional[date] = None
    major: Optional[str] = Field(default=None, max_length=200)
    honors: Optional[str] = Field(default=None, max_length=200)

    @model_validator(mode="after")
    def validate_dates(self) -> "Education":
        """Ensure graduation_date is not earlier than start_date, if provided."""
        if self.graduation_date and self.graduation_date < self.start_date:
            raise ValueError("graduation_date cannot be earlier than start_date")
        return self


class Experience(BaseModel):
    """Work or internship experience entry.

    NOTE: `id` is free-form for now (future enforcement can add patterns).
    """
    id: str = Field(
        ...,
        description=(
            "Free-form ID. Future strict format example: work_exp#<slug> or intern_exp#<slug>"
        ),
    )
    title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(..., min_length=1, max_length=200)
    start_date: date
    end_date: Optional[date] = None
    responsibilities: list[str] = Field(..., min_length=1, max_length=10)

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
        return normalized

    @model_validator(mode="after")
    def validate_end_date(self) -> "Experience":
        """Ensure end_date is not earlier than start_date, if provided."""
        if self.end_date and self.end_date < self.start_date:
            raise ValueError("end_date cannot be earlier than start_date")
        return self


class Skill(BaseModel):
    """Structured representation of a student's skill.

    NOTE: `id` is free-form for now (future enforcement can add patterns).
    """
    id: str = Field(
        ...,
        description=(
            "Free-form ID. Future strict format example: pattern='^skill#[a-zA-Z0-9_-]+$'"
        ),
    )
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, max_length=200)
    level: SkillLevel


class Award(BaseModel):
    """Award, scholarship, or notable achievement."""
    id: str = Field(
        ...,
        description=(
            "Free-form ID. Future strict format example: pattern='^award#[a-zA-Z0-9_-]+$'"
        ),
    )
    title: str = Field(..., max_length=200)
    issuer: str = Field(..., max_length=200)
    date: Optional[date] = None
    description: Optional[str] = Field(default=None, max_length=500)


class Extracurricular(BaseModel):
    """Extracurricular or co-curricular activity."""
    id: str = Field(
        ...,
        description=("Free-form ID. Future strict format example: '^extra#[slug]'"),
    )
    title: str = Field(..., max_length=200)
    organization: str = Field(..., max_length=200)
    role: Optional[str] = Field(default=None, max_length=100)
    duration: Optional[str] = Field(default=None, max_length=100)
    description: Optional[str] = Field(default=None, max_length=500)


class Publication(BaseModel):
    """Academic or professional publication."""
    id: str = Field(
        ...,
        description=("Free-form ID. Future strict pattern example: '^pub#[slug]'"),
    )
    title: str = Field(..., max_length=300)
    venue: Optional[str] = Field(default=None, max_length=200)
    year: Optional[int] = Field(default=None, ge=1900, le=2100)
    link: Optional[HttpUrl] = None
    description: Optional[str] = Field(default=None, max_length=500)


class Training(BaseModel):
    """Course, bootcamp, or professional training."""
    id: str = Field(
        ...,
        description=("Free-form ID. Future strict pattern example: '^training#[slug]'"),
    )
    title: str = Field(..., max_length=200)
    provider: Optional[str] = Field(default=None, max_length=200)
    training_date: Optional[date] = None
    description: Optional[str] = Field(default=None, max_length=500)


class Reference(BaseModel):
    """Professional or academic reference contact."""
    id: str = Field(
        ...,
        description=("Free-form ID. Future strict pattern example: '^ref#[slug]'"),
    )
    name: str = Field(..., max_length=200)
    title: Optional[str] = Field(default=None, max_length=200)
    company: Optional[str] = Field(default=None, max_length=200)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(default=None, max_length=50)
    relationship: Optional[str] = Field(default=None, max_length=200)
    note: Optional[str] = Field(default=None, max_length=300)


class AdditionalInfoItem(BaseModel):
    """Catch-all extra information as key-value items."""
    id: str = Field(
        ...,
        description=("Free-form ID. Future strict pattern example: '^add#[slug]'"),
    )
    label: str = Field(..., max_length=100)
    value: str = Field(..., max_length=300)


class StudentProfile(BaseModel):
    """Aggregated, sanitized student profile used as LLM ground truth."""
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

    model_config = {"extra": "forbid"}


class CompanyInfo(BaseModel):
    """Basic information about the target company."""
    name: str = Field(..., max_length=200)
    industry: Optional[str] = Field(default=None, max_length=100)


class RoleTaxonomy(BaseModel):
    """Structured description of a target role (job family / generic role)."""
    role_id: Optional[str] = None
    role_title: str = Field(..., min_length=1, max_length=200)
    role_description: str = Field(..., max_length=500)

    # Accept list[str] or list[dict] upstream; normalize to list[str] here
    role_required_skills: list[str] = Field(..., min_length=1, max_length=50)

    @field_validator("role_required_skills", mode="before")
    @classmethod
    def _coerce_role_required_skills(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
                elif isinstance(item, dict):
                    name = (item.get("role_required_skills_name") or "").strip()
                    if name:
                        out.append(name)
            return out
        return []


class JobTaxonomy(BaseModel):
    """Structured description of the target job / JD (Data API aware, tolerant).

    Accepts:
    - jd_id (preferred)
    - job_id (legacy/alternate) -> mapped into jd_id
    """
    jd_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("jd_id", "job_id"),
    )

    job_title: str = Field(..., min_length=1, max_length=200)

    # Accept list[str] or list[dict] upstream; normalize to list[str] here
    job_required_skills: list[str] = Field(..., min_length=1, max_length=50)

    # Accept list[str] or list[dict] upstream; normalize to list[str] here
    job_responsibilities: list[str] = Field(default_factory=list, max_length=50)

    # Optional company info, either flattened or nested
    company_name: Optional[str] = Field(default=None, max_length=200)
    company_industry: Optional[str] = Field(default=None, max_length=100)
    company_info: Optional[CompanyInfo] = None

    model_config = {"extra": "forbid"}

    @field_validator("job_required_skills", mode="before")
    @classmethod
    def _coerce_job_required_skills(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
                elif isinstance(item, dict):
                    name = (item.get("jd_required_skills_name") or item.get("name") or "").strip()
                    if name:
                        out.append(name)
            return out
        return []

    @field_validator("job_responsibilities", mode="before")
    @classmethod
    def _coerce_job_responsibilities(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if isinstance(v, list):
            out: list[str] = []
            for item in v:
                if isinstance(item, str):
                    s = item.strip()
                    if s:
                        out.append(s)
                elif isinstance(item, dict):
                    resp = (item.get("responsibility") or item.get("text") or "").strip()
                    if resp:
                        out.append(resp)
            return out
        return []

    @model_validator(mode="after")
    def _build_company_info(self) -> "JobTaxonomy":
        if self.company_info is None and (self.company_name or self.company_industry):
            self.company_info = CompanyInfo(
                name=self.company_name or "",
                industry=self.company_industry,
            )
        return self


class UserInputSkillItem(BaseModel):
    """User-provided skill override for the CV text."""
    name: Optional[str] = Field(default=None, max_length=100)
    level: Optional[str] = Field(default=None, max_length=50)
    model_config = {"extra": "forbid"}


class UserInputExperienceItem(BaseModel):
    """User-provided experience override for the CV text."""
    title: Optional[str] = Field(default=None, max_length=200)
    company: Optional[str] = Field(default=None, max_length=200)
    period: Optional[str] = Field(default=None, max_length=100)
    highlights: Optional[list[str]] = None

    @field_validator("highlights")
    @classmethod
    def normalize_highlights(cls, v: Optional[list[str]]) -> Optional[list[str]]:
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
    degree: Optional[str] = Field(default=None, max_length=200)
    institution: Optional[str] = Field(default=None, max_length=200)
    location: Optional[str] = Field(default=None, max_length=200)
    model_config = {"extra": "forbid"}


class UserInputCVTextBySection(BaseModel):
    """Optional user-provided overrides for CV sections."""
    profile_summary: Optional[str] = Field(default=None, max_length=2000)
    skills: Optional[list[UserInputSkillItem]] = None
    experience: Optional[list[UserInputExperienceItem]] = None
    education: Optional[list[UserInputEducationItem]] = None

    model_config = {"extra": "forbid"}


class CVGenerationRequest(BaseModel):
    """Top-level CV generation request payload (Stage-0 contract)."""
    user_id: str = Field(..., pattern=r"^[A-Za-z0-9_-]+$", max_length=50)
    language: Language = Language.EN
    language_tone: LanguageTone = LanguageTone.FORMAL
    template_id: str = Field(default="T_EMPLOYER_STD_V3")
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
        default=["profile_summary", "skills", "experience", "education"],
        min_length=1,
        max_length=20,
    )

    student_profile: StudentProfile
    user_input_cv_text_by_section: Optional[dict[str, Any]] = None
    target_role_taxonomy: Optional[RoleTaxonomy] = None
    target_jd_taxonomy: Optional[JobTaxonomy] = None
    jd_required_skills: Optional[list[str]] = None

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
