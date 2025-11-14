"""Schema definitions for CV generation."""

from schemas.input_schema import (
    CVGenerationRequest,
    Education,
    Experience,
    Extracurricular,
    JobTaxonomy,
    Language,
    PersonalInfo,
    Skill,
    SkillLevel,
    StudentProfile,
)

from .cv_template_schema import TemplateSchema

# from schemas.output_schema import (
#     CVGenerationResponse,
#     EvidenceMapping,
#     GenerationStatus,
#     Justification,
#     Metadata,
#     SectionContent,
# )

__all__ = [
    # Input schemas
    "CVGenerationRequest",
    "StudentProfile",
    "PersonalInfo",
    "Education",
    "Experience",
    "Skill",
    "SkillLevel",
    "Extracurricular",
    "JobTaxonomy",
    "Language",
    # Output schemas
    # "CVGenerationResponse",
    # "SectionContent",
    # "Metadata",
    # "Justification",
    # "EvidenceMapping",
    # "GenerationStatus",
]