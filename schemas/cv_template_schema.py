# schemas/cv_template_schema.py
from typing import Literal

from pydantic import BaseModel, Field


class TemplateSchema(BaseModel):
    """API-facing representation of a CV template.

    This is what you expose in the API so clients know:
    - which template IDs are available
    - what sections and styles they have
    - max character limits per section (for UI validation)
    """

    template_id: str = Field(..., description="Stable template identifier")
    name: str = Field(..., description="Human-readable template name")
    style: Literal["modern", "classic", "minimalist", "creative"] = Field(
        ..., description="High-level visual style"
    )
    sections_order: list[str] = Field(
        ...,
        description=(
            "Ordered section keys as used in CVGenerationResponse.sections "
            "(e.g. 'profile_summary', 'skills', ...)"
        ),
    )
    color_scheme: dict[str, str] = Field(
        ...,
        description=(
            "Semantic colors (primary, secondary, accent, background, text, muted) "
            "as hex codes"
        ),
    )
    font_family: str = Field(
        ..., description="CSS font-family string used in rendered HTML"
    )
    max_pages: int = Field(
        ...,
        ge=1,
        description="Recommended maximum number of pages for this template",
    )
    max_chars_per_section: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Character limits per section key for UI validation / truncation hints. "
            "Keys should match section keys in `sections_order`."
        ),
    )
