"""CV template rendering and management."""

from local_cv_templates.cv_templates import (
    EMPLOYER_STANDARD_V3,
    MINIMALIST_V1,
    CREATIVE_V1,
    get_template,
    render_cv_html,
    render_cv_markdown,
    save_cv_html,
    save_cv_markdown,
)

__all__ = [
    "EMPLOYER_STANDARD_V3",
    "MINIMALIST_V1",
    "CREATIVE_V1",
    "get_template",
    "render_cv_html",
    "render_cv_markdown",
    "save_cv_html",
    "save_cv_markdown",
]