# api.py
from __future__ import annotations

from typing import Optional, Any, Dict
from types import SimpleNamespace  # 👈 NEW

import structlog
from fastapi import FastAPI, HTTPException, Header, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from schemas.input_schema import CVGenerationRequest
from schemas.output_schema import CVGenerationResponse
from main import run_cv_generation
from functions.utils.common import model_validate_compat  # 👈 same as CLI

logger = structlog.get_logger().bind(module="api")

app = FastAPI(
    title="CV Generation Service",
    version="1.0.0",
    description="Multi-stage CV generation pipeline (Stages A–D) exposed via FastAPI.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def _build_cv_request_from_public_payload(payload: Dict[str, Any]) -> CVGenerationRequest:
    """
    Map the public API JSON (which may include extra keys like `template_info`
    and `user_input_cv_text_by_section`) into a strict CVGenerationRequest.

    This mirrors what mock_api_generate_cv_v2 does:
    - Use only the allowed fields to build CVGenerationRequest
    - Then attach rich `template_info` and `user_input_cv_text_by_section`
      as extra attributes via object.__setattr__.
    """

    # ---- 1) Core payload for strict validation ----
    core_payload: Dict[str, Any] = {
        "user_id": payload.get("user_id"),
        "language": payload.get("language", "en"),
        "template_id": payload.get("template_id", "T_EMPLOYER_STD_V3"),
        "sections": payload.get("sections")
        or [
            "profile_summary",
            "skills",
            "experience",
            "education",
            "awards",
            "extracurricular",
        ],
        "student_profile": payload.get("student_profile"),
        "target_role_taxonomy": payload.get("target_role_taxonomy"),
        "target_jd_taxonomy": payload.get("target_jd_taxonomy"),
    }

    # Let Pydantic do all the real validation here
    request_obj: CVGenerationRequest = model_validate_compat(
        CVGenerationRequest,
        core_payload,
    )

    # ---- 2) Attach template_info view (rich template YAML-style object) ----
    tmpl_raw: Dict[str, Any] = payload.get("template_info") or {}

    if tmpl_raw:
        # Prefer sections_order from template; fallback to request.sections
        sections_order = tmpl_raw.get("sections_order") or request_obj.sections
        max_chars = tmpl_raw.get("max_chars_per_section") or {}

        object.__setattr__(
            request_obj,
            "template_info",
            SimpleNamespace(
                template_id=tmpl_raw.get("template_id", request_obj.template_id),
                sections=sections_order,
                max_chars_per_section=max_chars,
                language=request_obj.language.value,
                # you *can* add more if Stage B/C later want them:
                # name=tmpl_raw.get("name"),
                # style=tmpl_raw.get("style"),
            ),
        )

    # ---- 3) Attach user drafts (optional) ----
    drafts = payload.get("user_input_cv_text_by_section")
    if drafts is not None:
        object.__setattr__(request_obj, "user_input_cv_text_by_section", drafts)

    return request_obj


@app.post("/generate_cv", response_model=CVGenerationResponse)
async def generate_cv(
    payload: Dict[str, Any] = Body(...),
    x_request_id: Optional[str] = Header(default=None, alias="X-Request-ID"),
) -> CVGenerationResponse:
    """
    Main CV generation endpoint.

    - Accepts raw JSON (dict), which may contain extra keys like:
        * template_info
        * user_input_cv_text_by_section
    - Adapts it into a strict CVGenerationRequest via _build_cv_request_from_public_payload
    - Runs Stage A → B → C → D
    """
    # ---- Step 1: adapt + validate into CVGenerationRequest ----
    try:
        request_obj: CVGenerationRequest = _build_cv_request_from_public_payload(payload)
    except ValidationError as e:
        logger.warning("request_validation_error", errors=e.errors())
        raise HTTPException(status_code=422, detail=e.errors())
    except Exception as e:
        logger.warning("request_validation_error_generic", error=str(e))
        raise HTTPException(status_code=422, detail=f"Invalid request: {e}")

    # ---- Step 2: run pipeline ----
    try:
        cv_response, final_request_id = run_cv_generation(
            request_obj,
            request_id=x_request_id,
            llm_usage=None,
        )
        logger.info(
            "api_generate_cv_success",
            request_id=final_request_id,
            user_id=getattr(request_obj, "user_id", "unknown"),
        )
        return cv_response

    except ValueError as e:
        # Stage A validation failure etc.
        logger.warning("api_generate_cv_stage_a_error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error("api_generate_cv_internal_error", error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")
