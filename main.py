# main.py
"""
Main access point for CV generation (Stages A–D).

Pipeline:
    Stage A: Guardrails (validation, sanitization, evidence plan)
    Stage B: LLM generation (CVGenerationEngine)
    Stage C: Post-validation / trimming
    Stage D: Packaging & metadata enrichment (finalize_cv_response)

Error packaging via Stage D's `build_error_response` is intentionally
left to the API layer (e.g. FastAPI route) so this module stays HTTP-agnostic.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import structlog

from schemas.input_schema import CVGenerationRequest
from schemas.output_schema import CVGenerationResponse
from schemas.internal_schema import EvidencePlan

from functions.stage_a_guardrails import GuardrailsProcessor
from functions.stage_b_generation import CVGenerationEngine
from functions.stage_c_validation import run_stage_c_validation
from functions.stage_d_packaging import (
    LLMUsageSummary,
    finalize_cv_response,
)

from functions.utils.common import (
    model_validate_compat,
    model_dump_compat,
    select_llm_client_and_params,
    ensure_llm_metrics_env,
    load_yaml_dict,
)

logger = structlog.get_logger().bind(module="main")

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Core pipeline: Stage A → B → C → D
# ---------------------------------------------------------------------------


def run_cv_generation(
    request: CVGenerationRequest,
    *,
    request_id: str | None = None,
    llm_usage: LLMUsageSummary | None = None,
) -> Tuple[CVGenerationResponse, str]:
    """
    Run the CV generation pipeline for a single request.

    Stages:
      A) GuardrailsProcessor.validate_and_sanitize
         + GuardrailsProcessor.build_evidence_plan
      B) CVGenerationEngine.generate_cv
      C) run_stage_c_validation
      D) finalize_cv_response

    Args
    ----
    request:
        CVGenerationRequest (Stage 0 mapped into internal request model).
    request_id:
        Optional external/correlation ID. If omitted, Stage D will generate one.
    llm_usage:
        Optional LLMUsageSummary produced by upstream (e.g. llm_metrics).
        If provided, Stage D will attach tokens/cost to metadata.

    Returns
    -------
    (cv_response, request_id)
        - cv_response: Final packaged CVGenerationResponse (ready for HTTP body).
        - request_id: Final request ID used for logging / HTTP headers.
    """
    # Ensure env vars for llm_metrics (if used under the hood)
    ensure_llm_metrics_env()

    user_id = getattr(request, "user_id", "unknown")
    logger.info("pipeline_start", user_id=user_id, request_id=request_id)

    generation_start = datetime.now(timezone.utc)

    # Load parameters.yaml and pass the validation block into GuardrailsProcessor
    root = Path(__file__).resolve().parent
    params_path = root / "parameters" / "parameters.yaml"
    params = load_yaml_dict(params_path)

    # ----------------------------- Stage A ------------------------------
    validation_cfg = params.get("validation", {}) or {}

    guardrails = GuardrailsProcessor(validation_config=validation_cfg)

    validation = guardrails.validate_and_sanitize(request)
    if not validation.is_valid:
        logger.error(
            "stage_a_validation_failed",
            errors=validation.errors,
            warnings=validation.warnings,
        )
        error_msg = "; ".join(validation.errors or ["Stage A validation failed"])
        raise ValueError(error_msg)

    evidence_plan: EvidencePlan = guardrails.build_evidence_plan(request)
    logger.info(
        "stage_a_completed",
        warnings=validation.warnings or None,
        evidence_count=len(evidence_plan.evidences),
    )

    # ----------------------------- Stage B ------------------------------
    llm_client, engine_params = select_llm_client_and_params()
    engine = CVGenerationEngine(llm_client=llm_client, generation_params=engine_params)

    logger.info("stage_b_generation_start")
    response: CVGenerationResponse = engine.generate_cv(
        request=request,
        evidence_plan=evidence_plan,
    )
    logger.info("stage_b_generation_done")
    try:
        llm_usage = LLMUsageSummary(
            model=getattr(engine, "_model_name", "unknown"),
            prompt_tokens=getattr(engine, "_stage_b_total_input_tokens", 0),
            completion_tokens=getattr(engine, "_stage_b_total_output_tokens", 0),
            total_tokens=getattr(engine, "_stage_b_total_tokens", 0),
            total_cost_thb=getattr(engine, "_stage_b_total_cost_thb", 0.0),
            total_cost_usd=getattr(engine, "_stage_b_total_cost_usd", 0.0),
        )
    except Exception:
        llm_usage = None

    # ----------------------------- Stage C ------------------------------
    logger.info("stage_c_validation_start")

    # Prefer explicit template_info if present (legacy YAML / orchestrator).
    # If it's not present (pure new-API shape), fall back to Stage A's resolver.
    template_info = getattr(request, "template_info", None)
    if template_info is None:
        # Uses the same logic as Stage A to build a lightweight view
        template_info = GuardrailsProcessor._resolve_template_info(request)  # type: ignore[attr-defined]

    response = run_stage_c_validation(
        response=response,
        template_info=template_info,
        original_request=request,
    )
    logger.info("stage_c_validation_done")

    # ----------------------------- Stage D ------------------------------
    # Use profile_info as plain dict for metadata/profile header.
    profile_info_dict: Dict[str, Any] | None = None
    if getattr(request, "profile_info", None) is not None:
        try:
            profile_info_dict = request.profile_info.model_dump(mode="json")  # type: ignore[attr-defined]
        except Exception:
            # Fallback – best-effort conversion
            try:
                profile_info_dict = dict(request.profile_info)  # type: ignore[arg-type]
            except Exception:
                profile_info_dict = None

    logger.info("stage_d_packaging_start")
    generation_end = datetime.now(timezone.utc)

    response, final_request_id = finalize_cv_response(
        response,
        request_id=request_id,
        user_id=user_id,
        profile_info=profile_info_dict,
        llm_usage=llm_usage,
        generation_start=generation_start,
        generation_end=generation_end,
        original_request=request,
    )


    logger.info("stage_d_packaging_done", request_id=final_request_id)

    logger.info("pipeline_completed", user_id=user_id, request_id=final_request_id)
    return response, final_request_id


# ---------------------------------------------------------------------------
# CLI wrapper (for debugging without FastAPI)
# ---------------------------------------------------------------------------


def _cli() -> int:
    """
    CLI usage:

        python main.py path/to/request.json

    Where request.json matches CVGenerationRequest schema.
    """
    if len(sys.argv) < 2:
        print("Usage: python main.py path/to/request.json", file=sys.stderr)
        return 1

    in_path = Path(sys.argv[1])
    if not in_path.is_file():
        print(f"[ERROR] Input file not found: {in_path}", file=sys.stderr)
        return 1

    raw = json.loads(in_path.read_text(encoding="utf-8"))
    try:
        req = model_validate_compat(CVGenerationRequest, raw)
    except Exception as e:
        print(f"[ERROR] Failed to parse CVGenerationRequest: {e}", file=sys.stderr)
        return 1

    try:
        cv, req_id = run_cv_generation(req)
    except Exception as e:
        print(f"[ERROR] Pipeline failed: {e}", file=sys.stderr)
        return 1

    out = model_dump_compat(cv)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    # Optionally print request_id to stderr for debugging
    print(f"[info] request_id={req_id}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
