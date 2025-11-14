# functions/utils/common.py
"""
common utility helpers used across the CV Generation pipeline.

This includes:
- Pydantic v1/v2 compatibility helpers
- Generation parameter loading from parameters.yaml
- LLM client selection (real vs stub)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
import structlog

logger = structlog.get_logger().bind(module="utils.common")

# Root of project (two dirs up from utils/)
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Pydantic helpers (v1 / v2)
# ---------------------------------------------------------------------------

def model_validate_compat(model_cls, data: Any):
    """Pydantic v1/v2 compatible model loader."""
    if hasattr(model_cls, "model_validate"):
        return model_cls.model_validate(data)  # pydantic v2
    return model_cls.parse_obj(data)  # pydantic v1


def model_dump_compat(model_obj: Any) -> Dict[str, Any]:
    """Pydantic v1/v2 compatible model dumper."""
    if hasattr(model_obj, "model_dump"):
        return model_obj.model_dump(mode="json")
    return model_obj.dict()

# ---------------------------------------------------------------------------
# ENV helper
# ---------------------------------------------------------------------------
def ensure_llm_metrics_env() -> None:
    """
    Ensure environment variables required by llm_metrics.py are set.

    These defaults keep llm_metrics fully functional for both:
      - local CLI / testing
      - FastAPI service runtime

    If the API gateway or orchestrator wants custom log paths,
    it can override these env vars before calling this function.
    """
    params_path = ROOT / "parameters" / "parameters.yaml"
    log_path = ROOT / "local_logs" / "llm_call_logs.csv"

    os.environ.setdefault("PARAMETERS_YAML", params_path.as_posix())
    os.environ.setdefault("LLM_METRICS_CSV", log_path.as_posix())

# ---------------------------------------------------------------------------
# Generation config loading
# ---------------------------------------------------------------------------

def load_generation_params() -> Dict[str, Any]:
    """Load `generation` config from parameters/parameters.yaml."""
    params_path = ROOT / "parameters" / "parameters.yaml"
    if not params_path.exists():
        logger.warning("parameters_yaml_missing", path=str(params_path))
        return {}

    with params_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    gen_cfg = cfg.get("generation", {}) or {}
    if not isinstance(gen_cfg, dict):
        logger.warning("generation_section_not_dict", raw=gen_cfg)
        return {}

    return gen_cfg


def map_engine_params(gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Map parameters.yaml → CVGenerationEngine-compatible params."""
    params = {}

    if "model_name" in gen_cfg:
        params["model"] = gen_cfg["model_name"]
    if "temperature" in gen_cfg:
        params["temperature"] = gen_cfg["temperature"]
    if "max_tokens" in gen_cfg:
        params["max_output_tokens"] = gen_cfg["max_tokens"]
    if "timeout_seconds" in gen_cfg:
        params["timeout_seconds"] = gen_cfg["timeout_seconds"]
    if "max_retries" in gen_cfg:
        params["max_retries"] = gen_cfg["max_retries"]

    return params

# ---------------------------------------------------------------------------
# Generic YAML loader
# ---------------------------------------------------------------------------
def load_yaml_file(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file and return a dict.

    - Returns {} if the file is missing or empty.
    - Logs helpful warnings instead of raising in most cases, so it is
      safe for CLIs / tests.
    """
    if not path.exists():
        logger.error("yaml_file_missing", path=str(path))
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        logger.warning(
            "yaml_root_not_mapping",
            path=str(path),
            root_type=type(data).__name__,
        )
        return {}

    return data

# ---------------------------------------------------------------------------
# LLM client selection
# ---------------------------------------------------------------------------

def dummy_llm_client(prompt: str, **kwargs: Any) -> str:
    """Simple mocked LLM client for local testing."""
    preview = prompt.replace("\n", " ")[:200]
    return f"(STUB LLM) Generated text based on: {preview!r}"


def select_llm_client_and_params() -> Tuple[Any, Dict[str, Any]]:
    """
    Decide which LLM client to use:
      - Stub (fast/no cost)
      - Real LLM (Gemini) via llm_client.call_llm
    """
    from functions.utils.llm_client import call_llm as real_client  # safe import

    gen_cfg = load_generation_params()
    engine_params = map_engine_params(gen_cfg)

    use_stub = gen_cfg.get("use_stub", False)

    if use_stub:
        logger.info("using_stub_llm_client")
        return dummy_llm_client, engine_params

    try:
        logger.info("using_real_llm_client", model=engine_params.get("model"))
        return real_client, engine_params
    except Exception as e:
        logger.error("real_llm_import_failed_fallback_stub", error=str(e))
        return dummy_llm_client, engine_params

# ---------------------------------------------------------------------------

__all__ = [
    "model_validate_compat",
    "model_dump_compat",
    "load_generation_params",
    "map_engine_params",
    "dummy_llm_client",
    "select_llm_client_and_params",
    "ensure_llm_metrics_env",
    "load_yaml_file",
]