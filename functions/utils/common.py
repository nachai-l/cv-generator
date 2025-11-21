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
from typing import Any, Dict, Tuple, Mapping
import yaml
import structlog

logger = structlog.get_logger().bind(module="utils.common")

# Root of project (two dirs up from utils/)
ROOT = Path(__file__).resolve().parents[2]

# Simple cache for full parameters.yaml
_PARAMETERS_CACHE: Dict[str, Any] | None = None

# ---------------------------------------------------------------------------
# yaml reader
# ---------------------------------------------------------------------------

def load_yaml_dict(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file into a dict. Accepts either a string path or a Path object.
    Returns {} on any error, and logs via structlog.
    """

    try:
        p = Path(path)
        if not p.exists():
            logger.error("yaml_file_not_found", path=p)
            return {}

        import yaml

        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            logger.error("yaml_file_not_a_mapping", path=p)
            return {}

        return data
    except Exception as exc:
        logger.error("yaml_file_load_error", path=path, error=str(exc))
        return {}

# ---------------------------------------------------------------------------
# Full parameters.yaml loader (cached)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Full parameters.yaml loader (cached)
# ---------------------------------------------------------------------------

_PARAMETERS_CACHE: Dict[str, Any] | None = None

def load_all_parameters() -> Dict[str, Any]:
    """
    Load and cache the entire parameters/parameters.yaml file.
    Used by pricing + currency conversion helpers.
    """
    global _PARAMETERS_CACHE
    if _PARAMETERS_CACHE is not None:
        return _PARAMETERS_CACHE

    params_path = ROOT / "parameters" / "parameters.yaml"

    if not params_path.exists():
        logger.warning("parameters_yaml_missing", path=str(params_path))
        _PARAMETERS_CACHE = {}
        return _PARAMETERS_CACHE

    try:
        with params_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            logger.warning(
                "parameters_root_not_mapping",
                path=str(params_path),
                root_type=type(cfg).__name__,
            )
            cfg = {}
    except Exception as exc:
        logger.error("parameters_yaml_load_error", path=str(params_path), error=str(exc))
        cfg = {}

    _PARAMETERS_CACHE = cfg
    return _PARAMETERS_CACHE

# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------
def get_pricing_for_model(model_name: str) -> dict[str, float]:
    """
    Resolve USD/token pricing from parameters.yaml.
    Falls back to 'default' pricing if model-specific entry is missing.
    """
    params = load_all_parameters()
    pricing_cfg = params.get("pricing", {}) or {}

    cfg = (
        pricing_cfg.get(model_name)
        or pricing_cfg.get("default")
        or {}
    )

    result = {}

    try:
        result["usd_per_input_token"] = float(
            cfg.get("usd_per_input_token", 0.0) or 0.0
        )
    except Exception:
        result["usd_per_input_token"] = 0.0

    try:
        result["usd_per_output_token"] = float(
            cfg.get("usd_per_output_token", 0.0) or 0.0
        )
    except Exception:
        result["usd_per_output_token"] = 0.0

    return result

def strip_redundant_section_heading(
    raw_text: str,
    section_id: str,
    removal_map: dict[str, str] | None = None,
) -> str:
    """
    Remove a meaningless first line that repeats the section heading.
    Keeps all bullets except when the whole line is just the heading itself.
    """

    # Safe default map (no mutable default argument)
    if removal_map is None:
        removal_map = {
            "references": "references",
            "additional_info": "additional information",
        }

    if not raw_text:
        return raw_text

    # Determine if this section has a redundant title to remove
    title = removal_map.get(section_id)
    if not title:
        return raw_text

    lines = raw_text.splitlines()
    if not lines:
        return raw_text

    cleaned: list[str] = []
    first_nonempty_seen = False

    for ln in lines:
        stripped = ln.strip()

        # Skip leading empty lines before real content
        if not stripped and not first_nonempty_seen:
            continue

        if not first_nonempty_seen:
            first_nonempty_seen = True

            # Normalize for comparison (do NOT modify original text)
            core = (
                stripped
                .lstrip("-•* ")   # strip bullets only for checking
                .rstrip(":")
                .strip()
                .lower()
            )

            # If this line is just the heading → remove it
            if core == title.lower():
                continue

        cleaned.append(ln)  # keep original line

    return "\n".join(cleaned).strip()


def get_thb_per_usd_from_params() -> float:
    """
    Look up THB-per-USD conversion from parameters.yaml.
    Supported keys:
        thb_per_usd
        usd_to_thb
        thb_rate
    Returns 1.0 if missing.
    """
    params = load_all_parameters() or {}
    candidates: list[dict] = []

    if isinstance(params, dict):
        candidates.append(params)
        if isinstance(params.get("currency"), dict):
            candidates.append(params["currency"])
        if isinstance(params.get("pricing"), dict):
            candidates.append(params["pricing"])

    for container in candidates:
        for key in ("thb_per_usd", "usd_to_thb", "thb_rate"):
            val = container.get(key)
            try:
                if val is not None:
                    return float(val)
            except Exception:
                continue

    return 1.0


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
      safe for CLIs / tests_utils.
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
# token budget selection based on section and params
# ---------------------------------------------------------------------------

def resolve_token_budget(section_id: str, attempt: int, params: dict) -> int | None:
    """
    Pure function for determining max_output_tokens for a given section+attempt.

    Inputs:
        section_id : str
        attempt    : int (1-based)
        params     : dict loaded from parameters.yaml

    Behaviour:
        - Looks at params["section_token_budgets"]
        - Accepts scalar (int) or list/tuple per section
        - Clamps attempt to last item for lists
        - Falls back to "default" if section-specific entry not found
    """
    budgets_cfg = (params.get("section_token_budgets") or {}).copy()
    raw = budgets_cfg.get(section_id, budgets_cfg.get("default"))

    if raw is None:
        return None

    if isinstance(raw, int):
        return raw if raw > 0 else None

    if isinstance(raw, (list, tuple)) and raw:
        idx = max(0, min(attempt - 1, len(raw) - 1))
        try:
            val = int(raw[idx])
            return val if val > 0 else None
        except Exception:
            return None

    return None

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
    "resolve_token_budget",
    "load_yaml_dict",
    "get_pricing_for_model",
    "get_thb_per_usd_from_params",
    "load_all_parameters",
]