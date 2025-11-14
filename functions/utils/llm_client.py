# functions/utils/llm_client.py
from __future__ import annotations

import os
import time
import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Callable, cast

import structlog
import yaml

logger = structlog.get_logger().bind(module="llm_client")

# -------- Optional: official Google SDK (preferred if present) ----------
try:  # pragma: no cover
    import google.generativeai as genai  # type: ignore
    try:
        from google.generativeai.types import (  # type: ignore
            GenerationConfig,
            RequestOptions,
        )
    except ImportError:  # very old SDKs may lack these
        GenerationConfig = None  # type: ignore
        RequestOptions = None  # type: ignore
except ImportError as import_err:  # pragma: no cover
    logger.info("google_generativeai_import_failed", error=str(import_err))
    genai = None  # type: ignore
    GenerationConfig = None  # type: ignore
    RequestOptions = None  # type: ignore

# -------- Optional: LangChain fallback -----------------------------------
try:  # pragma: no cover
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import HumanMessage  # type: ignore
except ImportError as import_err:  # pragma: no cover
    logger.info("langchain_google_genai_import_failed", error=str(import_err))
    ChatGoogleGenerativeAI = None  # type: ignore
    HumanMessage = None  # type: ignore


# ---------------------------------------------------------------------------
# String subclass that can carry usage metadata
# ---------------------------------------------------------------------------
class LLMText(str):
    """
    String that also exposes:
      - .usage: dict-like {prompt_tokens, completion_tokens, total_tokens, input_tokens, output_tokens}
      - .usage_metadata: SDK-specific usage object
      - .raw: raw SDK response object
    """
    def __new__(
        cls,
        text: str,
        usage: Optional[Dict[str, Any]] = None,
        usage_metadata: Any = None,
        raw: Any = None,
    ) -> "LLMText":
        obj = cast(LLMText, str.__new__(cls, text or ""))
        obj.usage = usage or {}
        obj.usage_metadata = usage_metadata
        obj.raw = raw
        return obj


def _as_llm_text(
    text: str,
    *,
    usage: Optional[Dict[str, Any]] = None,
    usage_metadata: Any = None,
    raw: Any = None,
) -> LLMText:
    """Single place to build LLMText (helps static analyzers)."""
    return LLMText(text, usage=usage, usage_metadata=usage_metadata, raw=raw)


def _safe_get_text(resp, prompt=None):
    params = load_parameters()
    verbose_main = params.get("generation", {}).get("verbose_main", False)

    # Unified helper for error logging
    def _log_problem(reason: str):
        logger.warning(
            "gemini_empty_text",
            reason=reason,
            finish_reason=str(getattr(resp, "finish_reason", None)),
        )
        if verbose_main and prompt:
            logger.error(
                "gemini_failed_prompt",
                prompt_preview=prompt[:1000],
            )

    # 1) SDK exception when accessing resp.text (e.g., finish_reason=2)
    try:
        txt = getattr(resp, "text", None)
    except Exception:
        _log_problem("exception_on_text_accessor")
        return "[API_ERROR] No resp.text (finish_reason=2)"

    # 2) resp.text exists but empty/blank
    if not txt or not str(txt).strip():
        _log_problem("blank_text_returned")
        return "[API_ERROR] .text exists but is empty/blank"

    cleaned = str(txt).strip()

    # 3) Detect STUB_ERROR or API_ERROR patterns
    if (
        cleaned.startswith("[API_ERROR")
        or cleaned.startswith("[STUB_ERROR")
        or cleaned.startswith("STUB_ERROR:")
    ):
        _log_problem("llm_client_returned_api_or_stub_error")
        return cleaned  # Return the error marker itself

    return cleaned


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        if not path.exists():
            logger.info("yaml_file_missing", path=str(path))
            return {}
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            logger.warning("yaml_file_not_dict", path=str(path), type=str(type(data)))
            return {}
        return data
    except Exception as e:  # pragma: no cover
        logger.exception("yaml_file_load_error", path=str(path), error=str(e))
        return {}


@lru_cache(maxsize=1)
def _load_credentials() -> Dict[str, Any]:
    root = _project_root()
    cred_path = root / "parameters" / "credentials.yaml"
    return _load_yaml(cred_path)


@lru_cache(maxsize=1)
def _load_parameters() -> Dict[str, Any]:
    root = _project_root()
    params_path = root / "parameters" / "parameters.yaml"
    return _load_yaml(params_path)


@lru_cache(maxsize=1)
def _get_api_key() -> Optional[str]:
    creds = _load_credentials()
    api_key = creds.get("GOOGLE_API_KEY") or creds.get("google_api_key")
    if not api_key:
        logger.warning("google_api_key_missing_in_credentials")
        return None
    return str(api_key)


def _ensure_env_api_key() -> Optional[str]:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if api_key:
        return api_key
    api_key = _get_api_key()
    if not api_key:
        logger.warning("google_api_key_missing_for_env")
        return None
    os.environ.setdefault("GOOGLE_API_KEY", api_key)
    return api_key


@lru_cache(maxsize=1)
def _use_stub_llm() -> bool:
    """
    Decide whether to use a stub LLM.

    Order:
      1) parameters.yaml model.use_stub or generation.use_stub → True
      2) No API key → True
      3) No usable SDKs → True
      4) Else False
    """
    params = _load_parameters()

    model_cfg = params.get("model", {}) or {}
    if isinstance(model_cfg, dict) and model_cfg.get("use_stub") is True:
        logger.info("llm_stub_enabled_by_model_config")
        return True

    gen_cfg = params.get("generation", {}) or {}
    if isinstance(gen_cfg, dict) and gen_cfg.get("use_stub") is True:
        logger.info("llm_stub_enabled_by_generation_config")
        return True

    if not _get_api_key():
        logger.info("llm_stub_enabled_no_api_key")
        return True

    if genai is None and (ChatGoogleGenerativeAI is None or HumanMessage is None):
        logger.info("llm_stub_enabled_no_sdk_available")
        return True

    return False


# ---------------------------------------------------------------------------
# Raw dump helpers (for saving SDK/raw responses from here)
# ---------------------------------------------------------------------------
def _raw_dump_enabled() -> bool:
    """
    Enable if:
      - env LLM_RAW_DUMP=1, or
      - parameters.yaml paths.llm_raw_dump is True
    """
    if os.getenv("LLM_RAW_DUMP") == "1":
        return True
    params = _load_parameters() or {}
    paths = params.get("paths") or {}
    return bool(paths.get("llm_raw_dump", False))


def _raw_dump_dir() -> Path:
    """
    Dump dir preference order:
      1) env LLM_RAW_DUMP_DIR (absolute or under project root)
      2) parameters.yaml → paths.llm_raw_dump_dir
      3) <project_root>/local_logs/raw_api_responses
    """
    root = _project_root()
    env_dir = os.getenv("LLM_RAW_DUMP_DIR")
    if env_dir:
        p = Path(env_dir)
        return p if p.is_absolute() else (root / p)
    params = _load_parameters() or {}
    paths = params.get("paths") or {}
    conf = paths.get("llm_raw_dump_dir")
    if isinstance(conf, str) and conf.strip():
        p = Path(conf)
        return p if p.is_absolute() else (root / p)
    return root / "local_logs" / "raw_api_responses"


def _safe_preview(txt: Optional[str], n: int = 400) -> Optional[str]:
    if not isinstance(txt, str):
        return None
    t = txt.strip()
    if not t:
        return None
    return t[:n] + ("…[truncated]" if len(t) > n else "")

def _format_usage_preview(usage_metadata: Any, usage_dict: Optional[Dict[str, Any]]):
    """
    Returns a tuple: (usage_preview_str, source_tag)
    Priority:
      1) usage_metadata (attr or dict -> prompt_token_count/candidates_token_count/total_token_count)
      2) canonical usage dict (prompt_tokens/completion_tokens/total_tokens)
      3) None
    """
    # 1) usage_metadata as object with attributes
    if usage_metadata is not None:
        try:
            pin = getattr(usage_metadata, "prompt_token_count", None)
            pout = getattr(usage_metadata, "candidates_token_count", None)
            ptot = getattr(usage_metadata, "total_token_count", None)
            if any(isinstance(x, int) for x in (pin, pout, ptot)):
                return (
                    f"prompt_token_count: {pin}\n"
                    f"candidates_token_count: {pout}\n"
                    f"total_token_count: {ptot}",
                    "usage_metadata_attrs",
                )
        except Exception:
            pass

        # 1b) usage_metadata as dict
        if isinstance(usage_metadata, dict):
            pin = usage_metadata.get("prompt_token_count")
            pout = usage_metadata.get("candidates_token_count")
            ptot = usage_metadata.get("total_token_count")
            if any(isinstance(x, int) for x in (pin, pout, ptot)):
                return (
                    f"prompt_token_count: {pin}\n"
                    f"candidates_token_count: {pout}\n"
                    f"total_token_count: {ptot}",
                    "usage_metadata_dict",
                )

        # 1c) usage_metadata as str – keep as-is
        if isinstance(usage_metadata, str) and usage_metadata.strip():
            return (usage_metadata.strip(), "usage_metadata_str")

    # 2) fallback to canonical usage dict
    if isinstance(usage_dict, dict):
        pt = usage_dict.get("prompt_tokens")
        ct = usage_dict.get("completion_tokens")
        tt = usage_dict.get("total_tokens")
        if any(isinstance(x, int) for x in (pt, ct, tt)):
            return (
                f"prompt_tokens: {pt}\n"
                f"completion_tokens: {ct}\n"
                f"total_tokens: {tt}",
                "canonical_usage",
            )

    # 3) nothing meaningful
    return (None, None)

def _to_shallow_dict(o: Any) -> Any:
    """Best-effort one-level JSONable conversion of SDK objects."""
    try:
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        if isinstance(o, dict):
            return {k: _to_shallow_dict(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_shallow_dict(x) for x in o]
        # Try attrs commonly present on Gemini responses
        d = {}
        for k in ("text", "usage", "usage_metadata", "candidates", "model_version"):
            if hasattr(o, k):
                v = getattr(o, k)
                # avoid huge nested proto objects; shallow only
                if k in ("candidates",):
                    try:
                        d[k] = len(v) if isinstance(v, (list, tuple)) else str(type(v).__name__)
                    except Exception:
                        d[k] = str(type(v).__name__)
                else:
                    d[k] = str(v) if not isinstance(v, (dict, list, tuple, str, int, float, bool, type(None))) else v
        if d:
            d["__type__"] = type(o).__name__
            return d
        # fallback: object dict (shallow)
        if hasattr(o, "__dict__"):
            d = dict(getattr(o, "__dict__", {}))
            d["__type__"] = type(o).__name__
            return d
    except Exception:
        pass
    # last resort string
    try:
        return str(o)
    except Exception:
        return f"<unserializable:{type(o).__name__}>"

def _dump_raw_response(
    *,
    channel: str,
    model: str,
    prompt: str,
    text: str,
    usage: Optional[Dict[str, Any]],
    usage_metadata: Any,
    raw: Any,
) -> None:
    if not _raw_dump_enabled():
        return
    try:
        out_dir = _raw_dump_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        fname = f"raw_response_{channel}_{model}_{ts}.json"
        out_path = out_dir / fname

        # NEW: compute a compact usage preview
        usage_preview, usage_preview_source = _format_usage_preview(usage_metadata, usage)

        payload = {
            "channel": channel,  # genai | langchain | stub
            "model": model,
            "timestamp_utc": ts,
            "prompt_preview": _safe_preview(prompt, 800),
            "text_preview": _safe_preview(text, 800),
            "usage": usage or {},
            "usage_metadata": _to_shallow_dict(usage_metadata),
            "raw": _to_shallow_dict(raw),

            # NEW: quick glance usage for humans
            "usage_preview": usage_preview,
            "usage_preview_source": usage_preview_source,
        }

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

        logger.info("llm_raw_dump_saved", path=str(out_path))
    except Exception as e:  # pragma: no cover
        logger.warning("llm_raw_dump_failed", error=str(e))

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_parameters() -> Dict[str, Any]:
    return _load_parameters()


def _genai_call(
    prompt: str,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
) -> LLMText:
    """Real call via google-generativeai with usage_metadata surfaced."""
    api_key = _ensure_env_api_key()
    if genai is None or not api_key:
        return _as_llm_text(f"[STUB_ERROR:{model}] google-generativeai unavailable or no API key")

    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)

    # Build typed configs when available; otherwise pass dicts.
    if GenerationConfig is not None:
        gen_config_cls = cast(Any, GenerationConfig)
        gen_cfg: Any = gen_config_cls(
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
    else:
        gen_cfg = {"temperature": temperature, "top_p": top_p, "max_output_tokens": max_output_tokens}

    if RequestOptions is not None:
        request_options_cls = cast(Any, RequestOptions)
        req_opts: Any = request_options_cls(timeout=timeout)
    else:
        req_opts = {"timeout": timeout}

    logger.info(
        "llm_real_call_start",
        module="google-generativeai",
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = gen_model.generate_content(
                prompt,
                generation_config=gen_cfg,
                request_options=req_opts,
                safety_settings=None,
            )
            text = _safe_get_text(resp, prompt=prompt)

            um = getattr(resp, "usage_metadata", None)
            usage = {
                "prompt_tokens": getattr(um, "prompt_token_count", None) if um else None,
                "completion_tokens": getattr(um, "candidates_token_count", None) if um else None,
                "total_tokens": getattr(um, "total_token_count", None) if um else None,
                "input_tokens": getattr(um, "prompt_token_count", None) if um else None,
                "output_tokens": getattr(um, "candidates_token_count", None) if um else None,
            }

            logger.info(
                "llm_real_call_success",
                module="google-generativeai",
                model=model,
                result_preview=text[:200],
                prompt_tokens=usage["prompt_tokens"],
                output_tokens=usage["completion_tokens"],
                total_tokens=usage["total_tokens"],
            )

            # 🔸 dump here
            _dump_raw_response(
                channel="genai",
                model=model,
                prompt=prompt,
                text=text,
                usage=usage,
                usage_metadata=um,
                raw=resp,
            )

            return _as_llm_text(text, usage=usage, usage_metadata=um, raw=resp)

        except Exception as e:  # pragma: no cover
            last_error = e
            logger.exception(
                "llm_real_call_failed",
                module="google-generativeai",
                error=str(e),
                attempt=attempt,
                model=model,
            )
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    error_preview = str(last_error)[:200] if last_error else "Unknown error"
    logger.warning("llm_fallback_to_stub_after_error", fallback_preview=error_preview)
    # also dump the error case as a stubbed record
    _dump_raw_response(
        channel="genai",
        model=model,
        prompt=prompt,
        text=f"[ERROR:{model}] {error_preview}",
        usage=None,
        usage_metadata=None,
        raw={"error": error_preview},
    )
    return _as_llm_text(f"[STUB_ERROR:{model}] LLM call failed: {error_preview}")


def _langchain_call(
    prompt: str,
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_output_tokens: int,
    timeout: int,
    max_retries: int,
) -> LLMText:
    """Fallback via langchain-google-genai (usage tokens usually unavailable)."""
    if ChatGoogleGenerativeAI is None or HumanMessage is None:
        return _as_llm_text(f"[STUB_ERROR:{model}] LangChain client unavailable")

    chat_cls: Any = cast(Any, ChatGoogleGenerativeAI)
    chat = chat_cls(
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
    )

    logger.info(
        "llm_real_call_start",
        module="langchain-google-genai",
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = chat.invoke([HumanMessage(content=prompt)])
            text = (getattr(response, "content", "") or "").strip()

            logger.info(
                "llm_real_call_success",
                module="langchain-google-genai",
                model=model,
                result_preview=text[:200],
            )

            usage = {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "input_tokens": None,
                "output_tokens": None,
            }

            # 🔸 dump here
            _dump_raw_response(
                channel="langchain",
                model=model,
                prompt=prompt,
                text=text,
                usage=usage,
                usage_metadata=None,
                raw=response,
            )

            return _as_llm_text(text, usage=usage, usage_metadata=None, raw=response)
        except Exception as e:  # pragma: no cover
            last_error = e
            logger.exception(
                "llm_real_call_failed",
                module="langchain-google-genai",
                error=str(e),
                attempt=attempt,
                model=model,
            )
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    error_preview = str(last_error)[:200] if last_error else "Unknown error"
    logger.warning("llm_fallback_to_stub_after_error", fallback_preview=error_preview)
    _dump_raw_response(
        channel="langchain",
        model=model,
        prompt=prompt,
        text=f"[ERROR:{model}] {error_preview}",
        usage=None,
        usage_metadata=None,
        raw={"error": error_preview},
    )
    return _as_llm_text(f"[STUB_ERROR:{model}] LLM call failed: {error_preview}")


def call_llm(prompt: str, **kwargs: Any) -> LLMText:
    """Call the LLM and return generated text (string subclass with usage)."""
    params = _load_parameters()
    generation_cfg = params.get("generation", {}) or {}

    default_model = generation_cfg.get("model_name", "gemini-2.5-flash")

    model = kwargs.get("model", default_model)
    temperature = float(kwargs.get("temperature", generation_cfg.get("temperature", 0.3)))
    top_p = float(kwargs.get("top_p", generation_cfg.get("top_p", 0.9)))
    max_output_tokens = int(kwargs.get("max_output_tokens", generation_cfg.get("max_tokens", 1024)))
    timeout = int(kwargs.get("timeout", kwargs.get("timeout_seconds", generation_cfg.get("timeout_seconds", 30))))
    max_retries = int(kwargs.get("max_retries", generation_cfg.get("max_retries", 1)))

    # Stub path
    if _use_stub_llm():
        logger.info(
            "llm_stub_call",
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
        )
        text = f"[STUB:{model}] " + prompt[:200]
        usage = {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        # 🔸 dump stub too (helps confirm control flow)
        _dump_raw_response(
            channel="stub",
            model=model,
            prompt=prompt,
            text=text,
            usage=usage,
            usage_metadata=None,
            raw={"note": "stubbed path"},
        )
        return _as_llm_text(text, usage=usage)

    # Prefer official SDK (usage available)
    if genai is not None and _ensure_env_api_key():
        return _genai_call(
            prompt,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

    # Fallback to LangChain
    return _langchain_call(
        prompt,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_output_tokens=max_output_tokens,
        timeout=timeout,
        max_retries=max_retries,
    )


# Backwards-compatible alias with explicit type to satisfy analyzer
llm_client: Callable[..., LLMText] = call_llm
