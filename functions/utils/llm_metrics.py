from __future__ import annotations

import csv
import json
import time
import uuid
import os
import re
import datetime  # module-level so tests can patch llm_metrics.datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import structlog
from functions.utils.llm_client import load_parameters

logger = structlog.get_logger(__name__)
timezone = datetime.timezone

# ---------------------- config loaders ----------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # repo root (parent of 'functions')

# regexes for parsing raw text proto dumps (Gemini ._pb etc.)
# Updated to accept both : and = separators for flexibility
_RE_PROMPT = re.compile(r"\b(prompt_token_count|prompt_tokens|input_tokens)\s*[:=]\s*(\d+)", re.IGNORECASE)
_RE_CAND = re.compile(r"\b(candidates_token_count|completion_tokens|output_tokens)\s*[:=]\s*(\d+)", re.IGNORECASE)
_RE_TOTAL = re.compile(r"\b(total_token_count|total_tokens)\s*[:=]\s*(\d+)", re.IGNORECASE)


def _parse_token_counts_from_text_blob(s: str) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """Parse token counts from Gemini's text proto format."""
    if not isinstance(s, str) or not s:
        return None, None, None

    m1 = _RE_PROMPT.search(s)
    m2 = _RE_CAND.search(s)
    m3 = _RE_TOTAL.search(s)

    return (
        int(m1.group(2)) if m1 else None,
        int(m2.group(2)) if m2 else None,
        int(m3.group(2)) if m3 else None,
    )


def _i(x: Optional[int]) -> int:
    try:
        if isinstance(x, (int, float)):
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
    except Exception:
        pass
    return 0

def _safe_add(a: Optional[int], b: Optional[int]) -> int:
    return _i(a) + _i(b)


def _to_deep_jsonable(o, _depth=0, _max=6):
    if _depth > _max:
        try:
            return str(o)
        except Exception:
            return f"<unserializable:{type(o).__name__}>"

    # primitives
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o

    # dict
    if isinstance(o, dict):
        return {str(k): _to_deep_jsonable(v, _depth + 1, _max) for k, v in o.items()}

    # list/tuple
    if isinstance(o, (list, tuple)):
        return [_to_deep_jsonable(x, _depth + 1, _max) for x in o]

    # try to project common SDK attrs
    try:
        if hasattr(o, "__dict__") and not isinstance(o, type):
            # avoid exploding objects: only shallow dict then recurse
            return _to_deep_jsonable(dict(o.__dict__), _depth + 1, _max)
    except Exception:
        pass

    # last resort
    try:
        return str(o)
    except Exception:
        return f"<unserializable:{type(o).__name__}>"


def _to_shallow_jsonable(o: Any) -> Any:
    """Best-effort, one-level JSON-safe view of SDK objects (e.g., Gemini protos)."""
    try:
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        if isinstance(o, dict):
            return {k: _to_shallow_jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_to_shallow_jsonable(x) for x in o]

        # Common Gemini attrs; keep shallow
        d = {}
        for k in ("text", "usage", "usage_metadata", "candidates", "model_version"):
            if hasattr(o, k):
                v = getattr(o, k)
                if k == "candidates":
                    try:
                        d[k] = len(v) if isinstance(v, (list, tuple)) else str(type(v).__name__)
                    except Exception:
                        d[k] = str(type(v).__name__)
                else:
                    if isinstance(v, (dict, list, tuple, str, int, float, bool, type(None))):
                        d[k] = v
                    else:
                        try:
                            d[k] = str(v)
                        except Exception:
                            d[k] = f"<unserializable:{type(v).__name__}>"
        if d:
            d["__type__"] = type(o).__name__
            return d

        if hasattr(o, "__dict__"):
            d = dict(getattr(o, "__dict__", {}))
            d["__type__"] = type(o).__name__
            return d
    except Exception:
        pass
    try:
        return str(o)
    except Exception:
        return f"<unserializable:{type(o).__name__}>"


def _canon_from_counts(
        prompt: Optional[int],
        completion: Optional[int],
        total: Optional[int],
) -> Dict[str, int]:
    pi = _i(prompt)
    po = _i(completion)
    tt = _i(total) if total is not None else (pi + po)
    return {
        "prompt_tokens": pi,
        "completion_tokens": po,
        "total_tokens": tt,
        "input_tokens": pi,
        "output_tokens": po,
    }


def _canonicalize_usage_from_snapshot(
        snapshot: Dict[str, Any] | None,
        fallback_in: Optional[int] = None,
        fallback_out: Optional[int] = None,
) -> Dict[str, int]:
    if not snapshot:
        return _canon_from_counts(_i(fallback_in), _i(fallback_out), None)

    # Case A: Google-style names present
    if any(k in snapshot for k in ("prompt_token_count", "candidates_token_count", "total_token_count")):
        pt = snapshot.get("prompt_token_count")
        ct = snapshot.get("candidates_token_count")
        tt = snapshot.get("total_token_count")
        c = _canon_from_counts(_i(pt), _i(ct), _i(tt))
        return {k: _i(v) for k, v in c.items()}

    # Case B: OpenAI-like names present
    pt = snapshot.get("prompt_tokens")
    ct = snapshot.get("completion_tokens")
    tt = snapshot.get("total_tokens")
    it = snapshot.get("input_tokens")
    ot = snapshot.get("output_tokens")
    if it is not None:
        pt = it
    if ot is not None:
        ct = ot
    c = _canon_from_counts(_i(pt), _i(ct), _i(tt))
    return {k: _i(v) for k, v in c.items()}


def _extract_raw_usage_str_from_raw_blob(raw_blob: Any) -> Optional[str]:
    """
    Enhanced version: Extracts usage text from Gemini raw blobs.
    Handles:
      - String containing 'prompt_token_count: ...'
      - Dict with 'usage_metadata' being a string
      - Object with .usage_metadata that is either string or SDK object
      - Direct attributes on the raw_blob itself (GenerateContentResponse)
    """

    def _mk_block_from_um_obj(um_obj: Any) -> Optional[str]:
        """Build a text block from a UsageMetadata-like object."""
        try:
            pt = getattr(um_obj, "prompt_token_count", None)
            ct = getattr(um_obj, "candidates_token_count", None)
            tt = getattr(um_obj, "total_token_count", None)
            if any(v is not None for v in (pt, ct, tt)):
                lines = []
                if pt is not None: lines.append(f"prompt_token_count: {pt}")
                if ct is not None: lines.append(f"candidates_token_count: {ct}")
                if tt is not None: lines.append(f"total_token_count: {tt}")
                return "\n".join(lines)
        except Exception:
            return None
        return None

    # Case 1: Direct string
    if isinstance(raw_blob, str):
        return raw_blob if raw_blob.strip() else None

    # Case 2: Dict → try key 'usage_metadata'
    if isinstance(raw_blob, dict):
        um = raw_blob.get("usage_metadata")
        if isinstance(um, str) and um.strip():
            return um
        if um is not None and not isinstance(um, (str, dict, list, tuple, int, float, bool)):
            result = _mk_block_from_um_obj(um)
            if result:
                return result
        return None

    # Case 3: Object → try attr 'usage_metadata'
    try:
        um = getattr(raw_blob, "usage_metadata", None)
        if isinstance(um, str) and um.strip():
            return um
        if um is not None and not isinstance(um, (str, dict, list, tuple, int, float, bool)):
            result = _mk_block_from_um_obj(um)
            if result:
                return result
    except Exception:
        pass

    # Case 4: Check if raw_blob ITSELF has the token count attributes
    # This handles GenerateContentResponse objects directly
    try:
        result = _mk_block_from_um_obj(raw_blob)
        if result:
            return result
    except Exception:
        pass

    return None


def _deep_scan_for_tokens(obj: Any, max_depth: int = 3) -> tuple[int, int]:
    """
    Lightweight recursive scan to extract (prompt/input, completion/output) tokens
    from any nested dict/object that contains *_token_count or *_tokens keys.
    """
    seen: set[int] = set()

    def _as_dict(x: Any) -> dict[str, Any]:
        if isinstance(x, dict):
            return x
        if hasattr(x, "__dict__"):
            return dict(getattr(x, "__dict__", {}))
        return {}

    def _walk(x: Any, depth: int = 0) -> tuple[Optional[int], Optional[int]]:
        if x is None or depth > max_depth:
            return None, None
        obj_id = id(x)
        if obj_id in seen:
            return None, None
        seen.add(obj_id)

        # string blob like: "prompt_token_count: 11\n..."
        if isinstance(x, str):
            pi, po, _ = _parse_token_counts_from_text_blob(x)
            return pi, po

        # dict / object
        if isinstance(x, (dict, object)):
            d = x if isinstance(x, dict) else _as_dict(x)
            pi = d.get("prompt_token_count") or d.get("prompt_tokens") or d.get("input_tokens")
            po = d.get("candidates_token_count") or d.get("completion_tokens") or d.get("output_tokens")
            if pi or po:
                return int(pi or 0), int(po or 0)

            # search common subkeys
            for k in ("usage", "usage_metadata", "raw", "response", "result", "data", "payload", "snapshot_wo_text"):
                v = d.get(k) or getattr(x, k, None)
                if v is not None:
                    a, b = _walk(v, depth + 1)
                    if a or b:
                        return a, b

            # fallback to values
            for v in d.values():
                a, b = _walk(v, depth + 1)
                if a or b:
                    return a, b
        return None, None

    i, o = _walk(obj)
    return (i or 0), (o or 0)


def _best_usage_from_sources(resp: Any) -> Dict[str, Optional[int]]:
    """
    Merge usage counts from multiple sources and return the best non-zero result.
    Sources (priority by signal quality):
      A) usage_metadata (string/object with google-style fields)
      B) raw.usage_metadata (string/object)
      C) top-level usage (openai-style)
      D) snapshot_wo_text.usage_metadata
      E) deep recursive scan across the whole object graph
    Rule: take the max of each field found across sources, then derive totals/parts if needed.
    """
    best_i = best_o = best_t = None

    def _update(i, o, t):
        nonlocal best_i, best_o, best_t
        if i is not None:
            best_i = (max(_i(best_i), _i(i)) if (best_i is not None) else _i(i))
        if o is not None:
            best_o = (max(_i(best_o), _i(o)) if (best_o is not None) else _i(o))
        if t is not None:
            best_t = (max(_i(best_t), _i(t)) if (best_t is not None) else _i(t))

    # A) usage_metadata at top-level
    um = resp.get("usage_metadata") if isinstance(resp, dict) else getattr(resp, "usage_metadata", None)
    raw_str = _extract_raw_usage_str_from_raw_blob({"usage_metadata": um} if um is not None else None)
    if raw_str:
        pi, po, pt = _parse_token_counts_from_text_blob(raw_str)
        _update(pi, po, pt)

    # B) raw.usage_metadata
    raw_like = resp.get("raw") if isinstance(resp, dict) else getattr(resp, "raw", None)
    raw_str = _extract_raw_usage_str_from_raw_blob(raw_like)
    if raw_str:
        pi, po, pt = _parse_token_counts_from_text_blob(raw_str)
        _update(pi, po, pt)

    # C) top-level usage (openai-style keys)
    u = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
    if u:
        if isinstance(u, dict):
            pi = u.get("prompt_tokens") or u.get("input_tokens")
            po = u.get("completion_tokens") or u.get("output_tokens")
            pt = u.get("total_tokens")
        else:
            pi = getattr(u, "prompt_tokens", None) or getattr(u, "input_tokens", None)
            po = getattr(u, "completion_tokens", None) or getattr(u, "output_tokens", None)
            pt = getattr(u, "total_tokens", None)
        _update(pi, po, pt)

    # D) snapshot_wo_text
    snap = resp.get("snapshot_wo_text") if isinstance(resp, dict) else getattr(resp, "snapshot_wo_text", None)
    if isinstance(snap, dict):
        proto_txt = snap.get("usage_metadata") or snap.get("_pb")
        if isinstance(proto_txt, str):
            pi, po, pt = _parse_token_counts_from_text_blob(proto_txt)
            _update(pi, po, pt)

    # E) deep recursive scan anywhere
    di, do = _deep_scan_for_tokens(resp, max_depth=4)
    if (di or 0) > 0 or (do or 0) > 0:
        _update(di, do, _safe_add(di, do))

    # Derive missing piece
    if best_t is None and best_i is not None and best_o is not None:
        best_t = _i(best_i) + _i(best_o)
    if best_i is None and best_t is not None and best_o is not None:
        best_i = max(0, _i(best_t) - _i(best_o))
    if best_o is None and best_t is not None and best_i is not None:
        best_o = max(0, _i(best_t) - _i(best_i))

    bi = _i(best_i)
    bo = _i(best_o)
    bt = best_t if isinstance(best_t, int) else (bi + bo)

    return {
        "prompt_tokens": bi,
        "completion_tokens": bo,
        "total_tokens": bt,
        "input_tokens": bi,
        "output_tokens": bo,
    }


@lru_cache(maxsize=1)
def _load_config() -> Dict[str, Any]:
    try:
        return load_parameters() or {}
    except (OSError, ValueError, TypeError) as e:  # pragma: no cover
        logger.warning("llm_metrics_params_load_failed", error=str(e))
        return {}


@lru_cache(maxsize=1)
def _load_pricing() -> Dict[str, Any]:
    cfg = _load_config() or {}
    return cfg.get("pricing", {}) or {}


def _get_params() -> Dict[str, Any]:
    try:
        return load_parameters() or {}
    except (OSError, ValueError, TypeError) as e:  # pragma: no cover
        logger.warning("llm_metrics_params_load_failed", error=str(e))
        return {}


@lru_cache(maxsize=1)
def _get_log_path() -> Path:
    """
    Determine CSV log path for LLM cost/usage.

    Priority:
      1) Env var LLM_METRICS_CSV (absolute or relative to project root)
      2) parameters.yaml → paths.llm_log_csv (relative to project root if not absolute)
      3) Default: <project_root>/local_logs/llm_call_logs.csv
    """
    project_root = Path(__file__).resolve().parents[2]

    env_path = os.getenv("LLM_METRICS_CSV")
    if env_path:
        p = Path(env_path)
        return (p if p.is_absolute() else (project_root / p)).resolve()

    params = _load_config() or _get_params()
    paths = params.get("paths", {}) or {}
    path_str = paths.get("llm_log_csv") or "local_logs/llm_call_logs.csv"

    p = Path(path_str)
    if p.is_absolute():
        return p

    return (project_root / p).resolve()


@lru_cache(maxsize=1)
def _get_csv_path() -> Path:
    return _get_log_path()


def _is_verbose() -> bool:
    if os.getenv("LLM_METRICS_VERBOSE") == "1":
        return True
    cfg = _load_config() or {}
    # try logging.verbose_main or generation.verbose_main
    if bool((cfg.get("logging") or {}).get("verbose_main")):
        return True
    if bool((cfg.get("generation") or {}).get("verbose_main")):
        return True
    return False


def _get_rates(model: Optional[str]) -> Tuple[float, float]:
    pricing = _load_pricing() or (_get_params().get("pricing", {}) or {})
    key = model or "default"
    plan = (pricing.get(key) or pricing.get("default") or {}) if pricing else {}
    try:
        in_rate = float(plan.get("usd_per_input_token", 0.0) or 0.0)
    except (TypeError, ValueError):
        in_rate = 0.0
    try:
        out_rate = float(plan.get("usd_per_output_token", 0.0) or 0.0)
    except (TypeError, ValueError):
        out_rate = 0.0
    return in_rate, out_rate

def _get_section_max_tokens(section_id: str, attempt_index: int = 0) -> int:
    """
    Decide max_output_tokens for this section + attempt, based on parameters.yaml.

    Uses:
      - root.section_token_budgets if present (list like [1024, 2048])
      - generation.max_tokens as a hard ceiling
    """
    cfg = _get_params() or {}

    gen_cfg = cfg.get("generation", {}) or {}
    global_max = int(gen_cfg.get("max_tokens", 4096))

    # NOTE: your YAML has section_token_budgets at ROOT (not under generation)
    budgets = cfg.get("section_token_budgets", {}) or {}

    seq = budgets.get(section_id) or budgets.get("default") or [global_max]

    # allow scalar or list in YAML
    if not isinstance(seq, (list, tuple)):
        seq = [seq]

    # clamp attempt_index to [0, len(seq)-1]
    idx = min(max(int(attempt_index), 0), len(seq) - 1)

    try:
        chosen = int(seq[idx])
    except (TypeError, ValueError):
        chosen = global_max

    return min(chosen, global_max)

def _safe_preview(s: Any, limit: int = 600) -> str:
    try:
        text = str(s)
    except Exception:  # best-effort stringify
        return "<unrepresentable>"
    return text[:limit] + " …[truncated]" if len(text) > limit else text


def _build_usage_preview(
        resp_dict: Dict[str, Any],
        *,
        usage_snapshot: Dict[str, Any] | None = None,
        canonical_usage: Dict[str, Any] | None = None,
        limit: int = 240,
) -> Dict[str, Any]:
    """
    Produce a short preview string of raw usage info (like text_preview, but for usage).
    Prefer sources in order:
      1) resp_dict["usage_metadata"] if it's a string or has ._pb-like str
      2) resp_dict["raw"].usage_metadata string
      3) usage_snapshot (convert to google-style lines)
      4) canonical_usage (convert to google-style lines)
    Returns: {"usage_preview": str|None, "usage_preview_source": str|None}
    """
    # 1) usage_metadata at top-level
    um = resp_dict.get("usage_metadata")
    if isinstance(um, str) and um.strip():
        return {
            "usage_preview": _safe_preview(um, limit),
            "usage_preview_source": "usage_metadata_str",
        }
    if isinstance(um, dict):
        # sometimes parameters.yaml/raw dump may store the google __type__/ _pb
        pb = um.get("_pb")
        if isinstance(pb, str) and pb.strip():
            return {
                "usage_preview": _safe_preview(pb, limit),
                "usage_preview_source": "usage_metadata._pb",
            }

    # 2) raw.usage_metadata
    raw_blob = resp_dict.get("raw")
    raw_str = _extract_raw_usage_str_from_raw_blob(raw_blob)
    if isinstance(raw_str, str) and raw_str.strip():
        return {
            "usage_preview": _safe_preview(raw_str, limit),
            "usage_preview_source": "raw.usage_metadata_str",
        }

    # 3) from usage_snapshot (google names)
    if usage_snapshot:
        lines = []
        pt = usage_snapshot.get("prompt_token_count")
        ct = usage_snapshot.get("candidates_token_count")
        tt = usage_snapshot.get("total_token_count")
        if any(v is not None for v in (pt, ct, tt)):
            if pt is not None: lines.append(f"prompt_token_count: {pt}")
            if ct is not None: lines.append(f"candidates_token_count: {ct}")
            if tt is not None: lines.append(f"total_token_count: {tt}")
            blob = "\n".join(lines)
            return {
                "usage_preview": _safe_preview(blob, limit),
                "usage_preview_source": "usage_snapshot",
            }

    # 4) from canonical_usage (map back to google-ish names)
    if canonical_usage:
        lines = []
        pt = canonical_usage.get("prompt_tokens")
        ct = canonical_usage.get("completion_tokens")
        tt = canonical_usage.get("total_tokens")
        if any(v is not None for v in (pt, ct, tt)):
            if pt is not None: lines.append(f"prompt_token_count: {pt}")
            if ct is not None: lines.append(f"candidates_token_count: {ct}")
            if tt is not None: lines.append(f"total_token_count: {tt}")
            blob = "\n".join(lines)
            return {
                "usage_preview": _safe_preview(blob, limit),
                "usage_preview_source": "canonical_usage",
            }

    return {"usage_preview": None, "usage_preview_source": None}


def _obj_attrs(o: Any) -> Dict[str, bool]:
    """
    Shallow capability map: which common attributes/keys exist?
    """
    if isinstance(o, dict):
        keys = set(o.keys())
        return {
            "is_dict": True,
            "has_usage": "usage" in keys,
            "has_usage_metadata": "usage_metadata" in keys,
            "has_response": "response" in keys,
            "has_result": "result" in keys,
            "has_data": "data" in keys,
            "has_payload": "payload" in keys,
            "has_candidates": "candidates" in keys,
            "has_text": "text" in keys,
            "has_prompt_token_count": "prompt_token_count" in keys,
            "has_total_token_count": "total_token_count" in keys,
        }
    # object
    return {
        "is_dict": False,
        "has_usage": hasattr(o, "usage"),
        "has_usage_metadata": hasattr(o, "usage_metadata"),
        "has_response": hasattr(o, "response"),
        "has_result": hasattr(o, "result"),
        "has_data": hasattr(o, "data"),
        "has_payload": hasattr(o, "payload"),
        "has_candidates": hasattr(o, "candidates"),
        "has_text": hasattr(o, "text"),
        "has_prompt_token_count": hasattr(o, "prompt_token_count"),
        "has_total_token_count": hasattr(o, "total_token_count"),
    }


def _to_dict_best_effort(o: Any) -> Optional[Dict[str, Any]]:
    """
    Try to convert SDK object into a dict for inspection.
    """
    try:
        if isinstance(o, dict):
            return o
        if hasattr(o, "to_dict"):
            # type: ignore[attr-defined]
            return o.to_dict()
        if hasattr(o, "__dict__"):
            return dict(getattr(o, "__dict__", {}))
    except (AttributeError, TypeError, ValueError):
        return None
    return None


def _verbose_dump_response_for_zero_tokens(resp: Any, call_id: str) -> None:
    """
    Only called when tokens == (0,0) and env LLM_METRICS_VERBOSE=1.
    Logs structure + a shallow preview, and digs one level into common wrappers.
    """
    try:
        info: Dict[str, Any] = {
            "type": type(resp).__name__,
            "attrs": _obj_attrs(resp),
        }
        d = _to_dict_best_effort(resp)
        if d is not None:
            info["dict_top_keys"] = list(d.keys())[:50]
            for k in ("usage", "usage_metadata", "response", "result", "data", "payload"):
                sub = d.get(k)
                if isinstance(sub, dict):
                    info[f"{k}_keys"] = list(sub.keys())[:50]
                elif isinstance(sub, list):
                    info[f"{k}_keys"] = f"list[{len(sub)}]"

        text_preview = None
        try:
            t_attr = getattr(resp, "text", None)
            if isinstance(t_attr, str) and t_attr:
                text_preview = _safe_preview(t_attr, limit=180)
        except (AttributeError, TypeError, ValueError):
            text_preview = None
        if text_preview:
            info["text_preview"] = text_preview

        wrappers: List[Dict[str, Any]] = []
        for w in ("response", "result", "raw", "data", "payload"):
            inner = getattr(resp, w, None) if not isinstance(resp, dict) else resp.get(w)
            if inner is not None:
                inner_dict = _to_dict_best_effort(inner)
                wrappers.append(
                    {
                        "wrapper": w,
                        "type": type(inner).__name__,
                        "attrs": _obj_attrs(inner),
                        "dict_top_keys": list(inner_dict.keys())[:50] if isinstance(inner_dict, dict) else None,
                    }
                )
        if wrappers:
            info["wrappers"] = wrappers

        logger.info("llm_metrics_zero_token_verbose", call_id=call_id, debug=info)
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        # keep best-effort; do not break main flow
        logger.warning("llm_metrics_zero_token_verbose_failed", call_id=call_id, error=str(e))


def _now_iso() -> str:
    # timezone-aware UTC ISO8601 with trailing Z; simple & deterministic
    return (
        datetime.datetime.now(datetime.UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ------------------------- CSV logging -------------------------

_HEADERS = [
    "timestamp",
    "user_id",
    "purpose",
    "section_id",
    "model",
    "response_time_ms",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "input_cost_usd",
    "output_cost_usd",
    "total_cost_usd",
    "call_id",
]


def _to_path(p: Path | str) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_parent_dir(path: Path | str) -> None:
    p = _to_path(path)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # pragma: no cover
        logger.warning("llm_metrics_mkdir_failed", error=str(e), dir=str(p.parent))


def _append_csv_row(path: Path | str, row: Dict[str, Any]) -> None:
    p = _to_path(path)
    _ensure_parent_dir(p)
    file_exists = p.exists()
    try:
        with p.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_HEADERS)
            if not file_exists:
                writer.writeheader()
            safe_row = {k: row.get(k) for k in _HEADERS}
            writer.writerow(safe_row)
    except (OSError, csv.Error) as e:  # pragma: no cover
        logger.warning("llm_metrics_csv_write_failed", error=str(e), path=str(p))


# -------------------- response parsing helpers --------------------

def _has_attr(obj: Any, name: str) -> bool:
    return hasattr(obj, name)


def _get_attr(obj: Any, name: str, default=None):
    return getattr(obj, name, default)


def _extract_tokens_from_usage_like(
        usage_like: Any,
        tok_in_keys: Tuple[str, ...],
        tok_out_keys: Tuple[str, ...],
        tok_tot_keys: Tuple[str, ...],
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    def _to_int(x: Any) -> Optional[int]:
        try:
            return int(x) if x is not None else None
        except (TypeError, ValueError):
            return None

    if usage_like is None:
        return None, None, None

    in_val = out_val = tot_val = None

    if isinstance(usage_like, dict):
        for k in tok_in_keys:
            if in_val is None:
                in_val = _to_int(usage_like.get(k))
        for k in tok_out_keys:
            if out_val is None:
                out_val = _to_int(usage_like.get(k))
        for k in tok_tot_keys:
            if tot_val is None:
                tot_val = _to_int(usage_like.get(k))
    else:
        for k in tok_in_keys:
            if in_val is None:
                in_val = _to_int(getattr(usage_like, k, None))
        for k in tok_out_keys:
            if out_val is None:
                out_val = _to_int(getattr(usage_like, k, None))
        for k in tok_tot_keys:
            if tot_val is None:
                tot_val = _to_int(getattr(usage_like, k, None))

    # derive if possible
    if out_val is None and tot_val is not None and in_val is not None:
        out_val = max(0, tot_val - in_val)
    if in_val is None and tot_val is not None and out_val is not None:
        in_val = max(0, tot_val - out_val)

    return in_val, out_val, tot_val


def extract_usage_tokens(resp: Any) -> Tuple[int, int]:
    """
    Enhanced token extraction with better prioritization for Gemini responses.
    """
    tok_in_keys = ("prompt_token_count", "prompt_tokens", "input_tokens")
    tok_out_keys = ("candidates_token_count", "completion_tokens", "output_tokens")
    tok_tot_keys = ("total_token_count", "total_tokens")

    def _accept(i: Optional[int], o: Optional[int], t: Optional[int]) -> bool:
        return ((i or 0) > 0) or ((o or 0) > 0) or ((t or 0) > 0)

    # 🔹 NEW: Handle proxy-style top-level *_token_count keys first
    # e.g. {"prompt_token_count": 30, "total_token_count": 44}
    if isinstance(resp, dict):
        tin_proxy, tout_proxy, ttot_proxy = _extract_tokens_from_usage_like(
            resp,
            tok_in_keys,
            tok_out_keys,
            tok_tot_keys,
        )
        if _accept(tin_proxy, tout_proxy, ttot_proxy):
            if _is_verbose():
                logger.info("llm_metrics_usage_source", source="top.level_token_counts")
            return (tin_proxy or 0), (tout_proxy or 0)

    # If we have an original response object stored, try that FIRST
    original_resp = None
    if isinstance(resp, dict):
        original_resp = resp.get("_original_response_object")

    # Normalize response to dict if needed
    if isinstance(resp, str):
        resp = {
            "text": str(resp),
            "usage": getattr(resp, "usage", None),
            "usage_metadata": getattr(resp, "usage_metadata", None),
            "raw": getattr(resp, "raw", None),
        }
    elif not isinstance(resp, dict):
        resp = {
            "text": getattr(resp, "text", None),
            "usage": getattr(resp, "usage", None),
            "usage_metadata": getattr(resp, "usage_metadata", None),
            "choices": getattr(resp, "choices", None),
            "response": getattr(resp, "response", None),
            "result": getattr(resp, "result", None),
            "data": getattr(resp, "data", None),
            "payload": getattr(resp, "payload", None),
            "raw": getattr(resp, "raw", None),
            "snapshot_wo_text": getattr(resp, "snapshot_wo_text", None),
        }

    # PRIORITY 0: If we have the original response object, try extracting from it directly
    if original_resp is not None and original_resp is not resp:
        # Try raw.usage_metadata from original
        raw_like = getattr(original_resp, "raw", None)
        if raw_like is not None:
            raw_txt = _extract_raw_usage_str_from_raw_blob(raw_like)
            if raw_txt:
                pin, pout, ptot = _parse_token_counts_from_text_blob(raw_txt)
                if _accept(pin, pout, ptot):
                    if _is_verbose():
                        logger.info("llm_metrics_usage_source", source="original.raw.usage_metadata")
                    return (pin or 0), (pout or 0)

        # Try usage_metadata from original
        um = getattr(original_resp, "usage_metadata", None)
        if um is not None:
            if isinstance(um, str) and um.strip():
                pin, pout, ptot = _parse_token_counts_from_text_blob(um)
                if _accept(pin, pout, ptot):
                    if _is_verbose():
                        logger.info("llm_metrics_usage_source", source="original.usage_metadata_str")
                    return (pin or 0), (pout or 0)

            tin, tout, ttot = _extract_tokens_from_usage_like(um, tok_in_keys, tok_out_keys, tok_tot_keys)
            if _accept(tin, tout, ttot):
                if _is_verbose():
                    logger.info("llm_metrics_usage_source", source="original.usage_metadata_obj")
                return (tin or 0), (tout or 0)

    # PRIORITY 1: raw.usage_metadata (most reliable for Gemini)
    raw_like = resp.get("raw")
    if raw_like is not None:
        # Try to extract from raw object's usage_metadata
        raw_txt = _extract_raw_usage_str_from_raw_blob(raw_like)
        if raw_txt:
            pin, pout, ptot = _parse_token_counts_from_text_blob(raw_txt)
            if _accept(pin, pout, ptot):
                if _is_verbose():
                    logger.info("llm_metrics_usage_source", source="raw.usage_metadata_str")
                return (pin or 0), (pout or 0)

    # PRIORITY 2: top-level usage_metadata
    um = resp.get("usage_metadata")
    if um is not None:
        # Try as string first
        if isinstance(um, str) and um.strip():
            pin, pout, ptot = _parse_token_counts_from_text_blob(um)
            if _accept(pin, pout, ptot):
                if _is_verbose():
                    logger.info("llm_metrics_usage_source", source="top.usage_metadata_str")
                return (pin or 0), (pout or 0)

        # Try as object with attributes
        tin, tout, ttot = _extract_tokens_from_usage_like(um, tok_in_keys, tok_out_keys, tok_tot_keys)
        if _accept(tin, tout, ttot):
            if _is_verbose():
                logger.info("llm_metrics_usage_source", source="top.usage_metadata_obj",
                            types={"usage_metadata": type(um).__name__})
            return (tin or 0), (tout or 0)

    # PRIORITY 3: top-level usage (OpenAI style, but also works for normalized Gemini)
    tin, tout, ttot = _extract_tokens_from_usage_like(resp.get("usage"), tok_in_keys, tok_out_keys, tok_tot_keys)
    if _accept(tin, tout, ttot):
        if _is_verbose():
            logger.info("llm_metrics_usage_source", source="top.usage")
        return (tin or 0), (tout or 0)

    # PRIORITY 4: Crawl through wrapper objects
    def _dig(obj: Any, key: str) -> Any:
        if isinstance(obj, dict):
            return obj.get(key)
        return getattr(obj, key, None)

    wrapper_attrs = ("response", "result", "data", "payload", "snapshot_wo_text")
    frontier = []
    for a in wrapper_attrs:
        v = _dig(resp, a)
        if v is not None:
            frontier.append(v)

    # Add nested wrappers (depth ≤ 2)
    for node in tuple(frontier):
        for a in wrapper_attrs:
            v = _dig(node, a)
            if v is not None:
                frontier.append(v)

    for node in frontier:
        # Try usage_metadata first (Gemini prefers this)
        um_node = _dig(node, "usage_metadata")
        if um_node is not None:
            tin, tout, ttot = _extract_tokens_from_usage_like(um_node, tok_in_keys, tok_out_keys, tok_tot_keys)
            if _accept(tin, tout, ttot):
                if _is_verbose():
                    logger.info(
                        "llm_metrics_usage_source",
                        source="wrapper.usage_metadata",
                        wrapper_type=type(node).__name__,
                    )
                return (tin or 0), (tout or 0)

        # Then try usage
        u_node = _dig(node, "usage")
        if u_node is not None:
            tin, tout, ttot = _extract_tokens_from_usage_like(u_node, tok_in_keys, tok_out_keys, tok_tot_keys)
            if _accept(tin, tout, ttot):
                if _is_verbose():
                    logger.info(
                        "llm_metrics_usage_source",
                        source="wrapper.usage",
                        wrapper_type=type(node).__name__,
                    )
                return (tin or 0), (tout or 0)

        # Check flat fields on the node itself
        tin, tout, ttot = _extract_tokens_from_usage_like(node, tok_in_keys, tok_out_keys, tok_tot_keys)
        if _accept(tin, tout, ttot):
            if _is_verbose():
                logger.info(
                    "llm_metrics_usage_source",
                    source="wrapper.flat",
                    wrapper_type=type(node).__name__,
                )
            return (tin or 0), (tout or 0)

    # 🔚 FINAL FALLBACK: use the more powerful deep-scan helper
    try:
        best = _best_usage_from_sources(resp)
        bi = _i(best.get("input_tokens"))
        bo = _i(best.get("output_tokens"))
        if (bi or 0) > 0 or (bo or 0) > 0:
            if _is_verbose():
                logger.info("llm_metrics_usage_source", source="best_usage_fallback")
            return bi, bo
    except Exception:
        # don't break metrics if deep scan fails for weird objects
        pass

    if _is_verbose():
        logger.info("llm_metrics_usage_source", source="none_found")

    return 0, 0


def _join_openai_content(content: Any) -> str:
    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict):
                t = part.get("text")
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    if isinstance(content, str):
        return content
    return ""


def _extract_text_from_openai_attr(resp: Any) -> str:
    if not _has_attr(resp, "choices"):
        return ""
    choices = _get_attr(resp, "choices", None)
    try:
        if isinstance(choices, list) and choices:
            first = choices[0]
            message = _get_attr(first, "message", None)
            if isinstance(message, dict):
                content = message.get("content")
                joined = _join_openai_content(content)
                if joined:
                    return joined
            elif message is not None:
                content = _get_attr(message, "content", None)
                joined = _join_openai_content(content)
                if joined:
                    return joined
            text = first.get("text") if isinstance(first, dict) else _get_attr(first, "text", None)
            if isinstance(text, str):
                return text
    except (AttributeError, KeyError, TypeError, ValueError):
        return ""
    return ""


def extract_text(resp: Any) -> str:
    text = _get_attr(resp, "text", None)
    if isinstance(text, str):
        return text

    if isinstance(resp, dict):
        # direct text
        if isinstance(resp.get("text"), str):
            return resp["text"]

        choices = resp.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0] if choices else None

            # --- dict-style choice ---
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    joined = _join_openai_content(msg.get("content"))
                    if joined:
                        return joined
                if isinstance(first.get("text"), str):
                    return first["text"]

            # --- object-style choice (when resp is dict but items are objects) ---
            if first is not None:
                try:
                    msg = getattr(first, "message", None)
                    if msg is not None:
                        content = msg["content"] if isinstance(msg, dict) else getattr(msg, "content", None)
                        joined = _join_openai_content(content)
                        if joined:
                            return joined
                    # fallback: .text on the choice object
                    t = getattr(first, "text", None)
                    if isinstance(t, str) and t:
                        return t
                except Exception:
                    pass

    # Final fallback: try OpenAI attribute style on the whole object
    from_attr = _extract_text_from_openai_attr(resp)
    if from_attr:
        return from_attr
    return ""


def _extract_finish_reason(resp: Any) -> Optional[str]:
    try:
        candidates = _get_attr(resp, "candidates", None)
        if isinstance(candidates, list) and candidates:
            fr = getattr(candidates[0], "finish_reason", None)
            return getattr(fr, "name", str(fr)) if fr is not None else None
    except (AttributeError, TypeError, ValueError):
        return None
    return None


# ------------------------ debug dump helpers ------------------------

def _as_bool(x):
    s = str(x).strip().lower()
    if s in ("1", "true", "yes", "on"):  return True
    if s in ("0", "false", "no", "off"): return False
    return None  # unknown

def _sdk_dump_dir() -> Optional[Path]:
    # 1) Env flag has highest priority
    env_flag = _as_bool(os.getenv("LLM_SDK_DUMP", ""))
    env_dir  = os.getenv("LLM_SDK_DUMP_DIR")

    # 2) Config flags
    params = _load_config() or _get_params()
    paths  = (params.get("paths") or {})
    cfg_flag = _as_bool(paths.get("llm_sdk_dump", ""))  # boolean authoritative
    cfg_dir  = paths.get("llm_sdk_dump_dir")

    # --- Decide enablement first ---
    if env_flag is True or cfg_flag is True:
        # enabled → pick a directory
        base = Path(env_dir) if env_dir else Path(cfg_dir) if cfg_dir else (PROJECT_ROOT / "local_logs" / "sdk_objects")
        return base if base.is_absolute() else (PROJECT_ROOT / base).resolve()

    if env_flag is False or cfg_flag is False:
        # explicitly disabled
        return None

    # no explicit enable → disabled by default
    return None


def _ensure_dir(p: Path) -> None:
    try:
        p.mkdir(parents=True, exist_ok=True)
    except OSError as e:  # pragma: no cover
        logger.warning("llm_metrics_sdk_dump_mkdir_failed", dir=str(p), error=str(e))


def _maybe_dump_sdk_object(
        *,
        call_id: str,
        resp_dict: Dict[str, Any],
        text: str,
        usage_snapshot: Dict[str, Any],
        prompt: Optional[str] = None,
) -> None:
    dump_dir = _sdk_dump_dir()
    if not dump_dir:
        return
    try:
        _ensure_dir(dump_dir)

        # Pull raw & parse proto usage (for tokens)
        raw_blob = resp_dict.get("raw")
        raw_usage_text = _extract_raw_usage_str_from_raw_blob(raw_blob)
        parsed_from_raw = None
        if raw_usage_text:
            pin, pout, ptot = _parse_token_counts_from_text_blob(raw_usage_text)
            parsed_from_raw = {
                "prompt_token_count": pin,
                "candidates_token_count": pout,
                "total_token_count": ptot,
            }

        # Build canonical usage (so tokens are never zero for Gemini)
        canon: Dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
        }

        # A) top-level usage (often already correct)
        top_usage = resp_dict.get("usage")
        if isinstance(top_usage, dict):
            pt = _i(top_usage.get("prompt_tokens") or top_usage.get("input_tokens"))
            ct = _i(top_usage.get("completion_tokens") or top_usage.get("output_tokens"))
            tt = _i(top_usage.get("total_tokens")) or (pt + ct)
            canon.update({
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "total_tokens": tt,
                "input_tokens": pt,
                "output_tokens": ct,
            })

        # B) parsed raw (take max per field)
        if parsed_from_raw:
            pt = _i(parsed_from_raw.get("prompt_token_count"))
            ct = _i(parsed_from_raw.get("candidates_token_count"))
            tt = _i(parsed_from_raw.get("total_token_count"))
            canon["prompt_tokens"] = max(0, canon["prompt_tokens"])
            canon["completion_tokens"] = max(0, canon["completion_tokens"])
            canon["total_tokens"] = max(0, canon["total_tokens"])
            canon["input_tokens"] = max(0, canon["input_tokens"])
            canon["output_tokens"] = max(0, canon["output_tokens"])

        # C) snapshot (take max per field)
        if usage_snapshot:
            snap = _canonicalize_usage_from_snapshot(usage_snapshot, None, None)
            pt = _i(snap.get("prompt_tokens"))
            ct = _i(snap.get("completion_tokens"))
            tt = _i(snap.get("total_tokens"))
            canon["prompt_tokens"] = max(0, canon["prompt_tokens"])
            canon["completion_tokens"] = max(0, canon["completion_tokens"])
            canon["total_tokens"] = max(0, canon["total_tokens"])
            canon["input_tokens"] = max(0, canon["input_tokens"])
            canon["output_tokens"] = max(0, canon["output_tokens"])

        # D) finalize consistency
        if not canon["total_tokens"]:
            canon["total_tokens"] = canon["prompt_tokens"] + canon["completion_tokens"]

        simple_usage = {
            "prompt_tokens": canon["prompt_tokens"],
            "completion_tokens": canon["completion_tokens"],
            "total_tokens": canon["total_tokens"],
        }

        usage_preview_info = _build_usage_preview(
            resp_dict,
            usage_snapshot=usage_snapshot,
            canonical_usage=canon,
            limit=240,
        )

        out = {
            "type": "dict",
            "snapshot_wo_text": {
                "usage": simple_usage,
                "usage_metadata": _to_shallow_jsonable(resp_dict.get("usage_metadata")),
                "choices": _to_shallow_jsonable(resp_dict.get("choices")),
                "response": _to_shallow_jsonable(resp_dict.get("response")),
                "result": _to_shallow_jsonable(resp_dict.get("result")),
                "data": _to_shallow_jsonable(resp_dict.get("data")),
                "payload": _to_shallow_jsonable(resp_dict.get("payload")),
                "raw": _to_shallow_jsonable(raw_blob),
                "snapshot_wo_text": _to_shallow_jsonable(resp_dict.get("snapshot_wo_text")),
            },
            "usage": {
                "prompt_tokens": canon.get("prompt_tokens", 0),
                "completion_tokens": canon.get("completion_tokens", 0),
                "total_tokens": canon.get("total_tokens",
                                          _safe_add(canon.get("prompt_tokens"), canon.get("completion_tokens"))),
                "input_tokens": canon.get("input_tokens", canon.get("prompt_tokens", 0)),
                "output_tokens": canon.get("output_tokens", canon.get("completion_tokens", 0)),
            },
            "usage_metadata": _to_shallow_jsonable(resp_dict.get("usage_metadata")),
            "text_preview": _safe_preview(text, 240),
            "prompt_preview": _safe_preview(prompt, 240) if prompt else None,
            "usage_preview": usage_preview_info["usage_preview"],
            "usage_preview_source": usage_preview_info["usage_preview_source"],
            "parsed_tokens_from_raw": parsed_from_raw,
            "usage_snapshot_logged": usage_snapshot,
        }

        # Deep-sanitize to avoid non-serializable SDK objects anywhere
        out_deep = _to_deep_jsonable(out)
        path = dump_dir / f"{call_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(out_deep, f, ensure_ascii=False, indent=2)
    except Exception as e:  # pragma: no cover
        logger.warning(
            "llm_metrics_sdk_dump_failed",
            call_id=call_id,
            error=repr(e),
            has_raw=resp_dict.get("raw") is not None,
            has_usage=isinstance(resp_dict.get("usage"), dict),
        )

# ----------------------------- public API -----------------------------

def call_llm_section_with_metrics(
        *,
        llm_client,
        model: str | None,
        prompt: str | None,
        section_id: str,
        purpose: str,
        user_id: str,
        messages: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    llm_client must have: generate(model=..., messages=[...]) -> resp with .text and optional .usage
    """
    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]

    call_id = str(uuid.uuid4())
    start = time.monotonic()
    model_used = model or "unknown"

    logger.info(
        "llm_call_start",
        model=model_used,
        section_id=section_id,
        purpose=purpose,
        user_id=user_id,
        call_id=call_id,
    )

    # Decide max_output_tokens for this section (attempt_index=0 for now)
    max_tokens = _get_section_max_tokens(section_id, attempt_index=0)

    generate_kwargs = {
        "model": model_used,
        "messages": messages,
    }

    try:
        # Preferred path: client supports extra kwargs like max_output_tokens
        resp = llm_client.generate(max_output_tokens=max_tokens, **generate_kwargs)
    except TypeError:
        # Backwards-compat path: stub doesn't accept max_output_tokens
        logger.warning(
            "llm_client_generate_no_max_output_tokens",
            call_id=call_id,
            section_id=section_id,
            model=model_used,
        )
        resp = llm_client.generate(**generate_kwargs)

    logger.info(
        "llm_metrics_debug_resp_types",
        resp_type=type(resp).__name__,
        has_usage=((isinstance(resp, dict) and ("usage" in resp)) or (hasattr(resp, "usage"))),
        has_usage_md=((isinstance(resp, dict) and ("usage_metadata" in resp)) or (
            hasattr(resp, "usage_metadata"))),
    )

    # CRITICAL: Check for LLMText BEFORE checking isinstance(str) since LLMText is a str subclass
    if isinstance(resp, str) and not hasattr(resp, 'usage'):
        # Plain string without usage attrs
        logger.warning(
            "llm_client_returned_plain_text",
            call_id=call_id,
            model=model_used,
            note="Client returned plain string without usage metadata.",
        )
        resp = {
            "text": str(resp),
            "usage": None,
            "usage_metadata": None,
            "raw": None,
            "snapshot_wo_text": None,
        }
    elif isinstance(resp, str) and hasattr(resp, 'usage'):
        # LLMText or similar - preserve all attributes
        logger.info(
            "llm_client_returned_llmtext",
            call_id=call_id,
            model=model_used,
            note="Client returned LLMText with usage metadata.",
        )
        orig = resp  # keep before rewriting
        resp = {
            "text": str(orig),
            "usage": getattr(orig, "usage", None),
            "usage_metadata": getattr(orig, "usage_metadata", None),
            "raw": getattr(orig, "raw", None),
            "snapshot_wo_text": getattr(orig, "snapshot_wo_text", None),
            "_original_response_object": orig,  # <-- use orig, not orig.raw
        }
    elif not isinstance(resp, dict):
        # 🔧 Normalize SDK objects to a dict without losing important fields
        try:
            raw_choices = getattr(resp, "choices", None)
            norm_choices = None
            if isinstance(raw_choices, list):
                norm_choices = []
                for c in raw_choices:
                    if isinstance(c, dict):
                        norm_choices.append(c)
                    else:
                        msg = getattr(c, "message", None)
                        if msg is not None and not isinstance(msg, dict):
                            try:
                                msg = dict(getattr(msg, "__dict__", {}))
                            except Exception:
                                msg = {"content": getattr(msg, "content", None)}
                        norm_choices.append({
                            "message": msg,
                            "text": getattr(c, "text", None),
                        })

            # CRITICAL: Store the original response object so we can extract from it
            resp = {
                "text": getattr(resp, "text", None),
                "usage": getattr(resp, "usage", None),
                "usage_metadata": getattr(resp, "usage_metadata", None),
                "choices": norm_choices if norm_choices is not None else raw_choices,
                "response": getattr(resp, "response", None),
                "result": getattr(resp, "result", None),
                "data": getattr(resp, "data", None),
                "payload": getattr(resp, "payload", None),
                "raw": getattr(resp, "raw", None),
                "snapshot_wo_text": getattr(resp, "snapshot_wo_text", None),
                "_original_response_object": resp,  # NEW: Keep original for token extraction
            }
        except Exception:
            resp = {
                "text": getattr(resp, "text", None),
                "_original_response_object": resp,  # NEW: Keep original even on error
            }

    elapsed_ms = max(1, int(round((time.monotonic() - start) * 1000)))

    # --- token & text extraction ---
    input_tokens, output_tokens = extract_usage_tokens(resp)
    total_tokens = input_tokens + output_tokens
    text = extract_text(resp)

    if (input_tokens, output_tokens) == (0, 0) and _is_verbose():
        _verbose_dump_response_for_zero_tokens(resp, call_id)

    # --- cost calculation ---
    in_rate, out_rate = _get_rates(model_used)
    input_cost_usd = round(input_tokens * in_rate, 8)
    output_cost_usd = round(output_tokens * out_rate, 8)
    total_cost_usd = round(input_cost_usd + output_cost_usd, 8)

    # --- usage snapshot (debug aid) ---
    usage_snapshot: Dict[str, Any] = {}
    try:
        # Build a merged/best snapshot across all possible sources
        merged = _best_usage_from_sources(resp)

        # Keep track of where we *think* it came from (forensics)
        src = "merged"
        um = resp.get("usage_metadata") if isinstance(resp, dict) else getattr(resp, "usage_metadata", None)
        if isinstance(um, str) and um.strip():
            src = "usage_metadata_str"
        else:
            raw_like = resp.get("raw") if isinstance(resp, dict) else getattr(resp, "raw", None)
            raw_txt = _extract_raw_usage_str_from_raw_blob(raw_like)
            if isinstance(raw_txt, str) and raw_txt.strip():
                src = "raw.usage_metadata_str"

        usage_snapshot = dict(merged)
        usage_snapshot["_source"] = src
        usage_snapshot["input_cost_usd"] = input_cost_usd
        usage_snapshot["output_cost_usd"] = output_cost_usd
        usage_snapshot["total_cost_usd"] = total_cost_usd

    except (AttributeError, TypeError, ValueError) as e:
        logger.debug("llm_metrics_usage_snapshot_build_failed", error=str(e))

    logger.info(
        "llm_call_summary",
        call_id=call_id,
        model=model_used,
        section_id=section_id,
        purpose=purpose,
        user_id=user_id,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        input_cost_usd=input_cost_usd,
        output_cost_usd=output_cost_usd,
        total_cost_usd=total_cost_usd,
        response_time_ms=elapsed_ms,
        finish_reason=_extract_finish_reason(resp),
        usage_snapshot=usage_snapshot,
    )

    # Optional: dump a full SDK snapshot (including raw) for forensic debugging
    try:
        # 1) usage_snapshot (best when from usage_metadata)
        canon = _canonicalize_usage_from_snapshot(usage_snapshot or None, None, None)

        # 2) top-level resp['usage']
        if not canon.get("prompt_tokens") and not canon.get("completion_tokens"):
            u = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
            if isinstance(u, dict) or u is not None:
                canon = _canonicalize_usage_from_snapshot(
                    {
                        "prompt_tokens": u.get("prompt_tokens") if isinstance(u, dict) else getattr(u,
                                                                                                    "prompt_tokens",
                                                                                                    None),
                        "completion_tokens": u.get("completion_tokens") if isinstance(u, dict) else getattr(u,
                                                                                                            "completion_tokens",
                                                                                                            None),
                        "total_tokens": u.get("total_tokens") if isinstance(u, dict) else getattr(u,
                                                                                                  "total_tokens",
                                                                                                  None),
                        "input_tokens": u.get("input_tokens") if isinstance(u, dict) else getattr(u,
                                                                                                  "input_tokens",
                                                                                                  None),
                        "output_tokens": u.get("output_tokens") if isinstance(u, dict) else getattr(u,
                                                                                                    "output_tokens",
                                                                                                    None),
                    },
                    None, None,
                )

        # 3) last resort, parse the raw proto-ish string if present
        if not canon or all(v in (None, 0) for v in canon.values()):
            raw_like = resp.get("raw") if isinstance(resp, dict) else getattr(resp, "raw", None)
            raw_txt = _extract_raw_usage_str_from_raw_blob(raw_like)
            if raw_txt:
                pin, pout, ptot = _parse_token_counts_from_text_blob(raw_txt)
                canon = {
                    "prompt_tokens": _i(pin),
                    "completion_tokens": _i(pout),
                    "total_tokens": _i(ptot) if ptot is not None else _safe_add(pin, pout),
                    "input_tokens": _i(pin),
                    "output_tokens": _i(pout),
                }

        # Ensure the object we pass to the dumper *includes* this usage
        resp_for_dump = resp if isinstance(resp, dict) else {
            "text": getattr(resp, "text", None),
            "usage": getattr(resp, "usage", None),
            "usage_metadata": getattr(resp, "usage_metadata", None),
            "raw": getattr(resp, "raw", None),
            "snapshot_wo_text": getattr(resp, "snapshot_wo_text", None),
        }

        orig_usage = resp_for_dump.get("usage")
        orig_um = resp_for_dump.get("usage_metadata")
        orig_raw = resp_for_dump.get("raw")

        resp_for_dump["__orig_usage_type__"] = None if orig_usage is None else type(orig_usage).__name__
        resp_for_dump["__orig_usage_metadata_type__"] = None if orig_um is None else type(orig_um).__name__
        resp_for_dump["__orig_raw_type__"] = None if orig_raw is None else type(orig_raw).__name__

        # Overwrite/add the canonical usage for the SDK dump
        resp_for_dump["usage"] = canon or (resp_for_dump.get("usage") or {})

        _maybe_dump_sdk_object(
            call_id=call_id,
            resp_dict=resp_for_dump,
            text=text,
            usage_snapshot=usage_snapshot,
            prompt=prompt,
        )
    except Exception as e:  # pragma: no cover
        logger.debug("llm_metrics_sdk_dump_wrapper_failed", call_id=call_id, error=str(e))

    # CSV row
    log_path = _get_csv_path()
    _append_csv_row(
        log_path,
        {
            "timestamp": _now_iso(),
            "user_id": user_id,
            "purpose": purpose,
            "section_id": section_id,
            "model": model_used,
            "response_time_ms": elapsed_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "input_cost_usd": input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "total_cost_usd": total_cost_usd,
            "call_id": call_id,
        },
    )

    return text or ""