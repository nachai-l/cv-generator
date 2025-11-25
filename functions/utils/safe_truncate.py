"""
Utilities for safe, markdown-aware truncation with strong JSON protection.

Key rules
---------
1. Never truncate JSON:
   - If the text is (or contains) a JSON block, or the section_id hints JSON,
     we return the text unchanged and mark strategy_used = "no_truncation_json".

2. Otherwise, try to truncate intelligently:
   - Prefer sentence boundaries, with small allowed overflow beyond max_len.
   - Avoid removing more than `reduction_limit` of the original text.
   - Try to avoid breaking markdown structures (bullets, code fences, bold, backticks).
   - As a last resort, hard truncate and then try to fix broken markdown.

Public API
----------
- is_json_block(text: str, section_id: str | None = None) -> bool
- smart_truncate_markdown(
      text: str,
      max_len: int,
      lang: Literal["en", "th"],
      overflow_limit: int = 128,
      reduction_limit: float = 0.15,
      section_id: str | None = None,
  ) -> dict
"""

from __future__ import annotations

import json
import re
from typing import Literal, Optional


JSON_SECTION_HINTS = {
    "skills_structured",
    "experience_hybrid_json",
    "experience_justification",
    "experience_bullets_justification",
    "profile_summary_justification",
}

def is_json_block(text: str, section_id: Optional[str] = None) -> bool:
    """
    Heuristically detect whether `text` should be treated as JSON-only content.

    Rules (any match => True):
    - section_id ends with `_json` or is one of JSON_SECTION_HINTS
    - text stripped starts with `{` and ends with `}`, and json.loads succeeds
    - text contains a large JSON-looking block:
        * a substring from first '{' to last '}' json.loads successfully
        * and the substring is a significant portion of the text
    - text matches a JSON-like key pattern with only JSON-compatible characters

    This function is intentionally biased toward *false positives* (i.e., more
    likely to classify as JSON than not). False positives are safe because
    they only disable truncation.
    """
    if not text:
        return False

    stripped = text.strip()
    if not stripped:
        return False

    # 1. Section name hints
    if section_id:
        sid = section_id.lower()
        if sid.endswith("_json") or sid in JSON_SECTION_HINTS:
            return True

    # 2. Pure JSON object, quick path
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            json.loads(stripped)
            return True
        except Exception:
            # Fall through to more permissive heuristics
            pass

    # 3. Embedded top-level JSON block: first '{' .. last '}'
    first_brace = stripped.find("{")
    last_brace = stripped.rfind("}")
    if 0 <= first_brace < last_brace:
        candidate = stripped[first_brace : last_brace + 1]
        try:
            json.loads(candidate)
            # Treat as JSON if the candidate is a substantial portion of the text
            if len(candidate) >= 0.5 * len(stripped):
                return True
        except Exception:
            pass

    # 4. JSON-like key pattern with only JSON-safe characters
    key_pattern = re.compile(r'"\s*[^"]+"\s*:')
    if key_pattern.search(stripped):
        allowed_chars = re.compile(
            r'^[\s\{\}\[\]\:\,\"\'0-9A-Za-z_\-\.\+\@\#\$\%\&\*\(\)\\/]*$'
        )
        if allowed_chars.match(stripped):
            return True

    return False


def _detect_sentence_boundaries_en(text: str) -> list[int]:
    """
    Return a list of indices (1-based) that mark sentence ends in English.

    A simple heuristic:
    - '.', '?', '!' followed by whitespace or end of string.
    """
    boundaries: list[int] = []
    n = len(text)

    for i, ch in enumerate(text):
        if ch in ".!?":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt.isspace():
                boundaries.append(i + 1)
    return boundaries


def _detect_sentence_boundaries_th(text: str) -> list[int]:
    """
    Return heuristic sentence boundaries for Thai.

    Thai does not use '.', '?', '!' as consistently. Use:
    - any whitespace or newline as a boundary when preceded by non-whitespace.
    This is intentionally simple and conservative.
    """
    boundaries: list[int] = []
    n = len(text)

    for i, ch in enumerate(text):
        if ch.isspace() and i > 0 and not text[i - 1].isspace():
            boundaries.append(i)
    # Also treat end-of-text as a boundary
    boundaries.append(n)
    return boundaries


def _select_truncation_index_by_sentence(
    text: str,
    max_len: int,
    overflow_limit: int,
    lang: Literal["en", "th"],
) -> tuple[Optional[int], Optional[str]]:
    """
    Choose an index at which to truncate `text` based on sentence boundaries.

    Returns (index, strategy_tag) where:
    - index is the cut position (number of characters to keep) or None
    - strategy_tag is "allow_overflow" or "sentence_boundary_cut" or None
    """
    if lang == "th":
        boundaries = _detect_sentence_boundaries_th(text)
    else:
        boundaries = _detect_sentence_boundaries_en(text)

    if not boundaries:
        return None, None

    # Consider only boundaries up to max_len + overflow_limit
    upper = max_len + max(0, overflow_limit)
    valid = [b for b in boundaries if b <= upper]
    if not valid:
        return None, None

    # Choose the last boundary within the allowed range
    cut = max(valid)

    if cut >= max_len:
        return cut, "allow_overflow"
    else:
        return cut, "sentence_boundary_cut"


def _validate_reduction(
    original_len: int,
    truncated_len: int,
    reduction_limit: float,
) -> bool:
    """
    Ensure we haven't removed more than `reduction_limit` of the original.

    Example: reduction_limit=0.15 (15%)
    → truncated_len must be at least 85% of original_len.
    """
    if original_len <= 0:
        return True
    min_len = int(original_len * (1.0 - reduction_limit))
    return truncated_len >= min_len


def _safe_markdown_cleanup(text: str) -> str:
    """
    Perform simple, conservative cleanups on truncated markdown:

    - If code fences (``` ) count is odd → close with one more fence.
    - If '**' (bold markers) count is odd → append '**' at the end.
    - If '`' (inline code) count is odd → append '`' at the end.
    - Avoid ending with an obviously broken link or image marker.
    """
    cleaned = text

    # Fix code fences
    fence_count = cleaned.count("```")
    if fence_count % 2 == 1:
        cleaned = cleaned.rstrip() + "\n```"

    # Fix bold markers
    bold_count = cleaned.count("**")
    if bold_count % 2 == 1:
        cleaned = cleaned.rstrip() + "**"

    # Fix inline code markers
    backtick_count = cleaned.count("`")
    if backtick_count % 2 == 1:
        cleaned = cleaned.rstrip() + "`"

    # Avoid ending on "[" or "(" or "!["
    trimmed = cleaned.rstrip()
    if trimmed.endswith("[") or trimmed.endswith("(") or trimmed.endswith("!"):
        # Trim back to last space/newline
        last_space = trimmed.rfind(" ")
        last_nl = trimmed.rfind("\n")
        cut = max(last_space, last_nl)
        if cut > 0:
            cleaned = trimmed[:cut].rstrip()
        else:
            cleaned = trimmed

    return cleaned


def smart_truncate_markdown(
    text: str,
    max_len: int,
    lang: Literal["en", "th"],
    overflow_limit: int = 128,
    reduction_limit: float = 0.15,
    section_id: Optional[str] = None,
) -> dict:
    """
    Safely truncate markdown text to approximately `max_len` characters.

    Behaviour:
    ----------
    1. JSON fast-path (NEW, hard rule):
       - If `is_json_block(text, section_id)` is True:
         → skip truncation entirely and return:
             {
               "safe_text": text,
               "truncation_applied": False,
               "strategy_used": "no_truncation_json"
             }

    2. If not JSON and len(text) <= max_len:
         → return unchanged, strategy_used = "none".

    3. If not JSON and len(text) > max_len:
         a. Try sentence-based truncation with allowed overflow:
            - Find sentence boundaries (EN vs TH)
            - Select the last boundary ≤ max_len + overflow_limit
            - If boundary ≥ max_len → strategy "allow_overflow"
            - Else → strategy "sentence_boundary_cut"

         b. Validate reduction: do not remove more than `reduction_limit`
            fraction of the original text; if too much removed → fallback.

         c. Apply markdown cleanup on the truncated text.

         d. If sentence-based strategy fails:
            - Fallback to naive cut at max_len
            - Cleanup markdown
            - strategy_used = "markdown_safe" if cleanup changed text, else "naive"

    Parameters
    ----------
    text : str
        Original section text.
    max_len : int
        Target maximum length in characters.
    lang : {"en", "th"}
        Language hint for sentence boundary detection.
    overflow_limit : int, default 128
        Allowed overflow characters beyond `max_len` when the next sentence
        boundary is close.
    reduction_limit : float, default 0.15
        Maximum allowed fraction of the original text that may be removed
        by sentence-based truncation.
    section_id : str | None, optional
        Section identifier; used as a hint for JSON-only sections.

    Returns
    -------
    dict
        {
          "safe_text": "<final text>",
          "truncation_applied": bool,
          "strategy_used": "none"
                            | "no_truncation_json"
                            | "allow_overflow"
                            | "sentence_boundary_cut"
                            | "markdown_safe"
                            | "naive"
        }
    """
    if text is None:
        return {
            "safe_text": "",
            "truncation_applied": False,
            "strategy_used": "none",
        }

    # 1. JSON fast-path — NEVER truncate JSON
    if is_json_block(text, section_id=section_id):
        return {
            "safe_text": text,
            "truncation_applied": False,
            "strategy_used": "no_truncation_json",
        }

    original_len = len(text)

    # 2. No truncation needed
    if original_len <= max_len:
        return {
            "safe_text": text,
            "truncation_applied": False,
            "strategy_used": "none",
        }

    # 3. Sentence-based truncation
    cut_index, strategy = _select_truncation_index_by_sentence(
        text=text,
        max_len=max_len,
        overflow_limit=overflow_limit,
        lang=lang,
    )

    if cut_index is not None and strategy is not None:
        candidate = text[:cut_index]
        if _validate_reduction(original_len, len(candidate), reduction_limit):
            cleaned = _safe_markdown_cleanup(candidate)
            return {
                "safe_text": cleaned,
                "truncation_applied": True,
                "strategy_used": strategy,
            }

    # 4. Fallback: naive cut + markdown cleanup
    naive_cut = text[:max_len]
    cleaned = _safe_markdown_cleanup(naive_cut)
    strategy_used = "markdown_safe" if cleaned != naive_cut else "naive"

    return {
        "safe_text": cleaned,
        "truncation_applied": True,
        "strategy_used": strategy_used,
    }
