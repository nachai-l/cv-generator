"""
Prompt Injection Detection Module
=================================

Provides lightweight, dependency-free functions for detecting potential
prompt injection or malicious content in user-provided text or request
payloads.

This module implements three complementary strategies:

1. Pattern-based detection
   - Regex patterns (critical and suspicious) loaded dynamically from
     `parameters/parameters.yaml`, allowing updates without code changes.

2. Heuristic analysis
   - Simple statistical heuristics such as excessive special-character
     ratios and newline density to catch obfuscated or multi-block payloads.

3. Recursive scanning
   - Safe traversal of nested data structures (dicts, lists, strings) to
     aggregate detection results from complex inputs.

Primary Functions
-----------------
- detect_injection(text: str) -> InjectionDetectionResult
    Scan a single text input for known injection signatures or heuristics.
    Returns a structured result including `is_safe`, `risk_score`, and
    `detected_patterns`.

- scan_dict_for_injection(data: dict | list | str) -> InjectionDetectionResult
    Recursively scan JSON-like payloads for injection attempts, combining
    findings across nested structures.

Configuration
-------------
Patterns are loaded from `parameters/parameters.yaml` under:

    security:
      critical_patterns: [...]
      suspicious_patterns: [...]

If configuration loading fails, both pattern lists default to empty tuples
and a warning is printed to stdout.

The output model `InjectionDetectionResult` (defined in
`schemas.internal_schema.py`) standardizes security scan results for
downstream guardrail logic or logging.
"""

from __future__ import annotations

import re
import copy
from pathlib import Path
from typing import Any

import yaml

from schemas.internal_schema import InjectionDetectionResult

# ---------------------------------------------------------------------------
# Load regex patterns from parameters.yaml
# ---------------------------------------------------------------------------

PARAM_FILE = Path(__file__).resolve().parents[2] / "parameters" / "parameters.yaml"

try:
    with open(PARAM_FILE, "r", encoding="utf-8") as f:
        params = yaml.safe_load(f) or {}
        _security = params.get("security", {}) or {}
        CRITICAL_PATTERNS: tuple[str, ...] = tuple(
            _security.get("critical_patterns", []) or []
        )
        SUSPICIOUS_PATTERNS: tuple[str, ...] = tuple(
            _security.get("suspicious_patterns", []) or []
        )
        CONTROL_CHARS_EXCEPT_WHITESPACE = _security.get(
            "control_chars_except_whitespace",
            r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]",
        )
except Exception as e:  # noqa: BLE001
    # Fallback if config missing or broken
    CRITICAL_PATTERNS = ()
    SUSPICIOUS_PATTERNS = ()
    print(f"[WARN] Failed to load security patterns from {PARAM_FILE}: {e}")

# ---------------------------------------------------------------------------
# Detection logic
# ---------------------------------------------------------------------------


def detect_injection(text: str) -> InjectionDetectionResult:
    """
    Analyze a single string for potential prompt injection.

    This function applies:
    - Regex-based matching against configured critical and suspicious patterns.
    - Simple heuristics on special-character and newline ratios.

    The resulting `InjectionDetectionResult` encodes whether the input is
    considered safe to process (`is_safe`), which patterns or heuristics were
    triggered (`detected_patterns`), and an approximate normalized risk score
    (`risk_score` in [0.0, 1.0]).

    Args:
        text:
            User-provided text to scan. Non-string or empty values are treated
            as safe and return a zero-risk result.

    Returns:
        InjectionDetectionResult:
            - `is_safe` is False if risk_score >= 0.8 or any critical patterns
              are matched.
            - `detected_patterns` contains tagged pattern IDs, e.g.
              "CRITICAL: <regex>", "SUSPICIOUS: <regex>",
              "HEURISTIC: HIGH_SPECIAL_CHAR_RATIO".
            - `risk_score` reflects the highest severity finding:
                * 0.0  = no known issues
                * ~0.4 = low risk (heuristics only)
                * ~0.6 = medium risk (suspicious patterns)
                * 1.0  = high risk (critical patterns)
    """
    # Treat non-string or empty strings as safe
    if not isinstance(text, str) or not text.strip():
        return InjectionDetectionResult.safe()

    detected: list[str] = []
    risk_score = 0.0

    # -------------------------
    # Critical patterns
    # -------------------------
    for pattern in CRITICAL_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            detected.append(f"CRITICAL: {pattern}")
            risk_score = 1.0

    # -------------------------
    # Suspicious patterns
    # -------------------------
    if risk_score < 1.0:
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                detected.append(f"SUSPICIOUS: {pattern}")
                risk_score = max(risk_score, 0.6)

    # -------------------------
    # Heuristic checks
    # -------------------------
    total_len = len(text)

    # Special-character ratio (potential obfuscation)
    special_ratio = sum(
        1 for c in text if not c.isalnum() and not c.isspace()
    ) / max(total_len, 1)
    if special_ratio > 0.3:
        detected.append("HEURISTIC: HIGH_SPECIAL_CHAR_RATIO")
        risk_score = max(risk_score, 0.5)

    # Newline ratio (multi-block payloads, hidden instructions)
    newline_ratio = text.count("\n") / max(total_len, 1)
    if newline_ratio > 0.1:
        detected.append("HEURISTIC: EXCESSIVE_NEWLINES")
        risk_score = max(risk_score, 0.4)

    # De-duplicate patterns while preserving order
    detected = list(dict.fromkeys(detected))

    return InjectionDetectionResult.from_findings(
        is_safe=(risk_score < 0.8),
        detected_patterns=detected,
        risk_score=risk_score,
    )


def scan_dict_for_injection(
    data: dict[str, Any] | list[Any] | str,
) -> InjectionDetectionResult:
    """
    Recursively scan nested JSON-like data for prompt injection.

    This helper walks through dicts, lists, and strings, applying
    `detect_injection` to every string value it finds and aggregating the
    results into a single `InjectionDetectionResult`.

    Only string values are inspected; other scalar types (int, float, bool,
    None) are ignored.

    Args:
        data:
            Arbitrary payload to scan, typically a deserialized request body.
            Supported containers:
              - dict[str, Any]
              - list[Any]
              - str

    Returns:
        InjectionDetectionResult:
            - `is_safe` is False if the highest risk score across all strings
              is >= 0.8.
            - `detected_patterns` is the de-duplicated union of all patterns
              found in any nested string.
            - `risk_score` is the maximum `risk_score` observed across all
              scanned values.

    Notes:
        This function is intended for coarse-grained guardrails at API
        boundaries. For finer-grained checks, call `detect_injection`
        directly on specific fields of interest.
    """
    all_detected: list[str] = []
    max_risk = 0.0

    def _scan(v: Any) -> None:
        nonlocal all_detected, max_risk

        if isinstance(v, str):
            r = detect_injection(v)
            if r.has_findings:
                all_detected.extend(r.detected_patterns)
                max_risk = max(max_risk, r.risk_score)

        elif isinstance(v, dict):
            for vv in v.values():
                _scan(vv)

        elif isinstance(v, list):
            for vv in v:
                _scan(vv)

        # Other scalar types are ignored (int, float, bool, None, etc.)

    _scan(data)

    # No findings at all → explicitly return a safe result
    if not all_detected and max_risk == 0.0:
        return InjectionDetectionResult.safe()

    # De-duplicate patterns while preserving order
    all_detected = list(dict.fromkeys(all_detected))

    return InjectionDetectionResult.from_findings(
        is_safe=(max_risk < 0.8),
        detected_patterns=all_detected,
        risk_score=max_risk,
    )


def sanitize_dict(data: dict[str, Any] | list[Any] | str) -> Any:
    """
    Recursively sanitize text fields in nested data structures.

    - Normalizes all whitespace (spaces, tabs, newlines) to single spaces
    - Strips leading/trailing whitespace
    - Removes non-whitespace control characters (0x00–0x08, 0x0B–0x0C, 0x0E–0x1F, 0x7F)

    Returns a deep-copied, sanitized structure with the same shape as input.
    """

    def _sanitize(v: Any) -> Any:
        if isinstance(v, str):
            # 1) Collapse any whitespace (space, \n, \t, etc.) into a single space
            v = re.sub(r"\s+", " ", v).strip()
            # 2) Remove non-whitespace control characters
            v = re.sub(CONTROL_CHARS_EXCEPT_WHITESPACE, "", v)
            return v
        elif isinstance(v, dict):
            return {k: _sanitize(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [_sanitize(item) for item in v]
        else:
            # Non-string scalars are returned unchanged
            return v

    return _sanitize(copy.deepcopy(data))