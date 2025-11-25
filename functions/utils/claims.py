# functions/utils/claims.py

"""
Utilities for LLM-based claim justification:
- Splitting section text from justification JSON
- Parsing justification JSON safely
- Validating that justification claims actually appear in the section text
- Config helper to decide whether justification is required for a section

This module is deliberately **generic** and does not depend on Stage B internals:
Stage B should:
  1) Build a combined prompt (section + justification instructions).
  2) Call the LLM once.
  3) Use `split_section_and_justification` to separate text vs JSON.
  4) Use `parse_justification_json` to build a Justification object.
  5) Use `validate_justification_against_text` to clean/filter it.
  6) Attach the final Justification to the SectionContent / CVGenerationResponse.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

import structlog

from functions.utils.common import extract_loose_json_block
from schemas.output_schema import Justification

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

SECTION_SPLITTERS: List[str] = [
    "=== JUSTIFICATION_JSON ===",
    "JUSTIFICATION_JSON",
    "===JUSTIFICATION===",
]

# Small English-ish stopword list for token-overlap matching.
STOPWORDS = {
    "the",
    "and",
    "to",
    "for",
    "a",
    "an",
    "of",
    "in",
    "on",
    "with",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "by",
    "from",
    "this",
    "that",
    "it",
    "or",
    "be",
    "into",
    "over",
    "under",
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def build_empty_justification() -> Justification:
    """
    Build a default "empty" justification object.

    Shape (as requested):
      {
        "evidence_map": [],
        "unsupported_claims": [],
        "coverage_score": 1.0,
        "total_claims_analyzed": 0
      }

    This ensures we have a consistent baseline even when justification is
    disabled or fails.
    """
    j = Justification()
    setattr(j, "evidence_map", [])
    setattr(j, "unsupported_claims", [])
    setattr(j, "coverage_score", 1.0)
    setattr(j, "total_claims_analyzed", 0)
    return j


def should_require_justification(
    section_id: str,
    generation_cfg: Dict[str, Any] | None,
) -> bool:
    if not generation_cfg:
        return False

    cfg = generation_cfg.get("sections_req_justification", [])

    if cfg == []:
        return False

    section_norm = (section_id or "").strip().lower()

    if isinstance(cfg, str):
        return cfg.strip().lower() == "all"

    if isinstance(cfg, (list, tuple, set)):
        cfg_norm = {str(x).strip().lower() for x in cfg}
        return section_norm in cfg_norm

    return False



def split_section_and_justification(raw_output: str) -> Tuple[str, str]:
    """
    Split a combined LLM output into:
      - section_text
      - justification_raw_json (as string, may be empty)

    Strategy:
      1) Try multiple known separator tokens (SECTION_SPLITTERS).
      2) If not found, try to detect a JSON object containing "evidence_map"
         near the end of the output.
      2b) If still not found, try a loose JSON block extraction.
      3) If still not found, treat the entire output as section text only.
    """
    if not raw_output:
        return "", ""

    # 1) Try explicit separators
    for separator in SECTION_SPLITTERS:
        if separator in raw_output:
            before, _, after = raw_output.partition(separator)
            section_text = before.strip()
            justification_raw = after.strip()
            logger.info(
                "justification_separator_found",
                separator=separator,
                section_length=len(section_text),
                justification_length=len(justification_raw),
            )
            return section_text, justification_raw

    # 2) Fallback: detect JSON object containing "evidence_map"
    json_match = re.search(
        r'\{[^}]*"evidence_map".*\}',
        raw_output,
        re.DOTALL,
    )
    if json_match:
        json_start = json_match.start()
        section_text = raw_output[:json_start].strip()
        justification_raw = raw_output[json_start:].strip()
        logger.info(
            "justification_json_detected_fallback",
            section_length=len(section_text),
            justification_length=len(justification_raw),
        )
        return section_text, justification_raw

    # 2b) EXTRA fallback: loose JSON block extraction
    loose_block = extract_loose_json_block(raw_output)
    if loose_block:
        idx = raw_output.find(loose_block)
        if idx != -1:
            section_text = raw_output[:idx].strip()
            justification_raw = loose_block.strip()
            logger.info(
                "justification_loose_json_detected_fallback",
                section_length=len(section_text),
                justification_length=len(justification_raw),
            )
            return section_text, justification_raw

    # 3) No justification found
    logger.info(
        "justification_not_found_in_output",
        output_length=len(raw_output),
    )
    return raw_output.strip(), ""


def _strip_code_fences_and_language_hints(raw: str) -> str:
    """
    Remove common markdown code-fence wrappers such as:

        ```json
        { ... }
        ```

    and bare triple-backticks that often appear around JSON blocks.
    """
    if not raw:
        return ""

    text = raw.strip()

    # Remove leading ```json / ```JSON / ``` etc.
    text = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", text)

    # Remove trailing ``` fence if present.
    text = re.sub(r"\s*```$", "", text)

    # Fallback: strip any leftover ``` anywhere.
    text = text.replace("```", "")

    return text.strip()


def _detect_and_complete_truncated_json(raw: str) -> str:
    """
    Detect if JSON appears to be truncated (incomplete) and attempt to complete it.

    Common LLM truncation patterns:
    - Unterminated strings: "reason": "some text without closing quote
    - Missing closing braces: {"evidence_map": [{"section": "skills"
    - Cut off mid-array: "evidence_ids": ["item1", "item2

    Strategy:
    1. Count opening/closing braces and brackets
    2. Find the last complete object in evidence_map
    3. Close any unterminated strings
    4. Balance remaining structures
    """
    if not raw:
        return ""

    text = raw.strip()

    # Count structural elements
    open_braces = text.count('{')
    close_braces = text.count('}')
    open_brackets = text.count('[')
    close_brackets = text.count(']')

    # Check if JSON appears complete
    if open_braces == close_braces and open_brackets == close_brackets:
        return text  # Looks complete, no fix needed

    logger.info(
        "truncated_json_detected",
        open_braces=open_braces,
        close_braces=close_braces,
        open_brackets=open_brackets,
        close_brackets=close_brackets,
        text_length=len(text),
    )

    # Find the last complete evidence_map entry
    # Strategy: walk through and track depth, find last properly closed object
    last_complete_obj = -1
    depth = 0
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == '\\':
            escape_next = True
            continue

        if char == '"' and not in_string:
            in_string = True
        elif char == '"' and in_string:
            in_string = False
        elif not in_string:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                # Complete object at depth 2 = inside evidence_map array
                if depth == 2:
                    last_complete_obj = i

    # Truncate to last complete object if found
    if last_complete_obj > 0:
        text = text[:last_complete_obj + 1]
        logger.info(
            "truncated_to_last_complete_object",
            new_length=len(text),
        )

    # Close any unterminated string
    # Count unescaped quotes
    quote_count = 0
    escape_next = False
    for char in text:
        if escape_next:
            escape_next = False
            continue
        if char == '\\':
            escape_next = True
            continue
        if char == '"':
            quote_count += 1

    # Odd number of quotes = unterminated string
    if quote_count % 2 == 1:
        text += '"'
        logger.info("closed_unterminated_string")

    # Balance remaining structures
    # Note: recalculate after potential truncation
    missing_brackets = text.count('[') - text.count(']')
    missing_braces = text.count('{') - text.count('}')

    # Close arrays first (evidence_map, evidence_ids), then objects
    text += ']' * missing_brackets
    text += '}' * missing_braces

    if missing_brackets > 0 or missing_braces > 0:
        logger.info(
            "completed_truncated_json",
            added_brackets=missing_brackets,
            added_braces=missing_braces,
        )

    return text


def _try_loose_json_cleanup(raw: str) -> str:
    """
    Apply a comprehensive JSON cleanup pipeline before strict json.loads.

    Pipeline:
      1. Strip code fences / language hints (```json ... ```)
      2. Remove trailing commas before '}' or ']'
      3. **Detect and complete truncated JSON** (NEW)
      4. Remove control characters
      5. Fix common escape issues

    Returns the cleaned string (may be significantly modified).
    """
    if not raw:
        return ""

    # Step 1: Remove markdown fences
    cleaned = _strip_code_fences_and_language_hints(raw)

    # Step 2: Remove trailing commas
    cleaned = re.sub(r",(\s*[}\]])", r"\1", cleaned)

    # Step 3: Complete truncated JSON (MOST IMPORTANT)
    cleaned = _detect_and_complete_truncated_json(cleaned)

    # Step 4: Remove control characters that sometimes appear
    cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)

    # Step 5: Fix common double-escape issues: \\" -> \"
    cleaned = cleaned.replace('\\\\"', '\\"')

    return cleaned.strip()


def _extract_evidence_items_via_regex(raw: str) -> List[Dict[str, Any]]:
    """
    Last-resort extraction: use regex to pull out evidence_map items
    even if the JSON as a whole is unparseable.

    This is a safety net for severely malformed JSON.
    """
    items = []

    # Pattern to match individual evidence objects
    # Looks for: {"section": "...", "sentence": "...", ...}
    pattern = r'\{\s*"section"\s*:\s*"([^"]+)"\s*,\s*"sentence"\s*:\s*"([^"]+)"[^}]*\}'

    for match in re.finditer(pattern, raw, re.DOTALL):
        section = match.group(1)
        sentence = match.group(2)

        items.append({
            "section": section or "profile_summary",
            "sentence": sentence,
            # IMPORTANT: ensure evidence_ids is non-empty to satisfy the schema
            "evidence_ids": ["fallback.regex_extracted"],
            "match_score": 0.5,  # Low confidence for regex extraction
        })

    if items:
        logger.info(
            "evidence_extracted_via_regex_fallback",
            count=len(items),
        )

    return items



def parse_justification_json(raw: str | None) -> Justification:
    """
    Parse the justification JSON text into a Justification object.

    This function is extremely robust and handles:
    - Markdown code fences (```json ... ```)
    - Trailing commas
    - Truncated JSON (incomplete structures)
    - Unterminated strings
    - Control characters
    - Malformed escape sequences

    Fallback pipeline:
    1. Strict JSON parse after basic cleanup
    2. Comprehensive repair (truncation completion, etc.)
    3. Regex extraction as last resort
    4. Empty justification if all else fails

    Returns:
        Justification object, never raises exceptions.
    """
    if not raw:
        return build_empty_justification()

    loose = extract_loose_json_block(raw)
    if loose:
        raw = loose

    cleaned = _strip_code_fences_and_language_hints(raw)

    # --- First strict parse attempt ----------------------------------------
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:  # More specific exception
        # Calculate line/column in original for better debugging
        error_pos = getattr(exc, 'pos', None)
        error_line = getattr(exc, 'lineno', None)
        error_col = getattr(exc, 'colno', None)

        # Show context around error
        context = ""
        if error_pos and error_pos < len(cleaned):
            start = max(0, error_pos - 100)
            end = min(len(cleaned), error_pos + 100)
            context = cleaned[start:end]

        logger.warning(
            "justification_json_parse_failed",
            error=str(exc),
            error_line=error_line,
            error_col=error_col,
            error_context=context,  # NEW: show surrounding text
            raw_length=len(cleaned),
            note="strict json.loads(cleaned) failed; attempting repair",
        )

        # --- Repair pass ----------------------------------------------------
        repaired = _try_loose_json_cleanup(cleaned)

        if repaired == cleaned:
            logger.info(
                "justification_json_repair_noop",
                raw_length=len(cleaned),
                note="repair produced no changes; trying regex extraction",
            )

            # NEW: Try regex extraction as last resort
            items = _extract_evidence_items_via_regex(cleaned)
            if items:
                return Justification(
                    evidence_map=items,
                    unsupported_claims=[],
                    coverage_score=0.5,  # Low confidence
                    total_claims_analyzed=len(items),
                )

            return build_empty_justification()

        try:
            data = json.loads(repaired)
            logger.info(
                "justification_json_repair_succeeded",
                raw_length=len(repaired),
            )
        except Exception as exc2:
            logger.warning(
                "justification_json_repair_failed",
                error=str(exc2),
                raw_length=len(repaired),
            )

            # NEW: Try regex extraction as final fallback
            items = _extract_evidence_items_via_regex(repaired)
            if items:
                return Justification(
                    evidence_map=items,
                    unsupported_claims=[],
                    coverage_score=0.5,
                    total_claims_analyzed=len(items),
                )

            return build_empty_justification()

    # --- Model validation ---------------------------------------------------
    try:
        justif = Justification.model_validate(data)
        return justif
    except Exception as exc:
        logger.warning(
            "justification_model_validation_failed",
            error=str(exc),
            data_preview=str(data)[:200],
        )
        return build_empty_justification()

# ---------------------------------------------------------------------------
# Legacy → new-schema migration helper
# ---------------------------------------------------------------------------

def migrate_legacy_justification_schema(
    data: dict[str, Any],
    section_id: str,
    section_text: str,
) -> dict[str, Any]:
    """
    Convert legacy justification schema:

        evidence_map: [
            {"claim": "...", "source": "inferred"},
            ...
        ]

    into the new schema required by `EvidenceMapping`:

        {
          "section": section_id,
          "sentence": "...",
          "evidence_ids": [...],
          "match_score": float
        }

    This allows Stage C to accept old-style LLM outputs without breaking.
    """
    if not isinstance(data, dict):
        return data

    ev = data.get("evidence_map")
    if not isinstance(ev, list) or not ev:
        return data

    # Detect legacy shape
    first = ev[0]
    if "claim" in first and "source" in first and "sentence" not in first:
        # Split text into sentences
        sentences = [
            s.strip() for s in re.split(r"[.!?]\s+", section_text) if s.strip()
        ]
        if not sentences:
            sentences = [section_text.strip()] if section_text.strip() else []

        migrated = []
        for item in ev:
            claim = item.get("claim", "").strip()
            source = item.get("source", "").strip()

            # Pick best matching sentence
            sent = next(
                (s for s in sentences if claim and claim.lower() in s.lower()),
                sentences[0] if sentences else claim,
            )

            # Construct evidence_ids + match_score from legacy "source"
            if source.lower() in ("profile_info", "profile", "base"):
                evidence_ids = ["profile_info"]
                score = 0.95
            elif source:
                evidence_ids = [f"student_profile.{source}"]
                score = 0.80
            else:
                evidence_ids = []
                score = 0.60

            migrated.append(
                {
                    "section": section_id,
                    "sentence": sent,
                    "evidence_ids": evidence_ids,
                    "match_score": score,
                }
            )

        data["evidence_map"] = migrated
        # Ensure required keys exist
        data.setdefault("unsupported_claims", [])
        data.setdefault("coverage_score", 0.0)
        data.setdefault("total_claims_analyzed", len(migrated))

        logger.info(
            "justification_schema_migrated",
            migrated_count=len(migrated),
            section=section_id,
        )

    return data

def claim_appears_in_text(
        claim: str, section_text: str,
        min_token_length: int = 3,
        min_overlap_ratio: float = 0.7, # 70% of key terms should appear
) -> bool:
    """
    Heuristic check whether a claim appears (or clearly corresponds) in the
    section text.

    - First uses a simple normalized substring check.
    - Then falls back to token-overlap with a small stopword filter to handle
      mild paraphrasing.

    Returns:
        True  if the claim is likely represented in the text.
        False otherwise.
    """
    if not claim or not section_text:
        return False

    claim_norm = claim.lower().strip()
    text_norm = section_text.lower()

    # Direct substring match (fast path)
    if claim_norm in text_norm:
        return True

    # Token overlap (handles paraphrasing / reordering)
    tokens = {t for t in claim_norm.split() if t and t not in STOPWORDS}
    if len(tokens) < min_token_length:
        # With too few tokens, overlap is not very meaningful.
        return False

    matches = sum(1 for t in tokens if t in text_norm)
    overlap_ratio = matches / max(len(tokens), 1)

    return overlap_ratio >= min_overlap_ratio


def _extract_claim_from_evidence_entry(entry: Any) -> str | None:
    """
    Attempt to extract a claim string from a generic evidence entry.

    Supports both:
      - legacy schema: "claim", "text", "description"
      - new schema:    "sentence" (as in EvidenceMapping.sentence)
    """
    if entry is None:
        return None

    if isinstance(entry, dict):
        claim = (
            entry.get("claim")
            or entry.get("sentence")      # NEW: support new schema
            or entry.get("text")
            or entry.get("description")
        )
        return str(claim) if claim else None

    # Object-like
    for attr in ("claim", "sentence", "text", "description"):  # NEW: include "sentence"
        if hasattr(entry, attr):
            val = getattr(entry, attr)
            if val:
                return str(val)

    return None


def validate_justification_against_text(
    justification: Justification | None,
    section_text: str,
) -> Justification:
    """
    Validate and clean a Justification object by checking that its claims
    actually appear in the generated section text.

    Steps:
      1) If justification is None → return empty.
      2) Filter evidence_map entries whose claim is not found in section_text.
      3) Filter unsupported_claims that do not appear in section_text.
      4) Recompute:
           - total_claims_analyzed
           - coverage_score = evidence_claims / total_claims
      5) If no claims remain, return an empty justification.

    This helps defend against:
      - LLM hallucinating claims not present in the section.
      - Stale or misaligned justification JSON.
    """
    if justification is None:
        return build_empty_justification()

    if not section_text:
        # No text to compare against → treat as empty to be safe.
        logger.info(
            "justification_validation_no_section_text",
        )
        return build_empty_justification()

    # Safely access existing fields (they may or may not be present).
    evidence_map_raw = getattr(justification, "evidence_map", []) or []
    unsupported_raw = getattr(justification, "unsupported_claims", []) or []

    validated_evidence: List[Any] = []
    validated_unsupported: List[Any] = []

    # Filter evidence_map
    for entry in evidence_map_raw:
        claim_str = _extract_claim_from_evidence_entry(entry)
        if not claim_str:
            continue
        if claim_appears_in_text(claim_str, section_text):
            validated_evidence.append(entry)

    # Filter unsupported_claims (can be strings or dicts)
    for item in unsupported_raw:
        if isinstance(item, str):
            claim_str = item
        else:
            claim_str = _extract_claim_from_evidence_entry(item)

        if not claim_str:
            continue
        if claim_appears_in_text(claim_str, section_text):
            validated_unsupported.append(item)
    total_claims = len(validated_evidence) + len(validated_unsupported)

    if total_claims == 0:
        logger.info(
            "justification_all_claims_filtered_out",
        )
        return build_empty_justification()

    # Update the existing model in-place
    setattr(justification, "evidence_map", validated_evidence)
    setattr(justification, "unsupported_claims", validated_unsupported)
    setattr(justification, "total_claims_analyzed", total_claims)

    coverage_score = len(validated_evidence) / float(total_claims)
    setattr(justification, "coverage_score", coverage_score)

    logger.info(
        "justification_validated",
        evidence_count=len(validated_evidence),
        unsupported_count=len(validated_unsupported),
        total_claims=total_claims,
        coverage_score=coverage_score,
    )

    return justification

# ---------------------------------------------------------------------------
# Token budget adjustment for justification
# ---------------------------------------------------------------------------
def adjust_section_token_budget(
        section_id: str,
        base_budget: int | None,
        generation_cfg: Dict[str, Any] | None,
) -> int | None:
    """
    Adjust the token budget when justification is enabled.

    Handles section_id normalization (e.g., skills_structured → skills)
    to match the sections_req_justification config.
    """
    if base_budget is None:
        return None

    if not generation_cfg:
        return base_budget

    # 🔹 NEW: Normalize section_id to canonical form
    # This ensures "skills_structured" is treated as "skills"
    canonical_section_id = section_id
    if section_id.endswith("_structured"):
        canonical_section_id = section_id[:-len("_structured")]

    # Try both the exact section_id AND the canonical form
    needs_justification = (
            should_require_justification(section_id, generation_cfg)
            or should_require_justification(canonical_section_id, generation_cfg)
    )

    if not needs_justification:
        return base_budget

    # Additional justification token allowance
    extra = generation_cfg.get("justification_additional_tokens")
    try:
        extra = int(extra)
    except Exception:
        return base_budget

    if extra <= 0:
        return base_budget

    return int(base_budget) + extra
