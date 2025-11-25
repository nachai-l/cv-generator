# utils/skills_formatting.py

"""
Utility helpers for rendering and parsing skill sections.

This module is intentionally kept free of any LLM, Stage B, or I/O dependencies.
It focuses purely on **presentation**, **simple parsing**, and **canonical-matching**
logic that Stage B relies on.

Responsibilities
----------------
1. Presentation / Rendering
   - pretty_skill_level()
   - format_plain_skill_bullets()

2. Fallback Parsing
   - parse_skills_from_bullets()

3. Canonical Skill Matching & Validation
   - is_combined_canonical_name()
   - match_canonical_skill()  ← hybrid alias/token/fuzzy matcher

Design Principles
-----------------
- Pure functions only: no YAML loading, no LLM calls.
- Stage B supplies alias maps + thresholds.
"""

from typing import List, Optional, Dict
from difflib import SequenceMatcher

from schemas.output_schema import OutputSkillItem
from schemas.internal_schema import CanonicalSkill
import re

_WORD_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}

def _contains_word_case_insensitive(text: str, word: str) -> bool:
    """
    Return True if `word` appears as a full word in `text` (case-insensitive).
    Prevents cases like canonical='Java' matching 'JavaScript'.
    """
    if not word:
        return False

    key = word.strip().lower()
    pat = _WORD_BOUNDARY_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(key)}\b", re.IGNORECASE)
        _WORD_BOUNDARY_CACHE[key] = pat

    return pat.search(text) is not None


# ---------------------------------------------------------------------------
# Presentation helpers
# ---------------------------------------------------------------------------

def pretty_skill_level(level: Optional[str]) -> Optional[str]:
    """Convert taxonomy-style level labels into human-readable labels."""
    if not level:
        return None

    s = str(level)
    if "_" in s:
        return s.split("_", 1)[1].replace("_", " ").title()
    return s.title()


def format_plain_skill_bullets(skills: List[OutputSkillItem]) -> str:
    """Render a simple bullet list from structured skills."""
    lines: List[str] = []

    for sk in skills:
        name = (sk.name or "").strip()
        if not name:
            continue

        level_str = pretty_skill_level(getattr(sk, "level", None))
        if level_str:
            lines.append(f"- {name} ({level_str})")
        else:
            lines.append(f"- {name}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Basic fallback parsing
# ---------------------------------------------------------------------------

def parse_skills_from_bullets(text: str) -> List[str]:
    """
    Parse a bullet-list 'skills' block into raw skill names.
    Very conservative parsing.
    """
    if not text:
        return []

    skills: List[str] = []

    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue

        if raw[0] in "-*•":
            raw = raw[1:].strip()

        if not raw:
            continue

        if raw.endswith("."):
            raw = raw[:-1].rstrip()

        if raw:
            skills.append(raw)

    return skills


# ---------------------------------------------------------------------------
# Canonical matching helpers
# ---------------------------------------------------------------------------

def is_combined_canonical_name(name: str, canonical_by_name: dict[str, CanonicalSkill]) -> bool:
    """
    Detect skills like "SQL & Python" that join multiple canonical names.
    """
    if not name:
        return False

    lower = name.strip().lower()
    separators = ["&", "/", ","]

    hits = 0
    for sep in separators:
        if sep in lower:
            parts = [p.strip() for p in lower.split(sep) if p.strip()]
            for p in parts:
                if p in canonical_by_name:
                    hits += 1

    return hits >= 2


def _normalize_skill_tokens(s: str) -> list[str]:
    """
    Normalize a skill string into comparable tokens.
    """
    if not s:
        return []

    s = s.lower().strip()

    for sep in ["&", "/", ",", "+", "|"]:
        s = s.replace(sep, " ")

    for ch in "()[].":
        s = s.replace(ch, " ")

    return [t for t in s.split() if t]


# ---------------------------------------------------------------------------
# Hybrid canonical matching logic
# ---------------------------------------------------------------------------

def match_canonical_skill(
    name: str,
    canonical_by_name: Dict[str, CanonicalSkill],
    *,
    alias_map: Optional[Dict[str, str]] = None,
    min_coverage: float = 0.5,
    fuzzy_threshold: Optional[float] = None,
) -> Optional[CanonicalSkill]:
    """
    Match an LLM-returned skill name to a canonical taxonomy skill.

    Matching rules (in order):
      1. Exact match (case-insensitive)
      2. Alias map (e.g., 'ml' → 'machine learning')
      3. Token-based extension matching (rename expansion)
      4. Optional fuzzy fallback for typos

    Stage B provides:
      - alias_map (from alias_mapping.yaml)
      - min_coverage (parameters.yaml)
      - fuzzy_threshold (parameters.yaml, or None to disable)
    """
    if not name:
        return None

    lowered = name.strip().lower()
    if not lowered:
        return None

    # 1) Exact match
    if lowered in canonical_by_name:
        return canonical_by_name[lowered]

    # 2) Alias match
    if alias_map:
        canonical_name = alias_map.get(lowered)
        if canonical_name:
            key = canonical_name.strip().lower()
            if key in canonical_by_name:
                return canonical_by_name[key]

    # 3) Token-based canonical detection
    candidate_tokens = _normalize_skill_tokens(name)
    if not candidate_tokens:
        return None

    candidate_set = set(candidate_tokens)

    best_match: Optional[CanonicalSkill] = None
    best_token_count = 0
    best_coverage = 0.0

    for key, canon in canonical_by_name.items():
        if not key:
            continue

        # Prevent false positives (Java vs JavaScript)
        if key in lowered and not _contains_word_case_insensitive(lowered, key):
            continue

        canon_tokens = _normalize_skill_tokens(key)
        if not canon_tokens:
            continue

        canon_set = set(canon_tokens)

        if not canon_set.issubset(candidate_set):
            continue

        coverage = len(canon_set) / max(len(candidate_set), 1)
        if coverage < min_coverage:
            continue

        # Prefer longer canonical names + higher coverage
        if len(canon_set) > best_token_count or (
            len(canon_set) == best_token_count and coverage > best_coverage
        ):
            best_match = canon
            best_token_count = len(canon_set)
            best_coverage = coverage

    if best_match is not None:
        return best_match

    # 4) Optional fuzzy fallback (handle spelling mistakes)
    if fuzzy_threshold is not None and len(lowered) >= 4:
        best_similarity = 0.0
        best_canon = None

        for key, canon in canonical_by_name.items():
            if not key or len(key) < 3:
                continue

            similarity = SequenceMatcher(None, lowered, key).ratio()
            if similarity >= fuzzy_threshold and similarity > best_similarity:
                # extra guard for Java vs JavaScript
                if key in lowered and not _contains_word_case_insensitive(lowered, key):
                    continue
                best_similarity = similarity
                best_canon = canon

        if best_canon:
            return best_canon

    return None

# ---------------------------------------------------------------------------
# JD matching helpers (pure, no Stage B / LLM deps)
# ---------------------------------------------------------------------------

def compute_section_matched_jd_skills(
    section_id: str,
    section_text: str,
    skills_output: List[OutputSkillItem] | None,
    jd_target_skill_names: List[str] | set[str],
) -> List[str]:
    """
    Given:
      - section_id (e.g. 'profile_summary', 'skills', 'experience')
      - section_text: rendered text of the section
      - skills_output: final structured skills list (taxonomy + inferred)
      - jd_target_skill_names: iterable of JD skill names (strings)

    Return:
      - A list of skill *names* (canonical/pretty) that:
          * appear in jd_target_skill_names (case-insensitive), and
          * are evidenced by this section.

    Rules:
      - For 'skills' section:
          → any skills in `skills_output` that are in JD are considered matched.
      - For all other sections:
          → require that the skill name appears as a word in section_text
            (case-insensitive, word-boundary-aware).
    """
    if not jd_target_skill_names or not skills_output:
        return []

    # Normalize JD target skills to lowercase set
    jd_set = {str(name).strip().lower() for name in jd_target_skill_names if str(name).strip()}
    if not jd_set:
        return []

    text_lc = (section_text or "").lower()

    # Map lowercase name → pretty name from skills_output
    skill_by_lc: Dict[str, str] = {}
    for s in skills_output:
        name = (getattr(s, "name", None) or "").strip()
        if not name:
            continue
        skill_by_lc[name.lower()] = name

    matched_lc: set[str] = set()

    for name_lc, pretty in skill_by_lc.items():
        if name_lc not in jd_set:
            continue

        if section_id == "skills":
            # Structured skills section: being in skills_output + JD is enough
            matched_lc.add(name_lc)
        else:
            # Other sections: require mention in text as a word
            if _contains_word_case_insensitive(text_lc, pretty):
                matched_lc.add(name_lc)

    # Return pretty names in deterministic order
    pretty_results: List[str] = []
    for name_lc in sorted(matched_lc):
        pretty_results.append(skill_by_lc.get(name_lc, name_lc))

    return pretty_results
