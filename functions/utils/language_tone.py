"""
Utility helpers for normalizing language and tone descriptions
for use in LLM prompts.

These functions are used across multiple stages:
- Stage A: validation, inference
- Stage B: prompt construction
- Stage C: normalization, error messaging

Centralizing these helpers avoids cross-stage entanglement and keeps
language/tone handling consistent across the entire CV Generation pipeline.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Language Description Normalizer
# ---------------------------------------------------------------------------

def describe_language(language_code: str | None) -> str:
    """
    Convert a language code (e.g., 'en', 'th', 'ja') into a natural-language
    descriptor suitable for LLM prompts.

    Unknown values fall back to a generic descriptor:
        "the target language (code='xx')"

    Default:
        If language_code is None → English
    """
    if not language_code:
        return "English"

    code = language_code.lower()

    mapping = {
        "en": "English",
        "th": "Thai",
        "ja": "Japanese",
        "zh": "Chinese",
        "fr": "French",
        "de": "German",
        "es": "Spanish",
        "ko": "Korean",
    }

    if code in mapping:
        return mapping[code]

    # Fallback for unknown codes
    return f"the target language (code='{language_code}')"


# ---------------------------------------------------------------------------
# Tone Description Normalizer
# ---------------------------------------------------------------------------

def describe_tone(tone: str | None) -> str:
    """
    Convert a tone keyword into a natural-language description for use in prompts.

    Accepted tones:
        - formal
        - neutral
        - academic
        - casual
        - funny (for testing – intentionally unprofessional)

    Unknown values → neutral professional tone.
    """
    if not tone:
        return "a neutral, professional tone"

    t = tone.strip().lower()

    tone_map = {
        "formal": (
            "a highly formal, polished business tone: structured sentences, "
            "no contractions, precise vocabulary, and strictly professional wording"
        ),
        "neutral": (
            "a neutral, professional tone: clear, concise, factual, and business-oriented "
            "without emotional or stylistic embellishment"
        ),
        "academic": (
            "an academic, research-focused tone: objective phrasing, technical accuracy, "
            "cautious claims, and structured explanations typical of scholarly writing"
        ),
        "casual": (
            "a casual and friendly tone: light conversational phrasing, occasional "
            "contractions, and approachable wording while remaining professional"
        ),
        "funny": (
            # Testing-only comedic tone
            "a **wacky, over-the-top, zany, comedic tone** full of playful exaggeration, "
            "silly analogies, and harmless jokes 🤪. "
            "Example: instead of saying 'improved efficiency', say 'sped things up "
            "faster than Thai milk tea disappears at a company party.'"
        ),
    }

    return tone_map.get(t, "a neutral, professional tone")
