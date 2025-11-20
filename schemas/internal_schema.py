"""Internal schemas for security, monitoring, and guardrail utilities.

These models are not part of the external API contract. They are used
to structure internal signals (e.g., prompt injection detection results),
validation outcomes, and evidence planning in a consistent, type-safe way
across services.

**Skills Handling Philosophy:**

The skills system enforces a "canonical source of truth" principle:

1. **Stage A** extracts canonical skills from profile_info/student_profile and
   creates a SkillsSectionPlan with immutable name+level pairs.

2. **Stage B** generates structured skills via LLM, then:
   - Deduplicates by name only (case-insensitive)
   - Uses fuzzy matching + alias maps to snap similar names to canonical forms
   - Reconciles levels: any skill matching a canonical name gets its level
     restored to the canonical value (LLM cannot change taxonomy levels)
   - Optionally preserves all canonical skills (dropping_irrelevant_skills=False)

3. **Stage C** validates and deduplicates again (aligned with Stage B):
   - Deduplicates by name only (one record per skill name)
   - Never overwrites canonical levels coming from Stage B
   - Does not modify levels; the first occurrence for each name is kept

This ensures:
- **Name consistency**: Fuzzy variants map to canonical names
- **Level integrity**: Taxonomy levels are never silently downgraded
- **Single source of truth**: One canonical record per skill name
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Security / Prompt Injection
# ---------------------------------------------------------------------------


class InjectionDetectionResult(BaseModel):
    """Standardized result for prompt injection detection.

    Attributes:
        is_safe:
            Boolean flag indicating whether the scanned input is considered
            safe enough to continue processing (True) or should be blocked /
            flagged for review (False).
        detected_patterns:
            List of human-readable tags or pattern identifiers that triggered
            the detection logic (e.g., "CRITICAL: ignore\\s+previous",
            "SUSPICIOUS: \\{\\{.*\\}\\}", "HEURISTIC: HIGH_SPECIAL_CHAR_RATIO").
        risk_score:
            Normalized risk score in [0.0, 1.0], where:
              - 0.0  = no known issues
              - ~0.4 = low risk (monitor)
              - ~0.6 = medium risk (suspicious)
              - ≥0.8 = high risk (block)
    """

    is_safe: bool = Field(
        ...,
        description="True if the input is considered safe enough to process.",
    )
    detected_patterns: List[str] = Field(
        default_factory=list,
        description="List of patterns / heuristics that were triggered.",
    )
    risk_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Normalized risk score in [0.0, 1.0].",
    )

    @model_validator(mode="after")
    def _auto_is_safe_from_risk(
        cls, values: "InjectionDetectionResult"
    ) -> "InjectionDetectionResult":
        """Ensure is_safe is consistent with risk_score if not already enforced upstream."""
        if values.risk_score >= 0.8 and values.is_safe:
            # High-risk results should never be marked safe.
            values.is_safe = False
        return values

    @property
    def has_findings(self) -> bool:
        """Return True if any patterns or heuristics were triggered."""
        return bool(self.detected_patterns)

    @classmethod
    def safe(cls) -> "InjectionDetectionResult":
        """Convenience factory for a clean 'no issues' result."""
        return cls(is_safe=True, detected_patterns=[], risk_score=0.0)

    @classmethod
    def from_findings(
        cls,
        *,
        is_safe: bool,
        detected_patterns: list[str] | None = None,
        risk_score: float = 0.0,
    ) -> "InjectionDetectionResult":
        """Factory method to construct a result from raw findings."""
        return cls(
            is_safe=is_safe,
            detected_patterns=detected_patterns or [],
            risk_score=risk_score,
        )

    def merge(self, other: "InjectionDetectionResult") -> "InjectionDetectionResult":
        """Merge two detection results, keeping max risk and union of patterns."""
        combined_patterns = list(
            dict.fromkeys(self.detected_patterns + other.detected_patterns)
        )
        max_risk = max(self.risk_score, other.risk_score)
        return InjectionDetectionResult(
            is_safe=(max_risk < 0.8),
            detected_patterns=combined_patterns,
            risk_score=max_risk,
        )


# ---------------------------------------------------------------------------
# Stage A guardrails: validation + evidence planning
# ---------------------------------------------------------------------------


class ValidationResult(BaseModel):
    """Result of Stage A validation and guardrails checks.

    Produced by GuardrailsProcessor.validate_and_sanitize and consumed
    by the pipeline to decide whether to proceed to generation.

    Attributes:
        is_valid:
            True if the request passes all blocking checks.
        errors:
            List of blocking error messages. If non-empty, the pipeline should
            stop and return these to the caller.
        warnings:
            Non-blocking issues (e.g., missing optional fields, sanitized
            inputs, weak JD taxonomy). These can be logged or surfaced to
            the user but do not prevent generation.
    """

    is_valid: bool = Field(..., description="True if request can proceed.")
    errors: List[str] = Field(
        default_factory=list,
        description="Blocking validation errors. Non-empty implies is_valid=False.",
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Non-blocking warnings encountered during validation.",
    )

    @property
    def has_errors(self) -> bool:
        """Return True if there are any blocking errors."""
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        """Return True if there are any non-blocking warnings."""
        return bool(self.warnings)


class Evidence(BaseModel):
    """Atomic factual unit extracted from the student profile.

    Each Evidence item is a small, self-contained fact that the LLM can
    reference when generating CV sections.

    Attributes:
        evidence_id:
            Stable identifier for this fact (e.g., education.id, exp.id,
            or a synthetic ID like "<exp_id>_resp0").
        fact:
            Human-readable factual statement (e.g., "Holds BSc in Computer
            Science from MIT").
        source_type:
            Logical source of the fact, such as "education", "experience",
            "skill", "award", or "extracurricular".
    """

    evidence_id: str = Field(..., description="Stable identifier for this evidence.")
    fact: str = Field(..., description="Factual statement extracted from profile.")
    source_type: str = Field(
        ..., description="Category of source (education, experience, skill, etc.)."
    )


class EvidencePlan(BaseModel):
    """Aggregate evidence and hints for section-level generation.

    Built by GuardrailsProcessor.build_evidence_plan and passed to the
    generation stage to ground the LLM with concrete facts.

    Attributes:
        evidences:
            Flat list of all Evidence items extracted from the profile.
        section_hints:
            Mapping from CV section name (e.g., "education", "experience",
            "skills") to a list of evidence_ids that are most relevant for
            that section. This allows Stage B to prioritize which facts to
            surface when prompting the LLM.
    """

    evidences: List[Evidence] = Field(
        default_factory=list,
        description="All extracted evidence items from the profile.",
    )
    section_hints: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Map of section -> list of evidence_ids relevant to that section.",
    )

    def get_evidence_for_section(self, section: str) -> List[Evidence]:
        """Return evidences relevant to the given section name."""
        ids = set(self.section_hints.get(section, []))
        if not ids:
            return []
        return [ev for ev in self.evidences if ev.evidence_id in ids]


# ---------------------------------------------------------------------------
# Skills guardrail models (taxonomy-preserving, Stage A/B-centric)
# ---------------------------------------------------------------------------


class CanonicalSkill(BaseModel):
    """Canonical representation of a skill from the internal taxonomy.

    This is derived from profile_info + the skills taxonomy during Stage A
    and consumed primarily in Stage B's structured-skills pipeline.

    Downstream stages MUST treat `name` as immutable and only reorder or
    drop skills; renaming is not allowed at the business-logic level.

    Attributes:
        name:
            Exact taxonomy label for the skill (e.g., "Molecular Biology").
        level:
            Optional taxonomy level code such as "L3_Advanced" or "L4_Expert".
        taxonomy_id:
            Optional internal identifier for the skill in the taxonomy store.
    """

    name: str = Field(..., description="Canonical skill name from taxonomy.")
    level: str | None = Field(
        default=None,
        description="Optional skill level code, e.g. 'L3_Advanced'.",
    )
    taxonomy_id: str | None = Field(
        default=None,
        description="Optional internal taxonomy identifier for this skill.",
    )


class SkillSelectionItem(BaseModel):
    """LLM-friendly representation of a skill for the CV skills section.

    The LLM is allowed to:
      - reorder skills,
      - set `keep = false` to drop a skill,
      - optionally adjust `level` (subject to business rules and canonical
        level protection in Stage B).

    The LLM is *not* supposed to rename skills: Stage B enforces that the
    final structured skills list remains aligned with known canonical skills
    (or an allowed additional skill whitelist). Stage C then performs only
    light normalization (name-stripping, dedup by name, capping), without
    re-running taxonomy checks.
    """

    name: str = Field(
        ...,
        description="Canonical or allowed additional skill name.",
    )
    level: str | None = Field(
        default=None,
        description="Optional skill level code, e.g. 'L3_Advanced'.",
    )
    keep: bool = Field(
        default=True,
        description="If False, this skill should be omitted from the final CV.",
    )
    source: str = Field(
        default="taxonomy",
        description="Origin of this skill (taxonomy, user_input, inferred, etc.).",
    )

    @model_validator(mode="after")
    def _normalize_name(
        cls, values: "SkillSelectionItem"
    ) -> "SkillSelectionItem":
        """Light normalization to avoid spurious differences (e.g., trailing spaces)."""
        values.name = values.name.strip()
        return values


class SkillsSectionPlan(BaseModel):
    """Internal plan for enforcing taxonomy-preserving skills generation.

    Built in Stage A from profile_info + taxonomy and passed to Stage B/C.

    Current usage:
      - Stage A builds `canonical_skills` and `allowed_additional_skills`.
      - Stage B uses this to drive structured skills generation and canonical
        matching (including alias-based matching and dropping irrelevant skills).
      - Stage C assumes names are already canonicalized and focuses on light
        cleanup (normalize, dedup by name, cap list size, fallback from
        original request if needed).

    Attributes:
        canonical_skills:
            Skills coming directly from the taxonomy/profile.
        allowed_additional_skills:
            Optional whitelist of additional skill names that the LLM may add
            (still taxonomy-backed). If empty, no new skills may be introduced.
    """

    canonical_skills: List[CanonicalSkill] = Field(
        default_factory=list,
        description="Canonical skills sourced from taxonomy/profile.",
    )
    allowed_additional_skills: List[str] = Field(
        default_factory=list,
        description="Whitelist of skill names that may be added by the LLM.",
    )

    def all_allowed_names(self) -> set[str]:
        """Return the full set of skill names that are allowed in the final output."""
        base_names = {s.name for s in self.canonical_skills}
        return base_names | set(self.allowed_additional_skills)
