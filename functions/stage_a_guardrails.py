"""Stage A: Input validation, sanitization, and guardrails.

This module is responsible for:
- Scanning incoming requests for prompt injection attempts
- Sanitizing all user-controlled text fields
- Performing basic completeness checks on the profile and JD/role info
- Determining which CV sections can be reliably generated
- Building an evidence plan (facts) for downstream LLM generation

Supported input formats:

1) Legacy / internal shape (still valid):
   - request.profile_info          (required)
   - request.job_role_info         (optional)
   - request.job_position_info     (optional)
   - request.company_info          (optional)
   - request.template_info         (required)
   - request.user_input_cv_text_by_section (optional)

2) New public API shape (CVGenerationRequest from schemas.input_schema):
   - request.student_profile       (required → treated as profile_info)
   - request.target_role_taxonomy  (optional → job_role_info analogue)
   - request.target_jd_taxonomy    (optional → job_position_info analogue)
   - request.template_id           (required → part of template_info)
   - request.sections              (required → part of template_info)
   - request.language              (used to enrich template_info)
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, List, Set, Tuple, Dict

from pydantic import BaseModel
import structlog

from functions.utils.security_functions import (
    scan_dict_for_injection,
    sanitize_dict,
)
from schemas.input_schema import CVGenerationRequest
from schemas.internal_schema import EvidencePlan, Evidence, ValidationResult

logger = structlog.get_logger()


class GuardrailsProcessor:
    """Handles Stage A processing: validation, sanitization, and profile assembly."""

    def __init__(self) -> None:
        self.logger = logger.bind(stage="A_guardrails")

    # -------------------------------------------------------------------------
    # Helper: user_id resolution
    # -------------------------------------------------------------------------
    @staticmethod
    def _get_user_id(request: Any) -> str:
        """Extract user_id from either top-level request or profile_info."""
        uid = getattr(request, "user_id", None)
        if not uid:
            profile = getattr(request, "profile_info", None)
            if profile is not None:
                uid = getattr(profile, "user_id", None)
        return uid or "unknown"

    # -------------------------------------------------------------------------
    # Helpers: resolve "profile" and "template" views from either schema
    # -------------------------------------------------------------------------
    @staticmethod
    def _resolve_profile(request: Any) -> Any | None:
        """
        Return a profile-like object for validation/evidence.

        For new API:
          - prefer request.student_profile (richer Pydantic model)
        For legacy:
          - fall back to request.profile_info
        """
        # New API shape (preferred for validation/evidence)
        student_profile = getattr(request, "student_profile", None)
        if student_profile is not None:
            return student_profile

        # Legacy / normalized view
        profile = getattr(request, "profile_info", None)
        if profile is not None:
            return profile

        return None

    @staticmethod
    def _resolve_template_info(request: Any) -> Any | None:
        """
        Return a template_info-like object for validation.

        Prefers:
          - request.template_info (legacy; may already include max_chars_per_section)
        Falls back to a lightweight view built from:
          - request.template_id
          - request.sections
          - request.language

        For the new CVGenerationRequest shape, this does NOT invent
        max_chars_per_section; that should come from an external
        template service / YAML and be threaded in by the caller if needed.
        """
        tmpl = getattr(request, "template_info", None)
        if tmpl is not None:
            return tmpl

        template_id = getattr(request, "template_id", None)
        sections = getattr(request, "sections", None)
        if not template_id:
            return None

        # SimpleNamespace gives us attribute-style access compatible with _validate_template_info
        language = getattr(request, "language", None)
        lang_value = getattr(language, "value", None) if language is not None else None

        return SimpleNamespace(
            template_id=template_id,
            sections=sections,              # list[str] – enough for "at least one section"
            max_chars_per_section=None,     # not available from bare CVGenerationRequest
            language=lang_value,
        )

    def resolve_template_info(self, request: Any) -> Any | None:
        """
        Public helper for the pipeline to obtain a template_info-like object.

        This mirrors the internal _resolve_template_info logic so callers
        (e.g. main.run_cv_generation, Stage B, Stage C) don't need to know
        whether they're dealing with the legacy shape or the new
        CVGenerationRequest shape.

        In production, callers can replace or enrich this object with a
        fully-loaded template (e.g. from YAML/DB) that includes
        max_chars_per_section, color_scheme, etc., while keeping the same
        interface.
        """
        return self._resolve_template_info(request)

    # -------------------------------------------------------------------------
    # Helper: normalize shape for downstream stages (B, C, D)
    # -------------------------------------------------------------------------
    @staticmethod
    def _normalize_request_shape_for_downstream(request: Any) -> Any:
        """
        Ensure that downstream stages (B, C, D) can rely on a lightweight
        `request.profile_info` view even when using the new CVGenerationRequest
        shape that only has `student_profile`.

        - For legacy callers that already provide request.profile_info, this is a NO-OP.
        - For CVGenerationRequest, we create a FLAT dict with at least `name` and `email`
          at the top level (plus a few optional fields) so:
              * Stage B can attach it to metadata
              * Stage C can validate required metadata fields: name, email

        Stage A itself still uses student_profile (via _resolve_profile), so this
        normalization does not change existing validation/evidence behaviour.
        """
        # Legacy callers / tests already provide profile_info
        if getattr(request, "profile_info", None) is not None:
            return request

        # New API shape (CVGenerationRequest with student_profile)
        student_profile = getattr(request, "student_profile", None)
        if student_profile is None:
            return request

        personal = getattr(student_profile, "personal_info", None)
        name = getattr(personal, "name", None) if personal is not None else None
        email = getattr(personal, "email", None) if personal is not None else None
        phone = getattr(personal, "phone", None) if personal is not None else None
        linkedin = getattr(personal, "linkedin", None) if personal is not None else None

        profile_info: Dict[str, Any] = {
            "name": name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
        }

        # Optional: give some lightweight structure for prompts (non-breaking)
        try:
            edu = getattr(student_profile, "education", None) or []
            skills = getattr(student_profile, "skills", None) or []
            profile_info["education_count"] = len(edu)
            profile_info["skills_count"] = len(skills)

            if skills:
                profile_info["skills"] = []
                for s in skills:
                    skill_dict = {
                        "name": getattr(s, "name", None),
                    }
                    level = getattr(s, "level", None)
                    # Handle Enum
                    if level is not None:
                        if hasattr(level, "value"):
                            skill_dict["level"] = level.value
                        else:
                            skill_dict["level"] = level
                    profile_info["skills"].append(skill_dict)
        except Exception:
            # Best-effort only
            pass

        # Attach without fighting pydantic's extra="forbid"
        try:
            if isinstance(request, BaseModel):
                object.__setattr__(request, "profile_info", profile_info)
            else:
                setattr(request, "profile_info", profile_info)
        except Exception:
            logger.debug(
                "normalize_request_shape_profile_info_attach_failed",
                exc_info=True,
            )

        return request

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def validate_and_sanitize(self, request: CVGenerationRequest | Any) -> ValidationResult:
        """Validate the incoming request, scan for injection, and sanitize inputs."""
        uid = self._get_user_id(request)
        self.logger.info("starting_validation", user_id=uid)

        errors: List[str] = []
        warnings: List[str] = []

        # ------------------------------------------------------------------
        # Step 1: Prompt injection / security scan (on raw payload)
        # ------------------------------------------------------------------
        if hasattr(request, "model_dump"):
            raw_payload = request.model_dump(mode="python")  # type: ignore[call-arg]
        elif hasattr(request, "dict"):
            raw_payload = request.model_dump()  # type: ignore[call-arg]
        else:
            raw_payload = request

        injection_result = scan_dict_for_injection(raw_payload)

        if not injection_result.is_safe:
            self.logger.warning(
                "injection_detected",
                user_id=uid,
                patterns=injection_result.detected_patterns,
                risk_score=injection_result.risk_score,
            )
            errors.append(
                f"Security violation detected (risk: {injection_result.risk_score:.2f})"
            )
            return ValidationResult(is_valid=False, errors=errors)

        # ------------------------------------------------------------------
        # Step 2: Sanitization (now safe to normalize text)
        # ------------------------------------------------------------------
        sanitized_payload: Any = sanitize_dict(raw_payload)

        if isinstance(sanitized_payload, dict) and sanitized_payload != raw_payload:
            warnings.append(
                "Inputs were sanitized to remove control characters and unsafe whitespace."
            )
            # For Pydantic requests, avoid overwriting nested models with dicts.
            # We only need the sanitization for scanning + logging here.
            if not isinstance(request, BaseModel):
                self._apply_sanitized_payload(request, sanitized_payload)

        # ------------------------------------------------------------------
        # Step 2b: Normalize request shape for downstream stages
        #         (ensures request.profile_info exists for new API shape)
        # ------------------------------------------------------------------
        request = self._normalize_request_shape_for_downstream(request)

        # ------------------------------------------------------------------
        # Step 3: Validate profile (required: profile_info or student_profile)
        # ------------------------------------------------------------------
        profile = self._resolve_profile(request)
        if profile is None:
            # Keep legacy error string for test compatibility
            errors.append("Profile info is required (missing `profile_info`).")
        else:
            profile_errors, profile_warnings = self._validate_profile(profile)
            errors.extend(profile_errors)
            warnings.extend(profile_warnings)

        # ------------------------------------------------------------------
        # Step 3b: Validate template_info (required: template_info or template_id+sections)
        # ------------------------------------------------------------------
        template_info = self._resolve_template_info(request)
        if template_info is None:
            # Keep legacy error string for test compatibility
            errors.append("Template info is required (missing `template_info`).")
        else:
            tmpl_errors, tmpl_warnings = self._validate_template_info(template_info)
            errors.extend(tmpl_errors)
            warnings.extend(tmpl_warnings)

        # ------------------------------------------------------------------
        # Step 4: Validate JD / role information (optional)
        # ------------------------------------------------------------------
        jd_objects: List[Any] = []

        # Legacy attributes
        for attr in ("job_role_info", "job_position_info"):
            if hasattr(request, attr):
                value = getattr(request, attr)
                if value is not None:
                    jd_objects.append(value)

        # New API attributes (mapped conceptually)
        if hasattr(request, "target_role_taxonomy"):
            value = getattr(request, "target_role_taxonomy")
            if value is not None:
                jd_objects.append(value)
        if hasattr(request, "target_jd_taxonomy"):
            value = getattr(request, "target_jd_taxonomy")
            if value is not None:
                jd_objects.append(value)

        if not jd_objects:
            # Keep legacy warning string for test compatibility
            warnings.append(
                "No job role / job position information provided "
                "(expected `job_role_info` and/or `job_position_info`). "
                "Targeting quality may be reduced."
            )
        else:
            for jd in jd_objects:
                jd_errors, jd_warnings = self._validate_jd_taxonomy(jd)
                errors.extend(jd_errors)
                warnings.extend(jd_warnings)

        # company_info and user_input_cv_text_by_section are optional and
        # currently do not have hard validation rules beyond sanitization.

        # ------------------------------------------------------------------
        # Step 5: Validate requested sections vs available data
        # ------------------------------------------------------------------
        available_sections: Set[str] = set()
        if profile is not None:
            available_sections = self._get_available_sections(profile)

        requested_sections = getattr(request, "sections", []) or []
        section_warnings = self._validate_requested_sections(
            requested_sections=list(requested_sections),
            available_sections=available_sections,
        )
        warnings.extend(section_warnings)

        # ------------------------------------------------------------------
        # Step 6: Finalize result
        # ------------------------------------------------------------------
        if errors:
            self.logger.error(
                "validation_failed",
                user_id=uid,
                errors=errors,
                warnings=warnings or None,
            )
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        self.logger.info(
            "validation_passed",
            user_id=uid,
            warnings=warnings or None,
        )
        return ValidationResult(is_valid=True, warnings=warnings)

    def build_evidence_plan(self, request: CVGenerationRequest | Any) -> EvidencePlan:
        """Extract facts from the sanitized profile to build an evidence base."""
        uid = self._get_user_id(request)
        self.logger.info("building_evidence_plan", user_id=uid)

        evidences: List[Evidence] = []
        section_hints: dict[str, list[str]] = {}

        profile = self._resolve_profile(request)
        if profile is None:
            self.logger.warning("no_profile_for_evidence_plan", user_id=uid)
            return EvidencePlan(evidences=evidences, section_hints=section_hints)

        # Drafts (user_input_cv_text_by_section) – best-effort dict
        drafts_raw = getattr(request, "user_input_cv_text_by_section", None) or {}
        if hasattr(drafts_raw, "model_dump"):  # Pydantic model
            drafts = drafts_raw.model_dump(mode="python")
        elif isinstance(drafts_raw, dict):
            drafts = drafts_raw
        else:
            drafts = {}

        # ------------------------------
        # Profile summary (structured + draft)
        # ------------------------------
        summary = getattr(profile, "profile_summary", None) or getattr(profile, "summary", None)
        summary_ids: List[str] = []

        if summary:
            eid = "profile_summary_0"
            evidences.append(
                Evidence(
                    evidence_id=eid,
                    fact=f"Profile summary provided: {summary[:160]}...",
                    source_type="profile_summary",
                )
            )
            summary_ids.append(eid)

        draft_summary = drafts.get("profile_summary")
        if isinstance(draft_summary, str) and draft_summary.strip():
            eid = "profile_summary_draft_0"
            evidences.append(
                Evidence(
                    evidence_id=eid,
                    fact=f"User draft profile summary: {draft_summary[:160]}...",
                    source_type="profile_summary_draft",
                )
            )
            summary_ids.append(eid)

        if summary_ids:
            section_hints["profile_summary"] = summary_ids

        # ------------------------------
        # Education (structured)
        # ------------------------------
        education_ids: List[str] = []
        for idx, edu in enumerate(getattr(profile, "education", []) or []):
            edu_id = getattr(edu, "id", None) or f"education_{idx}"
            fact = f"Holds {getattr(edu, 'degree', '')} from {getattr(edu, 'institution', '')}".strip()
            if getattr(edu, "gpa", None) is not None:
                fact += f" (GPA: {edu.gpa})"
            evidences.append(Evidence(evidence_id=edu_id, fact=fact, source_type="education"))
            education_ids.append(edu_id)

        # Education (drafts)
        draft_education = drafts.get("education")
        if isinstance(draft_education, list):
            for idx, edu in enumerate(draft_education):
                if not isinstance(edu, dict):
                    continue
                edu_id = f"education_draft_{idx}"
                degree = edu.get("degree", "")
                inst = edu.get("institution", "")
                loc = edu.get("location", "")
                fact = f"User draft education: {degree} at {inst}".strip()
                if loc:
                    fact += f" in {loc}"
                evidences.append(
                    Evidence(
                        evidence_id=edu_id,
                        fact=fact,
                        source_type="education_draft",
                    )
                )
                education_ids.append(edu_id)

        if education_ids:
            section_hints["education"] = education_ids

        # ------------------------------
        # Experience (structured)
        # ------------------------------
        experience_ids: List[str] = []
        for idx, exp in enumerate(getattr(profile, "experience", []) or []):
            exp_id = getattr(exp, "id", None) or f"experience_{idx}"
            fact = f"Worked as {getattr(exp, 'title', getattr(exp, 'role', ''))} at {getattr(exp, 'company', '')}".strip()
            if getattr(exp, "duration", None):
                fact += f" for {exp.duration}"
            evidences.append(Evidence(evidence_id=exp_id, fact=fact, source_type="experience"))
            experience_ids.append(exp_id)

            for r_idx, resp in enumerate(
                getattr(exp, "responsibilities", getattr(exp, "highlights", [])) or []
            ):
                resp_id = f"{exp_id}_resp{r_idx}"
                evidences.append(Evidence(evidence_id=resp_id, fact=resp, source_type="experience"))

        # Experience (drafts)
        draft_experience = drafts.get("experience")
        if isinstance(draft_experience, list):
            for idx, exp in enumerate(draft_experience):
                if not isinstance(exp, dict):
                    continue
                exp_id = f"experience_draft_{idx}"
                title = exp.get("title") or exp.get("role") or ""
                company = exp.get("company", "")
                period = exp.get("period") or exp.get("years") or ""
                fact = f"User draft experience: {title} at {company}".strip()
                if period:
                    fact += f" ({period})"
                evidences.append(
                    Evidence(evidence_id=exp_id, fact=fact, source_type="experience_draft")
                )
                experience_ids.append(exp_id)

                for r_idx, hl in enumerate(exp.get("highlights") or []):
                    if not isinstance(hl, str):
                        continue
                    resp_id = f"{exp_id}_resp{r_idx}"
                    evidences.append(
                        Evidence(
                            evidence_id=resp_id,
                            fact=hl,
                            source_type="experience_draft",
                        )
                    )

        if experience_ids:
            section_hints["experience"] = experience_ids

        # ------------------------------
        # Skills (structured)
        # ------------------------------
        skill_ids: List[str] = []
        for idx, skill in enumerate(getattr(profile, "skills", []) or []):
            skill_id = getattr(skill, "id", None) or f"skill_{idx}"

            level = getattr(skill, "level", None)
            if level is not None and hasattr(level, "value"):
                level_display = level.value
            else:
                level_display = level or "unspecified"

            fact = f"Proficient in {getattr(skill, 'name', '')} at {level_display} level"
            evidences.append(Evidence(evidence_id=skill_id, fact=fact, source_type="skill"))
            skill_ids.append(skill_id)

        # Skills (drafts)
        draft_skills = drafts.get("skills")
        if isinstance(draft_skills, list):
            for idx, s in enumerate(draft_skills):
                if not isinstance(s, dict):
                    continue
                skill_id = f"skill_draft_{idx}"
                name = s.get("name", "")
                level_display = s.get("level", "unspecified")
                fact = f"User draft skill: {name} at {level_display} level"
                evidences.append(
                    Evidence(
                        evidence_id=skill_id,
                        fact=fact,
                        source_type="skill_draft",
                    )
                )
                skill_ids.append(skill_id)

        if skill_ids:
            section_hints["skills"] = skill_ids

        # ------------------------------
        # Awards (structured)
        # ------------------------------
        award_ids: List[str] = []
        for idx, award in enumerate(getattr(profile, "awards", []) or []):
            award_id = getattr(award, "id", None) or f"award_{idx}"
            fact = f"Received {getattr(award, 'title', '')}".strip()
            if getattr(award, "issuer", None):
                fact += f" from {award.issuer}"
            if getattr(award, "description", None):
                fact += f": {award.description}"
            evidences.append(Evidence(evidence_id=award_id, fact=fact, source_type="award"))
            award_ids.append(award_id)

        # Awards (drafts)
        draft_awards = drafts.get("awards")
        if isinstance(draft_awards, list):
            for idx, award in enumerate(draft_awards):
                if not isinstance(award, dict):
                    continue
                award_id = f"award_draft_{idx}"
                title = award.get("title", "")
                org = award.get("organization") or award.get("issuer") or ""
                fact = f"User draft award: {title}".strip()
                if org:
                    fact += f" from {org}"
                evidences.append(
                    Evidence(
                        evidence_id=award_id,
                        fact=fact,
                        source_type="award_draft",
                    )
                )
                award_ids.append(award_id)

        if award_ids:
            section_hints["awards"] = award_ids

        # ------------------------------
        # Extracurriculars (structured)
        # ------------------------------
        extra_ids: List[str] = []
        for idx, extra in enumerate(getattr(profile, "extracurriculars", []) or []):
            extra_id = getattr(extra, "id", None) or f"extracurricular_{idx}"
            fact = f"Participated in {getattr(extra, 'title', '')}".strip()
            if getattr(extra, "organization", None):
                fact += f" at {extra.organization}"
            if getattr(extra, "role", None):
                fact += f" as {extra.role}"
            evidences.append(Evidence(evidence_id=extra_id, fact=fact, source_type="extracurricular"))
            extra_ids.append(extra_id)

        if extra_ids:
            section_hints["extracurricular"] = extra_ids

        # ------------------------------
        # Projects (structured)
        # ------------------------------
        project_ids: List[str] = []
        for idx, proj in enumerate(getattr(profile, "projects", []) or []):
            proj_id = getattr(proj, "id", None) or f"project_{idx}"
            title = getattr(proj, "title", "") or getattr(proj, "name", "")
            fact = f"Completed project: {title}"
            if getattr(proj, "description", None):
                fact += f" – {proj.description[:100]}..."
            # try to include related_skills if present
            related_skills = getattr(proj, "related_skills", None)
            if related_skills:
                try:
                    rs = ", ".join(list(related_skills))
                    fact += f" (related skills: {rs})"
                except Exception:
                    pass
            evidences.append(Evidence(evidence_id=proj_id, fact=fact, source_type="project"))
            project_ids.append(proj_id)

        # Projects (drafts)
        draft_projects = drafts.get("projects")
        if isinstance(draft_projects, list):
            for idx, proj in enumerate(draft_projects):
                if not isinstance(proj, dict):
                    continue
                proj_id = f"project_draft_{idx}"
                title = proj.get("title") or proj.get("name") or ""
                desc = proj.get("description", "")
                fact = f"User draft project: {title}".strip()
                if desc:
                    fact += f" – {desc[:100]}..."
                rs = proj.get("related_skills") or []
                if isinstance(rs, list) and rs:
                    try:
                        fact += f" (related skills: {', '.join(rs)})"
                    except Exception:
                        pass
                evidences.append(
                    Evidence(
                        evidence_id=proj_id,
                        fact=fact,
                        source_type="project_draft",
                    )
                )
                project_ids.append(proj_id)

        if project_ids:
            section_hints["projects"] = project_ids

        # ------------------------------
        # Certifications (structured)
        # ------------------------------
        cert_ids: List[str] = []
        for idx, cert in enumerate(getattr(profile, "certifications", []) or []):
            cid = getattr(cert, "id", None) or f"cert_{idx}"
            t = getattr(cert, "title", "") or getattr(cert, "name", "")
            fact = f"Certified in {t}"
            if getattr(cert, "issuer", None):
                fact += f" by {cert.issuer}"
            evidences.append(Evidence(evidence_id=cid, fact=fact, source_type="certification"))
            cert_ids.append(cid)

        # Certifications (drafts)
        draft_certs = drafts.get("certifications")
        if isinstance(draft_certs, list):
            for idx, cert in enumerate(draft_certs):
                if not isinstance(cert, dict):
                    continue
                cid = f"cert_draft_{idx}"
                t = cert.get("title") or cert.get("name") or ""
                org = cert.get("organization") or cert.get("issuer") or ""
                fact = f"User draft certification: {t}"
                if org:
                    fact += f" by {org}"
                evidences.append(
                    Evidence(
                        evidence_id=cid,
                        fact=fact,
                        source_type="certification_draft",
                    )
                )
                cert_ids.append(cid)

        if cert_ids:
            section_hints["certifications"] = cert_ids

        # ------------------------------
        # Interests (drafts only – cheap but useful)
        # ------------------------------
        draft_interests = drafts.get("interests")
        interest_ids: List[str] = []
        if isinstance(draft_interests, list):
            for idx, intr in enumerate(draft_interests):
                if not isinstance(intr, str):
                    continue
                iid = f"interest_draft_{idx}"
                evidences.append(
                    Evidence(
                        evidence_id=iid,
                        fact=f"Interest: {intr}",
                        source_type="interest_draft",
                    )
                )
                interest_ids.append(iid)

        if interest_ids:
            section_hints["interests"] = interest_ids

        plan = EvidencePlan(evidences=evidences, section_hints=section_hints)
        self.logger.info(
            "evidence_plan_built",
            user_id=uid,
            evidence_count=len(evidences),
            sections=list(section_hints.keys()),
        )
        return plan

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _apply_sanitized_payload(request: Any, sanitized_payload: dict[str, Any]) -> None:
        """Apply sanitized dict back onto the request model (best-effort, in-place)."""
        for field_name, value in sanitized_payload.items():
            try:
                setattr(request, field_name, value)
            except Exception:
                continue

    # ---- Validation helpers ------------------------------------------------
    @staticmethod
    def _validate_profile(profile: Any) -> Tuple[List[str], List[str]]:
        """Validate baseline completeness of the profile block."""
        errors: List[str] = []
        warnings: List[str] = []

        personal = getattr(profile, "personal_info", None) or profile
        name = getattr(personal, "name", "").strip()
        email = getattr(personal, "email", "")

        if not name:
            errors.append("Personal info: name is required.")
        if not email:
            errors.append("Personal info: email is required.")

        education = getattr(profile, "education", []) or []
        if not education:
            errors.append("At least one education entry is required.")

        skills = getattr(profile, "skills", []) or []
        if len(skills) < 3:
            errors.append("At least 3 skills are required.")

        return errors, warnings

    @staticmethod
    def _validate_template_info(template_info: Any) -> Tuple[List[str], List[str]]:
        """Validate the template configuration (supports multiple naming variants)."""
        errors: List[str] = []
        warnings: List[str] = []

        template_id = getattr(template_info, "template_id", None)
        sections_map = (
            getattr(template_info, "max_chars_per_section", None)
            or getattr(template_info, "sections", None)
            or getattr(template_info, "max_lengths", None)
            or getattr(template_info, "section_max_lengths", None)
        )

        if not template_id:
            errors.append("template_info: `template_id` is required.")
        if not sections_map:
            errors.append("template_info: at least one section must be specified.")
        return errors, warnings

    @staticmethod
    def _validate_jd_taxonomy(jd: Any) -> Tuple[List[str], List[str]]:
        """Validate basic completeness of job role / position info."""
        errors: List[str] = []
        warnings: List[str] = []

        required_skills = (
            getattr(jd, "required_skills", None)
            or getattr(jd, "skills_required", None)
            or getattr(jd, "skills", None)
            or getattr(jd, "job_required_skills", None)
        )
        role_title = (
            getattr(jd, "role_title", None)
            or getattr(jd, "title", None)
            or getattr(jd, "job_title", None)
        )

        if not required_skills:
            warnings.append("No required skills specified in job role / job position information.")
        if not role_title:
            warnings.append("No role title specified – may affect targeting quality.")
        return errors, warnings

    @staticmethod
    def _validate_requested_sections(requested_sections: List[str], available_sections: Set[str]) -> List[str]:
        """Check that requested sections are supported by profile data."""
        warnings: List[str] = []
        always_allowed = {"profile_summary", "interests"}

        for section in requested_sections:
            if section in always_allowed:
                continue
            if section not in available_sections:
                warnings.append(
                    f"Section '{section}' requested but no data available – "
                    f"generated content may be omitted or generic."
                )
        return warnings

    @staticmethod
    def _get_available_sections(profile: Any) -> Set[str]:
        """Determine which sections have data available on the profile."""
        available: Set[str] = {"profile_summary", "education", "skills"}
        if getattr(profile, "experience", []):
            available.add("experience")
        if getattr(profile, "awards", []):
            available.add("awards")
        if getattr(profile, "extracurriculars", []):
            available.add("extracurricular")
        return available
