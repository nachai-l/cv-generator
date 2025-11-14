from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict
from types import SimpleNamespace

from functions.utils.baseline_renderer import build_baseline_response
from tests.test_utils import (  # type: ignore  # noqa: E402
    load_legacy_yaml_payload,
    compute_sections_from_legacy_template,
)

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_cv_templates.cv_templates import save_cv_html  # type: ignore  # noqa: E402
from schemas.input_schema import (  # type: ignore  # noqa: E402
    CVGenerationRequest,
    StudentProfile,
    PersonalInfo,
    Education,
    Experience,
    Skill,
    Award,
    Extracurricular,
    SkillLevel,
    Language,
    CompanyInfo as TargetCompanyInfo,
    RoleTaxonomy,
    JobTaxonomy,
)
from main import run_cv_generation  # type: ignore  # noqa: E402
from functions.utils.common import (  # type: ignore  # noqa: E402
    model_dump_compat,
    ensure_llm_metrics_env,
)

# ---------------------------------------------------------------------------
# YAML loading + mock API payload assembly
# ---------------------------------------------------------------------------


def build_mock_api_payload() -> Dict[str, Any]:
    """
    Load the legacy YAML fixtures from tests/json_test_inputs/ and
    combine them into a single dict.

    Wrapper around tests.test_utils.load_legacy_yaml_payload(), kept here
    so existing callers don't need to change.
    """
    return load_legacy_yaml_payload()


# ---------------------------------------------------------------------------
# Build CVGenerationRequest from mock payload (for Stage A–D pipeline)
# ---------------------------------------------------------------------------


def _build_cv_request_from_mock_payload(mock_payload: Dict[str, Any]) -> CVGenerationRequest:
    """
    Map the YAML-based mock payload into a minimal-but-valid CVGenerationRequest.

    Notes:
    - This is a *best-effort* mapping for testing the Stage A–D pipeline.
    - Some fields (dates, IDs, levels) are synthesized to satisfy validators.
    """
    profile_info = (
        mock_payload.get("profile_info")
        or mock_payload.get("student_profile")
        or mock_payload.get("profile")
        or {}
    )
    if "personal_info" in profile_info:
        # un-nest legacy shape
        profile_info = profile_info["personal_info"] | {
            "education": profile_info.get("education", []),
            "skills": profile_info.get("skills", []),
        }

    template_info: Dict[str, Any] = mock_payload.get("template_info", {}) or {}
    job_role_info: Dict[str, Any] = mock_payload.get("job_role_info", {}) or {}
    job_position_info: Dict[str, Any] = mock_payload.get("job_position_info", {}) or {}
    company_info_raw: Dict[str, Any] = mock_payload.get("company_info", {}) or {}
    user_sections: Dict[str, Any] = mock_payload.get("user_input_cv_text_by_section", {}) or {}

    # ------------------ Personal Info ------------------
    name = profile_info.get("name") or "Mock User"
    email = profile_info.get("email") or "mock@example.com"  # must be valid
    phone = profile_info.get("phone")
    linkedin = profile_info.get("linkedin")

    personal = PersonalInfo(
        name=name,
        email=email,
        phone=phone,
        linkedin=linkedin,
    )

    # ------------------ Education (best-effort) ------------------
    edu_entries: list[Education] = []
    raw_edu = profile_info.get("education") or []
    if isinstance(raw_edu, list):
        for idx, e in enumerate(raw_edu):
            if not isinstance(e, dict):
                continue
            degree = (e.get("degree") or "Degree").strip() or "Degree"
            institution = (
                e.get("institution")
                or e.get("school")
                or "Institution"
            ).strip() or "Institution"

            edu_entries.append(
                Education(
                    id=f"edu#mock_{idx}",
                    degree=degree[:200],
                    institution=institution[:200],
                    gpa=None,
                    start_date=date(2020, 1, 1),
                    graduation_date=None,
                    major=e.get("major"),
                )
            )

    if not edu_entries:
        # Fallback minimal education to satisfy validators
        edu_entries.append(
            Education(
                id="edu#mock_default",
                degree="B.Sc. (Mock)",
                institution="Mock University",
                gpa=3.5,
                start_date=date(2020, 1, 1),
                graduation_date=date(2024, 1, 1),
                major="General",
            )
        )

    # ------------------ Skills (best-effort) ------------------
    skill_entries: list[Skill] = []
    raw_skills = profile_info.get("skills") or []
    if isinstance(raw_skills, list):
        for idx, s in enumerate(raw_skills):
            if isinstance(s, dict):
                s_name = s.get("name")
                s_level_str = s.get("level")  # Get the level from YAML
            else:
                s_name = str(s)
                s_level_str = None

            if not s_name:
                continue

            # Parse level from string to enum
            skill_level = SkillLevel.L2_INTERMEDIATE  # default
            if s_level_str:
                try:
                    # Handle both "L4_Expert" and "L4_EXPERT" formats
                    level_upper = str(s_level_str).upper()
                    if not level_upper.startswith("L"):
                        level_upper = "L2_INTERMEDIATE"
                    skill_level = SkillLevel[level_upper]
                except (KeyError, ValueError):
                    skill_level = SkillLevel.L2_INTERMEDIATE

            skill_entries.append(
                Skill(
                    id=f"skill#mock_{idx}",
                    name=s_name[:100],
                    description=(s_name or "Skill")[:200],
                    level=skill_level,  # ← Use the parsed level!
                )
            )

    # Ensure at least 3 skills for validators
    defaults = ["Communication", "Teamwork", "Problem Solving"]
    i = 0
    while len(skill_entries) < 3 and i < len(defaults):
        s_name = defaults[i]
        skill_entries.append(
            Skill(
                id=f"skill#auto_{i}",
                name=s_name,
                description=s_name,
                level=SkillLevel.L2_INTERMEDIATE,
            )
        )
        i += 1

    # ------------------ Experience / Awards / Extras (optional) ------------------
    exp_entries: list[Experience] = []
    award_entries: list[Award] = []
    extra_entries: list[Extracurricular] = []

    student_profile = StudentProfile(
        personal_info=personal,
        education=edu_entries,
        experience=exp_entries,
        skills=skill_entries,
        awards=award_entries,
        extracurriculars=extra_entries,
    )

    # ------------------ Role Taxonomy ------------------
    role_title = (
        job_role_info.get("role_title")
        or job_role_info.get("title")
        or "Target Role"
    )
    role_description = (
        job_role_info.get("role_description")
        or job_role_info.get("description")
        or "Generic target role description."
    )
    raw_role_skills = (
        job_role_info.get("role_required_skills")
        or job_role_info.get("required_skills")
        or job_role_info.get("skills_required")
        or []
    )
    role_required_skills: list[str] = []
    if isinstance(raw_role_skills, list):
        role_required_skills = [str(s) for s in raw_role_skills if s]

    if not role_required_skills:
        role_required_skills = ["GenericSkill_L2"]

    role_taxonomy = RoleTaxonomy(
        role_title=role_title[:200],
        role_description=role_description[:500],
        role_required_skills=role_required_skills[:20],
    )

    # ------------------ Job Taxonomy ------------------
    job_title = (
        job_position_info.get("job_title")
        or job_position_info.get("title")
        or role_title
        or "Target Job"
    )
    raw_job_req_skills = (
        job_position_info.get("job_required_skills")
        or job_position_info.get("required_skills")
        or role_required_skills
        or []
    )
    job_required_skills: list[str] = []
    if isinstance(raw_job_req_skills, list):
        job_required_skills = [str(s) for s in raw_job_req_skills if s]

    if not job_required_skills:
        job_required_skills = ["GenericSkill_L2"]

    raw_job_responsibilities = (
        job_position_info.get("job_responsibilities")
        or job_position_info.get("responsibilities")
        or []
    )
    job_responsibilities: list[str] = []
    if isinstance(raw_job_responsibilities, list):
        for r in raw_job_responsibilities:
            if not r:
                continue
            job_responsibilities.append(str(r)[:500])

    if len(job_responsibilities) > 10:
        job_responsibilities = job_responsibilities[:10]

    company_info_obj: TargetCompanyInfo | None = None
    if company_info_raw:
        cname = company_info_raw.get("name")
        if cname:
            company_info_obj = TargetCompanyInfo(
                name=cname[:200],
                industry=(company_info_raw.get("industry") or None),
            )

    job_taxonomy = JobTaxonomy(
        job_title=job_title[:200],
        job_required_skills=job_required_skills[:20],
        job_responsibilities=job_responsibilities,
        company_info=company_info_obj,
    )

    # ------------------ Sections & language ------------------
    template_sections = compute_sections_from_legacy_template(template_info)
    allowed_sections = {
        "profile_summary",
        "skills",
        "experience",
        "education",
        "projects",
        "certifications",
        "awards",
        "extracurricular",
        "volunteering",
        "interests",
    }
    sections = [s for s in template_sections if s in allowed_sections]
    if not sections:
        sections = [
            "profile_summary",
            "skills",
            "experience",
            "education",
            "awards",
            "extracurricular",
        ]

    lang_raw = str(template_info.get("language", "en")).lower()
    language = Language.TH if lang_raw == "th" else Language.EN

    template_id = template_info.get("template_id", "T_EMPLOYER_STD_V3")

    # ------------------ Build request (only allowed schema fields) ------------------
    cv_request = CVGenerationRequest(
        user_id="MOCK_USER_001",
        language=language,
        template_id=template_id,
        sections=sections,  # type: ignore[arg-type]
        student_profile=student_profile,
        target_role_taxonomy=role_taxonomy,
        target_jd_taxonomy=job_taxonomy,
    )

    # ------------------ Attach extra views for internal stages ------------------
    # Pass through user drafts so Stage B can see them
    if user_sections:
        object.__setattr__(cv_request, "user_input_cv_text_by_section", user_sections)

    # Attach a template_info-like view so Stage A/B/C can use it
    max_chars = template_info.get("max_chars_per_section") or {}
    lang_value = language.value if isinstance(language, Language) else str(language)

    object.__setattr__(  # <-- use object.__setattr__ here too
        cv_request,
        "template_info",
        SimpleNamespace(
            template_id=template_id,
            sections=sections,
            max_chars_per_section=max_chars,
            language=lang_value,
        ),
    )

    return cv_request


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> None:
    # Pretty header similar to unittest output style
    print("\n" + "=" * 90, file=sys.stderr)
    print("🧪 STARTING MOCK API CV GENERATION (v2: Stage A–D) FROM YAML FIXTURES", file=sys.stderr)
    print("=" * 90, file=sys.stderr)

    # Ensure llm_metrics reads the correct config & path BEFORE importing it
    ensure_llm_metrics_env()

    from functions.utils import llm_metrics  # noqa: E402  (import after env setup)

    # Clear caches so our env vars are respected
    try:
        llm_metrics._load_config.cache_clear()
        llm_metrics._load_pricing.cache_clear()
        llm_metrics._get_log_path.cache_clear()
        llm_metrics._get_csv_path.cache_clear()
    except Exception:
        pass

    # 1) Build full mock API payload from YAML files
    print("[1/6] Loading YAML fixtures and building mock payload...", file=sys.stderr)
    mock_payload = build_mock_api_payload()

    out_dir = THIS_FILE.parent / "generated_test_cvs"
    out_dir.mkdir(exist_ok=True)

    # Show where logs will go (before generation)
    csv_path = llm_metrics._get_csv_path()
    print(f"[info] LLM cost/usage CSV will be written to → {Path(csv_path).resolve()}", file=sys.stderr)

    # 2) Clean up old files
    print(f"[2/6] Cleaning previous files in: {out_dir}", file=sys.stderr)
    for old_file in out_dir.iterdir():
        try:
            if old_file.is_file():
                old_file.unlink()
        except Exception as e:
            print(f"⚠️  Skipped {old_file.name}: {e}", file=sys.stderr)

    # 3) Save full payload for debugging
    full_payload_path = out_dir / "mock_api_payload.json"
    full_payload_path.write_text(
        json.dumps(mock_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[3/6] Saved full mock payload → {full_payload_path}", file=sys.stderr)

    # 4) Build NO-LLM response (baseline, no Stage B)
    print("[4/6] Building CVGenerationResponse WITHOUT LLM (baseline)...", file=sys.stderr)

    response_wo_llm = build_baseline_response(mock_payload)

    # Save JSON (no-LLM) – exclude metadata
    data_wo_llm = model_dump_compat(response_wo_llm)
    data_wo_llm.pop("metadata", None)

    output_wo_llm_json = out_dir / "output_wo_llm.json"
    output_wo_llm_json.write_text(
        json.dumps(data_wo_llm, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save HTML (no-LLM)
    output_cv_wo_llm_html = out_dir / "output_cv_wo_llm.html"
    save_cv_html(response_wo_llm, output_cv_wo_llm_html.as_posix())

    print(
        f"[4/6] Saved NO-LLM CV JSON → {output_wo_llm_json}\n"
        f"      Saved NO-LLM CV HTML → {output_cv_wo_llm_html}",
        file=sys.stderr,
    )

    # 5) Build CVGenerationRequest and call full Stage A–D pipeline
    print("[5/6] Building CVGenerationRequest and calling run_cv_generation()...", file=sys.stderr)

    try:
        request = _build_cv_request_from_mock_payload(mock_payload)
    except Exception as e:
        print(f"[ERROR] Failed to build CVGenerationRequest from mock payload: {e}", file=sys.stderr)
        return

    try:
        response_w_llm, request_id = run_cv_generation(request)
    except Exception as e:
        print(f"[ERROR] Stage A–D pipeline failed: {e}", file=sys.stderr)
        return

    # Save JSON (with-LLM, full pipeline)
    data_w_llm = model_dump_compat(response_w_llm)
    output_w_llm_json = out_dir / "output_w_llm.json"
    output_w_llm_json.write_text(
        json.dumps(data_w_llm, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Save HTML (with-LLM) across styles
    for style_name in ("nostyle", "modern", "minimalist", "double"):
        output_cv_w_llm_html = out_dir / f"output_cv_w_llm_{style_name}.html"
        save_cv_html(
            response_w_llm,
            output_cv_w_llm_html.as_posix(),
            style=None if style_name == "nostyle" else style_name,
        )

    print(
        f"[5/6] Saved WITH-LLM CV JSON → {output_w_llm_json}\n"
        f"      Saved WITH-LLM CV HTML → {output_cv_w_llm_html}\n"
        f"      request_id={request_id}",
        file=sys.stderr,
    )

    print("[6/6] ✅ Mock API CV generation (baseline + Stage A–D) completed.", file=sys.stderr)
    print("-" * 90 + "\n", file=sys.stderr)

    # Summary of CSV rows
    try:
        csv_path = Path(llm_metrics._get_csv_path()).resolve()
        if csv_path.exists():
            with csv_path.open("r", encoding="utf-8") as f:
                rows = sum(1 for _ in f) - 1  # exclude header
            size_kb = csv_path.stat().st_size / 1024.0
            print(f"[summary] LLM log: {csv_path}  ({rows} rows, {size_kb:.1f} KB)", file=sys.stderr)
        else:
            print(f"[summary] LLM log not found at {csv_path}", file=sys.stderr)
    except Exception as e:
        print(f"[summary] Could not read LLM log: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
