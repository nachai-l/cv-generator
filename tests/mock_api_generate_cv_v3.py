from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import requests

# -------------------------------------------------------------
# PATHS
# -------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]

TESTCASE_DIR = THIS_FILE.parent / "api_payload_tests"
OUT_DIR = THIS_FILE.parent / "generated_test_cvs"

API_URL = "http://127.0.0.1:8000/generate_cv"


# -------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------

def clear_output_dir() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for f in OUT_DIR.iterdir():
        if f.is_file():
            f.unlink()


def load_testcases() -> Dict[str, Dict[str, Any]]:
    testcases: Dict[str, Dict[str, Any]] = {}
    for file in TESTCASE_DIR.glob("*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        testcases[file.stem] = data
    return testcases


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _text_from_student_profile_section(
    section_id: str,
    student_profile: Any,
) -> str:
    """
    Very simple formatter that turns student_profile.* into plain text.
    NO LLM involved.
    """
    if student_profile is None:
        return ""

    # Defensive: both dict and Pydantic model supported
    def _get(obj, field, default=None):
        if isinstance(obj, dict):
            return obj.get(field, default)
        return getattr(obj, field, default)

    # ---- profile_summary ----
    if section_id == "profile_summary":
        name = _get(_get(student_profile, "personal_info", {}), "name", "")
        headline_parts = []
        if name:
            headline_parts.append(name)
        headline = " ".join(headline_parts).strip()
        if not headline:
            headline = "Professional summary"
        # keep it deterministic & non-LLM
        return f"{headline} — auto-generated summary from student profile data."

    # ---- skills ----
    if section_id == "skills":
        skills = _get(student_profile, "skills", []) or []
        lines: list[str] = []
        for sk in skills:
            sk_name = _get(sk, "name", "")
            level = _get(sk, "level", None)
            if hasattr(level, "value"):
                level = level.value
            if sk_name and level:
                lines.append(f"- {sk_name} ({level})")
            elif sk_name:
                lines.append(f"- {sk_name}")
        return "\n".join(lines)

    # ---- education ----
    if section_id == "education":
        edus = _get(student_profile, "education", []) or []
        lines: list[str] = []
        for edu in edus:
            degree = _get(edu, "degree", "")
            inst = _get(edu, "institution", "")
            field = _get(edu, "field_of_study", "")
            parts = [p for p in [degree, field, inst] if p]
            if parts:
                lines.append("- " + ", ".join(parts))
        return "\n".join(lines)

    # ---- experience ----
    if section_id == "experience":
        exps = _get(student_profile, "experience", []) or []
        blocks: list[str] = []
        for exp in exps:
            title = _get(exp, "title", "")
            company = _get(exp, "company", "")
            start = _get(exp, "start_date", "")
            end = _get(exp, "end_date", "")
            resp_list = _get(exp, "responsibilities", []) or []

            header_parts = []
            if title:
                header_parts.append(f"**{title}**")
            company_period_parts = []
            if company:
                company_period_parts.append(company)
            if start or end:
                company_period_parts.append(f"{start}–{end}".strip("-"))
            if company_period_parts:
                header_parts.append(" *" + ", ".join(company_period_parts) + "*")

            lines: list[str] = []
            if header_parts:
                lines.append(" ".join(header_parts))
            for r in resp_list:
                lines.append(f"- {r}")

            if lines:
                blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    return ""


def build_wo_llm_response(payload: Dict[str, Any]) -> "CVGenerationResponse":
    from schemas.output_schema import (
        CVGenerationResponse,
        SectionContent,
        OutputSkillItem,
        Metadata,
        Justification,
        GenerationStatus,
    )

    student_profile = payload.get("student_profile")
    template_info = payload.get("template_info") or {}

    # Prefer top-level, fall back to template_info
    template_id = payload.get("template_id") or template_info.get("template_id", "UNKNOWN_TEMPLATE")
    language = payload.get("language", "en")
    user_id = payload.get("user_id", "UNKNOWN_USER")

    # Prefer explicit sections; otherwise use template_info.sections_order
    sections_requested = payload.get("sections")
    if not sections_requested:
        sections_requested = template_info.get("sections_order") or []
    # Defensive: ensure list of strings
    if not isinstance(sections_requested, list):
        sections_requested = []
    else:
        sections_requested = [str(s) for s in sections_requested]

    # ---- Build sections from student_profile ----
    sections_dict: Dict[str, SectionContent] = {}

    for section_id in sections_requested:
        text = _text_from_student_profile_section(section_id, student_profile)
        if not text:
            # Make sure it passes min_length=10 if validated later
            text = f"{section_id.capitalize()} information not provided."
        word_count = len(text.split())

        sec = SectionContent.model_construct(
            text=text,
            word_count=word_count,
            matched_jd_skills=[],
            confidence_score=1.0,
        )
        sections_dict[section_id] = sec

    # ---- Build skills list from student_profile.skills ----
    skills_output: list[OutputSkillItem] = []
    if student_profile is not None:
        def _get(obj, field, default=None):
            if isinstance(obj, dict):
                return obj.get(field, default)
            return getattr(obj, field, default)

        sp_skills = _get(student_profile, "skills", []) or []
        for sk in sp_skills:
            name = _get(sk, "name", None)
            if not name:
                continue
            level = _get(sk, "level", None)
            if hasattr(level, "value"):
                level = level.value
            skills_output.append(
                OutputSkillItem.model_construct(
                    name=name,
                    level=level,
                    source="profile",
                )
            )

    metadata = Metadata.model_construct(
        generation_time_ms=0,
        model_version="wo_llm",
        retry_count=0,
        cache_hit=False,
        sections_requested=len(sections_requested),
        sections_generated=len(sections_dict),
        tokens_used=0,
        cost_estimate_thb=0.0,
        profile_info=student_profile,
        request_id=f"REQ_wo_llm_{user_id}",
        stage_c_validated=True,
        stage_d_completed=True,
    )

    justification = Justification.model_construct()

    cv = CVGenerationResponse.model_construct(
        job_id=f"JOB_{user_id}",
        template_id=template_id,
        language=language,
        status=GenerationStatus.COMPLETED,
        sections=sections_dict,
        skills=skills_output or None,
        metadata=metadata,
        justification=justification,
        quality_metrics=None,
        warnings=[],
        error=None,
        error_details=None,
    )
    return cv


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main() -> None:
    print("\n" + "=" * 80)
    print("🧪  RUNNING REAL API CV GENERATION (v3)")
    print("=" * 80)

    # 1. Clear output directory
    print("[1/4] Clearing tests/generated_test_cvs/ ...")
    clear_output_dir()

    # 2. Load JSON testcases
    print("[2/4] Loading testcases from tests/api_payload_tests/ ...")
    testcases = load_testcases()
    print(f"     → Found {len(testcases)} testcases")

    from local_cv_templates.cv_templates import save_cv_html
    from schemas.output_schema import CVGenerationResponse  # for type hints / validation

    # 3. Run pipeline
    print("[3/4] Running pipeline on each testcase ...")

    for name, payload in testcases.items():
        user_id = payload.get("user_id", name)
        print(f"     → Processing testcase: {name} (user_id={user_id})")

        # -------------------------------------------------
        # 3A. Build WO-LLM response from student_profile
        # -------------------------------------------------
        try:
            cv_wo = build_wo_llm_response(payload)
        except Exception as exc:
            print(f"       ❌ Failed to build wo_llm response: {exc}")
            cv_wo = None

        if cv_wo is not None:
            # JSON
            wo_llm_json_path = OUT_DIR / f"{user_id}_wo_llm.json"
            save_json(wo_llm_json_path, cv_wo.model_dump(mode="json"))

            # HTML (modern)
            wo_llm_html_path = OUT_DIR / f"{user_id}_wo_llm.html"
            save_cv_html(cv_wo, wo_llm_html_path.as_posix(), style="modern")

            print(f"       ✓ Saved WO-LLM JSON → {wo_llm_json_path.name}")
            print(f"       ✓ Saved WO-LLM HTML → {wo_llm_html_path.name}")

        # -------------------------------------------------
        # 3B. Call REAL API to get WITH-LLM response
        # -------------------------------------------------
        try:
            resp = requests.post(API_URL, json=payload, timeout=180)
        except Exception as exc:
            print(f"       ❌ API call failed: {exc}")
            continue

        try:
            raw_json = resp.json()
        except Exception:
            raw_json = {"raw_text": resp.text, "status_code": resp.status_code}

        w_llm_json_path = OUT_DIR / f"{user_id}_w_llm.json"
        save_json(w_llm_json_path, raw_json)

        if resp.status_code != 200:
            print(f"       ❌ API returned {resp.status_code}, skipping W-LLM HTML")
            continue

        try:
            cv_w = CVGenerationResponse.model_validate(raw_json)
        except Exception as exc:
            print(f"       ❌ Response schema validation failed (w_llm): {exc}")
            continue

        w_llm_html_path = OUT_DIR / f"{user_id}_w_llm.html"
        save_cv_html(cv_w, w_llm_html_path.as_posix(), style="modern")

        print(f"       ✓ Saved W-LLM JSON → {w_llm_json_path.name}")
        print(f"       ✓ Saved W-LLM HTML → {w_llm_html_path.name}")

    print("[4/4] DONE. All testcases processed.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
