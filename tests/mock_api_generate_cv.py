# tests_utils/mock_api_generate_cv.py
from __future__ import annotations

import json
import os
import sys
import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import yaml

# ---------------------------------------------------------------------------
# Make project root importable
# ---------------------------------------------------------------------------

THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parents[1]  # project root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from local_cv_templates.cv_templates import save_cv_html  # type: ignore  # noqa: E402
from schemas.output_schema import (  # type: ignore  # noqa: E402
    CVGenerationResponse,
    SectionContent,
    OutputSkillItem,
)
from functions.stage_b_generation import CVGenerationEngine  # type: ignore  # noqa: E402
from functions.stage_c_validation import run_stage_c_validation

# ---------------------------------------------------------------------------
# Helpers to construct schema objects without caring about Pydantic version
# ---------------------------------------------------------------------------

def _format_section_value(section_key: str, value: Any) -> str:
    """
    Convert structured section data (dicts/lists) into a Markdown-like string.
    """
    if isinstance(value, str):
        return value

    if section_key == "skills" and isinstance(value, list):
        lines: list[str] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            level = item.get("level")
            if name and level:
                lines.append(f"- {name} ({level})")
            elif name:
                lines.append(f"- {name}")
        return "\n".join(lines)

    if section_key == "experience" and isinstance(value, list):
        blocks: list[str] = []
        for job in value:
            if not isinstance(job, dict):
                continue
            title = job.get("title")
            company = job.get("company")
            period = job.get("period")
            highlights = job.get("highlights") or []
            header_parts = []
            if title:
                header_parts.append(f"**{title}**")
            if company or period:
                company_period = ", ".join(p for p in [company, period] if p)
                header_parts.append(f"*{company_period}*")
            block_lines: list[str] = []
            if header_parts:
                block_lines.extend(header_parts)
            for h in highlights:
                block_lines.append(f"- {h}")
            if block_lines:
                blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)

    if section_key == "education" and isinstance(value, list):
        lines: list[str] = []
        for edu in value:
            if not isinstance(edu, dict):
                continue
            degree = edu.get("degree")
            institution = edu.get("institution")
            location = edu.get("location")
            parts = [p for p in [degree, institution, location] if p]
            if parts:
                lines.append("- " + ", ".join(parts))
        return "\n".join(lines)

    if section_key == "certifications" and isinstance(value, list):
        lines: list[str] = []
        for cert in value:
            if isinstance(cert, dict):
                name = cert.get("name")
                org = cert.get("organization")
                if name and org:
                    lines.append(f"- {name}, {org}")
                elif name:
                    lines.append(f"- {name}")
            else:
                lines.append(f"- {cert}")
        return "\n".join(lines)

    if section_key == "awards" and isinstance(value, list):
        lines: list[str] = []
        for award in value:
            if isinstance(award, dict):
                title = award.get("title")
                org = award.get("organization")
                if title and org:
                    lines.append(f"- {title}, {org}")
                elif title:
                    lines.append(f"- {title}")
            else:
                lines.append(f"- {award}")
        return "\n".join(lines)

    if section_key == "interests" and isinstance(value, list):
        return "\n".join(f"- {item}" for item in value)

    return str(value)


def _build_section_content(section_key: str, value: Any) -> SectionContent:
    """Construct SectionContent from possibly structured section data."""
    text = _format_section_value(section_key, value)

    if hasattr(SectionContent, "model_construct"):
        return SectionContent.model_construct(text=text)

    raise TypeError(
        "SectionContent does not support model_construct; "
        "update _build_section_content to match its implementation."
    )


def _build_response_from_mock_payload(mock_payload: Dict[str, Any]) -> CVGenerationResponse:
    """
    Build a CVGenerationResponse from the (minimal) mock API payload.
    """
    template_info: Dict[str, Any] = mock_payload.get("template_info", {}) or {}
    user_sections: Dict[str, Any] = mock_payload.get("user_input_cv_text_by_section", {}) or {}
    profile_info: Dict[str, Any] = mock_payload.get("profile_info", {}) or {}

    template_id = template_info.get("template_id", "T_EMPLOYER_STD_V3")

    def _non_empty(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, (list, dict, tuple, set)):
            return len(value) > 0
        return True

    profile_key_map = {
        "profile_summary": "summary",
        "skills": "skills",
        "experience": "experience",
        "education": "education",
        "projects": "projects",
        "certifications": "certifications",
        "awards": "awards",
        "extracurricular": "extracurricular",
        "volunteering": "volunteering",
        "interests": "interests",
    }

    available: set[str] = set()

    for section_id, key in profile_key_map.items():
        if key in profile_info and _non_empty(profile_info.get(key)):
            available.add(section_id)

    for section_id, draft in user_sections.items():
        if _non_empty(draft):
            available.add(section_id)

    ordered_section_ids = _compute_sections_from_template_and_user(template_info, user_sections)

    sections: Dict[str, SectionContent] = {}
    for section_id in ordered_section_ids:
        if section_id not in available:
            continue

        if section_id in user_sections and _non_empty(user_sections[section_id]):
            value = user_sections[section_id]
        else:
            key = profile_key_map.get(section_id)
            if not key:
                continue
            value = profile_info.get(key)
            if not _non_empty(value):
                continue

        sections[section_id] = _build_section_content(section_id, value)

    # Metadata stub
    meta_fields = {
        "generated_at": datetime.now(),
        "name": profile_info.get("name"),
        "email": profile_info.get("email"),
        "phone": profile_info.get("phone"),
        "linkedin": profile_info.get("linkedin"),
        "profile_info": profile_info,
        "template_info": template_info,
    }

    meta_cls = type("Meta", (), {})
    metadata_stub = meta_cls()
    for k, v in meta_fields.items():
        setattr(metadata_stub, k, v)

    # Structured skills from profile_info.skills
    raw_skills = profile_info.get("skills") or []
    skills_structured: list[OutputSkillItem] = []
    if isinstance(raw_skills, list):
        for item in raw_skills:
            if isinstance(item, dict):
                name = item.get("name")
                if not name:
                    continue
                level = item.get("level")
                skills_structured.append(
                    OutputSkillItem(name=name, level=level, source="profile")
                )
            else:
                skills_structured.append(
                    OutputSkillItem(name=str(item), level=None, source="profile")
                )

    base_kwargs = {
        "template_id": template_id,
        "sections": sections,
        "job_id": "JOB_MOCK_001",
        "metadata": metadata_stub,
        "skills": skills_structured or None,
    }

    if hasattr(CVGenerationResponse, "model_construct"):
        return CVGenerationResponse.model_construct(**base_kwargs)
    if hasattr(CVGenerationResponse, "construct"):
        return CVGenerationResponse.model_construct(**base_kwargs)  # type: ignore[call-arg]

    raise TypeError(
        "CVGenerationResponse does not support model_construct/construct; "
        "update _build_response_from_mock_payload to match its implementation."
    )


# ---------------------------------------------------------------------------
# YAML loading + mock API payload assembly
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_mock_api_payload() -> Dict[str, Any]:
    """Load each YAML file under tests_utils/json_test_inputs and combine them."""
    json_input_dir = THIS_FILE.parent / "json_test_inputs"

    company_info = _load_yaml(json_input_dir / "company_info.yaml")
    job_position_info = _load_yaml(json_input_dir / "job_position_info.yaml")
    job_role_info = _load_yaml(json_input_dir / "job_role_info.yaml")
    profile_info = _load_yaml(json_input_dir / "profile_info.yaml")
    template_info = _load_yaml(json_input_dir / "template_info.yaml")
    user_sections = _load_yaml(json_input_dir / "user_input_cv_text_by_section.yaml")

    return {
        "company_info": company_info,
        "job_position_info": job_position_info,
        "job_role_info": job_role_info,
        "profile_info": profile_info,
        "template_info": template_info,
        "user_input_cv_text_by_section": user_sections,
    }


# ---------------------------------------------------------------------------
# Section list helper (template-aware, no extra user-only sections)
# ---------------------------------------------------------------------------

def _compute_sections_from_template_and_user(
    template_info: Dict[str, Any],
    user_sections: Dict[str, Any],
) -> List[str]:
    """
    Determine the ordered list of sections for generation.

    RULE:
    - Use ONLY the order defined in template_info["sections_order"]
      (or fallback to template_info["sections"]).
    - Do NOT append extra keys from user_sections; user input is treated
      as content for those template-defined sections, not as new sections.
    """
    result: List[str] = []
    seen: set[str] = set()

    raw_order = (
        template_info.get("sections_order")
        or template_info.get("sections")
        or []
    )

    if isinstance(raw_order, list):
        for item in raw_order:
            section_id: str | None
            if isinstance(item, str):
                section_id = item
            elif isinstance(item, dict):
                section_id = item.get("id")
            else:
                section_id = None

            if not section_id:
                continue

            if section_id not in seen:
                seen.add(section_id)
                result.append(section_id)

    # NOTE: intentionally NOT appending user_sections keys here.
    # This ensures "always only generate sections in sections_order",
    # while still allowing user content to feed those sections.

    return result


# ---------------------------------------------------------------------------
# Load generation params from parameters/parameters.yaml
# ---------------------------------------------------------------------------

def _load_generation_params() -> Dict[str, Any]:
    params_path = ROOT / "parameters" / "parameters.yaml"
    if not params_path.exists():
        print(
            f"⚠️  parameters.yaml not found at {params_path}, "
            f"using default generation params and STUB LLM.",
            file=sys.stderr,
        )
        return {}

    with params_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    gen_cfg = cfg.get("generation", {}) or {}
    if not isinstance(gen_cfg, dict):
        print("⚠️  'generation' key in parameters.yaml is not a dict, ignoring.", file=sys.stderr)
        return {}

    return gen_cfg


def _map_generation_params_for_engine(gen_cfg: Dict[str, Any]) -> Dict[str, Any]:
    engine_params: Dict[str, Any] = {}

    if "model_name" in gen_cfg:
        engine_params["model"] = gen_cfg["model_name"]
    if "temperature" in gen_cfg:
        engine_params["temperature"] = gen_cfg["temperature"]
    if "max_tokens" in gen_cfg:
        engine_params["max_output_tokens"] = gen_cfg["max_tokens"]
    if "timeout_seconds" in gen_cfg:
        engine_params["timeout_seconds"] = gen_cfg["timeout_seconds"]
    if "max_retries" in gen_cfg:
        engine_params["max_retries"] = gen_cfg["max_retries"]

    return engine_params


# ---------------------------------------------------------------------------
# Dummy LLM client for local testing
# ---------------------------------------------------------------------------

def _dummy_llm_client(prompt: str, **kwargs: Any) -> str:
    preview = prompt.replace("\n", " ")[:200]
    return f"This is a mocked LLM-generated CV section based on: {preview!r}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Pretty header similar to unittest output style
    print("\n" + "=" * 90, file=sys.stderr)
    print("🧪 STARTING MOCK API CV GENERATION FROM YAML FIXTURES", file=sys.stderr)
    print("=" * 90, file=sys.stderr)

    # Ensure llm_metrics reads the correct config & path BEFORE importing it
    os.environ.setdefault("PARAMETERS_YAML", (ROOT / "parameters" / "parameters.yaml").as_posix())
    os.environ.setdefault("LLM_METRICS_CSV", (ROOT / "local_logs" / "llm_call_logs.csv").as_posix())

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

    # 4) Build NO-LLM response
    print("[4/6] Building CVGenerationResponse WITHOUT LLM...", file=sys.stderr)

    template_info: Dict[str, Any] = mock_payload.get("template_info", {}) or {}
    profile_info: Dict[str, Any] = mock_payload.get("profile_info", {}) or {}
    user_sections: Dict[str, Any] = mock_payload.get("user_input_cv_text_by_section", {}) or {}

    minimal_payload: Dict[str, Any] = {
        "template_info": template_info,
        "profile_info": {
            "name": profile_info.get("name"),
            "email": profile_info.get("email"),
            "phone": profile_info.get("phone"),
            "linkedin": profile_info.get("linkedin"),
        },
        "user_input_cv_text_by_section": user_sections,
    }
    minimal_payload["profile_info"] = profile_info  # reattach full info for skills extraction

    response_wo_llm = _build_response_from_mock_payload(minimal_payload)

    # Save JSON (no-LLM) – exclude metadata
    data_wo_llm = response_wo_llm.model_dump(mode="json", exclude={"metadata"})
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

    # 5) Build request + call Stage B generate_cv() WITH LLM/STUB
    print("[5/6] Building request object and calling generate_cv()...", file=sys.stderr)

    gen_cfg = _load_generation_params()
    engine_params = _map_generation_params_for_engine(gen_cfg)

    # Verbosity: env overrides parameters.yaml
    env_verbose = (os.getenv("MAIN_VERBOSE") == "1") or (os.getenv("LLM_METRICS_VERBOSE") == "1")
    param_verbose = bool(gen_cfg.get("verbose_main", False))
    VERBOSE_MAIN: bool = env_verbose or param_verbose

    use_stub = gen_cfg.get("use_stub")
    if use_stub is None:
        use_stub = True

    if use_stub:
        print("[5/6] parameters.generation.use_stub = true → using STUB LLM client.", file=sys.stderr)
        llm_client = _dummy_llm_client
    else:
        try:
            from functions.utils.llm_client import call_llm as _real_llm_client  # type: ignore  # noqa: E402
            llm_client = _real_llm_client
            print("[5/6] parameters.generation.use_stub = false → using REAL LLM client.", file=sys.stderr)
        except Exception as e:
            print(
                f"⚠️  Failed to import real llm_client, falling back to STUB. Error: {e}",
                file=sys.stderr,
            )
            llm_client = _dummy_llm_client

    # ------------------------------------------------------------
    # 🔍 Step 1: Probe the LLM path to confirm token reporting works
    # ------------------------------------------------------------
    try:
        from functions.utils.llm_client import call_llm as real_llm_call
        probe_resp = real_llm_call(
            "Token probe: please say 'ok'.",
            model=engine_params.get("model", "gemini-2.5-flash"),
        )
        print(
            f"[probe] LLM usage: pt={probe_resp.usage.get('prompt_tokens')} "
            f"ct={probe_resp.usage.get('completion_tokens')} "
            f"tt={probe_resp.usage.get('total_tokens')}",
            file=sys.stderr,
        )
    except Exception as e:
        print(f"[probe] LLM call failed: {e}", file=sys.stderr)

    # ------------------------------------------------------------
    # 🧭 Step 3: Diagnostics – verify SDK + API key status
    # ------------------------------------------------------------
    print(f"[info] use_stub={use_stub}", file=sys.stderr)
    try:
        import google.generativeai as _g  # noqa: F401
        print("[info] google-generativeai SDK: AVAILABLE", file=sys.stderr)
    except Exception:
        print("[info] google-generativeai SDK: MISSING", file=sys.stderr)
    print(f"[info] GOOGLE_API_KEY present? {bool(os.getenv('GOOGLE_API_KEY'))}", file=sys.stderr)

    # NOTE: API dump/verbose SDK object logging is intentionally disabled here,
    # because llm_client.py already handles optional verbose dumping and usage tracing.

    chosen_model = engine_params.get("model", "gemini-2.5-flash")
    csv_path = llm_metrics._get_csv_path()
    print(f"[5/6] parameters.generation.model_name = {chosen_model}", file=sys.stderr)
    print(f"[5/6] LLM cost/usage CSV → {Path(csv_path).resolve()}", file=sys.stderr)

    sections: List[str] = _compute_sections_from_template_and_user(template_info, user_sections)

    class DummyTemplateInfo:
        """Minimal stub for request.template_info."""
        pass

    tmpl_stub = DummyTemplateInfo()
    setattr(tmpl_stub, "template_id", template_info.get("template_id", "T_EMPLOYER_STD_V3"))
    setattr(tmpl_stub, "max_chars_per_section", template_info.get("max_chars_per_section", {}))

    class DummyRequest:
        """Minimal stub for CVGenerationEngine.generate_cv(request=...)."""
        pass

    request = DummyRequest()
    request.user_id = "MOCK-USER-001"
    request.language = template_info.get("language") or "en"
    request.template_info = tmpl_stub
    request.profile_info = profile_info
    request.job_role_info = mock_payload.get("job_role_info")
    request.job_position_info = mock_payload.get("job_position_info")
    request.company_info = mock_payload.get("company_info")
    request.user_input_cv_text_by_section = user_sections
    request.sections = sections
    request.job_id = "JOB_MOCK_001"

    # ------------------------------------------------------------
    # ⚙️ Step 4: Confirm llm_metrics pipeline is being triggered
    # ------------------------------------------------------------
    print("[debug] Initializing CVGenerationEngine...", file=sys.stderr)

    # Optional warm-up through llm_metrics to guarantee one sdk_object with tokens
    # (safe no-op; remove if you don’t want an extra call)
    from functions.utils import llm_metrics as _metrics

    class _Adapter:
        def generate(self, *, model, messages):
            # Reuse the real llm_client so tokens/usage flow through
            from functions.utils.llm_client import call_llm as _real_llm
            return _real_llm("\n".join(m.get("content", "") for m in messages), model=model)

    try:
        _ = _metrics.call_llm_section_with_metrics(
            llm_client=_Adapter(),
            model=engine_params.get("model", "gemini-2.5-flash"),
            prompt="(warmup) say 'ok' only.",
            section_id="warmup",
            purpose="stage_b_generation",
            user_id="MOCK-USER-001",
        )
        print("[debug] Warm-up via llm_metrics completed.", file=sys.stderr)
    except Exception as e:
        print(f"[debug] Warm-up via llm_metrics failed: {e}", file=sys.stderr)

    print("[debug] Engine will be created next; Stage B will call llm_metrics.", file=sys.stderr)

    engine = CVGenerationEngine(llm_client=llm_client, generation_params=engine_params)
    response_w_llm = engine.generate_cv(request=request, evidence_plan=None)  # type: ignore[arg-type]
    response_w_llm = run_stage_c_validation(
        response_w_llm,
        template_info=template_info,
        original_request=request,
    )

    # Save JSON (with-LLM)
    data_w_llm = response_w_llm.model_dump(mode="json")
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
        f"      Saved WITH-LLM CV HTML → {output_cv_w_llm_html}",
        file=sys.stderr,
    )

    print("[6/6] ✅ Mock API CV generation completed (wo/LLM and w/LLM).", file=sys.stderr)
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
