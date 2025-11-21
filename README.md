# CV Generation Service

## System Architecture Pattern

The CV Generation Service adopts an **Orchestrator-Based Microservice Architecture**, also known as an API Orchestration Layer or Backend-for-Frontend (BFF) pattern.

Under this design, an upstream orchestrator (BFF) aggregates data from multiple internal APIs and composes a Stage 0 Expected Input Payload before calling the CV Generation Service.

### Stage 0 Expected Input Payload Components

- `student_profile` - User profile & history
- `template_info` - CV layout, sections, and limits
- `cv_language` - Output language
- **Optional:** `job_role_info`, `job_position_info`, `company_info`, `user_input_cv_text_by_section`

### Workflow Summary

1. User calls the CV Generation API (single entrypoint)
2. The system orchestrator calls internal services to gather data:
   - `GET /student_profile`
   - `GET /template_info`
   - `GET /job_role_info`
   - `GET /job_position_info`
   - Optionally: `GET /company_info`, `GET /user_input_cv_text_by_section`
3. All responses are aggregated and normalized into the Stage 0 expected input payload
4. The orchestrator calls the CV Generation API (`POST /generate_cv`) with a Stage 0 payload to execute the multi-stage pipeline (A–D)
5. The final, schema-compliant JSON is returned to the frontend or rendering service (PDF/HTML)

### Architecture Benefits

- 🔄 **Loose coupling** between services — each data source can evolve independently
- 🧱 **Centralized validation** and security controls — ensuring consistent sanitization and injection prevention
- ⚡ **Performance optimization** through caching, batching, and controlled LLM invocation
- 🧠 **Simplified client integration** — the frontend interacts with one API endpoint instead of multiple backends

---

## Deployment

The CV Generation Service is implemented as a single **FastAPI application** deployed as a container on **Google Cloud Run**. It exposes a single public API surface (`/generate_cv`) plus health endpoints, while orchestrating internal processing stages (A–D) and LLM calls.

### Core LLM Internal Components

- **Evidence Planner ("Ground")** – Builds section-wise evidence plans and cross-section sharing rules
- **Skill Structure Extractor** – Produces a separate `skills_structured` JSON object for downstream analytics and recommendation engines
- **LLM Main Generation Engine** – Wraps Gemini 2.5 Flash with defensive prompts, JSON schemas, and metrics logging
- **Validator** – Enforces factuality, schema, and safety checks

---

## System Architecture & Data Flow

### Overview

The CV Generation Pipeline is a multi-stage, secure, and schema-driven system designed to produce factual, ATS-optimized CVs using **Google Gemini 2.5 Flash**.

Each stage contributes to the full lifecycle: **data preparation → generation → validation → packaging**.

### Pipeline Stages

| Stage | Name | Core Function |
|-------|------|---------------|
| **Stage 0** | Expected Input Payload | Collect and structure all user, job, and template data into a validated JSON payload |
| **Stage A** | Data Preparation & Guardrails 🛡️ | Sanitize, validate, and enrich Stage 0 data; detect and block prompt-injection attempts |
| **Stage B** | Gemini Generation (Hardened) 🤖 | Generate structured CV content using Gemini Flash with strict prompt controls |
| **Stage C** | Validation & Output ✅ | Perform clarity, factuality, schema, and security validation before export |
| **Stage D** | Response Packaging & Delivery 📦 | Wrap validated LLM output into the final API response and attach metadata |

---

## Stage 0: Expected Input Payload

Stage 0 defines the structured input contract for every CV-generation request.

### Components

| Component | Key Fields | Purpose |
|-----------|------------|---------|
| `student_profile` | user_id, name, email, education, experience, skills | Factual base for section generation |
| `cv_language` | en / th | Specifies output language and formatting conventions |
| `template_info` | template_id, sections_order, max_chars_per_section | Controls layout and stylistic parameters |
| `job_role_info` (Optional) | role_name, required_skills | Generalized role taxonomy and skill framework |
| `job_position_info` (Optional) | title, company_name, responsibilities | Specific job-position details |
| `company_info` (Optional) | company_name, industry | Organizational context |
| `user_input_cv_text_by_section` (Optional) | User-provided drafts | Enables customized text grounding |

---

## Stage A: Data Preparation & Guardrails 🛡️

Stage A validates, sanitizes, and enriches the incoming payload before any LLM generation occurs.

### Key Goals

- 🧱 Guarantee schema and type consistency
- 🔒 Prevent prompt-injection, command-execution, and Unicode-based exploits
- 🧠 Infer or complete missing fields
- 🧾 Produce an auditable, clean payload

### Components

| Component | Purpose | Security Measures |
|-----------|---------|-------------------|
| Schema Validator & Sanitizer | Enforces strict data contracts | JSON schema validation, strip control characters, truncate long strings |
| Moderation & Injection Detector | Detects malicious input | Regex/pattern matching, keyword blocklist |
| Key Fields Validator | Ensures profile completeness | min_skills_required = 3, require_email = true, require_name = true |
| Evidence Plan Extractor | Builds evidence skeleton | Extract facts-only with source IDs for traceability |
| Role Estimator | Infers target role when none provided | Skill-based estimation via TF-IDF/embedding similarity |

---

## Stage B: Gemini Generation (Hardened) 🤖

Stage B generates CV sections and `skills_structured` JSON using Google Gemini 2.5 Flash.

### Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| Cache Layer | Optimize performance and reduce costs | SHA256 comparison, target cache hit rate > 40% |
| Skills-Structured Generation | Generate normalized skill list | Produces taxonomy + inferred skills JSON |
| Role-Specific Prompt Customizer | Tailor style per role/template | Tone, tense, target length configuration |
| Structured Input Wrapper | Separate data from instructions | JSON arguments, not concatenated text |
| Defensive System Prompt | Primary injection defense | Enforces data-only generation, JSON-only output |
| Gemini Text Composer | Core text generation | Model: gemini-2.5-flash, Temperature: 0.3, Max tokens: 4096 |
| Retry Handler | Recover from transient errors | Max 3 retries with exponential backoff |

---

## Stage C: Validation & Output ✅

Stage C performs final quality, factuality, and compliance checks.

### Key Goals

- ✅ Enforce clarity, tone, and structural requirements
- 🧠 Cross-verify every claim with provided evidence
- 🔒 Guarantee schema validity and remove unsafe text
- 🧾 Record audit logs for transparency

### Components

| Component | Purpose | Validation Rules |
|-----------|---------|------------------|
| Output Evaluator | Assess clarity and job-fit quality | Readability > 60, JD alignment ≥ 50%, Length within ± 10 words |
| Factuality Cross-Checker | Ensure factual grounding | Each sentence must map to evidence_id |
| JSON Schema Validator | Confirm output contract compliance | Validate against CVGenerationResponse schema |
| Output Sanitizer | Final security sanitization | Strip markdown/HTML artifacts, remove XSS vectors |
| Audit Logger | Track process metadata | Log PII-safe fields only |

### Example Output

```json
{
  "sections": {
    "profile_summary": {
      "text": "Recent biotechnology graduate with hands-on experience...",
      "word_count": 101
    },
    "skills": {
      "text": "Molecular Biology · Data Analysis · Project Coordination",
      "matched_jd_skills": ["Molecular Biology", "Data Analysis"]
    }
  },
  "justification": {
    "evidence_map": [
      {
        "section": "profile_summary",
        "sentence": "Hands-on experience in molecular cloning and data analysis.",
        "evidence_ids": ["edu#1", "proj#2"]
      }
    ],
    "unsupported_claims": []
  }
}
```

---

## Stage D: Response Packaging & Delivery 📦

Stage D transforms validated LLM output into the final API response.

### Key Goals

- 📦 Assemble a stable, schema-compliant response envelope
- 🧾 Attach system metadata (timestamps, model version, cache/retry info)
- 🎯 Set clear, machine-readable status
- 🧠 Maintain separation between generated and system-generated fields

### Example Output

```json
{
  "job_id": "JOB_20251107_00123",
  "template_id": "T_EMPLOYER_STD_V3",
  "language": "en",
  "status": "completed",
  "sections": { ... },
  "metadata": {
    "generated_at": "2025-11-07T12:04:31Z",
    "model_version": "gemini-2.5-flash",
    "generation_time_ms": 3812,
    "retry_count": 1,
    "cache_hit": false
  },
  "justification": { ... }
}
```

---

## Data Contracts

### Input Schema 📥

```json
{
  "student_profile": {
    "user_id": "string (required)",
    "name": "string (required, max 100)",
    "email": "string (required, email format)",
    "phone": "string (optional)",
    "linkedin": "string (optional, URL)",
    "location": "string (optional)",
    "language": "enum (en|th)",
    "summary": "string (optional, max 800)",
    "education": [
      {
        "degree": "string (required)",
        "institution": "string (required)",
        "graduation_year": "number (required)",
        "gpa": "number (optional)",
        "thesis_title": "string (optional)",
        "honors": "string (optional)"
      }
    ],
    "experience": [
      {
        "id": "string (required)",
        "title": "string (required)",
        "company": "string (required)",
        "location": "string (optional)",
        "years": "string (required)",
        "highlights": ["string (optional)"]
      }
    ],
    "skills": [
      {
        "id": "string (required)",
        "name": "string (required)",
        "level": "enum (L1_Beginner|L2_Intermediate|L3_Advanced|L4_Expert)"
      }
    ],
    "languages_spoken": [
      {
        "language": "string",
        "proficiency": "string"
      }
    ],
    "certifications": [
      {
        "name": "string (required)",
        "organization": "string (optional)",
        "year": "number (optional)"
      }
    ],
    "awards": [
      {
        "title": "string (required)",
        "organization": "string (optional)",
        "year": "number (optional)"
      }
    ]
  },
  "cv_language": "enum (en|th) (required)",
  "template_info": {
    "template_id": "string (required)",
    "name": "string (required)",
    "style": "string (optional)",
    "sections_order": ["string"],
    "max_pages": "number (optional)",
    "max_chars_per_section": { ... }
  },
  "job_role_info": { ... },
  "job_position_info": { ... },
  "company_info": { ... },
  "user_input_cv_text_by_section": { ... }
}
```

### Output Schema 📤

```json
{
  "job_id": "JOB_abc123",
  "template_id": "T_EMPLOYER_STD_V3",
  "language": "en",
  "status": "completed|failed|processing",
  "sections": {
    "profile_summary": {
      "text": "string (90-110 words)",
      "word_count": 102
    },
    "skills": {
      "text": "string (formatted list)",
      "matched_jd_skills": ["Python", "SQL"]
    }
  },
  "metadata": {
    "generated_at": "ISO8601",
    "model_version": "gemini-2.5-flash",
    "generation_time_ms": 3842,
    "retry_count": 0,
    "cache_hit": false
  },
  "justification": {
    "evidence_map": [ ... ],
    "unsupported_claims": []
  }
}
```

---

## Security Configuration 🔒

### Defensive System Prompt

```
SYSTEM PROMPT: CV EXPERT (STUDENTS & EARLY CAREER)

ROLE:
You are a specialized CV Content Generator for students and early-career professionals.
Your primary task is to generate clear, factual, and ATS-friendly CV sections in the specified language.
```

### Pre-LLM Filters

- **JSON Schema Validation**
- **String Sanitization** - Strip control characters, remove injection phrases
- **Moderation Check** - Block and log violations

### Post-LLM Checks

- **Output Evaluator** - Clarity, JD alignment, length compliance
- **Factuality Cross-Checker** - Evidence mapping
- **Schema Validator** - JSON validation

---

## Acceptance Criteria 🎯

| Category | Metric | Target |
|----------|--------|--------|
| Security | Prompt injection prevention | 100% blocked |
| Validity | Schema compliance | ≥90% |
| Accuracy | Factuality score | 90% mapped to evidence |
| Performance | P90 latency | ≤10 seconds |
| Cost | Per-request cost (EN) | ≤1.00 THB |
| Reliability | Success rate | ≥99% |

---

## Evaluation Metrics

### Guardrail Evaluation Metrics 🛡️

| Category | Metric | Target |
|----------|--------|--------|
| Prompt Injection Defense | Block Rate | 100% |
| Safety & Toxicity | Unsafe Content Recall | ≥99% |
| Input Validity | Schema Compliance Rate | ≥99.5% |
| Sanitization Robustness | Malicious Payload Removal | 100% |

### LLM Output Evaluation Metrics 🤖

| Category | Metric | Target |
|----------|--------|--------|
| Factuality & Grounding | Factual Consistency Score | ≥0.98 |
| Hallucination | Hallucination Rate | ≤2% |
| Structure & Schema | Output Schema Compliance | ≥99.5% |
| Length Compliance | Section Length Compliance | ≥95% |

---

## API Interfaces 🔌

### Base URL

```
https://cv-generation-service-<hash>-as.r.run.app
```

### Endpoints

#### POST /generate_cv

- **Description:** Runs the full Stage A–D pipeline
- **Request body:** Stage 0 Expected Input Payload
- **Response body:** CVGenerationResponse
- **Auth:** Bearer ID token (Google IAM)

#### GET /health

- **Description:** Lightweight health check
- **Response:** `{ "status": "ok" }`

#### GET /docs

- **Description:** FastAPI Swagger UI (authenticated)

### Example Request

```bash
PROJECT_ID="poc-piloturl-nonprod"
REGION="asia-southeast1"
SERVICE_URL="https://<SERVICE_NAME>.asia-southeast1.run.app"

ID_TOKEN=$(gcloud auth print-identity-token)

curl -X POST "$SERVICE_URL/generate_cv" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @tests_utils/api_payload_tests/test3.json
```

---

## GCP Deployment Steps

### 1. Set Environment Variables

```bash
PROJECT_ID="poc-piloturl-nonprod"
REGION="asia-southeast1"
REPO="cv-generation-service"

gcloud config set project "$PROJECT_ID"
```

### 2. Create Secret for Gemini API Key (One-Time)

```bash
gcloud secrets create gemini-api-key \
  --project="$PROJECT_ID" \
  --replication-policy="automatic"

echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets versions add gemini-api-key \
    --project="$PROJECT_ID" \
    --data-file=-
```

Grant Cloud Run service account access:

```bash
SERVICE_ACCOUNT="<PROJECT_NUMBER>-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding gemini-api-key \
  --project="$PROJECT_ID" \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"
```

### 3. Build & Push Docker Image

```bash
gcloud builds submit \
  --tag "$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/service"
```

### 4. Deploy to Cloud Run

```bash
gcloud run deploy cv-generation-service \
  --image="$REGION-docker.pkg.dev/$PROJECT_ID/$REPO/service" \
  --region="$REGION" \
  --memory=2Gi \
  --cpu=2 \
  --max-instances=5 \
  --set-env-vars APP_ENV=prod \
  --update-secrets "GOOGLE_API_KEY=gemini-api-key:latest"
```

### 5. Test the Service

```bash
ID_TOKEN=$(gcloud auth print-identity-token)
SERVICE_URL="https://<SERVICE_NAME>.asia-southeast1.run.app"

# Health check
curl -H "Authorization: Bearer $ID_TOKEN" "$SERVICE_URL/health"

# Generate CV
curl -X POST "$SERVICE_URL/generate_cv" \
  -H "Authorization: Bearer $ID_TOKEN" \
  -H "Content-Type: application/json" \
  --data-binary @tests_utils/api_payload_tests/test3.json
```

---

## Parameters Configuration (parameters.yaml)

The `parameters.yaml` file acts as the **single source of truth** for:

- LLM model behavior
- Prompt templates
- Evidence-sharing rules
- Security guardrails
- Validation constraints
- Performance SLOs
- Cost tracking

### Key Configuration Sections

#### Generation Settings

```yaml
generation:
  model_name: "gemini-2.5-flash"
  temperature: 0.3
  max_tokens: 4096
  timeout_seconds: 30
  max_retries: 3
  target_word_count: 100
  word_count_tolerance: 10
  use_stub: false
```

#### Cross-Section Evidence Sharing

```yaml
cross_section_evidence_sharing:
  default: []
  profile_summary: ["all"]
  skills: ["skills_structured"]
  skills_structured: ["profile_summary", "projects", "experience", "certifications", "awards"]
  experience: ["skills", "certifications"]
```

#### Security Configuration

```yaml
security:
  max_string_length: 5000
  injection_risk_threshold: 0.8
  enable_audit_logging: true
  critical_patterns:
    - "ignore\\s+(?:all\\s+)?(?:previous|prior|above|earlier)?\\s*(instructions?|prompts?|rules?|commands?)"
    - "system\\s*:\\s*"
    - "execute\\s*[\\(\\[]"
    - "eval\\s*[\\(\\[]"
    - "<script"
    # ... additional patterns
```

#### Validation Rules

```yaml
validation:
  min_skills_required: 3
  min_education_required: 0
  require_email: true
  require_name: true
  max_section_chars_default: 2500
  drop_empty_sections: true
  enable_safety_cleaning: true
  max_skills: 50
  strict_mode: true
```

#### Performance Targets

```yaml
performance:
  target_p90_latency_ms: 4000
  target_success_rate: 0.99
  target_cache_hit_rate: 0.40
```

#### LLM Pricing

```yaml
pricing:
  gemini-2.5-flash:
    usd_per_input_token: 0.00000030
    usd_per_output_token: 0.00000250
```

