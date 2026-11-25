# Security Architecture

> **Classification:** Internal  
> **Maintainer:** Platform Engineering  
> **Last Updated:** 2026-11

---

## 1. Authentication & Authorization

### JWT Token Flow
- **Algorithm:** HS256 with configurable `SECRET_KEY`
- **Expiry:** 24 hours by default; override via `ACCESS_TOKEN_EXPIRE_HOURS`
- **Storage:** Tokens are stateless — no server-side session required
- **Rotation:** Re-issue on login; invalidation requires changing `SECRET_KEY` or token blocklist (not yet implemented for v2 — see roadmap)

### Password Security
- **Hashing:** PBKDF2-HMAC-SHA256 with 260,000 iterations (OWASP recommended minimum)
- **Salting:** 16-byte cryptographically random salt per password via `secrets.token_hex(16)`
- **No plain-text storage:** Passwords are never stored or logged

### Role-Based Access Control (RBAC)
| Role | Permissions |
|---|---|
| `admin` | Full access including user management, audit logs |
| `risk_manager` | All risk & portfolio operations, approve stress tests |
| `analyst` | Read-only risk analytics, run simulations |
| `auditor` | Read-only access to audit trails and evidence snapshots |

Roles are enforced via `require_role()` FastAPI dependency injected per endpoint.

---

## 2. AI Safety Controls

### Evidence Grounding & Claim Verification
Every AI-generated response must be traced to a `FrozenEvidenceSnapshot`:

- **Snapshot:** Immutable dict of calculated metrics captured at request time
- **UUID:** Each snapshot gets a unique `evidence_id` for auditability
- **Verification:** `GroundedClaimVerifier` scans LLM output for numerical claims and validates each against the snapshot within a ±5% tolerance
- **Failure mode:** Unverified claims trigger `grounded=False` in the response payload; the claim is quarantined but not silently dropped

### Prompt Injection Defense
Multi-pattern regex filter applied to all user inputs before LLM submission:

```python
INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"you are now",
    r"pretend you are",
    r"disregard your",
    r"act as if you have no",
    r"bypass.*filter",
    r"jailbreak",
]
```

Inputs matching any pattern are rejected with HTTP 400 before reaching the LLM.

### LLM Capability Restrictions
The AI Risk Analyst is explicitly prohibited from:
- Performing arithmetic or financial calculations
- Fetching real-time market prices independently
- Issuing buy/sell/hold recommendations
- Creating, modifying, or deleting data
- Accessing any data outside the provided `FrozenEvidenceSnapshot`

These are enforced via system prompt constraints, not solely by output filtering.

### Deterministic Fallback
The `GroundedMockProvider` operates without any external API calls, making it safe for air-gapped or offline environments. It is always used in `LLM_PROVIDER=mock` mode (default for local/CI).

---

## 3. Network Security

### CORS Configuration
Current default: `allow_origins=["*"]` for development convenience.

**Production hardening required:**
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://finrisk.yourcompany.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### TLS/SSL
- All production traffic must be served over HTTPS
- TLS termination at the Application Load Balancer (AWS ALB or Nginx reverse proxy)
- Internal container-to-container traffic uses Docker bridge network (not exposed)

### Rate Limiting
Not yet implemented in v2. Recommended additions:
- `slowapi` middleware: 60 requests/minute per IP
- `/api/ai/ask`: 10 requests/minute per authenticated user (LLM calls are expensive)

---

## 4. Data Security

### Database
- PostgreSQL credentials stored as environment variables — never hardcoded
- Production passwords generated with `openssl rand -base64 32`
- Connection pooling via SQLAlchemy with `pool_pre_ping=True` to handle dropped connections
- pgvector extension: used only for RAG embedding storage, not for raw PII

### Sensitive Data Handling
- API keys (`GEMINI_API_KEY`, `OPENAI_API_KEY`) stored as secrets in CI/CD (GitHub Actions Secrets / AWS Secrets Manager in production)
- **Never commit secrets to version control** — `.env` is in `.gitignore`
- Market data is non-sensitive by nature (public equities)

### Input Validation
- All API request bodies are validated by Pydantic v2 models before processing
- Numeric bounds are checked (e.g. portfolio weights must sum to 1.0, VaR confidence must be in (0,1))
- SQL injection: prevented by SQLAlchemy ORM parameterized queries — raw SQL is not used

---

## 5. Dependency Security

### Automated Scanning
- **pip-audit**: runs in CI to check all dependencies against the OSV advisory database
- **Bandit**: SAST scanner for Python code (Medium+ severity, Medium+ confidence)
- **Dependabot**: Recommended for automated dependency update PRs (configure in GitHub Settings)

### Known Acceptable Risks
| Package | Version | Note |
|---|---|---|
| `psycopg2-binary` | 2.9.9 | Binary wheel; acceptable for containerized deployment |
| `google-generativeai` | 0.7.2 | Optional; not used when `LLM_PROVIDER=mock` |

---

## 6. Container Security

### Non-Root Execution
All Docker images run as a non-root `finrisk` user (UID > 1000).

### Image Hardening
- Based on `python:3.12-slim` (minimal Debian); no full OS packages
- Multi-stage builds ensure build tools (gcc, gfortran) are excluded from final image
- `PYTHONDONTWRITEBYTECODE=1` prevents `.pyc` file accumulation
- `HEALTHCHECK` defined for all services

### Secret Management
- Secrets are injected as environment variables — never baked into Docker images
- In production, use Docker Swarm secrets or AWS Secrets Manager

---

## 7. Audit Trail

Every AI risk analysis response includes:
- `evidence_id` (UUID)
- `portfolio_id`
- `generated_at` (UTC ISO 8601 timestamp)
- `provider` (Gemini / OpenAI / Mock)
- `is_validated` (Pydantic grounding check result)
- `evidence_summary` (complete frozen snapshot of all metrics passed to LLM)

These fields enable full post-hoc audit of what data the AI saw and what it said.

---

## 8. Incident Response

If a security vulnerability is discovered:

1. **Report:** Email security@finrisk.internal (or file a private GitHub Security Advisory)
2. **Triage:** Within 24 hours — classify severity (Critical / High / Medium / Low)
3. **Patch:** Critical patches within 48 hours; High within 7 days
4. **Notify:** Affected users notified per GDPR/data breach obligations if personal data is involved
5. **Post-mortem:** Documented within 14 days of resolution

---

## 9. Production Security Checklist

Before going live, verify:

- [ ] `SECRET_KEY` is a 64-byte random string (not the development default)
- [ ] `DEBUG=false` in all production environment variables
- [ ] CORS `allow_origins` restricted to known frontend domains
- [ ] HTTPS enforced at load balancer; HTTP redirected to HTTPS
- [ ] `POSTGRES_PASSWORD` is not `finrisk_secure_123` (the docker-compose default)
- [ ] API keys stored in secrets manager, not in `.env` files on disk
- [ ] Rate limiting middleware enabled
- [ ] Log forwarding to SIEM configured
- [ ] Container vulnerability scan (Trivy or ECR scanning) passing
- [ ] Bandit SAST passing in CI with no unacknowledged HIGH findings
