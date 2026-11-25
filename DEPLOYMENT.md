# Deployment Guide

> **Supported environments:** Local development, Docker Compose (self-hosted), AWS ECS Fargate (production)

---

## 1. Prerequisites

| Tool | Minimum Version | Purpose |
|---|---|---|
| Python | 3.12 | Backend + Frontend |
| Docker | 24.x | Container runtime |
| Docker Compose | 2.x | Local orchestration |
| PostgreSQL | 16 | Primary database |
| Redis | 7.x | Cache & Celery broker |
| Node.js | 20.x | Only if building React frontend |

---

## 2. Environment Variables

Copy `.env.example` to `.env` and populate:

```bash
cp .env.example .env
```

### Required Variables

| Variable | Example | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql://finrisk:pass@db:5432/finrisk` | PostgreSQL connection string |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `SECRET_KEY` | `$(openssl rand -hex 32)` | JWT signing key — keep secret |
| `LLM_PROVIDER` | `mock` | `mock`, `gemini`, or `openai` |
| `GEMINI_API_KEY` | `AIza...` | Required only if `LLM_PROVIDER=gemini` |
| `OPENAI_API_KEY` | `sk-...` | Required only if `LLM_PROVIDER=openai` |

### Optional Variables

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_RISK_FREE_RATE` | `0.04` | Annual risk-free rate (decimal) |
| `DEBUG` | `false` | FastAPI debug mode — never `true` in production |
| `POSTGRES_PASSWORD` | `finrisk_secure_123` | Override in production |
| `ACCESS_TOKEN_EXPIRE_HOURS` | `24` | JWT token lifetime |

---

## 3. Local Development (Without Docker)

```bash
# 1. Clone repository
git clone https://github.com/your-org/finrisk-ai.git
cd finrisk-ai

# 2. Create virtual environment with Python 3.12
python3.12 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Set environment variables
export DATABASE_URL="sqlite:///./finrisk.db"  # SQLite fallback for local dev
export LLM_PROVIDER=mock
export SECRET_KEY=dev-local-secret-key
export DEFAULT_RISK_FREE_RATE=0.04

# 5. Start FastAPI backend (auto-creates tables + seeds data)
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

# 6. In a second terminal: start Streamlit frontend
source .venv/bin/activate
export BACKEND_API_URL=http://localhost:8000/api
streamlit run frontend/app.py --server.port 8501
```

**Access:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Frontend: http://localhost:8501

---

## 4. Docker Compose Deployment

### Quick Start

```bash
# Start entire stack (PostgreSQL + Redis + Backend + Celery + Frontend)
docker-compose up --build

# Run in background (detached)
docker-compose up -d --build

# View logs
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

### Environment Override

```bash
# Production-like settings
POSTGRES_PASSWORD=your_secure_password \
LLM_PROVIDER=gemini \
GEMINI_API_KEY=AIzaSy... \
SECRET_KEY=$(openssl rand -hex 32) \
docker-compose up -d
```

### Service URLs
| Service | URL |
|---|---|
| FastAPI Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Streamlit Frontend | http://localhost:8501 |
| Celery Flower Monitor | http://localhost:5555 |
| PostgreSQL | localhost:5432 |
| Redis | localhost:6379 |

### Useful Commands

```bash
# Scale Celery workers for heavier Monte Carlo loads
docker-compose up -d --scale celery-worker=3

# Run database migrations (future Alembic support)
docker-compose exec backend alembic upgrade head

# Access PostgreSQL shell
docker-compose exec db psql -U finrisk -d finrisk

# Access Redis CLI
docker-compose exec redis redis-cli

# Restart only the backend after code changes
docker-compose restart backend

# Stop all services and remove volumes (clean slate)
docker-compose down -v
```

---

## 5. Production Deployment — AWS ECS Fargate

> Infrastructure-as-Code Terraform templates are located in `infra/terraform/`.

### Architecture

```
Internet → AWS ALB → ECS Fargate (backend task, frontend task)
                                ↓                       ↓
                        RDS PostgreSQL 16     ElastiCache Redis 7
                        (pgvector enabled)    (cache.t3.micro)
                                ↓
                        S3 (data snapshots, audit logs)
```

### Deployment Steps

**Step 1: Push images to ECR**
```bash
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.ap-south-1.amazonaws.com

docker build -f docker/backend.Dockerfile -t finrisk-backend .
docker tag finrisk-backend:latest <account>.dkr.ecr.ap-south-1.amazonaws.com/finrisk-backend:latest
docker push <account>.dkr.ecr.ap-south-1.amazonaws.com/finrisk-backend:latest
```

**Step 2: Provision infrastructure**
```bash
cd infra/terraform
terraform init
terraform plan -var-file=prod.tfvars
terraform apply -var-file=prod.tfvars
```

**Step 3: Update ECS service**
```bash
aws ecs update-service \
  --cluster finrisk-prod \
  --service finrisk-backend \
  --force-new-deployment \
  --region ap-south-1
```

### Key AWS Services Required

| Service | Purpose | Tier |
|---|---|---|
| ECS Fargate | Container orchestration | Serverless |
| RDS PostgreSQL 16 | Primary database with pgvector | db.t3.medium |
| ElastiCache Redis 7 | Celery broker + result cache | cache.t3.micro |
| ALB | Load balancing + TLS termination | Standard |
| S3 | Data snapshots + audit log storage | Standard |
| Secrets Manager | API keys + DB passwords | Per secret/month |
| CloudWatch | Logs + metrics + alarms | Pay per use |
| ECR | Private container image registry | Per GB |

### RDS pgvector Setup

```sql
-- Run once after RDS provisioning
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;
```

---

## 6. Database Migrations

The current version uses SQLAlchemy `create_all()` for initial schema creation. For production schema evolution, use Alembic:

```bash
# Initialize Alembic (first time only)
alembic init alembic

# Generate migration from model changes
alembic revision --autogenerate -m "add_user_table"

# Apply all pending migrations
alembic upgrade head

# Rollback one revision
alembic downgrade -1
```

---

## 7. Health Monitoring

### Health Check Endpoints
- Backend: `GET /health` → `{"status": "healthy"}`
- Frontend: `GET /_stcore/health` (Streamlit built-in)

### Recommended Alarms (CloudWatch)
- Backend 5xx error rate > 1% → PagerDuty alert
- Celery queue depth > 100 jobs → scale out workers
- RDS CPU > 80% for 5 minutes → instance resize
- ECS task restart count > 3 in 10 minutes → incident

---

## 8. Backup & Recovery

### Database Backups
- **RDS automated backups:** 7-day retention enabled by default
- **Manual snapshot before major releases:** `aws rds create-db-snapshot ...`
- **Point-in-time recovery:** Enabled on RDS with 5-minute RPO

### Data Snapshots
Market data CSVs are mounted read-only from S3 in production. Snapshots are versioned and immutable to ensure backtesting reproducibility.

---

## 9. CI/CD Pipeline

See [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

**Pipeline Stages:**
1. `lint` — Ruff, Black, isort
2. `test` — pytest with PostgreSQL + Redis services, 70% coverage gate
3. `security` — Bandit SAST + pip-audit CVE scan
4. `build` — Docker multi-stage build + push to GHCR (main branch only)

**Merge requirements:** All `test` and `security` checks must pass before PR merge.
