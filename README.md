# 🛡️ FinRisk AI — Institutional Portfolio Risk Intelligence Platform

> **An end-to-end, explainable AI portfolio risk management system** built to demonstrate professional-grade quantitative finance, machine learning, generative AI grounding, backend engineering, and cloud-native deployment.

---

## ✨ Platform Capabilities

| Domain | Capability |
|---|---|
| **Quantitative Risk** | Parametric VaR, Monte Carlo VaR, CVaR/ES, EWMA, GARCH, Sharpe, Sortino, Calmar, Euler risk attribution |
| **Stress Testing** | Historical crisis replay (2008, 2020, 2022, 2023), hypothetical factor shocks, reverse stress (SLSQP) |
| **Portfolio Optimization** | Markowitz Max Sharpe, Minimum Volatility, Risk Parity, Efficient Frontier |
| **Model Validation** | Kupiec POF test, Christoffersen independence test, Basel Traffic Light (Green/Yellow/Red), rolling backtest |
| **ML Intelligence** | K-Means + HMM + GMM market regime detection, Isolation Forest + LOF + CUSUM anomaly detection |
| **AI Risk Analyst** | Gemini / OpenAI / Mock provider, grounded Q&A, formal risk memorandum, RAG policy retrieval |
| **Data Engineering** | Modular data provider pattern (CSV, Yahoo Finance, synthetic GBM), quality validation, normalization |
| **Backend** | FastAPI v2, versioned `/api/v1` endpoints, JWT RBAC, Pydantic v2, SQLAlchemy, async job manager |
| **Infrastructure** | Multi-stage Docker builds, PostgreSQL pgvector, Redis, Celery, GitHub Actions CI/CD |

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FinRisk AI v2.0                              │
│                                                                     │
│  ┌──────────────┐    ┌──────────────────────────────────────────┐  │
│  │  Streamlit   │    │            FastAPI Backend                │  │
│  │  Frontend    │───▶│  /api/v1/portfolios  /api/v1/market      │  │
│  │  (10 Tabs)   │    │  /api/v1/ai          /api/v1/auth        │  │
│  └──────────────┘    └────────┬─────────────────────────────────┘  │
│                               │                                     │
│         ┌─────────────────────┼────────────────────────┐           │
│         ▼                     ▼                        ▼           │
│  ┌─────────────┐  ┌──────────────────┐  ┌──────────────────────┐  │
│  │ Risk Engine │  │    ML Engine     │  │    AI Engine         │  │
│  │ VaR · ES   │  │ Regime · Anomaly │  │ Gemini/OpenAI/Mock  │  │
│  │ GARCH · MC  │  │ HMM · GMM · IF  │  │ RAG · Grounding     │  │
│  │ Stress Lab  │  │ Walk-Forward CV  │  │ Evidence Snapshot   │  │
│  └──────┬──────┘  └───────┬──────────┘  └──────────┬───────────┘  │
│         └─────────────────┴────────────────────────┘               │
│                            │                                        │
│            ┌───────────────┼──────────────────┐                    │
│            ▼               ▼                  ▼                    │
│      ┌──────────┐   ┌────────────┐   ┌──────────────┐             │
│      │PostgreSQL│   │   Redis    │   │  Data Engine │             │
│      │pgvector  │   │  (Celery   │   │  CSV/Yahoo/  │             │
│      │  (ORM)   │   │   Broker)  │   │  GBM Mock    │             │
│      └──────────┘   └────────────┘   └──────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option A: Local Python (Fastest)

```bash
# 1. Clone
git clone https://github.com/your-org/finrisk-ai.git
cd finrisk-ai

# 2. Create virtualenv with Python 3.12
python3.12 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment (SQLite fallback — no Postgres needed)
export DATABASE_URL="sqlite:///./finrisk.db"
export LLM_PROVIDER=mock
export SECRET_KEY=dev-local-secret

# 5. Start backend
uvicorn backend.app.main:app --port 8000 --reload

# 6. Start frontend (new terminal)
export BACKEND_API_URL=http://localhost:8000/api
streamlit run frontend/app.py --server.port 8501
```

**Access:** Frontend → http://localhost:8501 | API Docs → http://localhost:8000/docs

---

### Option B: Docker Compose (Full Stack)

```bash
# Start PostgreSQL + Redis + Celery + Backend + Frontend
docker-compose up --build

# Or with custom LLM provider
LLM_PROVIDER=gemini GEMINI_API_KEY=AIzaSy... docker-compose up -d
```

| Service | URL |
|---|---|
| Streamlit Frontend | http://localhost:8501 |
| FastAPI Backend | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Celery Flower | http://localhost:5555 |

---

## 🧪 Running Tests

```bash
# Full test suite (50 tests)
pytest backend/tests/ -v

# With coverage report
pytest backend/tests/ --cov=backend/app --cov-report=term-missing
```

**Current:** ✅ 50 passed, 0 failed — covers risk engine, ML, AI grounding, stress testing, data pipeline, and API contracts.

---

## 📊 Platform Walkthrough (10 Tabs)

| Tab | Functionality |
|---|---|
| **Holdings & Allocation** | Portfolio composition table, weight pie chart, market value bar |
| **Risk Profile & Tail Risk** | Sharpe, Sortino, Calmar, Beta, VaR 95/99%, CVaR, drawdown timeline |
| **Monte Carlo Suite** | Correlated paths (up to 20,000), Normal/Student-t, methodology comparison table |
| **Stress Lab** | Hypothetical shocks, historical crisis replay (2008–2023), reverse stress solver |
| **Volatility Attribution** | Euler PCR decomposition, correlation heatmap, concentration alerts |
| **Portfolio Optimization** | Max Sharpe, Min Vol, Risk Parity, Efficient Frontier scatter plot |
| **ML Market Regimes** | K-Means regime timeline, multi-model anomaly detection scatter |
| **Model Validation** | Basel traffic light, Kupiec LR test, Christoffersen test, rolling backtest plot |
| **AI Risk Assistant** | Conversational Q&A with grounding, formal risk memorandum, evidence audit |
| **Governance & Audit** | RAG policy docs, AI safety protocols, claim verification status |

---

## 📐 Quantitative Methodology

All financial formulas are documented in [`QUANTITATIVE_METHODOLOGY.md`](QUANTITATIVE_METHODOLOGY.md).

### Key Conventions
- **Annualization:** 252 trading days
- **VaR sign convention:** Reported as a **positive dollar loss**
- **Euler risk decomposition:** $\text{PCR}_i = \frac{w_i \cdot (\Sigma w)_i}{\sigma_p} \times \frac{1}{\sigma_p}$, with $\sum_i \text{PCR}_i = 100\%$
- **GARCH stationarity:** $\alpha + \beta < 1$ enforced during MLE estimation
- **Monte Carlo:** Cholesky factorized covariance; Normal or Student-t distribution; reproducible seed
- **VaR Backtesting:** Kupiec POF (unconditional coverage) + Christoffersen (conditional independence)

---

## 🛡️ AI Safety Model

The AI Risk Analyst operates under strict evidence-grounding constraints:

```
User Query
    │
    ▼
Prompt Injection Screen (regex filter)
    │
    ▼
FrozenEvidenceSnapshot (UUID, immutable metrics dict)
    │
    ▼
LLM (Gemini / OpenAI / Mock)   ← Receives ONLY the evidence snapshot
    │
    ▼
GroundedClaimVerifier (Pydantic)  ← Rejects unsupported numerical claims
    │
    ▼
Structured Response + is_validated flag
```

**The LLM never performs arithmetic or fetches external data.** All numbers are pre-computed by NumPy/SciPy.

---

## 📂 Repository Structure

```
finrisk-ai/
├── backend/
│   ├── app/
│   │   ├── risk_engine/      # VaR, ES, Monte Carlo, GARCH, Stress, Optimization
│   │   ├── ml_engine/        # Regimes (HMM/GMM/KMeans), Anomalies, Walk-forward
│   │   ├── ai_engine/        # Grounding, RAG, Provider (Gemini/OpenAI/Mock)
│   │   ├── data_engine/      # Provider pattern, Quality validation, Pipeline
│   │   ├── api/              # FastAPI routers (portfolios, market, ai, auth)
│   │   ├── auth/             # JWT, PBKDF2, RBAC roles
│   │   ├── models/           # SQLAlchemy ORM models
│   │   ├── workers/          # Thread pool + Celery job manager
│   │   └── main.py           # FastAPI application factory
│   └── tests/                # 50 pytest tests
├── frontend/
│   ├── app.py                # 10-tab Streamlit dashboard
│   └── utils/api_client.py   # HTTP client for all backend endpoints
├── docker/
│   ├── backend.Dockerfile    # Multi-stage production backend
│   ├── frontend.Dockerfile   # Multi-stage production frontend
│   └── celery.Dockerfile     # Celery worker image
├── .github/workflows/
│   └── ci.yml                # Lint → Test → Security → Build pipeline
├── docker-compose.yml        # Full stack (PostgreSQL, Redis, Celery, Backend, Frontend)
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Bandit, pytest, Ruff, Black, isort config
├── PROJECT_AUDIT.md          # Pre-upgrade codebase assessment
├── SYSTEM_ARCHITECTURE.md    # Architecture documentation
├── IMPLEMENTATION_ROADMAP.md # 8-phase upgrade roadmap
├── TECHNICAL_DECISIONS.md    # ADR-01 through ADR-10
├── QUANTITATIVE_METHODOLOGY.md  # Math equations and conventions
├── ML_METHODOLOGY.md         # ML models, features, drift monitoring
├── AI_GROUNDING.md           # AI safety architecture
├── DATA_PIPELINE.md          # Data engineering documentation
├── API_DOCUMENTATION.md      # Complete API reference
├── SECURITY.md               # Security architecture and checklist
├── DEPLOYMENT.md             # Local, Docker, AWS ECS deployment guide
└── TESTING.md                # Testing guide and coverage targets
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | SQLite fallback | PostgreSQL or SQLite |
| `LLM_PROVIDER` | `mock` | `mock` / `gemini` / `openai` |
| `GEMINI_API_KEY` | _(empty)_ | Google AI Studio key |
| `SECRET_KEY` | _(dev)_ | JWT signing secret |
| `DEFAULT_RISK_FREE_RATE` | `0.04` | Annual Rf = 4% |

Copy `.env.example` to `.env` to get started.

---

## 🧰 Technology Stack

| Layer | Technology |
|---|---|
| **Backend API** | FastAPI 0.111, Uvicorn, Python 3.12 |
| **Frontend** | Streamlit 1.35, Plotly 5.22 |
| **Quantitative** | NumPy 1.26, SciPy 1.13, pandas 2.2, scikit-learn 1.5 |
| **Database** | PostgreSQL 16 + pgvector, SQLAlchemy 2.0 |
| **Cache/Jobs** | Redis 7, Celery (optional), Thread pool (default) |
| **AI/LLM** | Google Gemini, OpenAI GPT, Mock (deterministic) |
| **Validation** | Pydantic v2, Pandera (data quality) |
| **Auth** | JWT (HS256), PBKDF2 password hashing, RBAC |
| **Containers** | Docker (multi-stage), Docker Compose |
| **CI/CD** | GitHub Actions (lint → test → security → build) |
| **Cloud** | AWS ECS Fargate, RDS, ElastiCache, S3 (Terraform) |

---

## 📄 Documentation Index

| Document | Purpose |
|---|---|
| [`PROJECT_AUDIT.md`](PROJECT_AUDIT.md) | Full pre-upgrade codebase assessment |
| [`SYSTEM_ARCHITECTURE.md`](SYSTEM_ARCHITECTURE.md) | Existing & proposed target architecture |
| [`IMPLEMENTATION_ROADMAP.md`](IMPLEMENTATION_ROADMAP.md) | 8-phase upgrade roadmap |
| [`TECHNICAL_DECISIONS.md`](TECHNICAL_DECISIONS.md) | Architectural decision records (ADR-01–10) |
| [`QUANTITATIVE_METHODOLOGY.md`](QUANTITATIVE_METHODOLOGY.md) | All financial formulas with derivations |
| [`ML_METHODOLOGY.md`](ML_METHODOLOGY.md) | ML models, features, walk-forward validation |
| [`AI_GROUNDING.md`](AI_GROUNDING.md) | AI safety, grounding, RAG architecture |
| [`DATA_PIPELINE.md`](DATA_PIPELINE.md) | Data engineering and quality framework |
| [`API_DOCUMENTATION.md`](API_DOCUMENTATION.md) | Complete REST API reference |
| [`SECURITY.md`](SECURITY.md) | Security architecture and production checklist |
| [`DEPLOYMENT.md`](DEPLOYMENT.md) | Local, Docker, and AWS deployment guide |
| [`TESTING.md`](TESTING.md) | Test suite structure and coverage guide |

---

## ⚠️ Disclaimers

- This platform is built for **educational and portfolio management research purposes**
- All risk metrics are based on historical data and do not constitute financial advice
- Monte Carlo simulations and stress scenarios are illustrative and not forecasts
- The AI Risk Analyst does not issue buy/sell/hold recommendations
- Always consult a qualified financial professional for investment decisions
