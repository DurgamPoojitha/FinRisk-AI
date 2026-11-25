# SYSTEM ARCHITECTURE: FinRisk AI Platform

## 1. Architectural Overview

FinRisk AI is an institutional-grade, multi-asset portfolio risk intelligence and stress testing platform. Its core architectural tenet is the **strict separation of concerns**:
- **Deterministic Computational Layer:** All quantitative finance formulas, risk statistics, and simulation paths are computed by vectorized, auditable Python code (NumPy, SciPy, pandas).
- **Evidence Layer:** Calculation results are frozen into immutable, versioned evidence payloads with unique cryptographic or UUID identifiers.
- **Explainability & Generative Layer:** Large Language Models (LLMs) interpret, summarize, and contextualize verified calculations without performing independent arithmetic.
- **Retrieval-Augmented Generation (RAG):** Context is enriched by retrieving relevant regulatory guidelines, investment mandates, and institutional risk limits stored in a vector-enabled database.

---

## 2. Existing vs. Proposed Architecture

### 2.1. Current (Baseline) Architecture

```
                                 [ Browser ]
                                      |
                           Streamlit Dashboard (8501)
                                      | (Synchronous HTTP)
                                      v
                             FastAPI Backend (8000)
                                      |
         +----------------------------+----------------------------+
         |                            |                            |
         v                            v                            v
    Risk Engine                   ML Engine                    AI Engine
(Historical VaR, ES,          (Isolation Forest,            (Prompt Template,
 Volatility, Sharpe, Beta)     K-Means Regimes)              Gemini/OpenAI/Mock)
         |                            |                            |
         +----------------------------+----------------------------+
                                      |
                                      v
                               Disk / CSV File
                       (data/sample/historical_prices.csv)
                                      |
                                 (Secondary)
                                SQLite / Postgres
```

#### Key Flaws of the Current Architecture:
1. **Synchronous Bottleneck:** Heavy statistical operations (Monte Carlo, rolling backtests) block the single FastAPI process.
2. **Data Decoupling:** Database tables (`MarketPrice`, `RiskMetric`) are bypassed; raw CSV files are parsed on every request.
3. **Absence of Tenant Isolation:** Portfolios are globally accessible by any user without authentication.
4. **Shallow AI Validation:** Validation returns `True` unconditionally, risking unchecked hallucination propagation.

---

### 2.2. Proposed Target Architecture

```
                                  [ Users & Risk Managers ]
                                              |
                          +-------------------+-------------------+
                          |                                       |
                          v                                       v
               React 18 + TypeScript + Vite             Streamlit Prototype
               (Institutional Risk Dashboard)           (Interactive Sandbox)
                          |                                       |
                          +-------------------+-------------------+
                                              |
                                              | (HTTPS / Bearer JWT)
                                              v
                                   Application Load Balancer
                                              |
                                              v
                                  FastAPI Gateway Cluster
                        [ Auth | RBAC | Tenant Isolation | Rate Limiting ]
                                              |
                       +----------------------+----------------------+
                       |                                             |
                       v (Synchronous Fast Path)                     v (Async Heavy Path)
              Read/Write Services                              Celery Task Queue
         [ Portfolio | Market | Reports ]                     [ Distributed Workers ]
                       |                                             |
                       |                                  +----------+----------+
                       |                                  |          |          |
                       |                                  v          v          v
                       |                             Monte Carlo  GARCH/HMM  Stress Test
                       |                             Simulation   Forecasting Engine
                       |                                  |          |          |
                       +----------------------------------+----------+----------+
                                              |
                                              v
                                      Core Service Mesh
             +--------------------+--------------------+--------------------+
             |                    |                    |                    |
             v                    v                    v                    v
     Quantitative Risk        ML Intelligence       Market Ingestion     AI Risk Analyst
          Engine                  Engine               Pipeline            (RAG & LLM)
     - Parametric VaR         - HMM / GMM Regimes   - YFinance API       - LangChain/Llama
     - Cornish-Fisher VaR     - LOF / Robust MAD    - FRED Macro Feed    - pgvector Embeddings
     - Monte Carlo (10k+)     - CUSUM Change-point  - Data Quality/      - Pydantic Guardrails
     - Kupiec/Christoffersen  - SHAP Explainability   Pandera Schema     - Full Audit Trail
     - GARCH(1,1) & EWMA      - Walk-Forward Split  - Corporate Actions  - Multi-turn Chat
     - Risk Attribution       - Drift Monitor       - S3 / MinIO Store   - Citation Verifier
             |                    |                    |                    |
             +--------------------+--------------------+--------------------+
                                              |
                       +----------------------+----------------------+
                       |                                             |
                       v                                             v
              PostgreSQL 16 + pgvector                       Redis 7 Cluster
          - Relational Portfolios & Holdings             - Celery Broker & Result Backend
          - Historical Daily Market Prices               - Distributed Caching (Prices/Vols)
          - Immutable Risk Calculations                  - API Rate Limiting States
          - Audited AI Reports & Evidence Snapshots
          - Vector Embeddings for Risk Mandates
```

---

## 3. Detailed Component Decomposition

### 3.1. API & Backend Service Layer (FastAPI)
- **Framework:** FastAPI with asynchronous ASGI execution (`uvicorn`).
- **Pydantic Validation (v2):** Strict type safety, input coercion, and response serialization.
- **Authentication & RBAC:**
  - JWT-based authentication supporting four distinct roles:
    - `Administrator`: Full system and user administration.
    - `RiskManager`: Portfolio creation, custom scenario modeling, stress parameter editing, mandate definition.
    - `Analyst`: Portfolio viewing, standard report generation, AI interaction.
    - `Auditor`: Read-only access to immutable risk snapshots and AI validation audit logs.
- **Tenant Isolation:** Multi-tenancy enforced at the database query layer (`tenant_id` indexed on all primary tables) with strict row-level security principles.

### 3.2. Asynchronous Execution Engine (Celery + Redis)
- **Message Broker:** Redis 7 with dedicated logical DB partitions for cache and broker.
- **Task Workers:** Celery workers configured for:
  - Parallel Monte Carlo simulations (batched vector random generation).
  - Nightly market data ingestion and quality reporting.
  - Model backtesting (rolling historical windows).
  - Vector document indexing.
- **Status Polling & Webhooks:** REST endpoints (`/api/jobs/{task_id}`) returning real-time execution progress, status, and payload pointers.

### 3.3. Quantitative Risk & Model Validation Engine
- **Core Math Library:** NumPy, SciPy (optimization, statistical distributions), pandas.
- **Risk Calculation Modules:**
  1. `VaREngine`: Historical, Parametric (Normal & Student's t), Cornish-Fisher modified VaR, Monte Carlo VaR.
  2. `ExpectedShortfallEngine`: Historical tail average, parametric Student's t CVaR, Monte Carlo CVaR.
  3. `VolatilityEngine`: Sample annualized, EWMA ($\lambda=0.94$), GARCH(1,1) maximum likelihood estimation.
  4. `AttributionEngine`: Marginal Contribution to Risk (MCR), Component Contribution to Risk (CCR), Percentage Contribution to Risk (PCR).
  5. `ValidationEngine`: Kupiec Proportion of Failures (POF) test, Christoffersen Independence test, Basel regulatory traffic light categorization.

### 3.4. Scenario Stress Testing & Optimization Engine
- **Historical Crisis Replay:** Real market return windows preserved and mapped:
  - 2008 Lehman GFC (2008-09-01 to 2008-11-30).
  - 2020 COVID Crash (2020-02-19 to 2020-03-23).
  - 2022 Inflation & Rate Shock (2022-01-03 to 2022-10-14).
  - 2023 SVB Regional Banking Crisis (2023-03-08 to 2023-03-24).
- **Hypothetical Multi-Factor Shocks:** Asset shocks, sector shocks, interest rate duration shifts, and correlation matrix stress (eigenvalue perturbation).
- **Reverse Stress Testing:** Constrained quadratic/SLSQP optimization solving for the minimal multi-asset shock vector required to breach a specified portfolio loss threshold.
- **Portfolio Optimization:** Markowitz mean-variance optimization, Maximum Sharpe ratio, Minimum Variance portfolio, and Risk Parity allocation.

### 3.5. Machine Learning Intelligence & Explainability Engine
- **Regime Classification:**
  - Hidden Markov Models (`hmmlearn`) for probabilistic regime transitions.
  - Gaussian Mixture Models (`sklearn.mixture.GaussianMixture`) with BIC/AIC selection.
  - K-Means with deterministic volatility-based sorting for fast baseline comparison.
- **Anomaly Detection:**
  - Isolation Forest with dynamic contamination.
  - Local Outlier Factor (LOF) for density-based anomaly scoring.
  - Robust Z-scores via Median Absolute Deviation (MAD).
  - CUSUM (cumulative sum) change-point detection for structural breaks in volatility.
- **Model Explainability & Drift:**
  - Feature contribution attribution via SHAP values or feature deltas.
  - Walk-forward out-of-sample backtesting to eliminate look-ahead bias.

### 3.6. Evidence Layer, RAG & AI Risk Analyst
- **Evidence Freezing:** Every analysis captures a point-in-time calculation snapshot containing:
  - Calculation UUID.
  - Full parameter set (confidence level, horizon, risk-free rate, data window).
  - Calculated risk statistics, risk contributions, anomalies, and regimes.
- **RAG Pipeline:**
  - Document parser for institutional risk policies, mandates, and regulatory docs.
  - Text splitting and embedding generation.
  - PostgreSQL `pgvector` index (Cosine similarity / IVFFlat / HNSW).
  - Hybrid retrieval: semantic vector search combined with metadata filtering.
- **AI Safety & Grounding Layer:**
  - Prompt injection screening.
  - Strict system prompt with forbidden actions (no trade execution, no price fabrication).
  - Pydantic structured output validation: parses LLM output into typed structures and verifies cited numbers against frozen evidence within configurable tolerance thresholds.

### 3.7. Data Engineering & Market Feeds
- **Provider Abstraction:** Abstract `MarketDataProvider` class implemented by:
  - `YahooFinanceProvider`: Real daily OHLCV and corporate actions.
  - `FREDProvider`: Federal Reserve macroeconomic indicators (Treasury yields, CPI, Fed Funds).
  - `DatabaseProvider`: Fast cached queries against PostgreSQL `market_prices`.
  - `MockProvider`: Deterministic synthetic multivariate geometric Brownian motion for offline/sandbox testing.
- **Data Quality & Normalization:**
  - Pandera schema validation.
  - Missing business day detection and interpolation policies.
  - Extreme outlier rejection and log-return consistency checks.

---

## 4. Database Schema Design (PostgreSQL + pgvector)

```
+-------------------------------------------------------------+
|                           tenants                           |
|-------------------------------------------------------------|
| id: UUID (PK)                                               |
| name: VARCHAR(100)                                          |
| plan: VARCHAR(50)                                           |
| created_at: TIMESTAMPTZ                                     |
+-------------------------------------------------------------+
                              | 1
                              |
                              | N
+-------------------------------------------------------------+
|                            users                            |
|-------------------------------------------------------------|
| id: UUID (PK)                                               |
| tenant_id: UUID (FK -> tenants.id)                          |
| email: VARCHAR(255) (UNIQUE)                                |
| hashed_password: VARCHAR(255)                               |
| role: VARCHAR(50) (Admin, RiskManager, Analyst, Auditor)    |
| is_active: BOOLEAN                                          |
| created_at: TIMESTAMPTZ                                     |
+-------------------------------------------------------------+
                              | 1
                              |
                              | N
+-------------------------------------------------------------+
|                          portfolios                         |
|-------------------------------------------------------------|
| id: UUID (PK)                                               |
| tenant_id: UUID (FK -> tenants.id)                          |
| user_id: UUID (FK -> users.id)                              |
| name: VARCHAR(150)                                          |
| benchmark_symbol: VARCHAR(10) (Default: 'SPY')              |
| description: TEXT                                           |
| created_at: TIMESTAMPTZ                                     |
| updated_at: TIMESTAMPTZ                                     |
+-------------------------------------------------------------+
         | 1                                      | 1
         |                                        |
         | N                                      | N
+-------------------------+       +------------------------------------+
|        holdings         |       |            risk_metrics            |
|-------------------------|       |------------------------------------|
| id: UUID (PK)           |       | id: UUID (PK)                      |
| portfolio_id: UUID (FK) |       | portfolio_id: UUID (FK)            |
| asset_symbol: VARCHAR   |       | calculation_type: VARCHAR(50)      |
| quantity: NUMERIC(18,6) |       | portfolio_value: NUMERIC(18,2)     |
| buy_price: NUMERIC(18,4)|       | annualized_return: FLOAT           |
| created_at: TIMESTAMPTZ |       | annualized_volatility: FLOAT       |
+-------------------------+       | sharpe_ratio: FLOAT                |
                                  | sortino_ratio: FLOAT               |
                                  | calmar_ratio: FLOAT                |
                                  | beta: FLOAT                        |
                                  | var_95_hist / var_95_param /       |
                                  | var_95_mc: NUMERIC(18,2)           |
                                  | expected_shortfall_95: NUMERIC     |
                                  | max_drawdown: FLOAT                |
                                  | full_metrics_payload: JSONB        |
                                  | calculated_at: TIMESTAMPTZ         |
                                  +------------------------------------+
                                                    | 1
                                                    |
                                                    | N
                                  +------------------------------------+
                                  |             ai_reports             |
                                  |------------------------------------|
                                  | id: UUID (PK)                      |
                                  | portfolio_id: UUID (FK)            |
                                  | risk_metric_id: UUID (FK)          |
                                  | model_provider: VARCHAR(50)        |
                                  | prompt_template_id: VARCHAR(50)    |
                                  | report_markdown: TEXT              |
                                  | cited_metrics: JSONB               |
                                  | is_grounded: BOOLEAN               |
                                  | validation_errors: JSONB           |
                                  | generated_at: TIMESTAMPTZ          |
                                  +------------------------------------+

+-------------------------------------------------------------+
|                       risk_documents                        |
|-------------------------------------------------------------|
| id: UUID (PK)                                               |
| tenant_id: UUID (FK -> tenants.id)                          |
| title: VARCHAR(255)                                         |
| doc_type: VARCHAR(50) (Policy, Mandate, Regulatory)         |
| content_chunk: TEXT                                         |
| chunk_metadata: JSONB                                       |
| embedding: VECTOR(768) (Google Gemini / OpenAI ada-002)     |
| created_at: TIMESTAMPTZ                                     |
+-------------------------------------------------------------+
```

---

## 5. Security & Governance Architecture

1. **Defense-in-Depth:**
   - Transport Layer: Strict HTTPS/TLS 1.3 encryption.
   - Network Security: Fargate containers deployed within private subnets, fronted by an Application Load Balancer with AWS WAF rate limiting and GeoIP filtering.
   - Secrets Management: Zero plaintext credentials in code or version control; dynamic injection via environment variables or AWS Secrets Manager.
2. **Auditability & Regulatory Lineage:**
   - Every quantitative metric record in `risk_metrics` contains the input market data range, portfolio weight snapshot, and calculation engine version.
   - AI outputs store the exact evidence JSON payload sent to the LLM, the model temperature, and validation pass/fail details.
