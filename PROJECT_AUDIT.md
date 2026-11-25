# PROJECT AUDIT: FinRisk AI Platform Assessment

**Auditor:** Quantitative Risk, ML, and Systems Architecture Engineering  
**Date:** November 2026  
**Repository:** FinRisk AI — Explainable AI Portfolio Risk & Stress Testing Platform  
**Target Architecture:** Production-Grade Multi-Asset Portfolio Risk Intelligence Platform

---

## 1. Executive Summary

This audit presents an exhaustive line-by-line inspection of the existing **FinRisk AI** codebase. The objective is to identify mathematical defects, architectural bottlenecks, data pipeline vulnerabilities, security gaps, and missing production components before embarking on incremental development.

The existing repository demonstrates a functional prototype combining FastAPI, Streamlit, SQLAlchemy, and basic statistical and machine learning scripts. However, it currently falls short of an institutional-grade financial risk management platform:
1. **Mathematical Scope & Validation:** Only basic historical VaR, Expected Shortfall, and single-factor Beta are implemented. Parametric VaR, Monte Carlo simulations, Kupiec/Christoffersen regulatory backtesting, downside risk metrics (Sortino/Calmar), and risk contributions under non-linear conditions are absent.
2. **Stress Testing Limitations:** The stress test engine applies static arbitrary percentages to a hardcoded set of 7 tickers (`AAPL, MSFT, NVDA, JPM, XOM, JNJ, SPY`). True historical scenario replays (2008 GFC, 2020 COVID, 2022 Rate Shock, 2023 SVB Crisis) and reverse stress testing do not exist.
3. **Data Pipeline Fragility:** The application relies on reading a static CSV file (`data/sample/historical_prices.csv`) from disk on almost every API call. Although an ORM model `MarketPrice` exists, it is bypassed by core risk calculations. Corporate actions, price adjustments, dividend yields, and macroeconomic factors are unhandled.
4. **Machine Learning Integrity:** The anomaly detection and regime classification engines fit models on the fly without walk-forward cross-validation, creating data leakage and look-ahead risks. Advanced regime models (GMM, HMM), alternative anomaly detectors (LOF, Robust Z-score/MAD, CUSUM change points), and SHAP-based feature attributions are missing.
5. **AI Grounding & Hallucination Prevention:** The current validation function (`validate_ai_output`) performs a rudimentary regex search for dollar signs and returns `True` unconditionally, providing no actual validation. There is no conversational interface, no Retrieval-Augmented Generation (RAG) over risk policies, and no structured JSON Pydantic output validation.
6. **Backend & Security:** The API lacks authentication, role-based access control (RBAC), multi-tenancy, rate limiting, and asynchronous background worker queues (Celery/Redis) for compute-intensive tasks like Monte Carlo simulations.

---

## 2. Inventory of Existing Codebase

| Path | Lines | Primary Purpose | Status / Quality Assessment |
| :--- | :--- | :--- | :--- |
| `backend/app/main.py` | 66 | FastAPI entry point, CORS, table creation, route mounting | Functional prototype; seeds DB on startup if empty; CORS wide open (`*`); lacks centralized error handling. |
| `backend/app/config.py` | 29 | Pydantic `BaseSettings` for app configuration | Minimal; missing Redis, Celery, JWT secrets, database connection pool tuning, and AWS parameters. |
| `backend/app/models/database.py` | 45 | SQLAlchemy engine & session factory | Basic; global singleton engine; silent fallback to SQLite masks PostgreSQL connection failures. |
| `backend/app/models/portfolio.py` | 66 | Models: `Portfolio`, `Holding`, `RiskMetric`, `AIReport` | Clean schema, but `RiskMetric` records are never persisted by calculations; `AIReport.risk_metrics_id` is left `NULL`. |
| `backend/app/models/market.py` | 35 | Models: `Asset`, `MarketPrice`, `MLPrediction` | Sound schema, but ignored by `portfolio_service.py` which reads CSV directly from disk. |
| `backend/app/schemas/portfolio.py` | 146 | Pydantic request/response models | Good baseline validations (no duplicate tickers, positive shares); lacks parametric VaR and Monte Carlo schemas. |
| `backend/app/schemas/ai.py` | 38 | Pydantic schemas for AI Analyst and ML endpoints | Static response models; lacks multi-turn chat, RAG sources, and structured validation. |
| `backend/app/risk_engine/returns.py` | 44 | Asset return calculation & portfolio weighted return | Simple return calculation $(P_t - P_{t-1})/P_{t-1}$; does not support log returns or currency alignment. |
| `backend/app/risk_engine/metrics.py` | 167 | Quant metrics: Volatility, Sharpe, Beta, VaR, ES, MDD, PCR | Mathematically sound for basics, but lacks parametric VaR, Monte Carlo, GARCH, Sortino, and backtesting. |
| `backend/app/risk_engine/stress.py` | 152 | Stress testing engine with predefined scenarios | Rigid hardcoded tickers; lacks true historical crisis time-series replay, macro factor shocks, and reverse stress testing. |
| `backend/app/ml_engine/features.py` | 80 | Technical features for ML models | Generates 7 basic rolling features on SPY; no cross-asset features, macro yields, or credit spreads. |
| `backend/app/ml_engine/anomaly.py` | 138 | Isolation Forest anomaly detector | Functional but static contamination; lacks LOF, MAD, change-point detection, and SHAP explainability. |
| `backend/app/ml_engine/regime.py` | 147 | K-Means regime detection with deterministic vol sorting | Ingenious deterministic cluster sorting, but K-Means lacks probabilistic transitions; missing GMM and HMM. |
| `backend/app/ai_engine/provider.py` | 365 | LLM Provider Strategy (Gemini, OpenAI, Mock) | Good fallback design, but validation is a dummy stub that always returns `True`; lacks RAG and strict citation. |
| `backend/app/repositories/portfolio_repo.py` | ~60 | Data access for portfolios and holdings | Standard CRUD operations; lacks tenant/user isolation. |
| `backend/app/services/portfolio_service.py` | 273 | Business logic orchestrator | Re-reads CSV on every call; recomputes all historical series synchronously; hardcodes 7 supported tickers. |
| `backend/app/services/market_service.py` | 142 | Market data provider with DB/CSV fallback | Module-level singleton session leaks DB connections; unused by `portfolio_service.py`. |
| `frontend/app.py` | 538 | Streamlit user interface | Good interactive dashboard, but single-threaded, slow on recalculations, and tied to Streamlit's execution model. |
| `frontend/utils/api_client.py` | 73 | HTTP client for backend API | Direct synchronous requests; lacks retry logic, token authentication headers, and timeout resilience. |
| `scripts/seed_db.py` | 104 | Database seeding script from CSV | Seeds database with 7 assets and 3 years of daily prices; basic batching. |
| `scripts/download_real_data.py` | 106 | Script downloading 2021-2024 data via yfinance | Functional; unadjusted for split edge cases beyond yfinance defaults; hardcoded ticker list. |
| `backend/tests/` | 299 | Pytest test suite for risk, ML, and API | Basic unit tests present; zero tests for edge cases (zero vol, negative weights, singular covariance, VaR backtests). |

---

## 3. Detailed Audit by Engineering Domain

### 3.1. Quantitative Finance & Mathematics

#### Mathematical Correctness & Unit Consistency
1. **Annualized Portfolio Volatility:**
   $$\sigma_p = \sqrt{w^T \Sigma w} \times \sqrt{252}$$
   - *Code Implementation:* Correctly computes sample daily covariance matrix $\Sigma_{\text{daily}}$ and scales by $\sqrt{252}$.
   - *Issue:* In `README.md`, $\Sigma$ was described as the "annualized covariance matrix", which created ambiguity. In code, if $\Sigma$ had been annualized, multiplying by $\sqrt{252}$ would double-annualize.
   - *Limitation:* Does not handle ill-conditioned or non-positive semi-definite covariance matrices (e.g., when assets exceed observations or assets are collinear). Needs Ledoit-Wolf shrinkage or eigenvalue clipping.
2. **Annualized Return & Sharpe Ratio:**
   $$\text{AnnReturn} = \bar{R}_p \times 252, \quad \text{Sharpe} = \frac{\text{AnnReturn} - R_f}{\sigma_p}$$
   - *Issue:* Uses simple arithmetic mean annualization rather than compound annual growth rate (CAGR) or log-return compounding. Over long horizons with high volatility, arithmetic annualization significantly overestimates realized geometric return.
3. **Historical Value at Risk (VaR):**
   - *Code Implementation:* Uses `np.percentile(portfolio_returns, (1 - confidence_level) * 100)`.
   - *Issue:* VaR is returned as `-var_percentile`. If market returns are overwhelmingly positive in a bull market, `-var_percentile` can be negative (implying a profit, not a risk of loss).
   - *Missing:* Parametric Gaussian VaR ($-\mu + z_{\alpha}\sigma$), Cornish-Fisher expansion for skewness/kurtosis, and Monte Carlo VaR.
4. **Expected Shortfall (CVaR):**
   - *Code Implementation:* Filters `portfolio_returns <= -var_threshold`.
   - *Issue:* In discrete sample percentiles, strict inequality vs weak inequality can bias ES. If no points fall below the threshold (e.g. single outlier), it falls back to `min()`.
   - *Missing:* Spectral risk measures and parametric ES.
5. **Portfolio Beta:**
   $$\beta_p = \frac{\text{Cov}(R_p, R_m)}{\text{Var}(R_m)}$$
   - *Issue:* Beta is computed against `SPY`. If `SPY` variance is $< 10^{-8}$, it returns `0.0`. It does not calculate rolling beta or multi-factor betas (Fama-French 3-factor / Carhart 4-factor).
6. **Risk Decomposition:**
   $$\text{MCR} = \frac{\Sigma w}{\sigma_p}, \quad \text{CCR}_i = w_i \times \text{MCR}_i, \quad \text{PCR}_i = \frac{\text{CCR}_i}{\sigma_p}$$
   - *Code Implementation:* Correctly implemented using Euler's theorem for homogeneous risk measures. $\sum \text{PCR}_i = 1.0$.
   - *Limitation:* Only valid for volatility. Does not decompose Component VaR or Component Expected Shortfall.
7. **Missing Model Validation:**
   - No Kupiec Proportion of Failures (POF) likelihood ratio test.
   - No Christoffersen Markov Independence test.
   - No rolling-window VaR backtesting reports (Basel regulatory traffic light framework).

### 3.2. Machine Learning & Statistical Modeling

1. **Lookahead Bias & Data Leakage:**
   - `train_regime_detector` and `train_anomaly_detector` fit scalers and models on the entire time series (`prices_df`) before predicting.
   - In production, models must be evaluated using walk-forward (expanding or rolling window) validation.
2. **Model Selection:**
   - *Regime Detection:* Only K-Means is implemented. K-Means imposes hard hyperspherical boundaries with no temporal persistence or regime transition probabilities. Hidden Markov Models (HMM) and Gaussian Mixture Models (GMM) are required for institutional modeling.
   - *Anomaly Detection:* Only Isolation Forest is provided with a fixed contamination parameter (0.03). It lacks density-based models (Local Outlier Factor), distribution-free robust statistics (MAD / Robust Z-score), and CUSUM change-point detection.
3. **Explainability:**
   - Anomaly detection outputs an `anomaly_score` without attributing *which features* (e.g., volume spike, return collapse, volatility expansion) triggered the isolation. TreeSHAP or Shapley values are missing.

### 3.3. Stress Testing Engine

1. **Hardcoded Asset Scenarios:**
   - `PREDEFINED_SCENARIOS` in `backend/app/risk_engine/stress.py` has shock dictionaries containing only `["AAPL", "MSFT", "NVDA", "JPM", "XOM", "JNJ", "SPY"]`.
   - Any portfolio containing other tickers (e.g., `AMZN`, `GOOGL`, `TLT`) receives 0% shock unless custom shocks are specified.
2. **Absence of Historical Replay:**
   - Scenarios like "Market Crash -10%" are synthetic toy shocks.
   - Institutional stress testing requires exact historical time-window replay:
     - 2008 Lehman Brothers Collapse (Sep 2008 – Nov 2008)
     - 2020 COVID-19 Liquidity Shock (Feb 2020 – Mar 2020)
     - 2022 Inflation & Rate Hike Shock (Jan 2022 – Oct 2022)
     - 2023 Silicon Valley Bank Regional Banking Crisis (Mar 2023)
3. **Absence of Reverse Stress Testing:**
   - No algorithm exists to search for the combination of shocks (or macro factor movement) that causes a given portfolio drawdown threshold (e.g., -25% loss).

### 3.4. AI Grounding & RAG Architecture

1. **Validation Vulnerability:**
   - `validate_ai_output()` in `provider.py` contains:
     ```python
     if not dollar_amounts:
         return True
     ...
     return True # Unconditionally returns True!
     ```
   - Even if the AI outputs completely fabricated dollar amounts or risk metrics, the system logs a silent warning and marks `is_validated = True` in the database!
2. **No Policy Grounding (Missing RAG):**
   - The platform has no retrieval layer to ground responses in institutional risk policies (e.g., maximum permissible single-asset concentration of 20%, stop-loss limits, VaR tolerance limits).
   - Responses are generated purely from prompt templates.
3. **Lack of Conversational Memory & Multi-Turn Capability:**
   - The API only offers a one-shot `POST /api/ai/risk-analysis`. Users cannot ask follow-up questions such as "Why is NVDA's contribution so high?" or "What hedge would reduce portfolio volatility to 12%?".

### 3.5. Data Engineering & Market Feeds

1. **Static CSV Dependency:**
   - `portfolio_service.py` executes `pd.read_csv("data/sample/historical_prices.csv")` on every request.
   - This discards the database, introduces disk I/O latency, prevents scaling across multiple workers, and limits the asset universe to the 7 tickers in the CSV.
2. **Lack of Ingestion Pipeline & Quality Checks:**
   - No automated pipeline to fetch new daily market data, handle missing business days, detect stock splits/dividends, or alert on anomalous zero prices.
   - No integration with macroeconomic feeds (e.g., FRED for Treasury yields, CPI, and Fed Funds Rate).

### 3.6. Backend Architecture & Security

1. **Authentication & Multi-Tenancy:**
   - Zero authentication: no JWT, OAuth2, or API keys.
   - Any user can list, view, modify, or delete any portfolio.
   - No concept of tenants, organizations, or user ownership.
2. **Database Engine & Connection Leaks:**
   - `market_service.py` instantiates `SessionLocal()` as a module-level attribute, which remains open indefinitely.
   - Database errors trigger an automatic fallback to a local SQLite database, masking configuration or infrastructure bugs in production.
3. **Synchronous Execution:**
   - Long-running quant calculations (e.g., 10,000 Monte Carlo paths, backtesting across 1,000 trading days) run synchronously on the FastAPI thread pool, blocking HTTP workers. Celery + Redis task queue is absent.

### 3.7. Frontend Architecture

1. **Streamlit Limitations:**
   - The current UI is a single monolithic Streamlit script (`frontend/app.py`, 538 lines).
   - Streamlit re-executes the entire script on any widget state change, making large portfolio views sluggish.
   - Does not meet modern institutional UX standards for financial dashboards (responsive multi-panel layouts, high-performance data grids, customizable workspaces).

---

## 4. Test Suite Audit

Running an inventory of existing tests in `backend/tests/`:
- `test_risk_engine.py`: 8 basic tests covering standard arithmetic cases.
- `test_ml_engine.py`: 3 tests covering feature creation, anomaly shape, and regime order.
- `test_api.py`: 3 tests for health check, CRUD lifecycle, and mock AI endpoint.

### Critical Missing Test Coverage:
1. **Mathematical Edge Cases:**
   - Singular covariance matrix (collinear assets).
   - Single-asset portfolio (weight = 1.0).
   - Zero volatility asset (cash / risk-free proxy).
   - Short positions or negative weights.
   - Insufficient historical observations ($N < K$).
2. **Regulatory Backtesting Tests:**
   - Kupiec test likelihood ratio accuracy.
   - Christoffersen independence test p-values.
3. **Stress Testing Tests:**
   - Tickers not present in predefined scenarios.
   - Extreme shocks (e.g., -100% loss).
4. **Security & Validation Tests:**
   - SQL injection / XSS / prompt injection attempts.
   - Unauthenticated route access.
   - Cross-tenant data isolation.

---

## 5. Audit Conclusions & Prioritized Action Plan

The platform has a clear conceptual premise: **deterministic quantitative calculation separated from natural language explanation**. However, the current implementation is an early-stage prototype with critical gaps in mathematical breadth, stress testing realism, data engineering, model validation, and security.

### Prioritized Upgrade Roadmap:
1. **Phase 1: Mathematical Foundations & Quant Risk Engine Upgrade**
   - Implement Parametric VaR, Cornish-Fisher VaR, and Monte Carlo VaR (10,000+ paths).
   - Implement Expected Shortfall under multiple distributional assumptions.
   - Add Sortino, Calmar, EWMA, GARCH(1,1), and rolling metrics.
   - Build Kupiec POF and Christoffersen backtesting validation suite.
2. **Phase 2: Advanced Stress Testing & Portfolio Optimization**
   - Real historical crisis replay (2008 GFC, 2020 COVID, 2022 Inflation, 2023 SVB).
   - Reverse stress testing with constrained optimization.
   - Markowitz mean-variance optimization, Efficient Frontier, and Risk Parity.
3. **Phase 3: Machine Learning Intelligence & Explainability**
   - Gaussian Mixture Models and Hidden Markov Models for regime switching.
   - Local Outlier Factor and Robust Z-score anomaly detection.
   - SHAP-based feature importance and anomaly attribution.
4. **Phase 4: Evidence-Grounded AI Risk Analyst & RAG**
   - Real Pydantic structured output validation with deterministic rejection.
   - Embeddings and pgvector-based RAG pipeline for risk mandates and policy limits.
   - Interactive conversational assistant with strict metric citations.
5. **Phase 5: Data Engineering & Market Data Ingestion**
   - Modular data provider architecture (Yahoo Finance, FRED, CSV, Mock).
   - Automated data validation using Pandera / strict schema checks.
   - Persistence of all calculations and market data in PostgreSQL.
6. **Phase 6: Backend Enterprise Hardening & Background Processing**
   - Multi-tenant data model with user authentication and RBAC.
   - Asynchronous job execution using Celery and Redis.
   - Database migrations with Alembic and structured logging.
7. **Phase 7: Professional React + TypeScript Frontend**
   - Modern institutional dashboard with Tailwind CSS and Plotly.
   - Modular views: Overview, Risk Analytics, Stress Testing, ML Intelligence, AI Analyst.
8. **Phase 8: Production Deployment, CI/CD & Verification**
   - Production Docker configurations, AWS ECS/RDS/Redis Terraform templates, and GitHub Actions CI/CD.
