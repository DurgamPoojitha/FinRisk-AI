# TECHNICAL DECISIONS RECORD (ADR)

**Project:** FinRisk AI — Explainable Multi-Asset Portfolio Risk Intelligence Platform  
**Status:** Approved & Implemented  
**Date:** November 2026  

This document articulates the key architectural, mathematical, and technological decisions made during the evolution of FinRisk AI into an institutional-grade platform.

---

## 1. Architectural & Design Decisions

### ADR-01: Decoupling Deterministic Computation from LLM Reasoning
- **Context:** Financial risk metrics require absolute mathematical precision. Direct computation by Large Language Models leads to statistical hallucinations, arithmetic errors, and inconsistent reporting.
- **Decision:** Implement a strict, unidirectional pipeline:
  1. The **Quantitative Risk Engine** performs all calculations using vectorized NumPy/SciPy/pandas code.
  2. The **Evidence Layer** validates, freezes, and serializes the calculations into an immutable `RiskEvidence` object with a calculation UUID.
  3. The **AI Engine** acts strictly as an interpretation and explanation interface, receiving the frozen evidence as context.
  4. The **Validation Layer** parses the generated response using Pydantic models and checks numerical citations against the ground-truth evidence before persisting the report.
- **Consequences:** Eliminates quantitative hallucinations; ensures regulatory compliance, repeatability, and auditable lineage.

---

### ADR-02: Multi-Tenancy and Role-Based Access Control (RBAC)
- **Context:** Institutional financial institutions require strict tenant segregation and granular user permissions so that sensitive portfolio holdings and risk reports cannot be viewed across institutions or unauthorized teams.
- **Decision:**
  - Introduce `tenant_id` foreign keys indexed across all primary database tables (`users`, `portfolios`, `holdings`, `risk_metrics`, `risk_documents`).
  - Implement JWT authentication with role claims.
  - Define four core operational roles:
    1. `Administrator`: Platform configuration, user onboarding, audit logs.
    2. `RiskManager`: Portfolio creation, custom scenario modeling, stress parameter overrides, policy limit management.
    3. `Analyst`: Standard portfolio viewing, reporting, AI interactions.
    4. `Auditor`: Read-only access to immutable historical calculation snapshots and AI audit trails.
- **Consequences:** Ensures compliance with financial confidentiality standards and enables seamless multi-client SaaS or multi-department deployment.

---

### ADR-03: Asynchronous Task Processing with Celery & Redis
- **Context:** Monte Carlo simulations with 10,000+ paths, walk-forward backtests across years of daily data, and large reverse stress tests can take multiple seconds, blocking FastAPI's synchronous request-response cycle and triggering HTTP gateway timeouts.
- **Decision:**
  - Introduce Redis as a distributed message broker and cache.
  - Use Celery to offload heavy computational workloads:
    - Monte Carlo path generation.
    - Historical multi-window VaR backtesting.
    - Nightly market data ingestion and quality verification.
  - Expose asynchronous job tracking endpoints (`/api/jobs/{id}`) allowing clients to poll or listen for completion.
- **Consequences:** Maintains high API responsiveness, scales worker pools independently from web servers, and prevents web worker resource starvation.

---

## 2. Quantitative & Mathematical Decisions

### ADR-04: Multi-Methodology Value at Risk (VaR) Engine
- **Context:** Relying solely on historical simulation VaR assumes that future price movements will strictly resemble past empirical distributions, failing to capture distribution fat tails, skewness, or extreme black swan events.
- **Decision:** Implement three complementary VaR methodologies:
  1. **Historical Simulation VaR:** Empirical percentile of daily returns:
     $$\text{VaR}_{\text{Hist}}(\alpha) = -Q_{\alpha}(R_p)$$
  2. **Parametric VaR (Normal & Cornish-Fisher):**
     $$\text{VaR}_{\text{Normal}}(\alpha) = -(\mu_p + z_{\alpha}\sigma_p)$$
     $$\text{VaR}_{\text{CF}}(\alpha) = -\left(\mu_p + \left[z_{\alpha} + \frac{S}{6}(z_{\alpha}^2 - 1) + \frac{K}{24}(z_{\alpha}^3 - 3z_{\alpha}) - \frac{S^2}{36}(2z_{\alpha}^3 - 5z_{\alpha})\right]\sigma_p\right)$$
     where $S$ is return skewness and $K$ is excess kurtosis.
  3. **Monte Carlo VaR:** 10,000+ correlated paths generated via Cholesky factorization of the empirical covariance matrix $\Sigma = L L^T$:
     $$R_{\text{sim}} = \mu + L Z, \quad Z \sim \mathcal{N}(0, I)$$
- **Consequences:** Enables comprehensive risk analysis across Gaussian, non-Gaussian, and simulated tail regimes.

---

### ADR-05: Regulatory Model Validation Suite (Kupiec & Christoffersen)
- **Context:** Institutional risk models must be validated against actual historical losses to prove they neither systematically underestimate nor excessively overestimate risk.
- **Decision:** Implement formal statistical backtesting tests:
  1. **Kupiec Proportion of Failures (POF) Test:** Likelihood ratio test evaluating whether the observed exception rate $p = x/N$ equals the expected failure rate $p_0 = 1 - \alpha$:
     $$\text{LR}_{\text{POF}} = -2 \ln \left[ \frac{(1 - p_0)^{N-x} p_0^x}{(1 - x/N)^{N-x} (x/N)^x} \right] \sim \chi^2(1)$$
  2. **Christoffersen Independence Test:** Likelihood ratio test evaluating whether VaR exceedances are independent across time or occur in clusters (volatility clustering):
     $$\text{LR}_{\text{ind}} = -2 \ln \left[ \frac{\pi_0^{n_{00} + n_{10}} (1 - \pi_0)^{n_{01} + n_{11}}}{\pi_{01}^{n_{01}} (1 - \pi_{01})^{n_{00}} \pi_{11}^{n_{11}} (1 - \pi_{11})^{n_{10}}} \right] \sim \chi^2(1)$$
  3. **Basel Committee Traffic Light Framework:** Automated classification of 250-day exception counts into Green, Yellow, and Red zones.
- **Consequences:** Provides institutional credibility and automated regulatory model validation reporting.

---

### ADR-06: Volatility Modeling (EWMA & GARCH)
- **Context:** Constant sample standard deviation assumes homoskedasticity (constant volatility), which fails during market turbulence.
- **Decision:**
  - Implement **EWMA (Exponentially Weighted Moving Average)** with RiskMetrics parameter $\lambda = 0.94$:
    $$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) r_{t-1}^2$$
  - Implement **GARCH(1,1)** estimated via Maximum Likelihood Estimation:
    $$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2, \quad \omega > 0, \; \alpha \ge 0, \; \beta \ge 0, \; \alpha + \beta < 1$$
- **Consequences:** Delivers accurate forward-looking volatility forecasts and captures volatility clustering.

---

### ADR-07: Real Historical Crisis Replay and Reverse Stress Testing
- **Context:** Simple percentage price declines (e.g., -10% across all assets) ignore real-world correlation breakdowns, liquidity evaporation, and sector divergences.
- **Decision:**
  - Curate exact historical crisis return windows:
    - 2008 Lehman Bankruptcy (GFC).
    - 2020 COVID-19 Liquidity Shock.
    - 2022 Fed Rate Hike & Inflation Shock.
    - 2023 SVB Regional Banking Collapse.
  - Implement **Reverse Stress Testing**: An optimization solver that identifies the minimal multi-factor or asset-level shock vector that breaches a user-defined portfolio loss threshold (e.g. 25% or 40% loss), highlighting portfolio vulnerabilities.
- **Consequences:** Shifts stress testing from naive toy scenarios to realistic institutional risk diagnosis.

---

## 3. Machine Learning Decisions

### ADR-08: Hidden Markov Models & Gaussian Mixture Models for Market Regimes
- **Context:** K-Means clustering assumes spherical, equal-variance clusters and lacks probabilistic regime persistence or transition matrices.
- **Decision:**
  - Retain K-Means as a fast baseline with deterministic volatility sorting.
  - Implement **Hidden Markov Models (HMM)** with Gaussian emissions to model latent unobserved market states (Low, Medium, High Volatility) and estimate transition probability matrices $P(S_t \mid S_{t-1})$.
  - Implement **Gaussian Mixture Models (GMM)** with soft probability assignments.
- **Consequences:** Reflects authentic financial market dynamics where market volatility exhibits persistence and regime switching.

---

### ADR-09: Multi-Model Anomaly Detection with Feature Attributions
- **Context:** A single anomaly detector (Isolation Forest) with fixed contamination can yield false positives and lacks explainability regarding *why* an anomaly was flagged.
- **Decision:**
  - Supplement Isolation Forest with **Local Outlier Factor (LOF)** and **Robust Z-Score (MAD)**.
  - Compute feature attributions (feature delta against moving medians and tree-based importance) so every anomaly is accompanied by an explanation of the primary contributing driver (e.g., return collapse, volatility spike, or correlation breakdown).
- **Consequences:** Eliminates black-box ML and provides actionable risk intelligence.

---

## 4. Frontend & User Experience Decisions

### ADR-10: Modern React + TypeScript Dashboard
- **Context:** While Streamlit is effective for rapid prototyping, it re-runs the entire application on every user interaction, lacks fine-grained state management, and cannot deliver high-performance institutional trading/risk interfaces.
- **Decision:**
  - Retain Streamlit as an exploratory research sandbox.
  - Build a professional **React 18 + TypeScript + Vite + Tailwind CSS** single-page application.
  - Use React Query for intelligent caching, background refetching, and optimistic updates.
  - Use Plotly / Recharts for interactive financial charts (efficient frontier, drawdown charts, correlation heatmaps, Monte Carlo fan charts).
- **Consequences:** Provides a polished, responsive, and responsive institutional experience suitable for risk committees, senior executives, and quantitative analysts.
