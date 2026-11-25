# FinRisk AI — API Documentation

> **Version:** 2.0.0  
> **Base URL:** `http://localhost:8000`  
> **OpenAPI Docs:** `http://localhost:8000/docs`  
> **ReDoc:** `http://localhost:8000/redoc`

All endpoints are mounted at both `/api` (backward-compatible) and `/api/v1` (versioned).

---

## Authentication

### `POST /api/auth/register`
Register a new user account.

**Request Body:**
```json
{
  "username": "analyst_one",
  "email": "analyst@finrisk.com",
  "password": "SecurePass123!",
  "role": "analyst"
}
```
**Roles:** `admin` | `risk_manager` | `analyst` | `auditor`

**Response `201`:**
```json
{ "id": "uuid", "username": "analyst_one", "email": "analyst@finrisk.com", "role": "analyst" }
```

---

### `POST /api/auth/login`
Exchange credentials for a JWT access token.

**Request Body:** `application/x-www-form-urlencoded`
```
username=analyst_one&password=SecurePass123!
```

**Response `200`:**
```json
{ "access_token": "eyJ...", "token_type": "bearer" }
```

---

### `GET /api/auth/me`
Return the currently authenticated user.

**Headers:** `Authorization: Bearer <token>`

---

## Portfolio Endpoints

### `GET /api/portfolios`
List all portfolios (filtered to current tenant in multi-tenant deployments).

**Response `200`:** Array of portfolio summary objects.

---

### `POST /api/portfolios`
Create a new portfolio.

**Request Body:**
```json
{
  "name": "Growth Tech Portfolio",
  "holdings": [
    { "asset_symbol": "AAPL", "quantity": 100.0, "buy_price": 145.00 },
    { "asset_symbol": "MSFT", "quantity":  80.0, "buy_price": 240.00 }
  ]
}
```

**Response `201`:** Full portfolio object with `id`.

---

### `GET /api/portfolios/{portfolio_id}`
Return detailed portfolio holdings with current prices, market values, unrealized P&L, and weights.

**Response Fields:** `total_value`, `holdings[]` (each with `current_price`, `market_value`, `unrealized_pnl`, `unrealized_pnl_pct`, `weight`)

---

### `DELETE /api/portfolios/{portfolio_id}`
Delete a portfolio and all its holdings.

---

### `GET /api/portfolios/{portfolio_id}/risk`
Return comprehensive risk metrics for the portfolio.

**Response Fields:**
| Field | Description | Unit |
|---|---|---|
| `annualized_return` | CAGR over data period | decimal |
| `annualized_volatility` | σ × √252 | decimal |
| `sharpe_ratio` | (R − Rf) / σ, Rf = 4% | dimensionless |
| `sortino_ratio` | (R − Rf) / downside-σ | dimensionless |
| `calmar_ratio` | CAGR / max_drawdown | dimensionless |
| `beta` | Portfolio beta vs SPY benchmark | dimensionless |
| `var_95` | Parametric 95% daily VaR | USD |
| `var_95_pct` | 95% VaR as % of portfolio | % |
| `var_99` | Parametric 99% daily VaR | USD |
| `expected_shortfall_95` | CVaR / ES at 95% | USD |
| `max_drawdown` | Peak-to-trough drawdown | decimal |
| `max_drawdown_start` | Drawdown peak date | ISO date |
| `max_drawdown_trough` | Drawdown trough date | ISO date |
| `data_start` | First data observation | ISO date |
| `data_end` | Last data observation | ISO date |
| `trading_days` | Observation count | integer |

---

### `GET /api/portfolios/{portfolio_id}/risk-contribution`
Return Euler risk decomposition — Marginal, Component, and Percentage Risk Contributions.

**Response Fields:** `contributions[]` with `symbol`, `weight`, `marginal_risk`, `component_risk`, `contribution_pct`

> **Note:** `∑ contribution_pct == 100.0` by construction (Euler homogeneity).

---

### `GET /api/portfolios/{portfolio_id}/correlations`
Return the full correlation matrix and flag pairs with |R| ≥ 0.70.

**Response Fields:** `symbols[]`, `correlation_matrix[][]`, `highly_correlated_pairs[]`

---

### `GET /api/portfolios/{portfolio_id}/monte-carlo`
Run correlated Monte Carlo simulation.

**Query Parameters:**
| Param | Type | Default | Description |
|---|---|---|---|
| `num_simulations` | int | 5000 | Number of paths |
| `horizon_days` | int | 1 | Simulation horizon in trading days |
| `distribution` | string | `normal` | `normal` or `t` |
| `seed` | int | 42 | Random seed for reproducibility |

**Response Fields:** `var_95`, `var_95_pct`, `var_99`, `var_99_pct`, `expected_shortfall_95`, `expected_shortfall_95_pct`, `worst_simulated_loss`, `simulated_returns[]` (sample of up to 2000 paths for charting)

---

### `GET /api/portfolios/{portfolio_id}/backtest`
Rolling out-of-sample VaR backtest with Kupiec POF and Christoffersen independence tests.

**Query Parameters:**
| Param | Type | Default |
|---|---|---|
| `window_size` | int | 250 |
| `confidence_level` | float | 0.95 |

**Response Fields:** `total_eval_days`, `exceptions`, `expected_exceptions`, `exception_rate_pct`, `kupiec_lr`, `kupiec_p_value`, `kupiec_accept`, `christoffersen_lr`, `christoffersen_p_value`, `christoffersen_accept`, `basel_zone` (`green`/`yellow`/`red`), `regulatory_multiplier`, `backtest_series[]`

---

### `POST /api/portfolios/{portfolio_id}/stress-test`
Run hypothetical factor shock scenario.

**Request Body:**
```json
{ "scenario_id": "market_crash_10", "custom_shocks": null }
```

For custom scenarios, pass `"scenario_id": "custom"` and provide `custom_shocks` as `{"AAPL": -0.15, "MSFT": -0.10}`.

**Response Fields:** `scenario_name`, `pre_shock_value`, `post_shock_value`, `total_loss`, `loss_pct`, `asset_breakdown[]`

---

### `GET /api/portfolios/scenarios/list`
List all built-in stress scenarios.

---

### `POST /api/portfolios/{portfolio_id}/historical-crisis-replay`
Replay empirical asset shocks from a named historical crisis event.

**Query Parameters:** `crisis_id` (string)

**Crisis IDs:** `2008_gfc`, `2020_covid_crash`, `2022_inflation_rate_shock`, `2023_svb_banking_crisis`

**Response Fields:** `crisis_name`, `description`, `start_date`, `end_date`, `pre_stress_value`, `post_stress_value`, `total_loss`, `portfolio_loss_pct`, `recovery_return_needed_pct`, `asset_impacts[]`

---

### `GET /api/portfolios/crises/list`
List all supported historical crisis events.

---

### `POST /api/portfolios/{portfolio_id}/reverse-stress-test`
Solve for the minimal shock vector that causes a target portfolio loss.

**Query Parameters:** `target_loss_pct` (float, e.g. `0.25` = 25%)

**Algorithm:** SLSQP minimization of the L2 norm of shock vector subject to portfolio loss ≥ target.

**Response Fields:** `target_loss_pct`, `achieved_loss_pct`, `solver_success`, `primary_vulnerability`, `required_asset_shocks[]`

---

### `POST /api/portfolios/{portfolio_id}/optimize`
Run Markowitz / Risk Parity portfolio optimization.

**Query Parameters:** `objective` = `max_sharpe` | `min_volatility` | `risk_parity`

**Response Fields:** `objective`, `optimized_weights{}`, `expected_return`, `expected_volatility`, `sharpe_ratio`

---

### `GET /api/portfolios/{portfolio_id}/efficient-frontier`
Generate Markowitz efficient frontier points.

**Query Parameters:** `num_points` (int, default 25)

**Response Fields:** `frontier_points[]` with `return`, `volatility`, `sharpe_ratio`

---

## Market Intelligence Endpoints

### `GET /api/market/regime`
Classify current market volatility regime via K-Means.

**Response Fields:** `current_regime`, `dominant_regime_30d`, `regime_history[]`

---

### `GET /api/market/anomalies`
Multi-model anomaly detection for recent market data.

**Query Parameters:** `n_days` (int, default 90)

**Response Fields:** `anomaly_days`, `anomaly_rate_pct`, `anomalies[]` (each with `date`, `daily_return`, `is_anomaly`, `anomaly_score`)

---

## AI Risk Analyst Endpoints

### `POST /api/ai/risk-analysis`
Generate a formal risk memorandum for a portfolio.

**Request Body:**
```json
{
  "portfolio_id": "uuid",
  "include_stress_test": true,
  "stress_scenario_id": "market_crash_10"
}
```

**Response Fields:** `analysis` (formatted markdown narrative), `is_validated`, `provider`, `generated_at`, `evidence_summary{}`

---

### `POST /api/ai/ask`
Conversational natural-language Q&A grounded in portfolio metrics.

**Request Body:**
```json
{
  "portfolio_id": "uuid",
  "question": "Which asset contributes most to portfolio volatility?"
}
```

**Response Fields:** `response` (plain-language answer), `grounded` (bool), `evidence_id`

> **Safety constraints:** The LLM never performs arithmetic, never fetches unverified prices, and never issues buy/sell recommendations. All cited figures are verified against `FrozenEvidenceSnapshot` with 5% tolerance via Pydantic validation.

---

## Background Job Endpoints

### `GET /api/jobs/{job_id}`
Poll status of a long-running background computation job.

**Response Fields:** `job_id`, `status` (`pending`/`running`/`completed`/`failed`), `result` (when completed), `error` (when failed)

---

## Infrastructure Endpoints

### `GET /`
Root index with links to documentation.

### `GET /health`
Health check endpoint for load balancers and orchestrators.

**Response:** `{"status": "healthy", "app_name": "...", "database_url": "..."}`

---

## Error Codes

| HTTP Code | Meaning |
|---|---|
| 400 | Bad request / invalid input |
| 401 | Unauthorized — missing or invalid JWT |
| 403 | Forbidden — insufficient role |
| 404 | Resource not found |
| 422 | Pydantic validation error |
| 500 | Internal server error |
| 503 | Service temporarily unavailable |
