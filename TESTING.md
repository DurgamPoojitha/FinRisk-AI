# Testing Guide

> **Test Runner:** pytest 8.x  
> **Coverage Tool:** pytest-cov  
> **Virtual Environment:** `.venv` (Python 3.12)

---

## 1. Quick Start

```bash
# From project root with venv activated
source .venv/bin/activate

# Run all tests with coverage
pytest backend/tests/ -v --tb=short

# Run with coverage report
pytest backend/tests/ --cov=backend/app --cov-report=term-missing
```

---

## 2. Test Suite Structure

```
backend/tests/
├── __init__.py
├── conftest.py                  # Shared fixtures (in-memory SQLite, test portfolios)
├── test_risk_engine.py          # Core risk metrics (parametric VaR, ES, Sharpe, drawdown)
├── test_advanced_risk.py        # Monte Carlo, EWMA, GARCH, Kupiec, Christoffersen
├── test_stress_optimization.py  # Stress testing, reverse stress, Markowitz optimization
├── test_advanced_ml.py          # HMM, GMM, K-Means regimes; LOF, IF, CUSUM anomalies
├── test_ai_grounding.py         # FrozenEvidenceSnapshot, GroundedClaimVerifier, RAG
├── test_data_pipeline.py        # DataQualityValidator, provider pattern, normalization
├── test_integration.py          # End-to-end portfolio lifecycle via FastAPI TestClient
├── test_api.py                  # HTTP endpoint contract tests
└── test_ml_engine.py            # Walk-forward validation, drift monitoring
```

**Current status:** 50 passed, 0 failed

---

## 3. Running Individual Test Modules

```bash
# Risk metrics only
pytest backend/tests/test_risk_engine.py -v

# Advanced quantitative tests
pytest backend/tests/test_advanced_risk.py -v

# Stress testing + optimization
pytest backend/tests/test_stress_optimization.py -v

# ML regimes + anomaly detection
pytest backend/tests/test_advanced_ml.py -v

# AI grounding + RAG safety
pytest backend/tests/test_ai_grounding.py -v

# Data pipeline validation
pytest backend/tests/test_data_pipeline.py -v

# Integration tests (requires FastAPI TestClient)
pytest backend/tests/test_integration.py -v

# Full API contract tests
pytest backend/tests/test_api.py -v
```

---

## 4. Test Categories & What They Cover

### `test_risk_engine.py` — Core Risk Metrics
| Test | Validates |
|---|---|
| `test_parametric_var` | Normal VaR at 95%/99%, sign convention (positive loss) |
| `test_expected_shortfall` | ES > VaR at same confidence level |
| `test_sharpe_ratio` | (R − Rf) / σ with annualized inputs |
| `test_max_drawdown` | Peak-to-trough calculation, non-negative result |
| `test_risk_contribution` | ∑ PCR == 100.0 (Euler homogeneity) |
| `test_beta_calculation` | Regression slope vs benchmark |

### `test_advanced_risk.py` — Quantitative Finance Engine
| Test | Validates |
|---|---|
| `test_monte_carlo_var` | MC VaR 99% > VaR 95%, reproducible seed |
| `test_monte_carlo_es` | ES > VaR at same confidence |
| `test_monte_carlo_t_distribution` | Student-t paths have heavier tails than Normal |
| `test_ewma_volatility` | EWMA forecast is positive, λ=0.94 decay |
| `test_garch_volatility` | GARCH(1,1) convergence, α+β < 1 stationarity constraint |
| `test_kupiec_pof` | LR statistic, p-value, accept/reject boundary |
| `test_christoffersen` | Independence test on exception sequence |
| `test_sortino_ratio` | Sortino uses only downside deviation |
| `test_calmar_ratio` | Calmar = annualized return / max drawdown |
| `test_cornish_fisher_var` | CF-VaR > Normal VaR for negative skew |

### `test_stress_optimization.py` — Stress Testing & Portfolio Optimization
| Test | Validates |
|---|---|
| `test_historical_crisis_replay_2008` | 2008 GFC loss > 0, all fields present |
| `test_historical_crisis_replay_svb` | SVB recovery return ≥ 0 (can be gain) |
| `test_reverse_stress_solver` | Solver converges, achieved_loss_pct ≥ target |
| `test_max_sharpe_optimization` | Sharpe ≥ equal-weight portfolio Sharpe |
| `test_min_vol_optimization` | Volatility ≤ equal-weight portfolio volatility |
| `test_risk_parity` | Risk contributions are equal across assets |
| `test_efficient_frontier` | Points are monotonically increasing return for increasing vol |

### `test_advanced_ml.py` — Machine Learning Engine
| Test | Validates |
|---|---|
| `test_kmeans_regime_determinism` | Same seed → same regime labels |
| `test_hmm_regime_transitions` | Transition matrix rows sum to 1.0 |
| `test_gmm_bic_selection` | BIC decreases with better model fit |
| `test_isolation_forest_anomalies` | Anomaly scores in [−1, 1] |
| `test_lof_anomalies` | LOF detects local outliers |
| `test_ensemble_consensus` | Ensemble anomalies ≤ individual model max |
| `test_walk_forward_no_leakage` | Train window never overlaps test window |
| `test_ks_drift_detection` | KS statistic significant for clearly drifted data |

### `test_ai_grounding.py` — AI Safety & RAG
| Test | Validates |
|---|---|
| `test_frozen_snapshot_immutable` | Modifying result dict does not alter snapshot |
| `test_grounding_verifier_pass` | Valid claim within 5% tolerance passes |
| `test_grounding_verifier_fail` | Fabricated number outside tolerance rejected |
| `test_prompt_injection_blocked` | Injection patterns return rejected status |
| `test_rag_vector_search` | Query returns relevant document chunk |
| `test_unsupported_asset_detection` | Claim about ticker not in portfolio is flagged |

### `test_data_pipeline.py` — Data Engineering
| Test | Validates |
|---|---|
| `test_quality_validator_non_positive_price` | Flags zero/negative prices |
| `test_quality_validator_spike_detection` | Flags >50% single-day return |
| `test_quality_validator_duplicate_dates` | Flags duplicate index entries |
| `test_normalization` | Returns are in (−1, 1), prices are float |
| `test_csv_provider` | Loads synthetic test CSV, returns DataFrame |

---

## 5. Shared Fixtures (`conftest.py`)

```python
@pytest.fixture(scope="session")
def db_session():
    """In-memory SQLite database shared across all tests in the session."""

@pytest.fixture
def test_portfolio(db_session):
    """Pre-seeded portfolio with AAPL, MSFT, NVDA, JPM, XOM (equal weights)."""

@pytest.fixture
def sample_returns():
    """252 trading days of synthetic returns drawn from N(0.0003, 0.015)."""
```

---

## 6. Running Against a Real Database

By default, tests use an in-memory SQLite database (no Postgres required locally).

To test against PostgreSQL:

```bash
export DATABASE_URL="postgresql://finrisk:password@localhost:5432/finrisk_test"
pytest backend/tests/ -v
```

> **Warning:** Use a dedicated test database, not the development database. The test suite may truncate and re-seed tables.

---

## 7. Coverage Report

```bash
# Generate HTML coverage report
pytest backend/tests/ \
  --cov=backend/app \
  --cov-report=html:htmlcov \
  --cov-report=term-missing

# Open report
open htmlcov/index.html
```

**Coverage targets:**
- `risk_engine/`: > 85%
- `ml_engine/`: > 75%
- `ai_engine/`: > 70%
- `data_engine/`: > 80%
- `api/`: > 65%

---

## 8. Mathematical Property Testing

The risk engine tests use **property-based assertions** rather than fixed expected values, to avoid hardcoding outputs that depend on seeded random data:

```python
# Example: ES must always exceed VaR at the same confidence level
assert result["expected_shortfall_95"] >= result["var_95"]

# Example: Euler risk contributions must sum to 100.0
assert abs(sum(c["contribution_pct"] for c in contributions) - 100.0) < 0.01

# Example: Optimization must not decrease Sharpe
assert optimized["sharpe_ratio"] >= baseline_sharpe - 0.001
```

This approach makes tests robust to data changes while still enforcing invariants.

---

## 9. Adding New Tests

1. Place test file in `backend/tests/test_<module>.py`
2. Import the module under test using the absolute path: `from backend.app.risk_engine.metrics import ...`
3. Use shared fixtures from `conftest.py` where possible
4. Write at least one **mathematical invariant** test for every new financial calculation
5. Run `pytest --co -q` to verify the test is discovered before pushing

---

## 10. CI Test Execution

Tests run automatically on every push and pull request via GitHub Actions (see `.github/workflows/ci.yml`).

The CI test job:
- Spins up PostgreSQL 16 and Redis 7 as service containers
- Sets `DATABASE_URL` to point at the CI Postgres instance
- Requires **70% line coverage** to pass (enforced with `--cov-fail-under=70`)
- Reports coverage to Codecov on every run
