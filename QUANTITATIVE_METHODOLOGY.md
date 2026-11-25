# QUANTITATIVE METHODOLOGY: Mathematical Foundations of FinRisk AI

**Author:** Quantitative Research & Financial Risk Engineering  
**Version:** 2.0  
**Status:** Institutional Standard  

This document details every mathematical equation, distributional assumption, risk attribution formulation, and regulatory validation procedure implemented in the FinRisk AI Quantitative Risk Engine.

---

## 1. Return Calculations & Sign Conventions

### Simple Daily Return
For an asset with price series $P_t$, simple returns are defined as:
$$R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

### Portfolio Weighted Daily Return
For a portfolio with $N$ assets, weights vector $w = [w_1, w_2, \dots, w_N]^T$ where $\sum_{i=1}^N w_i = 1$, the daily portfolio return is:
$$R_{p,t} = \sum_{i=1}^N w_i R_{i,t} = w^T R_t$$

### Sign Conventions for Risk Metrics
- **Returns ($R_t$):** Profits are positive ($R > 0$); losses are negative ($R < 0$).
- **Value at Risk (VaR):** Reported as a **positive loss** percentage or dollar amount. If the 95% 1-day VaR is 2.5%, it denotes that daily losses will exceed 2.5% on only 5% of trading days:
  $$\text{VaR}_{\alpha} = -Q_{1-\alpha}(R_p) > 0$$
- **Expected Shortfall (ES / CVaR):** Reported as a **positive loss** representing the conditional expectation of loss given that the loss breaches VaR:
  $$\text{ES}_{\alpha} = -\mathbb{E}[R_p \mid R_p \le -\text{VaR}_{\alpha}] \ge \text{VaR}_{\alpha}$$
- **Drawdown:** Reported as a **positive loss fraction** representing the decline from running historical peak:
  $$\text{Drawdown}_t = \frac{\max_{\tau \le t} V_{\tau} - V_t}{\max_{\tau \le t} V_{\tau}} \ge 0$$

---

## 2. Volatility Modeling

### Annualized Sample Portfolio Volatility
Given the empirical daily covariance matrix $\Sigma_{\text{daily}}$, annualized volatility is:
$$\sigma_p = \sqrt{w^T \Sigma_{\text{daily}} w} \times \sqrt{252}$$
To ensure stability against ill-conditioned matrices, $\Sigma$ is regularized via symmetric eigenvalue clipping ($\lambda_{\min} \ge 10^{-7}$).

### Exponentially Weighted Moving Average (EWMA)
Following the J.P. Morgan RiskMetrics standard with decay factor $\lambda = 0.94$:
$$\sigma_t^2 = \lambda \sigma_{t-1}^2 + (1 - \lambda) R_{t-1}^2$$
$$\sigma_{\text{ann},t} = \sqrt{\sigma_t^2 \times 252}$$
EWMA dynamically captures volatility clustering and reacts faster to market turbulence than simple rolling standard deviation.

### Generalized Autoregressive Conditional Heteroskedasticity: GARCH(1,1)
The conditional variance evolves according to:
$$R_t = \mu + \epsilon_t, \quad \epsilon_t = \sigma_t z_t, \quad z_t \sim \text{i.i.d. } \mathcal{N}(0, 1)$$
$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$
Subject to constraints:
$$\omega > 0, \quad \alpha \ge 0, \quad \beta \ge 0, \quad \alpha + \beta < 1 \quad (\text{stationarity})$$
- **Unconditional Long-Term Volatility:**
  $$\sigma_{\infty} = \sqrt{\frac{\omega}{1 - (\alpha + \beta)} \times 252}$$
- **Parameters Estimation:** Estimated via Maximum Likelihood Estimation (L-BFGS-B) maximizing the log-likelihood function:
  $$\ln L(\theta) = -\frac{1}{2} \sum_{t=1}^T \left[ \ln(2\pi) + \ln(\sigma_t^2) + \frac{(R_t - \mu)^2}{\sigma_t^2} \right]$$

---

## 3. Value at Risk (VaR) Methodologies

### 3.1. Historical Simulation VaR
Non-parametric calculation from the empirical empirical distribution of portfolio returns sorted ascendingly:
$$\text{VaR}_{\text{Hist}}(\alpha) = -Q_{1-\alpha}(R_p)$$
For $\alpha = 0.95$, this corresponds to the 5th percentile of daily returns.

### 3.2. Parametric Gaussian VaR
Assuming returns are normally distributed with sample mean $\mu$ and standard deviation $\sigma$:
$$\text{VaR}_{\text{Normal}}(\alpha) = -(\mu + z_{1-\alpha} \sigma)$$
where $z_{0.05} = -1.64485$ and $z_{0.01} = -2.32635$.

### 3.3. Cornish-Fisher Modified VaR
Adjusts the Gaussian quantile for empirical return skewness ($S$) and excess kurtosis ($K$):
$$z_{\text{CF}} = z + \frac{S}{6}(z^2 - 1) + \frac{K}{24}(z^3 - 3z) - \frac{S^2}{36}(2z^3 - 5z)$$
$$\text{VaR}_{\text{CF}}(\alpha) = -(\mu + z_{\text{CF}} \sigma)$$
This captures asymmetry and fat tails without requiring numerical optimization.

### 3.4. Monte Carlo Simulation VaR
Simulates $M \ge 10,000$ correlated return paths:
1. Decompose the asset covariance matrix $\Sigma = L L^T$ via Cholesky factorization.
2. Draw independent standard random vectors $Z \sim \mathcal{N}(0, I_N)$ or multivariate Student's $t$ with $\nu$ degrees of freedom.
3. Compute correlated asset returns $R_{\text{sim}} = \mu + L Z$.
4. Calculate simulated portfolio returns $R_{p,\text{sim}} = R_{\text{sim}} w$.
5. Compound over horizon $H$:
   $$R_{p,H} = \prod_{h=1}^H (1 + R_{p,h}) - 1$$
6. $\text{VaR}_{\text{MC}}(\alpha)$ is the $(1-\alpha)$ empirical quantile of the simulated distribution.

---

## 4. Risk Attribution (Euler Decomposition)

Because portfolio volatility $\sigma_p(w)$ is a homogeneous function of degree 1 in portfolio weights ($f(c w) = c f(w)$), by Euler's Theorem:
$$\sigma_p = \sum_{i=1}^N w_i \frac{\partial \sigma_p}{\partial w_i}$$

1. **Marginal Contribution to Risk (MCR):**
   $$\text{MCR}_i = \frac{\partial \sigma_p}{\partial w_i} = \frac{(\Sigma w)_i}{\sigma_p}$$
2. **Component Contribution to Risk (CCR):**
   $$\text{CCR}_i = w_i \times \text{MCR}_i$$
   $$\sum_{i=1}^N \text{CCR}_i = \sigma_p$$
3. **Percentage Contribution to Risk (PCR):**
   $$\text{PCR}_i = \frac{\text{CCR}_i}{\sigma_p} = \frac{w_i (\Sigma w)_i}{\sigma_p^2}$$
   $$\sum_{i=1}^N \text{PCR}_i = 1.0 \quad (100\%)$$

---

## 5. Risk-Adjusted Performance Metrics

### Sharpe Ratio
Measures excess return per unit of total risk:
$$\text{Sharpe} = \frac{\text{AnnReturn} - R_f}{\sigma_p}$$
where $\text{AnnReturn} = \bar{R}_p \times 252$.

### Sortino Ratio
Penalizes only downside volatility below a target return $R_{\text{target}}$ (default 0 or $R_f$):
$$\text{Sortino} = \frac{\text{AnnReturn} - R_f}{\sigma_{\text{downside}}}$$
$$\sigma_{\text{downside}} = \sqrt{\frac{1}{T} \sum_{t=1}^T \left( \min(0, R_t - R_{\text{target}}) \right)^2 \times 252}$$

### Calmar Ratio
Measures return relative to drawdown severity:
$$\text{Calmar} = \frac{\text{AnnReturn}}{\text{MaxDrawdown}}$$

### Maximum Drawdown & Durations
$$\text{DD}_t = \frac{\max_{\tau \le t} V_{\tau} - V_t}{\max_{\tau \le t} V_{\tau}}, \quad \text{MDD} = \max_{t} \text{DD}_t$$
- **Drawdown Duration:** Trading days from the peak date to the trough date.
- **Recovery Duration:** Trading days from the trough date until portfolio value recovers to the previous peak value.

---

## 6. Regulatory Model Validation & Backtesting

### Kupiec Proportion of Failures (POF) Test
Evaluates whether the observed number of VaR exceedances $x$ over $N$ days is statistically consistent with expected rate $p_0 = 1 - \alpha$:
$$\text{LR}_{\text{POF}} = -2 \ln \left[ \frac{(1 - p_0)^{N-x} p_0^x}{(1 - \hat{p})^{N-x} \hat{p}^x} \right] \sim \chi^2(1)$$
where $\hat{p} = x / N$. If $p\text{-value} < 0.05$, the model is rejected.

### Christoffersen Independence Test
Evaluates whether VaR exceptions occur independently or in clusters (Markov transition test):
$$\text{LR}_{\text{ind}} = -2 \ln \left[ \frac{\pi_0^{n_{00} + n_{10}} (1 - \pi_0)^{n_{01} + n_{11}}}{\pi_{01}^{n_{01}} (1 - \pi_{01})^{n_{00}} \pi_{11}^{n_{11}} (1 - \pi_{11})^{n_{10}}} \right] \sim \chi^2(1)$$
where $\pi_{ij}$ is the transition probability from state $i$ to state $j$ ($0 = \text{no exception}, 1 = \text{exception}$).

### Basel Committee Traffic Light Framework
For $N = 250$ observations at the 99% confidence level:
- **Green Zone ($0 \le x \le 4$ exceptions):** Model is statistically sound; no capital penalty.
- **Yellow Zone ($5 \le x \le 9$ exceptions):** Supervisory concern; capital multiplier increases progressively from +0.40 to +0.85.
- **Red Zone ($x \ge 10$ exceptions):** Model failure; mandatory model overhaul and maximum capital penalty.
