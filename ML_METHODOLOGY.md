# MACHINE LEARNING METHODOLOGY: FinRisk AI Intelligence Engine

**Author:** Quantitative AI/ML Engineering  
**Version:** 2.0  
**Status:** Institutional Standard  

This document articulates the mathematical formulations, model selection criteria, feature engineering pipelines, and validation rigor governing the Machine Learning Intelligence Engine within FinRisk AI.

---

## 1. Feature Engineering Pipeline

All technical features are computed purely on past empirical market observations without look-ahead bias:

| Feature Name | Formulation | Financial Interpretation |
| :--- | :--- | :--- |
| `daily_return` | $R_t = (P_t - P_{t-1}) / P_{t-1}$ | Instantaneous daily price shock. |
| `rolling_vol_20` | $\sigma_{20,t} = \text{std}(R_{t-19:t}) \times \sqrt{252}$ | Medium-term annualized volatility baseline. |
| `rolling_vol_5` | $\sigma_{5,t} = \text{std}(R_{t-4:t}) \times \sqrt{252}$ | Short-term annualized volatility. |
| `vol_ratio` | $\text{VR}_t = \sigma_{5,t} / (\sigma_{20,t} + \epsilon)$ | Volatility spike indicator ($> 1.5$ signals abrupt turbulence). |
| `rolling_return_5` | $(P_t - P_{t-5}) / P_{t-5}$ | Short-term momentum / trend. |
| `rolling_return_20`| $(P_t - P_{t-20}) / P_{t-20}$ | Medium-term momentum / trend. |
| `return_zscore` | $(R_t - \mu_{20,t}) / (\sigma_{20,t}^{\text{daily}} + \epsilon)$ | Standardized return anomaly indicator. |

---

## 2. Market Volatility Regime Detection

Financial returns exhibit structural regime shifts between quiet bull markets, choppy sideways markets, and high-volatility liquidity crises. We model regimes using three complementary algorithms:

### 2.1. Hidden Markov Models (HMM)
A first-order Markov chain with latent discrete states $S_t \in \{0, 1, 2\}$ governing the conditional Gaussian emission distribution of features $X_t$:
$$P(S_t = j \mid S_{t-1} = i) = A_{ij}$$
$$X_t \mid (S_t = k) \sim \mathcal{N}(\mu_k, \Sigma_k)$$
- **Transition Probability Matrix ($A$):** Estimates transition dynamics between low, medium, and high volatility states.
- **Expected Regime Duration:**
  $$D_i = \frac{1}{1 - A_{ii}}$$
- **Next-Period Regime Forecast:**
  $$P(S_{t+1} = j \mid X_{1:t}) = \sum_{i=1}^K P(S_t = i \mid X_{1:t}) A_{ij}$$

### 2.2. Gaussian Mixture Models (GMM)
Unsupervised clustering modeling feature distributions as a mixture of $K=3$ multivariate Gaussians:
$$p(X_t) = \sum_{k=1}^K \pi_k \mathcal{N}(X_t \mid \mu_k, \Sigma_k)$$
Provides soft posterior probabilities $P(S_t = k \mid X_t)$ representing the likelihood of each regime on any given day. Model complexity is evaluated via Bayesian Information Criterion (BIC) and Akaike Information Criterion (AIC).

### 2.3. Deterministic Volatility Sorting
Cluster labels assigned by unsupervised models are arbitrary (e.g. state 0 in HMM might correspond to extreme panic). To guarantee deterministic interpretability:
$$\bar{\sigma}_{\text{cluster}} = \frac{1}{|C_k|} \sum_{t \in C_k} \sigma_{20,t}$$
Clusters are sorted ascendingly by $\bar{\sigma}_{\text{cluster}}$ and mapped to:
- State 0: **Low Volatility** (lowest mean volatility)
- State 1: **Medium Volatility** (intermediate volatility)
- State 2: **High Volatility** (highest mean volatility)

---

## 3. Multi-Detector Anomaly Detection & Attribution

Relying on a single anomaly detector creates vulnerability to false positives. FinRisk AI implements an ensemble of 4 distinct detection paradigms:

1. **Isolation Forest:**
   Isolates outliers by recursively splitting feature spaces randomly. Anomalies require significantly shorter path lengths:
   $$s(x, n) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}$$
2. **Local Outlier Factor (LOF):**
   Compares local density of an observation to the densities of its $k=20$ nearest neighbors:
   $$\text{LOF}_k(p) = \frac{\sum_{o \in N_k(p)} \frac{\text{lrd}(o)}{\text{lrd}(p)}}{|N_k(p)|}$$
3. **Robust Z-Score via Median Absolute Deviation (MAD):**
   Distribution-free outlier detector resistant to extreme outliers:
   $$\text{MAD} = \text{median}(|x_i - \text{median}(x)|)$$
   $$Z_{\text{robust}} = \frac{0.6745 \times (x_i - \text{median}(x))}{\text{MAD}}$$
   Flags returns where $|Z_{\text{robust}}| > 3.0$.
4. **CUSUM Change-Point Detection:**
   Tracks cumulative positive deviations from mean volatility:
   $$S_t^+ = \max(0, S_{t-1}^+ + (\sigma_{20,t} - \mu - k))$$
   Flags structural regime breaks when $S_t^+ > h$.

### Anomaly Consensus & Severity
- **Normal:** 0 detectors flagged.
- **Low Severity:** 1 detector flagged.
- **Medium Severity:** 2 detectors flagged.
- **High Severity:** 3 detectors flagged.
- **Extreme Severity:** All 4 detectors flagged or return loss $> 4\sigma$.

### Feature Attribution & Explainability
For every flagged anomaly, the feature attribution engine measures the standardized deviation of each feature relative to its rolling median baseline:
$$\Delta_f = \frac{|x_f - \text{median}(X_f)|}{\text{IQR}(X_f)}$$
The top 3 deviating features are translated into a plain-English narrative (e.g., *"High severity anomaly driven by unusual daily price shock (-4.2%) and sharp volatility spike (2.8x)."*).

---

## 4. Prevention of Lookahead Bias & Model Drift

### Walk-Forward Cross-Validation
Standard k-fold cross validation shuffles data randomly, causing catastrophic lookahead leakage where future data informs past predictions. FinRisk AI enforces temporal walk-forward splits:
- Expanding window: Train on $[0, t)$, test on $[t, t + \Delta t)$.
- Rolling window: Train on $[t - W, t)$, test on $[t, t + \Delta t)$.
At no point does the model access observations subsequent to time $t$.

### Feature Drift Monitoring
Evaluates distribution stability between baseline training data and production market data:
- **Two-Sample Kolmogorov-Smirnov Test:** Non-parametric test of equality of one-dimensional probability distributions:
  $$D = \sup_x |F_{\text{ref}}(x) - F_{\text{curr}}(x)|$$
- **Wasserstein Distance:** Earth Mover's Distance measuring the minimal cost of transforming reference feature distributions into current feature distributions.
Alerts the risk manager when more than 25% of input features exhibit statistically significant drift ($p < 0.05$).
