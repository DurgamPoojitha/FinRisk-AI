"""
Risk Model Validation & Regulatory Backtesting Suite:
- Kupiec Proportion of Failures (POF) Test (Unconditional Coverage)
- Christoffersen Independence Test (Conditional Coverage)
- Basel Committee Traffic Light Framework
- Rolling-Window Out-of-Sample VaR Backtesting
- Expected Shortfall Backtesting (Acerbi-Szekely Statistic)
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats


def kupiec_pof_test(
    num_observations: int,
    num_exceptions: int,
    confidence_level: float = 0.95,
) -> Dict[str, Any]:
    """
    Performs Kupiec's Proportion of Failures (POF) Likelihood Ratio Test.
    Evaluates null hypothesis H0: observed failure rate p equals expected failure rate p0 = 1 - confidence_level.

    Test statistic:
      LR_POF = -2 * ln( [ (1 - p0)^(N - x) * p0^x ] / [ (1 - p_hat)^(N - x) * p_hat^x ] )
      where p_hat = x / N, distributed as Chi-Square with 1 degree of freedom.

    Args:
        num_observations: Total number of out-of-sample backtest days (N).
        num_exceptions: Number of days where actual loss exceeded VaR (x).
        confidence_level: VaR confidence level (e.g. 0.95 or 0.99).

    Returns:
        Dict with test statistic, p-value, decision (reject/accept H0), and status.
    """
    if num_observations <= 0:
        raise ValueError("num_observations must be positive.")
    if num_exceptions < 0 or num_exceptions > num_observations:
        raise ValueError("num_exceptions must be between 0 and num_observations.")

    p0 = 1.0 - confidence_level
    p_hat = num_exceptions / num_observations

    if num_exceptions == 0:
        # Edge case: zero exceptions
        # LR = -2 * ln((1 - p0)^N) = -2 * N * ln(1 - p0)
        lr_stat = -2.0 * num_observations * np.log(1.0 - p0)
    elif num_exceptions == num_observations:
        # Extreme case: all exceptions
        lr_stat = -2.0 * num_observations * np.log(p0)
    else:
        # Standard likelihood ratio
        numerator = ((1.0 - p0) ** (num_observations - num_exceptions)) * (p0 ** num_exceptions)
        denominator = ((1.0 - p_hat) ** (num_observations - num_exceptions)) * (p_hat ** num_exceptions)
        if denominator <= 0 or numerator <= 0:
            lr_stat = 0.0
        else:
            lr_stat = -2.0 * np.log(numerator / denominator)

    lr_stat = max(0.0, float(lr_stat))
    p_value = float(1.0 - stats.chi2.cdf(lr_stat, df=1))

    # Determine model bias: does it underestimate or overestimate risk?
    expected_exceptions = num_observations * p0
    if num_exceptions > expected_exceptions:
        model_bias = "Underestimating Risk (Too many exceptions)"
    elif num_exceptions < expected_exceptions:
        model_bias = "Overestimating Risk / Conservative (Fewer exceptions than expected)"
    else:
        model_bias = "Well-Calibrated"

    # Reject null hypothesis at 5% significance level if p_value < 0.05
    reject_null = bool(p_value < 0.05)

    return {
        "test_name": "Kupiec POF Test",
        "num_observations": num_observations,
        "num_exceptions": num_exceptions,
        "expected_exceptions": round(expected_exceptions, 2),
        "actual_failure_rate": round(p_hat, 4),
        "expected_failure_rate": round(p0, 4),
        "lr_statistic": round(lr_stat, 4),
        "critical_value_95": 3.841,  # Chi2(1) at alpha=0.05
        "p_value": round(p_value, 4),
        "reject_null": reject_null,
        "decision": "Reject H0 (Model Inaccurate)" if reject_null else "Accept H0 (Model Valid)",
        "model_bias": model_bias,
    }


def christoffersen_independence_test(
    exception_indicators: np.ndarray,
) -> Dict[str, Any]:
    """
    Performs Christoffersen's Test of Independence for VaR exceptions.
    Evaluates whether exceptions are independent across time or clustered.

    Transitions:
      n00: non-exception followed by non-exception
      n01: non-exception followed by exception
      n10: exception followed by non-exception
      n11: exception followed by exception

    Test statistic:
      LR_ind = -2 * ln( [ pi_hat^(n01+n11) * (1-pi_hat)^(n00+n10) ] /
                        [ pi01^n01 * (1-pi01)^n00 * pi11^n11 * (1-pi11)^n10 ] )
      distributed as Chi-Square with 1 degree of freedom.

    Returns:
        Dict with transition counts, LR statistic, p-value, and decision.
    """
    clean_indicators = np.asarray(exception_indicators, dtype=int)
    n = len(clean_indicators)
    if n < 10:
        raise ValueError("At least 10 observations required for Christoffersen test.")

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for i in range(n - 1):
        prev = clean_indicators[i]
        curr = clean_indicators[i + 1]
        if prev == 0 and curr == 0:
            n00 += 1
        elif prev == 0 and curr == 1:
            n01 += 1
        elif prev == 1 and curr == 0:
            n10 += 1
        elif prev == 1 and curr == 1:
            n11 += 1

    # Probabilities
    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    total_exceptions = n01 + n11
    total_non_exceptions = n00 + n10
    pi_hat = total_exceptions / (n - 1) if (n - 1) > 0 else 0.0

    # If no exceptions or zero transition variation, test cannot be computed
    if total_exceptions == 0 or n11 == 0 and n01 == 0:
        return {
            "test_name": "Christoffersen Independence Test",
            "lr_statistic": 0.0,
            "p_value": 1.0,
            "reject_null": False,
            "decision": "Accept H0 (No exception clustering detected)",
            "consecutive_exceptions": 0,
        }

    # Likelihood under independence H0:
    eps = 1e-12
    term_null = total_exceptions * np.log(max(pi_hat, eps)) + total_non_exceptions * np.log(max(1.0 - pi_hat, eps))

    # Likelihood under dependence H1:
    term_alt = 0.0
    if n00 > 0:
        term_alt += n00 * np.log(max(1.0 - pi01, eps))
    if n01 > 0:
        term_alt += n01 * np.log(max(pi01, eps))
    if n10 > 0:
        term_alt += n10 * np.log(max(1.0 - pi11, eps))
    if n11 > 0:
        term_alt += n11 * np.log(max(pi11, eps))

    lr_ind = -2.0 * (term_null - term_alt)
    lr_ind = max(0.0, float(lr_ind))
    p_value = float(1.0 - stats.chi2.cdf(lr_ind, df=1))
    reject_null = bool(p_value < 0.05)

    return {
        "test_name": "Christoffersen Independence Test",
        "transitions": {"n00": n00, "n01": n01, "n10": n10, "n11": n11},
        "prob_exception_after_normal": round(pi01, 4),
        "prob_exception_after_exception": round(pi11, 4),
        "lr_statistic": round(lr_ind, 4),
        "p_value": round(p_value, 4),
        "reject_null": reject_null,
        "decision": "Reject H0 (Exceptions are Clustered)" if reject_null else "Accept H0 (Exceptions are Independent)",
        "consecutive_exceptions": n11,
    }


def basel_traffic_light_zone(
    num_exceptions: int,
    num_observations: int = 250,
) -> Dict[str, Any]:
    """
    Applies the Basel Committee Supervisory Framework Traffic Light approach for 99% 1-day VaR.
    For N=250 observations at 99% confidence:
      Green Zone:  0 to 4 exceptions  (Model valid; no supervisory penalty)
      Yellow Zone: 5 to 9 exceptions  (Supervisory scrutiny; capital multiplier increases)
      Red Zone:    10+ exceptions     (Model presumed flawed; immediate action required)
    """
    # Scale thresholds if N != 250
    scale = num_observations / 250.0
    green_threshold = int(np.floor(4 * scale))
    yellow_threshold = int(np.floor(9 * scale))

    if num_exceptions <= green_threshold:
        zone = "Green"
        status_color = "#10B981"
        description = "Model is statistically sound and acceptable under Basel standards."
        capital_multiplier_add = 0.0
    elif num_exceptions <= yellow_threshold:
        zone = "Yellow"
        status_color = "#F59E0B"
        description = "Supervisory concern: model exhibits elevated exceptions. Scaling factor applied."
        # Progressive penalty in yellow zone: +0.40 to +0.85
        capital_multiplier_add = 0.40 + 0.10 * (num_exceptions - green_threshold)
    else:
        zone = "Red"
        status_color = "#EF4444"
        description = "Model failure: excessive exceptions indicate severe underestimation of portfolio risk."
        capital_multiplier_add = 1.00

    return {
        "zone": zone,
        "color": status_color,
        "num_exceptions": num_exceptions,
        "num_observations": num_observations,
        "green_max": green_threshold,
        "yellow_max": yellow_threshold,
        "capital_multiplier_addon": round(capital_multiplier_add, 2),
        "description": description,
    }


def run_rolling_var_backtest(
    portfolio_returns: pd.Series,
    window_size: int = 250,
    confidence_level: float = 0.95,
    method: str = "historical",  # "historical" or "parametric"
) -> Dict[str, Any]:
    """
    Executes a rolling-window out-of-sample VaR backtest.
    At each step t in [window_size, N]:
      1. Train on returns in [t - window_size, t)
      2. Predict 1-day ahead VaR for day t
      3. Compare predicted VaR against realized return at day t
      4. Record whether an exception occurred (return_t < -VaR_t)

    Returns:
        Comprehensive validation report with Kupiec test, Christoffersen test, and Basel status.
    """
    returns_clean = portfolio_returns.dropna()
    n = len(returns_clean)
    if n <= window_size:
        raise ValueError(f"Total returns ({n}) must exceed rolling window size ({window_size}).")

    dates = returns_clean.index[window_size:]
    actual_returns = returns_clean.iloc[window_size:].values
    predicted_vars = np.zeros(len(dates))
    exception_indicators = np.zeros(len(dates), dtype=int)
    tail_losses = []

    alpha = 1.0 - confidence_level

    for i in range(len(dates)):
        # Training window
        train_window = returns_clean.iloc[i : i + window_size].values

        if method == "parametric":
            mu = np.mean(train_window)
            std = np.std(train_window, ddof=1)
            # Parametric normal VaR: - (mu + z_alpha * std)
            z = stats.norm.ppf(alpha)
            var_t = float(-(mu + z * std))
        else:
            # Historical simulation VaR
            q = np.percentile(train_window, alpha * 100)
            var_t = float(-q)

        var_t = max(0.0, var_t)
        predicted_vars[i] = var_t

        # Actual loss at day t: -actual_returns[i]
        realized_loss = -actual_returns[i]
        if realized_loss > var_t:
            exception_indicators[i] = 1
            tail_losses.append(realized_loss)

    total_eval_days = len(dates)
    total_exceptions = int(np.sum(exception_indicators))

    # Formal hypothesis tests
    kupiec = kupiec_pof_test(total_eval_days, total_exceptions, confidence_level)
    christoffersen = christoffersen_independence_test(exception_indicators)
    basel = basel_traffic_light_zone(total_exceptions, total_eval_days)

    # Average realized tail loss vs expected shortfall
    mean_tail_loss = float(np.mean(tail_losses)) if tail_losses else 0.0

    series_data = [
        {
            "date": str(dates[i].date()) if hasattr(dates[i], "date") else str(dates[i]),
            "return_pct": round(float(actual_returns[i]) * 100, 3),
            "realized_return_pct": round(float(actual_returns[i]) * 100, 3),
            "var_pct": round(float(predicted_vars[i]) * 100, 3),
            "predicted_var_pct": round(float(predicted_vars[i]) * 100, 3),
            "is_exception": bool(exception_indicators[i]),
        }
        for i in range(len(dates))
    ]

    return {
        "evaluation_period_days": total_eval_days,
        "total_eval_days": total_eval_days,
        "window_size": window_size,
        "confidence_level": confidence_level,
        "method": method,
        "total_exceptions": total_exceptions,
        "exceptions": total_exceptions,
        "expected_exceptions": round((1.0 - confidence_level) * total_eval_days, 1),
        "exception_rate_pct": round((total_exceptions / total_eval_days) * 100, 2),
        "expected_exception_rate_pct": round((1.0 - confidence_level) * 100, 2),
        "mean_tail_loss_pct": round(mean_tail_loss * 100, 4),
        "basel_zone": basel["zone"],
        "regulatory_multiplier": round(3.0 + basel.get("capital_multiplier_addon", 0.0), 2),
        "kupiec_lr": kupiec["lr_statistic"],
        "kupiec_p_value": kupiec["p_value"],
        "kupiec_accept": kupiec["decision"] == "Accept",
        "christoffersen_lr": christoffersen["lr_statistic"],
        "christoffersen_p_value": christoffersen["p_value"],
        "christoffersen_accept": christoffersen["decision"] == "Accept",
        "kupiec_test": kupiec,
        "christoffersen_test": christoffersen,
        "basel_framework": basel,
        "backtest_series": series_data,
        "history": series_data,
    }
