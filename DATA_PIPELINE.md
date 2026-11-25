# DATA PIPELINE & MARKET DATA ARCHITECTURE: FinRisk AI

**Author:** Financial Data Engineering & Infrastructure  
**Version:** 2.0  
**Status:** Institutional Standard  

This document outlines the modular market data ingestion architecture, data normalization rules, automated quality controls, and persistence mechanisms of FinRisk AI.

---

## 1. Modular Provider Architecture

To decouple the platform from static CSV files or single data vendors, FinRisk AI utilizes the Strategy pattern via `BaseMarketDataProvider`:

```
                 +--------------------------------+
                 |     BaseMarketDataProvider     |
                 +--------------------------------+
                                 |
         +-----------------------+-----------------------+
         |                       |                       |
         v                       v                       v
+------------------+    +------------------+    +------------------+
|  YahooFinance    |    |  CSVMarketData   |    |  MockMarketData  |
|     Provider     |    |     Provider     |    |     Provider     |
| (Live / Dev feed)|    | (Historical file)|    | (Deterministic   |
|                  |    |                  |    |  correl. GBM)    |
+------------------+    +------------------+    +------------------+
```

---

## 2. Ingestion & Quality Validation Pipeline

Before any raw market data feeds into quantitative risk calculations, it passes through the automated `DataQualityValidator`:

1. **Non-Positive Price Detection:**
   Verifies that close prices are strictly positive ($P_t > 0$). Any zero or negative prices trigger an immediate data quality warning.
2. **Extreme Return Spike Detection:**
   Flags any single-day return exceeding $\pm 50\%$:
   $$|R_t| = \left| \frac{P_t - P_{t-1}}{P_{t-1}} \right| > 0.50$$
   Unadjusted stock splits or bad ticks are caught before distorting covariance matrices.
3. **Duplicate Timestamp Filtering:**
   Deduplicates rows sharing identical calendar dates, retaining the latest valid trade observation.
4. **Calendar Alignment & Missing Day Handling:**
   Missing observations due to market holidays or trading halts are forward-filled up to a maximum threshold of 3 consecutive business days before row exclusion:
   $$P_t = P_{t-1} \quad \text{for } t - (t-1) \le 3 \text{ days}$$

---

## 3. Data Quality Reporting

Every ingestion run yields an auditable `DataQualityReport`:
- `status`: `PASSED` or `WARNING`
- `total_observations`: Total trading days evaluated
- `date_range`: Empirical start and end date
- `missing_values_count`: Total null observations filled
- `extreme_return_spikes`: List of flagged date/asset spikes
- `is_valid`: Boolean gate determining whether data can be safely passed to the Risk Engine

---

## 4. Database Persistence

Cleaned price vectors are persisted to the PostgreSQL `market_prices` table using bulk chunked transactions (500 records per batch) to ensure high write throughput and zero memory exhaustion.
Calculation pipelines query indexed database records, enabling reproducible risk audits traceable to exact timestamped market snapshots.
