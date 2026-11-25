# AI GROUNDING & RAG ARCHITECTURE: FinRisk AI Risk Analyst

**Author:** Quantitative AI Safety & Systems Architecture  
**Version:** 2.0  
**Status:** Institutional Standard  

This document outlines the safety architecture, grounding protocols, Retrieval-Augmented Generation (RAG) pipelines, and verification mechanisms implemented to guarantee zero hallucinations in FinRisk AI.

---

## 1. Core Principles of Financial AI Safety

1. **Separation of Computation and Natural Language Interpretation:**
   Under no circumstances does the LLM compute mathematical risk metrics, estimate correlations, or calculate portfolio losses. All numbers originate from verified, deterministic Python quantitative modules.
2. **Immutable Evidence Snapshots:**
   Calculations are frozen into a `FrozenEvidenceSnapshot` bearing a unique `calculation_id` and `evidence_id`. The LLM receives this snapshot as read-only context.
3. **Structured Pydantic Validation:**
   AI outputs are parsed into strict Pydantic schemas (`StructuredRiskAnalysis`, `RiskQuestionResponse`) enforcing required sections, valid metric types, and mandatory disclaimers.
4. **Deterministic Post-Generation Auditing:**
   Every generated response is passed to `GroundedClaimVerifier`, which checks cited dollar amounts, percentages, and asset tickers against ground-truth evidence before persisting or presenting the report.
5. **No Speculative or Trading Advice:**
   The AI is barred from providing buy/sell recommendations, predicting market peaks/bottoms, or claiming guaranteed returns.

---

## 2. Evidence Grounding & Verification Architecture

```
       [ Quantitative Risk Engine & ML Engine ]
                           |
                           v
              [ FrozenEvidenceSnapshot ]
         - calculation_id (UUID)
         - evidence_id (UUID)
         - Portfolio Valuation & Holdings
         - Value at Risk (Hist, Param, MC)
         - Expected Shortfall & Max Drawdown
         - Marginal / Component Risk Contributions
         - Market Volatility Regimes & Anomalies
         - Policy Limit Breaches (Evaluated deterministically)
                           |
       +-------------------+-------------------+
       |                                       |
       v                                       v
[ System Prompt & Context ]           [ GroundedClaimVerifier ]
- Strict Citation Rules               - Extracts all $ and % figures
- Forbidden Actions                   - Compares against evidence index
- Retrieved RAG Policy Context        - Flags unheld assets
       |                               - Computes Grounding Score (0.0 to 1.0)
       v                                       |
  [ LLM Inference ]                           |
(Gemini / OpenAI / Mock)                       |
       |                                       |
       +---------------------------------------+
                           |
                           v
                 [ Audit & Report Storage ]
           (Persisted in PostgreSQL `ai_reports`)
```

---

## 3. Retrieval-Augmented Generation (RAG) Pipeline

The RAG engine connects qualitative institutional mandates, investment policies, and regulatory standards with quantitative portfolio metrics.

### Document Corpus
1. **Global Portfolio Risk Policy & Investment Mandate:**
   - Single asset concentration limits ($\le 25.0\%$).
   - Maximum 1-day 95% VaR ($\le 5.0\%$).
   - Annualized volatility threshold ($\le 25.0\%$).
   - Drawdown warning ($15.0\%$) and mandatory de-risking trigger ($20.0\%$).
2. **Basel Committee Market Risk Backtesting Standards:**
   - Basel Traffic Light classification rules for 250-day 99% VaR.
   - Supervisory capital multiplier penalties (Green: 3.0x, Yellow: 3.4x–3.85x, Red: 4.0x).
   - Kupiec and Christoffersen backtesting criteria.
3. **Stress Testing Guidelines:**
   - Mandatory quarterly replay of 2008 GFC, 2020 COVID, 2022 Rate Shock, and 2023 SVB Crisis.
   - Reverse stress testing methodology and capital breach diagnosis.

### Semantic Search & Citations
- Input questions (e.g. *"Which risk limits have been breached?"*) trigger vector similarity search against pre-indexed policy chunks.
- The retrieved document titles, sections, and excerpts are passed alongside quantitative evidence.
- The AI includes institutional policy citations in its final response.

---

## 4. Prompt Injection & Jailbreak Defense

The system screens incoming user queries using regular expression matching against known adversarial jailbreak patterns:
- Instruction overrides (*"ignore all previous instructions"*).
- Persona / roleplaying bypasses (*"DAN mode"*, *"pretend you have no limits"*).
- System prompt exfiltration attempts.
- Arbitrary SQL, shell, or code execution commands.

Queries failing security screening are immediately rejected without calling the LLM.

---

## 5. Conversational Natural Language Q&A

The platform supports interactive natural language queries, translating complex user intent into verified quantitative retrievals:

| Sample User Query | Automated Routing & Grounding |
| :--- | :--- |
| *"Which asset contributes most to portfolio volatility?"* | Queries Euler percentage contribution to risk (PCR); cites top asset and volatility percentage. |
| *"Which risk limits have been breached?"* | Evaluates deterministic policy breach engine; lists specific violated mandate rules, current levels, and limit thresholds. |
| *"How did my portfolio perform during the 2008 GFC?"* | Retrieves historical crisis replay simulation; reports dollar loss, percentage drawdown, and recovery return needed. |
| *"What are my largest risk exposures?"* | Sorts portfolio holdings by market value weight; cites top weights and portfolio equity beta. |
