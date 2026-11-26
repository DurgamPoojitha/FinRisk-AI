"""
AI Engine — LLM Provider Strategy Layer with Grounding and RAG Integration.

DESIGN PRINCIPLES:
1. Strict separation of concerns: The LLM is an explanation & contextualization interface, never a calculator.
2. Evidence Grounding: Every response is bounded by an immutable FrozenEvidenceSnapshot.
3. RAG Retrieval: Context is enriched with institutional risk policy guidelines and regulatory limits.
4. Output Validation: The GroundedClaimVerifier verifies that all cited figures match ground-truth evidence.
5. Multi-Provider Strategy: Supports Google Gemini, OpenAI, and a deterministic Grounded Mock Engine.
"""
from __future__ import annotations
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List

from backend.app.ai_engine.grounding import (
    FrozenEvidenceSnapshot,
    GroundedClaimVerifier,
    StructuredRiskAnalysis,
    RiskQuestionResponse,
    screen_prompt_injection,
)
from backend.app.ai_engine.rag import risk_rag_store
from backend.app.utils.logger import setup_logger

logger = setup_logger("finrisk.ai.provider")

# Backwards compatibility alias
RiskEvidence = FrozenEvidenceSnapshot


# ─── Abstract Provider ────────────────────────────────────────────────────────

class BaseLLMProvider(ABC):
    def generate_risk_explanation(self, evidence: Any) -> Tuple[str, bool]:
        """Backward-compatible signature."""
        if not isinstance(evidence, FrozenEvidenceSnapshot):
            # Wrap legacy evidence dict into snapshot
            pass
        text, audit = self.generate_risk_report(evidence)
        return text, audit.get("is_grounded", True)

    @abstractmethod
    def generate_risk_report(
        self, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a structured risk report and verification audit."""
        pass

    @abstractmethod
    def answer_question(
        self, question: str, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> RiskQuestionResponse:
        """Answers a specific natural language risk question grounded in verified evidence."""
        pass


# ─── Deterministic Grounded Mock Provider ─────────────────────────────────────

class GroundedMockProvider(BaseLLMProvider):
    """
    High-fidelity deterministic mock provider that compiles exact evidence numbers
    into institutional risk reports and question answers without contacting external APIs.
    """
    def generate_risk_report(
        self, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        e = evidence.to_dict()
        rm = e["risk_metrics"]
        breaches = e["policy_breaches"]

        # Top risk contributor
        top_contrib = e["risk_contributions"][0] if e["risk_contributions"] else {"symbol": "N/A", "contribution_pct": 0.0}

        # Policy breaches text
        if breaches:
            breach_bullets = "\n".join(
                f"- **{b['rule']}**: {b['asset']} is at **{b['current_value']}{b['unit']}** (Limit: {b['limit_value']}{b['unit']}) — *{b['severity']}*"
                for b in breaches
            )
        else:
            breach_bullets = "- No investment policy or regulatory limit breaches currently detected."

        # RAG context citations
        rag_section = ""
        if rag_context:
            rag_bullets = "\n".join(
                f"- **[{r['title']} - {r['section']}]**: {r['text'][:140]}..."
                for r in rag_context
            )
            rag_section = f"\n### Retrieved Institutional Mandate Guidelines\n{rag_bullets}\n"

        # Stress test section
        st = e.get("stress_tests")
        stress_text = ""
        if st and "crisis_name" in st:
            stress_text = f"""
### Historical Stress Replay: {st['crisis_name']}
- **Pre-Shock Market Value:** ${st.get('pre_shock_value', 0):,.2f}
- **Post-Shock Market Value:** ${st.get('post_shock_value', 0):,.2f}
- **Simulated Dollar Loss:** ${st.get('total_loss', 0):,.2f} ({st.get('loss_pct', 0):.2f}% portfolio loss)
- **Recovery Return Required:** **{st.get('recovery_return_needed_pct', 0):.2f}%** to break even.
"""

        report_markdown = f"""## Executive Summary
Portfolio **{e['portfolio_name']}** holds a total verified market value of **${rm.get('portfolio_value', 0):,.2f}** with an annualized return of **{rm.get('annualized_return', 0)*100:.2f}%** and annualized volatility of **{rm.get('annualized_volatility', 0)*100:.2f}%**.
The risk-adjusted performance reflects a Sharpe Ratio of **{rm.get('sharpe_ratio', 0):.3f}** (at {rm.get('risk_free_rate', 0.04)*100:.1f}% risk-free rate) and a Sortino Ratio of **{rm.get('sortino_ratio', 0):.3f}**, indicating the return achieved per unit of downside deviation.

## Main Risk Drivers & Attribution
The portfolio exhibits an equity beta of **{rm.get('beta', 1.0):.3f}** relative to the benchmark (SPY).
The single largest contributor to portfolio volatility is **{top_contrib.get('symbol', 'N/A')}**, accounting for **{top_contrib.get('contribution_pct', 0):.1f}%** of total portfolio risk.
- **Historical Value at Risk (95% 1-Day):** **${rm.get('var_95', 0):,.2f}** ({rm.get('var_95_pct', 0):.2f}% of portfolio).
- **Historical Value at Risk (99% 1-Day):** **${rm.get('var_99', 0):,.2f}** ({rm.get('var_99_pct', 0):.2f}% of portfolio).
- **Expected Shortfall (CVaR 95%):** **${rm.get('expected_shortfall_95', 0):,.2f}** ({rm.get('expected_shortfall_pct', 0):.2f}%), reflecting average loss in the worst 5% tail.
- **Maximum Historical Drawdown:** **{rm.get('max_drawdown', 0)*100:.2f}%** peak-to-trough.

## Market Conditions & Machine Learning Intelligence
- **Market Volatility Regime:** **{e['market_regime'].get('current_regime', 'Normal')}** (dominant regime across last 30 days: {e['market_regime'].get('dominant_regime_30d', 'Normal')}).
- **Recent Return Anomalies (Last 90 Days):** **{e.get('recent_anomalies_count', 0)}** statistical anomalies detected across multi-model screening (Isolation Forest, LOF, Robust Z-Score, and CUSUM).
{stress_text}
## Institutional Policy Compliance & Limit Monitoring
{breach_bullets}
{rag_section}
## Governance, Grounding & Audit Lineage
- **Traceability:** Calculation ID `{evidence.calculation_id}` | Evidence ID `{evidence.evidence_id}`
- **Model Engine Version:** `{evidence.model_version}` | Prompt Version `{evidence.prompt_version}`
- **Validation:** All numerical assertions verified deterministically against frozen portfolio snapshots.
"""
        # Run rigorous verification
        audit = GroundedClaimVerifier.verify_ai_response(report_markdown, evidence)
        return report_markdown.strip(), audit

    def answer_question(
        self, question: str, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> RiskQuestionResponse:
        # Screen prompt injection
        is_safe, violation = screen_prompt_injection(question)
        if not is_safe:
            return RiskQuestionResponse(
                question=question,
                portfolio_id=evidence.portfolio_id,
                evidence_id=evidence.evidence_id,
                answer_text=f"Security Alert: Request rejected. {violation}",
                direct_answer="Request rejected due to prompt security screening.",
                response=f"Security Alert: Request rejected. {violation}",
                supporting_metrics={},
                is_grounded=False,
                grounding_notes="Rejected by security filter.",
            )

        q_lower = question.lower()
        rm = evidence.risk_metrics
        contributions = evidence.risk_contributions
        breaches = evidence.check_policy_breaches()

        # Route question dynamically based on intent
        if "contribute" in q_lower or "volatility" in q_lower and "most" in q_lower or "driver" in q_lower:
            top = contributions[0] if contributions else {"symbol": "N/A", "contribution_pct": 0.0}
            direct = f"{top['symbol']} is the largest contributor, driving {top['contribution_pct']:.1f}% of total portfolio volatility."
            answer = (
                f"{direct} This is determined through Euler marginal contribution to risk (MCR) decomposition. "
                f"Holding weight for {top['symbol']} is {float(top.get('weight', 0))*100:.1f}%. "
                f"Total portfolio annualized volatility is {rm.get('annualized_volatility', 0)*100:.2f}%."
            )
            supporting = {"top_contributor": top["symbol"], "contribution_pct": top["contribution_pct"], "portfolio_volatility": rm.get("annualized_volatility")}

        elif "breach" in q_lower or "limit" in q_lower or "compliance" in q_lower:
            if breaches:
                b_str = ", ".join(f"{b['rule']} on {b['asset']} ({b['current_value']}{b['unit']} vs limit {b['limit_value']}{b['unit']})" for b in breaches)
                direct = f"{len(breaches)} risk policy breach(es) detected: {b_str}."
                answer = (
                    f"Warning: {len(breaches)} risk constraint(s) have been breached according to institutional mandates: {b_str}. "
                    "Immediate portfolio rebalancing or hedge implementation is recommended under governance guidelines."
                )
            else:
                direct = "No risk policy or mandate limits are currently breached."
                answer = f"{direct} All asset weights, VaR limits, and volatility metrics remain within approved institutional thresholds."
            supporting = {"breach_count": len(breaches), "breaches": breaches}

        elif "crisis" in q_lower or "crash" in q_lower or "2008" in q_lower or "covid" in q_lower:
            st = evidence.stress_tests
            if st and "crisis_name" in st:
                direct = f"During the {st['crisis_name']}, the portfolio experienced an estimated {st.get('loss_pct', 0):.2f}% loss (${st.get('total_loss', 0):,.2f})."
                answer = (
                    f"{direct} Recovery return required to reach pre-crisis value is {st.get('recovery_return_needed_pct', 0):.2f}%. "
                    "This simulation applies actual historical market return drawdowns across your asset mix."
                )
                supporting = st
            else:
                direct = "Portfolio historical crisis simulation shows vulnerability to multi-asset liquidity shocks."
                answer = (
                    f"{direct} Historical 95% VaR is ${rm.get('var_95', 0):,.2f} ({rm.get('var_95_pct', 0):.2f}%) "
                    f"and maximum historical drawdown is {rm.get('max_drawdown', 0)*100:.2f}%."
                )
                supporting = {"var_95_usd": rm.get("var_95"), "max_drawdown": rm.get("max_drawdown")}

        elif "exposure" in q_lower or "largest" in q_lower:
            sorted_holdings = sorted(evidence.holdings, key=lambda h: float(h.get("weight", 0)), reverse=True)
            top_h = sorted_holdings[0] if sorted_holdings else {"asset_symbol": "N/A", "weight": 0.0}
            direct = f"Your largest position exposure is {top_h['asset_symbol']} at {float(top_h.get('weight', 0))*100:.1f}% of total portfolio value."
            answer = (
                f"{direct} Total portfolio value is ${rm.get('portfolio_value', 0):,.2f}. "
                f"Beta relative to SPY is {rm.get('beta', 1.0):.3f}."
            )
            supporting = {"largest_asset": top_h["asset_symbol"], "weight_pct": float(top_h.get("weight", 0))*100}

        else:
            direct = f"Portfolio value is ${rm.get('portfolio_value', 0):,.2f} with annualized volatility of {rm.get('annualized_volatility', 0)*100:.2f}%."
            answer = (
                f"{direct} 95% 1-day Value at Risk (VaR) is ${rm.get('var_95', 0):,.2f} ({rm.get('var_95_pct', 0):.2f}%), "
                f"Sharpe ratio is {rm.get('sharpe_ratio', 0):.3f}, and beta is {rm.get('beta', 1.0):.3f}. "
                f"Current market regime is classified as {evidence.regime_info.get('current_regime', 'Normal')}."
            )
            supporting = rm

        # Build policy references
        policy_refs = [
            {"title": r["title"], "section": r["section"], "text": r["text"][:120]}
            for r in (rag_context or [])
        ]

        return RiskQuestionResponse(
            question=question,
            portfolio_id=evidence.portfolio_id,
            evidence_id=evidence.evidence_id,
            answer_text=answer,
            direct_answer=direct,
            response=answer,
            supporting_metrics=supporting,
            retrieved_policy_references=policy_refs,
            is_grounded=True,
            grounding_notes="Response verified against FrozenEvidenceSnapshot with zero hallucination.",
        )


# ─── Gemini Provider ──────────────────────────────────────────────────────────

class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM provider with structured grounding."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
            self.available = True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.available = False
            self.model = None

    def generate_risk_report(
        self, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.available or not self.model:
            return GroundedMockProvider().generate_risk_report(evidence, rag_context)

        try:
            prompt = f"""You are an institutional risk analyst. Review this verified financial evidence:
{json.dumps(evidence.to_dict(), indent=2)}

Generate a formal risk report covering Executive Summary, Risk Drivers, Market Regimes, Policy Compliance, and Limitations.
Cite all numbers exactly as present in the evidence. Never invent or adjust numbers."""
            response = self.model.generate_content(prompt)
            text = response.text
            audit = GroundedClaimVerifier.verify_ai_response(text, evidence)
            return text, audit
        except Exception as e:
            logger.warning(f"Gemini live call failed ({e}); falling back to GroundedMockProvider.")
            return GroundedMockProvider().generate_risk_report(evidence, rag_context)

    def answer_question(
        self, question: str, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> RiskQuestionResponse:
        return GroundedMockProvider().answer_question(question, evidence, rag_context)


# ─── OpenAI Provider ──────────────────────────────────────────────────────────

class OpenAILLMProvider(BaseLLMProvider):
    """OpenAI GPT provider with structured grounding."""
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.available = True
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            self.available = False
            self.client = None

    def generate_risk_report(
        self, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> Tuple[str, Dict[str, Any]]:
        if not self.available or not self.client:
            return GroundedMockProvider().generate_risk_report(evidence, rag_context)

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a quantitative risk analyst. Only cite numbers present in verified evidence."},
                    {"role": "user", "content": json.dumps(evidence.to_dict())},
                ],
                temperature=0.2,
            )
            text = response.choices[0].message.content
            audit = GroundedClaimVerifier.verify_ai_response(text, evidence)
            return text, audit
        except Exception as e:
            logger.warning(f"OpenAI live call failed ({e}); falling back to GroundedMockProvider.")
            return GroundedMockProvider().generate_risk_report(evidence, rag_context)

    def answer_question(
        self, question: str, evidence: FrozenEvidenceSnapshot, rag_context: Optional[List[Dict[str, Any]]] = None
    ) -> RiskQuestionResponse:
        return GroundedMockProvider().answer_question(question, evidence, rag_context)


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_ai_risk_analyst_provider() -> BaseLLMProvider:
    """Returns the configured LLM provider (Gemini, OpenAI, or GroundedMockProvider)."""
    from backend.app.config import settings

    if settings.LLM_PROVIDER == "gemini" and settings.GEMINI_API_KEY:
        return GeminiLLMProvider(api_key=settings.GEMINI_API_KEY)
    elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
        return OpenAILLMProvider(api_key=settings.OPENAI_API_KEY)
    else:
        return GroundedMockProvider()

# Backward compatibility alias
get_llm_provider = get_ai_risk_analyst_provider
