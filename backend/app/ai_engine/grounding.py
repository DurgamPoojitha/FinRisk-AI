"""
AI Safety, Grounding, and Output Verification Engine:
- Pydantic Structured Output Validation
- Prompt Injection & Jailbreak Defense
- Unsupported Numerical Claim Detection & Citation Verification
- Frozen Evidence Snapshots with Traceability UUIDs
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any, Set
import re
import uuid
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator


# ─── Pydantic Structured AI Output Schemas ────────────────────────────────────

class RiskMetricCitation(BaseModel):
    metric_name: str = Field(..., description="Name of the cited metric, e.g. VaR 95%, Sharpe Ratio")
    cited_value: float = Field(..., description="Numerical value cited in the text")
    evidence_value: float = Field(..., description="Ground-truth value in the frozen evidence")
    unit: str = Field(default="", description="USD, %, ratio, etc.")
    is_verified: bool = Field(default=False)


class StructuredRiskAnalysis(BaseModel):
    """Strictly typed structured analysis required from LLM providers."""
    portfolio_id: str
    evidence_id: str
    calculation_id: str
    executive_summary: str = Field(..., min_length=20)
    top_risk_drivers: List[str] = Field(..., min_length=1)
    market_conditions_summary: str = Field(...)
    stress_test_interpretation: Optional[str] = None
    policy_compliance_summary: str = Field(...)
    policy_breaches_identified: List[str] = Field(default_factory=list)
    cited_metrics: List[RiskMetricCitation] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    is_grounded: bool = Field(default=False)
    grounding_score: float = Field(default=1.0, ge=0.0, le=1.0)
    hallucination_warnings: List[str] = Field(default_factory=list)


class RiskQuestionResponse(BaseModel):
    """Structured response for natural language Q&A risk assistant."""
    question: str
    portfolio_id: str
    evidence_id: str
    answer_text: str = Field(..., min_length=10)
    direct_answer: str
    supporting_metrics: Dict[str, Any]
    retrieved_policy_references: List[Dict[str, Any]] = Field(default_factory=list)
    is_grounded: bool = True
    grounding_notes: Optional[str] = None


# ─── Prompt Injection Protection ─────────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?rules",
    r"you\s+are\s+now\s+(an?\s+)?unrestricted",
    r"system\s*prompt\s*override",
    r"dan\s+mode",
    r"jailbreak",
    r"pretend\s+you\s+have\s+no\s+limits",
    r"bypass\s+(safety|rules|guardrails)",
    r"execute\s+(arbitrary|shell|sql|command)",
    r"delete\s+all\s+records",
    r"drop\s+table",
]


def screen_prompt_injection(user_input: str) -> Tuple[bool, Optional[str]]:
    """
    Screens input text for malicious prompt injection and system prompt override attempts.
    Returns: (is_safe, violation_reason)
    """
    clean_text = user_input.strip()
    if len(clean_text) > 3000:
        return False, "Input exceeds maximum character length of 3000."

    lower_text = clean_text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower_text):
            return False, f"Potential prompt injection attempt detected matching security rule: '{pattern}'."

    return True, None


# ─── Frozen Evidence Snapshot ─────────────────────────────────────────────────

class FrozenEvidenceSnapshot:
    """
    Immutable snapshot of all quantitative calculations, market regimes,
    anomalies, and policy limits for a single analysis instance.
    """
    def __init__(
        self,
        portfolio_id: str,
        portfolio_name: str = "",
        holdings: Optional[List[Dict[str, Any]]] = None,
        risk_metrics: Optional[Dict[str, Any]] = None,
        risk_contributions: Optional[List[Dict[str, Any]]] = None,
        regime_info: Optional[Dict[str, Any]] = None,
        recent_anomalies: Optional[List[Dict[str, Any]]] = None,
        stress_tests: Optional[Dict[str, Any]] = None,
        policy_limits: Optional[Dict[str, Any]] = None,
        model_version: str = "2.0.0",
        prompt_version: str = "2.0.0",
        *args,
        **kwargs,
    ):
        self.evidence_id = f"evid_{uuid.uuid4().hex[:12]}"
        self.calculation_id = f"calc_{uuid.uuid4().hex[:12]}"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.portfolio_id = portfolio_id
        self.portfolio_name = portfolio_name
        self.holdings = holdings or kwargs.get("holdings", [])
        
        # Build risk_metrics if passed individually or in kwargs
        if risk_metrics is not None:
            self.risk_metrics = risk_metrics
        else:
            self.risk_metrics = {
                "portfolio_value": kwargs.get("portfolio_value", 0.0),
                "annualized_return": kwargs.get("annualized_return", 0.0),
                "annualized_volatility": kwargs.get("annualized_volatility", 0.0),
                "sharpe_ratio": kwargs.get("sharpe_ratio", 0.0),
                "beta": kwargs.get("beta", 1.0),
                "var_95": kwargs.get("var_95", 0.0),
                "var_95_pct": kwargs.get("var_95_pct", 0.0),
                "var_99": kwargs.get("var_99", 0.0),
                "expected_shortfall_95": kwargs.get("expected_shortfall", 0.0),
                "max_drawdown": kwargs.get("max_drawdown", 0.0),
                "risk_free_rate": kwargs.get("risk_free_rate", 0.04),
            }

        self.risk_contributions = risk_contributions or kwargs.get("risk_contributions", [])
        self.regime_info = regime_info or {"current_regime": kwargs.get("current_regime", "Normal")}
        self.recent_anomalies = recent_anomalies or []
        self.stress_tests = stress_tests or kwargs.get("stress_test") or {}
        self.policy_limits = policy_limits or {
            "max_single_asset_weight_pct": 25.0,
            "max_var_95_pct": 5.0,
            "max_annual_volatility_pct": 25.0,
            "min_sharpe_ratio": 0.50,
            "max_drawdown_pct": 20.0,
        }
        self.model_version = model_version
        self.prompt_version = prompt_version

        # Build numerical lookup table for grounding verification
        self._ground_truth_numbers = self._build_number_index()
        self._allowed_assets = set(h["asset_symbol"].upper() for h in self.holdings) if self.holdings else set()

    def _build_number_index(self) -> Dict[str, float]:
        idx = {}
        rm = self.risk_metrics
        if "portfolio_value" in rm:
            idx["portfolio_value"] = float(rm["portfolio_value"])
        if "annualized_volatility" in rm:
            idx["volatility_pct"] = float(rm["annualized_volatility"]) * 100
        if "annualized_return" in rm:
            idx["return_pct"] = float(rm["annualized_return"]) * 100
        if "sharpe_ratio" in rm:
            idx["sharpe"] = float(rm["sharpe_ratio"])
        if "sortino_ratio" in rm:
            idx["sortino"] = float(rm["sortino_ratio"])
        if "calmar_ratio" in rm:
            idx["calmar"] = float(rm["calmar_ratio"])
        if "beta" in rm:
            idx["beta"] = float(rm["beta"])
        if "var_95" in rm:
            idx["var_95_usd"] = float(rm["var_95"])
        if "var_95_pct" in rm:
            idx["var_95_pct"] = float(rm["var_95_pct"])
        if "var_99" in rm:
            idx["var_99_usd"] = float(rm["var_99"])
        if "var_99_pct" in rm:
            idx["var_99_pct"] = float(rm["var_99_pct"])
        if "expected_shortfall_95" in rm:
            idx["es_95_usd"] = float(rm["expected_shortfall_95"])
        if "expected_shortfall_pct" in rm:
            idx["es_95_pct"] = float(rm["expected_shortfall_pct"])
        if "max_drawdown" in rm:
            idx["max_drawdown_pct"] = float(rm["max_drawdown"]) * 100

        # Holding weights
        for h in self.holdings:
            sym = h["asset_symbol"].upper()
            if "weight" in h:
                idx[f"{sym}_weight_pct"] = float(h["weight"]) * 100

        # Risk contributions
        for c in self.risk_contributions:
            sym = c.get("symbol", "").upper()
            if "contribution_pct" in c:
                idx[f"{sym}_risk_contrib_pct"] = float(c["contribution_pct"])

        return idx

    def check_policy_breaches(self) -> List[Dict[str, Any]]:
        """Evaluates portfolio metrics against institutional risk limits."""
        breaches = []
        rm = self.risk_metrics
        limits = self.policy_limits

        # 1. Single asset concentration
        max_allowed_weight = limits.get("max_single_asset_weight_pct", 25.0)
        for h in self.holdings:
            w_pct = float(h.get("weight", 0.0)) * 100
            if w_pct > max_allowed_weight:
                breaches.append({
                    "rule": "Single Asset Concentration Limit",
                    "asset": h["asset_symbol"],
                    "current_value": round(w_pct, 2),
                    "limit_value": max_allowed_weight,
                    "severity": "Warning" if w_pct <= max_allowed_weight * 1.2 else "Critical Breach",
                    "unit": "%",
                })

        # 2. VaR 95% Limit
        max_allowed_var = limits.get("max_var_95_pct", 5.0)
        var_pct = float(rm.get("var_95_pct", 0.0))
        if var_pct > max_allowed_var:
            breaches.append({
                "rule": "Value at Risk (95% 1-Day) Limit",
                "asset": "Portfolio",
                "current_value": round(var_pct, 2),
                "limit_value": max_allowed_var,
                "severity": "Critical Breach",
                "unit": "%",
            })

        # 3. Volatility Limit
        max_allowed_vol = limits.get("max_annual_volatility_pct", 25.0)
        vol_pct = float(rm.get("annualized_volatility", 0.0)) * 100
        if vol_pct > max_allowed_vol:
            breaches.append({
                "rule": "Annualized Volatility Limit",
                "asset": "Portfolio",
                "current_value": round(vol_pct, 2),
                "limit_value": max_allowed_vol,
                "severity": "Warning",
                "unit": "%",
            })

        return breaches

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "calculation_id": self.calculation_id,
            "created_at": self.created_at,
            "portfolio_id": self.portfolio_id,
            "portfolio_name": self.portfolio_name,
            "model_version": self.model_version,
            "prompt_version": self.prompt_version,
            "holdings": self.holdings,
            "risk_metrics": self.risk_metrics,
            "risk_contributions": self.risk_contributions,
            "market_regime": self.regime_info,
            "recent_anomalies_count": len(self.recent_anomalies),
            "policy_limits": self.policy_limits,
            "policy_breaches": self.check_policy_breaches(),
            "stress_tests": self.stress_tests,
        }


# ─── Grounded Claim Verifier ──────────────────────────────────────────────────

class GroundedClaimVerifier:
    """
    Verifies that all numerical assertions and asset references in generated AI text
    are grounded in the frozen evidence snapshot.
    """
    @classmethod
    def verify_ai_response(
        cls,
        ai_text: str,
        evidence: FrozenEvidenceSnapshot,
        tolerance_pct: float = 0.05,  # 5% relative tolerance
    ) -> Dict[str, Any]:
        warnings = []
        verified_citations = []
        ground_truth = evidence._ground_truth_numbers

        # 1. Check for unheld assets mentioned as holdings
        # Find potential tickers (2 to 5 uppercase letters preceded by a space or punctuation)
        found_tickers = set(re.findall(r'\b[A-Z]{2,5}\b', ai_text))
        # Common English words that look like tickers
        common_words = {"THE", "AND", "FOR", "VAR", "ALL", "NEW", "RISK", "USD", "GDP", "FED", "CPI", "USA", "SEC", "CEO", "CFO", "LOW", "HIGH", "MED"}
        suspect_tickers = found_tickers - common_words - evidence._allowed_assets - {"SPY", "QQQ"}

        if suspect_tickers:
            for st in suspect_tickers:
                # Check context: if mentioned as an asset
                if re.search(rf'\b{st}\b\s+(stock|shares|holding|position|contribution)', ai_text, re.IGNORECASE):
                    warnings.append(f"AI mentions asset '{st}' as a holding, but it is not in the portfolio.")

        # 2. Extract numerical figures with currency or percentage
        dollar_matches = re.findall(r'\$[\d,]+(?:\.\d+)?', ai_text)
        pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', ai_text)

        # Check dollar figures against ground truth USD figures
        usd_ground_truth = [v for k, v in ground_truth.items() if "usd" in k or "value" in k]
        for dm in dollar_matches:
            val = float(dm.replace("$", "").replace(",", ""))
            if val < 100:  # ignore trivial dollar amounts
                continue
            matched = any(abs(val - gt) / max(gt, 1.0) <= tolerance_pct for gt in usd_ground_truth)
            if matched:
                verified_citations.append({"type": "USD", "value": val, "verified": True})
            else:
                warnings.append(f"Cited dollar figure {dm} does not match any metric in the verified evidence.")

        # Check percentage figures against ground truth percentage figures
        pct_ground_truth = [v for k, v in ground_truth.items() if "pct" in k or "ratio" in k]
        for pm in pct_matches:
            val = float(pm)
            if val == 0.0 or val > 500:
                continue
            matched = any(abs(val - gt) <= max(0.5, gt * tolerance_pct) for gt in pct_ground_truth)
            if matched:
                verified_citations.append({"type": "%", "value": val, "verified": True})

        # Calculate grounding score
        total_claims = len(dollar_matches) + len(warnings)
        if total_claims == 0:
            grounding_score = 1.0
        else:
            grounding_score = max(0.0, 1.0 - (len(warnings) / max(total_claims, 1)))

        is_grounded = bool(len(warnings) == 0 or (len(warnings) <= 1 and grounding_score >= 0.85))

        return {
            "is_grounded": is_grounded,
            "grounding_score": round(grounding_score, 2),
            "warnings": warnings,
            "verified_citations_count": len(verified_citations),
            "unsupported_claims_count": len(warnings),
        }
