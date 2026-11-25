from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from backend.app.models.database import get_db
from backend.app.models.portfolio import AIReport, RiskMetric
from backend.app.schemas.ai import AIAnalysisRequest, AIAnalysisResponse
from backend.app.services.portfolio_service import (
    get_portfolio_detail, compute_risk_metrics, compute_risk_contributions,
    compute_correlations, run_portfolio_stress_test
)
from backend.app.ml_engine.anomaly import detect_anomalies
from backend.app.ml_engine.regime import get_current_regime
from backend.app.ai_engine.provider import RiskEvidence, get_llm_provider, get_ai_risk_analyst_provider
from backend.app.ai_engine.grounding import FrozenEvidenceSnapshot, RiskQuestionResponse
from backend.app.ai_engine.rag import risk_rag_store
from backend.app.utils.logger import setup_logger
from pydantic import BaseModel, Field
import pandas as pd

logger = setup_logger("finrisk.api.ai")
router = APIRouter(prefix="/ai", tags=["AI Analyst"])

def _load_prices() -> pd.DataFrame:
    df = pd.read_csv("data/sample/historical_prices.csv", parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df

@router.post("/risk-analysis", response_model=AIAnalysisResponse, summary="Generate explainable AI risk analyst report")
def generate_portfolio_analysis(payload: AIAnalysisRequest, db: Session = Depends(get_db)):
    try:
        portfolio_id = payload.portfolio_id
        
        # 1. Gather all quantitative evidence
        detail = get_portfolio_detail(db, portfolio_id)
        metrics = compute_risk_metrics(db, portfolio_id)
        contributions = compute_risk_contributions(db, portfolio_id)
        correlations = compute_correlations(db, portfolio_id)
        
        # ML metrics
        prices_df = _load_prices()
        regime_info = get_current_regime(prices_df, symbol="SPY")
        
        # Anomaly count in last 90 days
        anomalies_df = detect_anomalies(prices_df, symbol="SPY")
        recent_anomalies = int(anomalies_df.tail(90)["is_anomaly"].sum())
        
        # Optional Stress test metrics
        stress_results = None
        if payload.include_stress_test and payload.stress_scenario_id:
            stress_results = run_portfolio_stress_test(
                db, portfolio_id, payload.stress_scenario_id
            )
            
        # 2. Build structured RiskEvidence
        evidence = RiskEvidence(
            portfolio_id=portfolio_id,
            portfolio_name=detail["name"],
            portfolio_value=detail["total_value"],
            annualized_return=metrics["annualized_return"],
            annualized_volatility=metrics["annualized_volatility"],
            sharpe_ratio=metrics["sharpe_ratio"],
            beta=metrics["beta"],
            var_95=metrics["var_95"],
            var_95_pct=metrics["var_95_pct"],
            var_99=metrics["var_99"],
            expected_shortfall=metrics["expected_shortfall_95"],
            max_drawdown=metrics["max_drawdown"],
            risk_contributions=contributions["contributions"],
            highly_correlated_pairs=correlations["highly_correlated_pairs"],
            current_regime=regime_info["current_regime"],
            recent_anomaly_count=recent_anomalies,
            stress_test=stress_results,
            risk_free_rate=metrics["risk_free_rate"],
            data_start=metrics["data_start"],
            data_end=metrics["data_end"]
        )
        
        # 3. Request explanation from LLM provider
        provider = get_llm_provider()
        analysis_text, is_validated = provider.generate_risk_explanation(evidence)
        
        # 4. Save analysis to the database for auditability
        ai_report = AIReport(
            portfolio_id=portfolio_id,
            prompt=f"Explain risk for portfolio {detail['name']}",
            report_content=analysis_text,
            evidence_used=evidence.to_dict(),
            is_validated=is_validated
        )
        db.add(ai_report)
        db.commit()
        db.refresh(ai_report)
        
        # Limit list of limitations
        limitations = [
            "Backward-looking model limitations based on historical price correlations.",
            "Historical VaR assumes return stationary and fails in flash crash events.",
            "Market regime mapping is statistical (K-Means) and ignores global macro inputs."
        ]
        
        return AIAnalysisResponse(
            portfolio_id=portfolio_id,
            generated_at=ai_report.generated_at,
            provider=provider.__class__.__name__,
            is_validated=is_validated,
            analysis=analysis_text,
            evidence_summary=evidence.to_dict(),
            limitations=limitations
        )
        
    except ValueError as e:
        logger.error(f"Validation error generating analysis: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Unexpected error in AI explainability layer")
        raise HTTPException(status_code=500, detail=f"AI Explanation failed: {str(e)}")


class AskAIRequest(BaseModel):
    portfolio_id: str
    question: str = Field(..., min_length=3, max_length=1000)


@router.post("/ask", response_model=RiskQuestionResponse, summary="Ask a natural language risk question grounded in verified portfolio evidence")
def ask_portfolio_risk_question(payload: AskAIRequest, db: Session = Depends(get_db)):
    try:
        portfolio_id = payload.portfolio_id
        detail = get_portfolio_detail(db, portfolio_id)
        metrics = compute_risk_metrics(db, portfolio_id)
        contributions = compute_risk_contributions(db, portfolio_id)

        prices_df = _load_prices()
        regime_info = get_current_regime(prices_df, symbol="SPY")
        anomalies_df = detect_anomalies(prices_df, symbol="SPY")
        recent_anom = anomalies_df.tail(90).to_dict(orient="records")

        # Create frozen evidence snapshot
        snapshot = FrozenEvidenceSnapshot(
            portfolio_id=portfolio_id,
            portfolio_name=detail["name"],
            holdings=detail["holdings"],
            risk_metrics=metrics,
            risk_contributions=contributions["contributions"],
            regime_info=regime_info,
            recent_anomalies=recent_anom,
        )

        # Retrieve RAG context for question
        rag_ctx = risk_rag_store.search(payload.question, top_k=2)

        # Answer via provider
        provider = get_ai_risk_analyst_provider()
        answer = provider.answer_question(payload.question, snapshot, rag_context=rag_ctx)
        return answer

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error answering AI risk question: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")
