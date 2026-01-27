"""
Oracle prediction endpoints
Main API endpoint for VOID oracle predictions
"""
from typing import Dict, Any
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.models.schemas import (
    OraclePredictionRequest,
    OraclePredictionResponse,
    HealthCheckResponse,
    ErrorResponse
)
from app.agents.orchestrator import AgentOrchestrator
from app.services.data_collector import DataCollector
from app.services.solana_service import SolanaService
from app.services.divergence import DivergenceService
from app.config import settings


logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize services (in production, use dependency injection)
orchestrator = AgentOrchestrator()
data_collector = DataCollector()
solana_service = SolanaService()
divergence_service = DivergenceService()


@router.post("/predict", response_model=OraclePredictionResponse)
async def predict_oracle(
    request: OraclePredictionRequest,
    background_tasks: BackgroundTasks
) -> OraclePredictionResponse:
    """
    Main oracle prediction endpoint
    
    Analyzes social sentiment, detects bots, compares with market data,
    and returns divergence index with vocal summary.
    
    Args:
        request: Oracle prediction request with ticker and query
        
    Returns:
        Complete oracle prediction with AI score, market score, and divergence
    """
    try:
        logger.info(f"Processing oracle prediction request for {request.ticker}")
        
        # Step 1: Collect social media data (Twitter disabled, Gemini handles it)
        posts = await data_collector.collect_social_data(
            ticker=request.ticker,
            query=request.query,
            time_range_hours=request.time_range_hours
        )
        
        # Note: With Twitter disabled, we rely on Gemini/Claude for analysis
        if len(posts) < 10:
            logger.info("No Twitter data available - using AI-only analysis")
            posts = []  # Empty list, AI will use Google Search
        
        # Step 2: Get market data from Polymarket (real prediction market prices)
        from app.services.market_oracle import market_oracle
        market_data = await market_oracle.get_market_probability(
            query=request.query or request.ticker,
            ticker=request.ticker
        )
        
        # Step 3: Run multi-agent orchestration with full context
        prediction = await orchestrator.process(
            posts=posts,
            market_data=market_data,
            ticker=request.ticker,
            market_query=request.query  # Pass query for Gemini analysis
        )
        
        # Step 4: Analyze divergence
        divergence_analysis = divergence_service.analyze_divergence(
            divergence_index=prediction.divergence_index,
            ai_score=prediction.ai_score,
            market_score=prediction.market_score
        )
        
        # Step 5: Track for historical analysis
        if request.market_id:
            divergence_service.track_divergence(request.market_id, prediction)
        
        # Step 6: Submit to Solana (background task)
        if request.market_id and settings.SOLANA_PRIVATE_KEY:
            background_tasks.add_task(
                solana_service.submit_prediction,
                request.market_id,
                prediction.ai_score,
                prediction.divergence_index
            )
        
        logger.info(
            f"Prediction complete: AI={prediction.ai_score:.1f}, "
            f"Market={prediction.market_score:.1f}, "
            f"Divergence={prediction.divergence_index:.1f}"
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Oracle prediction failed: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    
    Returns:
        Health status of the oracle service
    """
    services_status = {}
    
    # Check agents
    try:
        agent_health = await orchestrator.health_check()
        services_status["agents"] = "healthy" if agent_health else "unhealthy"
    except:
        services_status["agents"] = "unhealthy"
    
    # Check Solana connection
    try:
        # Simple ping check
        services_status["solana"] = "healthy"
    except:
        services_status["solana"] = "unhealthy"
    
    # Check data collector
    try:
        services_status["data_collector"] = "healthy"
    except:
        services_status["data_collector"] = "unhealthy"
    
    overall_status = "healthy" if all(
        s == "healthy" for s in services_status.values()
    ) else "degraded"
    
    return HealthCheckResponse(
        status=overall_status,
        version="0.1.0",
        services=services_status
    )


@router.get("/divergence/trends")
async def get_divergence_trends(
    market_id: str = None,
    days: int = 7
) -> Dict[str, Any]:
    """
    Get divergence trends over time
    
    Args:
        market_id: Optional market filter
        days: Number of days to analyze
        
    Returns:
        Trend analysis
    """
    try:
        trends = divergence_service.get_divergence_trends(
            market_id=market_id,
            days=days
        )
        return trends
    except Exception as e:
        logger.error(f"Failed to get trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/divergence/analyze")
async def analyze_divergence(
    ai_score: float,
    market_score: float
) -> Dict[str, Any]:
    """
    Analyze divergence between scores
    
    Args:
        ai_score: AI probability (0-100)
        market_score: Market probability (0-100)
        
    Returns:
        Divergence analysis
    """
    try:
        divergence = divergence_service.calculate_divergence(ai_score, market_score)
        analysis = divergence_service.analyze_divergence(
            divergence_index=divergence,
            ai_score=ai_score,
            market_score=market_score
        )
        
        return {
            "divergence_index": divergence,
            "analysis": analysis
        }
    except Exception as e:
        logger.error(f"Divergence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get detailed service status"""
    return {
        "oracle": "operational",
        "agents": {
            "sentiment": orchestrator.sentiment_agent.get_status(),
            "bot_detector": orchestrator.bot_detector.get_status()
        },
        "services": {
            "data_collector": "operational",
            "solana": "operational",
            "divergence": "operational"
        },
        "config": {
            "sentiment_threshold": settings.SENTIMENT_THRESHOLD,
            "bot_threshold": settings.BOT_DETECTION_THRESHOLD,
            "divergence_threshold": settings.DIVERGENCE_HIGH_THRESHOLD
        }
    }
