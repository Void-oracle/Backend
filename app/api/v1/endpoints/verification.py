"""
Event Verification API Endpoints
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime
import logging

from app.services.oracle_engine import oracle_engine

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/verify/{market_id}")
async def verify_market_event(
    market_id: str,
    deadline: Optional[str] = Query(None, description="Event deadline in ISO format")
):
    """
    Verify if a market's event condition has been met
    
    This endpoint allows checking if the oracle has detected
    that the market's condition has occurred.
    
    Example: "Has Bitcoin reached $150,000?"
    
    Returns:
        - decision: YES / NO / UNCERTAIN
        - confidence: Confidence score (0-1)
        - reasoning: Explanation of decision
        - evidence: Supporting data
    """
    try:
        # Parse deadline if provided
        deadline_dt = None
        if deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format. Use ISO 8601 format.")
        
        # Verify event
        result = await oracle_engine.verify_market_event(
            market_id=market_id,
            deadline=deadline_dt
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Market {market_id} not found or not initialized"
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error verifying market {market_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{market_id}")
async def get_market_status(market_id: str):
    """
    Get current monitoring status of a market
    
    Returns:
        - monitoring_active: Is the market being monitored?
        - last_update: When was the last prediction update?
        - last_check: When was the last event check?
        - update_count: How many updates have been made?
        - current_prediction: Latest AI prediction
    """
    try:
        status = oracle_engine.get_market_status(market_id)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Market {market_id} not found"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting status for {market_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/markets")
async def list_monitored_markets():
    """
    List all markets currently being monitored
    
    Returns list of market IDs and their basic info
    """
    try:
        markets = []
        for market_id in oracle_engine.market_states.keys():
            status = oracle_engine.get_market_status(market_id)
            if status:
                markets.append(status)
        
        return {
            "count": len(markets),
            "markets": markets
        }
        
    except Exception as e:
        logger.error(f"Error listing markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
