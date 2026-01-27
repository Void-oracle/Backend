"""
API endpoints for prediction history
"""
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
import logging

from app.core.market_database import get_market_db, DB_DIR
from app.services.market_oracle import market_oracle

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/history/{market_id}")
async def get_market_history(
    market_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    time_range_hours: Optional[int] = Query(default=None)
):
    """
    Get prediction history for a market from its dedicated database
    
    - **market_id**: Market identifier (e.g., "market_1")
    - **limit**: Maximum number of records to return
    - **time_range_hours**: Filter by time range (optional)
    """
    try:
        market_db = get_market_db(market_id)
        history = market_db.get_predictions(
            limit=limit,
            since_hours=time_range_hours
        )
        
        return {
            "market_id": market_id,
            "count": len(history),
            "history": history
        }
        
    except Exception as e:
        logger.error(f"Error fetching history for {market_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/latest/{market_id}")
async def get_market_latest(market_id: str):
    """
    Get latest prediction for a market
    
    - **market_id**: Market identifier
    """
    try:
        market_db = get_market_db(market_id)
        latest = market_db.get_latest_prediction()
        
        if not latest:
            raise HTTPException(
                status_code=404,
                detail=f"No predictions found for market {market_id}"
            )
        
        return latest
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching latest for {market_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/markets")
async def list_markets():
    """Get list of all tracked markets with latest AI scores"""
    try:
        markets = []
        
        # Scan for market database files
        if DB_DIR.exists():
            for db_file in sorted(DB_DIR.glob("market_*.db")):
                market_id = db_file.stem
                market_db = get_market_db(market_id)
                
                info = market_db.get_market_info()
                stats = market_db.get_stats()
                latest = market_db.get_latest_prediction()
                
                if info:
                    markets.append({
                        "market_id": market_id,
                        "ticker": info["ticker"],
                        "query": info["query"],
                        "description": info.get("query", ""),  # Use query as description
                        "status": info["status"],
                        "deadline": info.get("deadline"),
                        "category": info.get("category", "markets"),
                        # AI data from latest prediction
                        "ai_score": latest.get("ai_score", 50.0) if latest else 50.0,
                        "market_score": latest.get("market_score", 50.0) if latest else 50.0,
                        "divergence": latest.get("divergence_index", 0.0) if latest else 0.0,
                        "confidence": latest.get("confidence", 0.5) if latest else 0.5,
                        "last_update": latest.get("timestamp") if latest else None,
                        "vocal_summary": latest.get("vocal_summary", "") if latest else "",
                        # Stats
                        "predictions_count": stats["predictions_count"],
                        "events_count": stats["events_count"]
                    })
        
        # Sort by market number
        markets.sort(key=lambda x: int(x["market_id"].replace("market_", "")))
        
        return {
            "count": len(markets),
            "markets": markets
        }
    except Exception as e:
        logger.error(f"Error listing markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/search")
async def search_polymarket(query: str, ticker: str = ""):
    """
    Search Polymarket for matching prediction markets
    
    - **query**: Market question to search for
    - **ticker**: Optional ticker for context
    """
    try:
        result = await market_oracle.get_market_probability(query, ticker)
        return result
    except Exception as e:
        logger.error(f"Polymarket search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/polymarket/trending")
async def get_trending_markets(limit: int = Query(default=10, ge=1, le=50)):
    """
    Get trending prediction markets from Polymarket
    
    - **limit**: Number of markets to return
    """
    try:
        markets = await market_oracle.get_popular_markets(limit)
        return {
            "count": len(markets),
            "source": "polymarket",
            "markets": markets
        }
    except Exception as e:
        logger.error(f"Error fetching trending markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))
