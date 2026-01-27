"""
Markets Management API Endpoints
Create, list, and manage prediction markets dynamically
"""
from fastapi import APIRouter, HTTPException, Body
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
import logging
import asyncio

from app.core.markets_manager import markets_manager
from app.services.oracle_engine import oracle_engine

logger = logging.getLogger(__name__)
router = APIRouter()


class CreateMarketRequest(BaseModel):
    """Request to create a new market"""
    ticker: str = Field(..., description="Ticker symbol (e.g., BTC-200K, ETH-10K)")
    query: str = Field(..., description="Market question (e.g., 'Will Bitcoin reach $200k by 2026?')")
    description: Optional[str] = Field(None, description="Market description")
    category: str = Field("crypto", description="Category: crypto, tech, politics, sports, other")
    deadline: Optional[str] = Field(None, description="Deadline in ISO format (e.g., 2026-12-31T23:59:59)")
    check_interval_minutes: int = Field(30, description="How often to update predictions (minutes)")


@router.post("/create")
async def create_market(request: CreateMarketRequest):
    """
    Create a new prediction market dynamically
    
    The market will be:
    1. Saved to database with auto-generated ID
    2. Initialized with AI analysis
    3. Monitored automatically
    
    All without server restart!
    
    Example request:
    ```json
    {
        "ticker": "BTC-200K",
        "query": "Will Bitcoin reach $200,000 by end of 2026?",
        "description": "Bitcoin price prediction for 2026",
        "category": "crypto",
        "deadline": "2026-12-31T23:59:59"
    }
    ```
    """
    try:
        # Parse deadline
        deadline = None
        if request.deadline:
            try:
                deadline = datetime.fromisoformat(request.deadline.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format. Use ISO 8601 (e.g., 2026-12-31T23:59:59)")
        
        # Create market in registry (market_id auto-generated)
        market = markets_manager.create_market(
            ticker=request.ticker,
            query=request.query,
            description=request.description,
            category=request.category,
            deadline=deadline,
            check_interval_minutes=request.check_interval_minutes,
            created_by="user"
        )
        
        market_id = market["market_id"]
        
        # Start monitoring immediately (hot reload!)
        logger.info(f"Starting monitoring for new market: {market_id}")
        
        # Initialize and start monitoring the new market
        asyncio.create_task(oracle_engine.initialize_market(
            market_id=market_id,
            ticker=market["ticker"],
            query=market["query"],
            deadline=market["deadline"],
            target_tweets=500
        ))
        asyncio.create_task(oracle_engine.monitor_market(market_id))
        
        return {
            "success": True,
            "message": f"Market {market_id} created and monitoring started",
            "market": market
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating market: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_markets(
    status: Optional[str] = None,
    category: Optional[str] = None,
    monitoring_active: Optional[bool] = None
):
    """
    List all markets with optional filters and AI scores
    
    - **status**: Filter by status (active, completed, archived)
    - **category**: Filter by category (crypto, tech, politics, sports, other)
    - **monitoring_active**: Filter by monitoring status
    """
    try:
        from app.core.market_database import get_market_db
        
        markets = markets_manager.list_markets(
            status=status,
            category=category,
            monitoring_active=monitoring_active
        )
        
        # Enrich with AI scores from market databases
        enriched_markets = []
        for market in markets:
            market_data = dict(market)
            market_id = market["market_id"]
            
            try:
                market_db = get_market_db(market_id)
                latest = market_db.get_latest_prediction()
                
                if latest:
                    market_data["ai_score"] = latest.get("ai_score", 50.0)
                    market_data["market_score"] = latest.get("market_score", 50.0)
                    market_data["divergence_index"] = latest.get("divergence_index", 0.0)
                    market_data["confidence"] = latest.get("confidence", 0.5)
                    market_data["vocal_summary"] = latest.get("vocal_summary", "")
                    market_data["last_prediction"] = latest.get("timestamp")
                else:
                    # No predictions yet - use defaults
                    market_data["ai_score"] = 50.0
                    market_data["market_score"] = 50.0
                    market_data["divergence_index"] = 0.0
                    market_data["confidence"] = 0.5
                    market_data["vocal_summary"] = "Awaiting first AI analysis..."
                    market_data["last_prediction"] = None
            except Exception as e:
                logger.debug(f"No market DB for {market_id}: {e}")
                market_data["ai_score"] = 50.0
                market_data["market_score"] = 50.0
                market_data["divergence_index"] = 0.0
                market_data["confidence"] = 0.5
                market_data["vocal_summary"] = "Initializing..."
                market_data["last_prediction"] = None
            
            enriched_markets.append(market_data)
        
        return {
            "count": len(enriched_markets),
            "markets": enriched_markets
        }
        
    except Exception as e:
        logger.error(f"Error listing markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{market_id}")
async def get_market(market_id: str):
    """Get details for a specific market"""
    try:
        market = markets_manager.get_market(market_id)
        
        if not market:
            raise HTTPException(status_code=404, detail=f"Market {market_id} not found")
        
        return market
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting market {market_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{market_id}/pause")
async def pause_market(market_id: str):
    """Pause monitoring for a market"""
    try:
        market = markets_manager.get_market(market_id)
        
        if not market:
            raise HTTPException(status_code=404, detail=f"Market {market_id} not found")
        
        markets_manager.update_market_status(
            market_id=market_id,
            status="paused",
            monitoring_active=False
        )
        
        # Stop monitoring in oracle engine
        if oracle_engine and market_id in oracle_engine.market_states:
            # The monitoring task will naturally stop when it checks status
            logger.info(f"Paused monitoring for {market_id}")
        
        return {
            "success": True,
            "message": f"Market {market_id} paused"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error pausing market: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{market_id}/resume")
async def resume_market(market_id: str):
    """Resume monitoring for a paused market"""
    try:
        market = markets_manager.get_market(market_id)
        
        if not market:
            raise HTTPException(status_code=404, detail=f"Market {market_id} not found")
        
        markets_manager.update_market_status(
            market_id=market_id,
            status="active",
            monitoring_active=True
        )
        
        # Restart monitoring
        asyncio.create_task(oracle_engine.monitor_market(market_id))
        
        return {
            "success": True,
            "message": f"Market {market_id} resumed"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resuming market: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{market_id}")
async def delete_market(market_id: str):
    """
    Delete a market
    
    This will:
    1. Stop monitoring
    2. Remove from registry
    3. Archive the market database
    """
    try:
        market = markets_manager.get_market(market_id)
        
        if not market:
            raise HTTPException(status_code=404, detail=f"Market {market_id} not found")
        
        # Mark as archived first
        markets_manager.update_market_status(market_id, "archived", monitoring_active=False)
        
        # Archive the market database
        from app.core.market_database import get_market_db
        market_db = get_market_db(market_id)
        market_db.mark_completed()
        market_db.archive()
        
        # Remove from registry
        markets_manager.delete_market(market_id)
        
        return {
            "success": True,
            "message": f"Market {market_id} deleted and archived"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting market: {e}")
        raise HTTPException(status_code=500, detail=str(e))
