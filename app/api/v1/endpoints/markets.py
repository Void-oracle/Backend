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
    market_id: str = Field(..., description="Unique market identifier")
    ticker: str = Field(..., description="Ticker symbol (e.g., SOL, BTC)")
    query: str = Field(..., description="Market question")
    description: Optional[str] = Field(None, description="Market description")
    category: str = Field("markets", description="Category: markets, sports, news, other")
    deadline: Optional[str] = Field(None, description="Deadline in ISO format")
    target_tweets: int = Field(500, description="Number of tweets to collect initially")
    check_interval_minutes: int = Field(30, description="How often to check for updates")


@router.post("/create")
async def create_market(request: CreateMarketRequest):
    """
    Create a new prediction market dynamically
    
    The market will be:
    1. Saved to database
    2. Initialized with AI analysis
    3. Monitored automatically
    
    All without server restart!
    """
    try:
        # Parse deadline
        deadline = None
        if request.deadline:
            try:
                deadline = datetime.fromisoformat(request.deadline.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid deadline format. Use ISO 8601.")
        
        # Create market in registry
        market = markets_manager.create_market(
            market_id=request.market_id,
            ticker=request.ticker,
            query=request.query,
            description=request.description,
            category=request.category,
            deadline=deadline,
            target_tweets=request.target_tweets,
            check_interval_minutes=request.check_interval_minutes,
            created_by="api"
        )
        
        # Start monitoring immediately (hot reload!)
        logger.info(f"Starting monitoring for new market: {request.market_id}")
        # Initialize and start monitoring the new market
        asyncio.create_task(oracle_engine.initialize_market(
            market_id=market["market_id"],
            ticker=market["ticker"],
            query=market["query"],
            deadline=market["deadline"],
            target_tweets=market.get("target_tweets", 500)
        ))
        asyncio.create_task(oracle_engine.monitor_market(market["market_id"]))
        
        return {
            "success": True,
            "message": f"Market {request.market_id} created and monitoring started",
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
    List all markets with optional filters
    
    - **status**: Filter by status (active, completed, archived)
    - **category**: Filter by category (markets, sports, news, other)
    - **monitoring_active**: Filter by monitoring status
    """
    try:
        markets = markets_manager.list_markets(
            status=status,
            category=category,
            monitoring_active=monitoring_active
        )
        
        return {
            "count": len(markets),
            "markets": markets
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
