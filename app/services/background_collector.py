"""
Background data collector - runs predictions for all markets automatically
"""
import asyncio
import logging
from datetime import datetime
from typing import List, Dict, Any

from app.agents.orchestrator import AgentOrchestrator
from app.core.database_sqlite import save_prediction, get_all_markets

logger = logging.getLogger(__name__)

# Define markets to track
TRACKED_MARKETS = [
    {
        "market_id": "market_1",
        "ticker": "SOL",
        "query": "Will Solana reach $300 by February 2026?",
        "description": "Will Solana reach $300 USD by February 28, 2026?",
    },
    {
        "market_id": "market_2",
        "ticker": "ETH",
        "query": "Will an Ethereum spot ETF be approved in Q1 2026?",
        "description": "Will an Ethereum spot ETF be approved in Q1 2026?",
    },
    {
        "market_id": "market_3",
        "ticker": "BTC",
        "query": "Will Bitcoin reach $150,000 by June 2026?",
        "description": "Will Bitcoin reach $150,000 USD by June 21, 2026?",
    },
]

async def collect_market_data(
    market: Dict[str, Any],
    time_range_hours: int = 24
) -> bool:
    """Collect data for a single market"""
    try:
        logger.info(f"Collecting data for {market['market_id']} ({market['ticker']})")
        
        # Import data collector to get posts
        from app.services.data_collector import DataCollector
        
        # Collect social data
        collector = DataCollector()
        posts = await collector.collect_social_data(
            ticker=market["ticker"],
            query=market["query"],
            time_range_hours=time_range_hours
        )
        
        logger.info(f"Collected {len(posts)} posts for {market['ticker']}")
        
        # Run orchestrator with posts
        orchestrator = AgentOrchestrator()
        result = await orchestrator.process(
            posts=posts,
            market_data=None,
            ticker=market["ticker"]
        )
        
        # Save to database
        prediction_dict = {
            "ai_score": result.ai_score,
            "market_score": result.market_score,
            "divergence_index": result.divergence_index,
            "confidence": result.confidence,
            "vocal_summary": result.vocal_summary,
            "data_sources": {
                "total_posts": result.data_sources.twitter_posts,
                "total_engagement": result.data_sources.total_engagement,
                "unique_sources": result.data_sources.unique_accounts,
            },
        }
        
        saved_id = save_prediction(
            market_id=market["market_id"],
            ticker=market["ticker"],
            query=market["query"],
            time_range_hours=time_range_hours,
            prediction_result=prediction_dict
        )
        
        if saved_id:
            logger.info(f"Saved prediction {saved_id} for {market['market_id']}")
        else:
            logger.warning(f"Duplicate prediction for {market['market_id']}, skipped")
        
        return True
        
    except Exception as e:
        logger.error(f"Error collecting data for {market['market_id']}: {e}")
        return False

async def collect_all_markets(time_range_hours: int = 24):
    """Collect data for all tracked markets"""
    logger.info(f"Starting background collection for {len(TRACKED_MARKETS)} markets")
    start_time = datetime.now()
    
    results = []
    for market in TRACKED_MARKETS:
        success = await collect_market_data(market, time_range_hours)
        results.append(success)
        
        # Small delay between markets to avoid API rate limits
        await asyncio.sleep(2)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    success_count = sum(results)
    
    logger.info(
        f"Background collection completed: "
        f"{success_count}/{len(TRACKED_MARKETS)} successful in {elapsed:.1f}s"
    )
    
    return results

async def run_periodic_collection(interval_minutes: int = 10):
    """Run collection periodically"""
    logger.info(f"Starting periodic collection every {interval_minutes} minutes")
    
    while True:
        try:
            await collect_all_markets()
        except Exception as e:
            logger.error(f"Error in periodic collection: {e}")
        
        # Wait for next interval
        await asyncio.sleep(interval_minutes * 60)
