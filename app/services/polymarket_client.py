"""
Polymarket API Client
Official integration using Gamma API for markets and CLOB for prices
Docs: https://docs.polymarket.com/quickstart/overview
"""
import httpx
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# API Endpoints
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


class PolymarketClient:
    """
    Official Polymarket API client
    - Gamma API: Market discovery & metadata
    - CLOB API: Real-time prices
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl = 300  # 5 minutes
        logger.info("Polymarket client initialized")
    
    async def get_active_markets(
        self, 
        limit: int = 50,
        order_by: str = "volume",
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get active markets from Polymarket
        
        Args:
            limit: Number of markets to fetch
            order_by: Sort by 'volume', 'liquidity', 'created'
            tag: Filter by tag (crypto, politics, sports, etc.)
        
        Returns:
            List of market data
        """
        try:
            params = {
                "closed": "false",
                "limit": limit,
                "order": order_by,
                "ascending": "false"
            }
            
            if tag:
                params["tag_slug"] = tag
            
            response = await self.client.get(
                f"{GAMMA_API}/markets",
                params=params
            )
            
            if response.status_code != 200:
                logger.error(f"Gamma API error: {response.status_code}")
                return []
            
            markets = response.json()
            logger.info(f"Fetched {len(markets)} markets from Polymarket")
            
            return markets
            
        except Exception as e:
            logger.error(f"Failed to fetch markets: {e}")
            return []
    
    async def get_market_by_id(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Get specific market by condition ID"""
        try:
            response = await self.client.get(
                f"{GAMMA_API}/markets/{condition_id}"
            )
            
            if response.status_code == 200:
                return response.json()
            return None
            
        except Exception as e:
            logger.error(f"Failed to fetch market {condition_id}: {e}")
            return None
    
    async def search_markets(self, query: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search markets by text query"""
        try:
            response = await self.client.get(
                f"{GAMMA_API}/markets",
                params={
                    "closed": "false",
                    "limit": limit,
                    "_q": query  # Text search
                }
            )
            
            if response.status_code == 200:
                return response.json()
            return []
            
        except Exception as e:
            logger.error(f"Market search failed: {e}")
            return []
    
    def parse_market(self, market: Dict) -> Dict[str, Any]:
        """
        Parse Polymarket market data into our format
        
        Returns standardized market data with probability
        """
        # Get YES probability from outcomePrices
        probability = 50.0
        outcome_prices = market.get("outcomePrices")
        
        if outcome_prices:
            # Can be string "[0.65, 0.35]" or list
            if isinstance(outcome_prices, str):
                try:
                    import json
                    outcome_prices = json.loads(outcome_prices)
                except:
                    outcome_prices = None
            
            if isinstance(outcome_prices, list) and len(outcome_prices) > 0:
                try:
                    probability = float(outcome_prices[0]) * 100
                except:
                    pass
        
        # Parse deadline
        deadline = None
        end_date = market.get("endDate") or market.get("endDateIso")
        if end_date:
            try:
                deadline = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            except:
                pass
        
        return {
            "polymarket_id": market.get("id") or market.get("conditionId"),
            "question": market.get("question", ""),
            "description": market.get("description", ""),
            "probability": probability,
            "volume": market.get("volume", 0),
            "liquidity": market.get("liquidity", 0),
            "deadline": deadline,
            "slug": market.get("slug", ""),
            "image": market.get("image", ""),
            "tags": market.get("tags", []),
            "url": f"https://polymarket.com/event/{market.get('slug', '')}"
        }
    
    async def get_popular_markets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most popular markets by volume"""
        markets = await self.get_active_markets(limit=limit, order_by="volume")
        return [self.parse_market(m) for m in markets]
    
    async def get_crypto_markets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get crypto-related markets"""
        markets = await self.get_active_markets(limit=limit, tag="crypto")
        return [self.parse_market(m) for m in markets]
    
    async def get_politics_markets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get politics-related markets"""
        markets = await self.get_active_markets(limit=limit, tag="politics")
        return [self.parse_market(m) for m in markets]
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
polymarket = PolymarketClient()
