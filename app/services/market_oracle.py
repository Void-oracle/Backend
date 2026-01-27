"""
Market Oracle Service
Fetches REAL market probabilities from Polymarket API
"""
import asyncio
import httpx
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import re

logger = logging.getLogger(__name__)

# Search config for finding markets on Polymarket
POLYMARKET_MARKETS = {
    "BTC-200K": {
        "search": "bitcoin 200000 2026",
        "keywords": ["bitcoin", "btc", "200", "2026"],
    },
    "ETH-10K": {
        "search": "ethereum 10000 2026",
        "keywords": ["ethereum", "eth", "10000", "2026"],
    },
    "SOL-ETF": {
        "search": "solana etf sec approval",
        "keywords": ["solana", "sol", "etf", "sec", "approved"],
    },
    "GPT5": {
        "search": "openai gpt-5 release 2026",
        "keywords": ["openai", "gpt", "gpt-5", "release"],
    },
    "FED-2026": {
        "search": "federal reserve rate cuts 2026",
        "keywords": ["fed", "federal reserve", "rate", "cut", "2026"],
    },
}


class MarketOracle:
    """
    Fetches real market probabilities from Polymarket API
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.cache: Dict[str, Any] = {}
        self.cache_ttl_seconds = 60  # 1 minute cache
        
        # Polymarket API
        self.polymarket_api = "https://gamma-api.polymarket.com"
        
        logger.info("MarketOracle initialized - fetching real Polymarket data")
    
    async def get_market_probability(
        self,
        query: str,
        ticker: str = ""
    ) -> Dict[str, Any]:
        """
        Get real probability from Polymarket
        """
        # Check cache first
        cache_key = f"{ticker}:{query[:50]}"
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if (datetime.now() - cached["timestamp"]).seconds < self.cache_ttl_seconds:
                logger.info(f"Cache hit for {ticker}: {cached['data'].get('market_probability', 50):.1f}%")
                return cached["data"]
        
        # Get search config for this ticker
        search_config = POLYMARKET_MARKETS.get(ticker.upper(), {}) if ticker else {}
        search_terms = search_config.get("search", query)
        required_keywords = search_config.get("keywords", [])
        
        # Search Polymarket with specific terms
        try:
            result = await self._search_polymarket_strict(search_terms, required_keywords, ticker)
            if result and result.get("found"):
                self.cache[cache_key] = {"data": result, "timestamp": datetime.now()}
                return result
        except Exception as e:
            logger.error(f"Polymarket API error: {e}")
        
        # No market found
        return {
            "found": False,
            "market_probability": 50.0,
            "source": "not_found",
            "message": "No matching market on Polymarket"
        }
    
    async def _search_polymarket_strict(
        self, 
        search_terms: str, 
        required_keywords: List[str],
        ticker: str
    ) -> Optional[Dict[str, Any]]:
        """Search Polymarket with strict keyword matching"""
        try:
            logger.info(f"Searching Polymarket for: {search_terms}")
            
            response = await self.client.get(
                f"{self.polymarket_api}/markets",
                params={
                    "closed": "false",
                    "limit": 100,
                    "active": "true"
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Polymarket API returned {response.status_code}")
                return None
            
            markets = response.json()
            logger.info(f"Polymarket returned {len(markets)} markets")
            
            # Find market that matches ALL required keywords
            best_match = None
            best_score = 0
            
            for market in markets:
                title = market.get("question", "").lower()
                
                # Count how many required keywords match
                matched = 0
                for kw in required_keywords:
                    if kw.lower() in title:
                        matched += 1
                
                # Need at least 2 keywords to match, or 50%+ of keywords
                min_matches = max(2, len(required_keywords) // 2)
                
                if matched >= min_matches and matched > best_score:
                    best_score = matched
                    best_match = market
            
            if best_match:
                prob = self._extract_probability(best_match)
                title = best_match.get("question", "")
                
                logger.info(f"Found matching market: {title[:60]}...")
                logger.info(f"   Probability: {prob:.1f}% (matched {best_score}/{len(required_keywords)} keywords)")
                
                return {
                    "found": True,
                    "market_probability": prob,
                    "source": "polymarket",
                    "market_title": title,
                    "market_id": best_match.get("id", ""),
                    "matched_keywords": best_score
                }
            
            logger.info(f"No market matched required keywords for {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Polymarket search error: {e}")
            return None
    
    def _extract_probability(self, market: Dict) -> float:
        """Extract YES probability from market data"""
        prob = 50.0
        
        # Try outcomePrices - can be string "[0.65, 0.35]" or list
        prices = market.get("outcomePrices")
        if prices:
            if isinstance(prices, str):
                try:
                    # Parse string like "[0.65, 0.35]" or '["0.65", "0.35"]'
                    import json
                    prices = json.loads(prices)
                except:
                    try:
                        import ast
                        prices = ast.literal_eval(prices)
                    except:
                        prices = None
            
            if isinstance(prices, list) and len(prices) > 0:
                try:
                    # First element is YES probability
                    prob = float(prices[0]) * 100
                    logger.debug(f"Extracted probability: {prob}% from {prices}")
                except Exception as e:
                    logger.warning(f"Failed to parse prices {prices}: {e}")
        
        # Also check for individual "yesPrice" or "price" fields
        if prob == 50.0:
            yes_price = market.get("yesPrice") or market.get("bestBid")
            if yes_price:
                try:
                    prob = float(yes_price) * 100
                except:
                    pass
        
        return prob
    
    async def _search_polymarket(
        self,
        query: str,
        ticker: str
    ) -> Optional[Dict[str, Any]]:
        """
        Search Polymarket for matching markets
        """
        try:
            # Build search keywords
            search_terms = self._extract_search_terms(query, ticker)
            logger.info(f"Searching Polymarket for: {search_terms}")
            
            # Fetch active markets
            response = await self.client.get(
                f"{self.polymarket_api}/markets",
                params={
                    "closed": "false",
                    "limit": 100,  # Get more markets to search through
                    "active": "true"
                }
            )
            
            if response.status_code != 200:
                logger.warning(f"Polymarket API returned {response.status_code}: {response.text[:200]}")
                return None
            
            markets = response.json()
            logger.info(f"Polymarket returned {len(markets)} markets")
            
            # Find best matching market
            best_match = self._find_best_match(markets, query, ticker)
            
            if best_match:
                # Extract probability - Polymarket uses different formats
                prob = 50.0
                
                # Try outcomePrices first
                if "outcomePrices" in best_match and best_match["outcomePrices"]:
                    prices = best_match["outcomePrices"]
                    if isinstance(prices, list) and len(prices) > 0:
                        prob = float(prices[0]) * 100
                
                # Try outcomes array
                elif "outcomes" in best_match:
                    outcomes = best_match.get("outcomes", [])
                    if outcomes and "price" in outcomes[0]:
                        prob = float(outcomes[0]["price"]) * 100
                
                # Try clobTokenIds and get price from tokens
                elif "tokens" in best_match:
                    tokens = best_match.get("tokens", [])
                    if tokens and "price" in tokens[0]:
                        prob = float(tokens[0]["price"]) * 100
                
                market_title = best_match.get("question", best_match.get("title", ""))
                logger.info(f"Found market: {market_title[:50]}... = {prob:.1f}%")
                
                return {
                    "found": True,
                    "market_probability": prob,
                    "source": "polymarket",
                    "market_title": market_title,
                    "market_id": best_match.get("id", best_match.get("conditionId", "")),
                    "volume": best_match.get("volume", best_match.get("volumeNum", 0)),
                    "liquidity": best_match.get("liquidity", 0),
                    "end_date": best_match.get("endDate", best_match.get("endDateIso", ""))
                }
            
            logger.info(f"No matching market found for: {ticker}")
            return None
            
        except Exception as e:
            logger.error(f"Polymarket search error: {e}")
            return None
    
    def _extract_search_terms(self, query: str, ticker: str) -> str:
        """Extract key search terms from query"""
        # Remove common words
        stop_words = ["will", "the", "a", "an", "by", "in", "on", "at", "to", "for", "be", "is", "?"]
        words = query.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        if ticker and ticker.lower() not in [k.lower() for k in keywords]:
            keywords.insert(0, ticker)
        
        return " ".join(keywords[:5])
    
    def _find_best_match(
        self,
        markets: List[Dict],
        query: str,
        ticker: str
    ) -> Optional[Dict]:
        """Find best matching market from list"""
        query_lower = query.lower()
        ticker_lower = ticker.lower() if ticker else ""
        
        # Build comprehensive keyword list
        keywords = []
        
        # Add ticker variations
        ticker_map = {
            "btc": ["bitcoin", "btc"],
            "eth": ["ethereum", "eth"],
            "sol": ["solana", "sol"],
            "trump": ["trump", "donald"],
            "fed": ["federal reserve", "fed", "interest rate"],
            "gpt": ["openai", "gpt", "chatgpt"],
            "apple": ["apple", "iphone"],
            "superbowl": ["super bowl", "nfl", "chiefs", "football"],
            "oscars": ["oscar", "academy award"],
        }
        
        # Add keywords from ticker
        if ticker_lower:
            for key, variations in ticker_map.items():
                if key in ticker_lower:
                    keywords.extend(variations)
            if not keywords:
                keywords.append(ticker_lower)
        
        # Extract important words from query
        important_words = ["tariff", "rate", "cut", "etf", "reach", "win", 
                          "release", "announce", "approved", "100k", "5000", "500"]
        for word in important_words:
            if word in query_lower:
                keywords.append(word)
        
        # Extract price targets
        price_match = re.search(r'\$(\d{1,3}(?:,\d{3})*)', query)
        if price_match:
            price_str = price_match.group(1).replace(",", "")
            keywords.append(price_str)
            # Also add formatted version
            keywords.append(f"${price_str}")
        
        logger.debug(f"Matching keywords: {keywords}")
        
        best_score = 0
        best_market = None
        
        for market in markets:
            title = market.get("question", market.get("title", "")).lower()
            description = market.get("description", "").lower()
            tags = " ".join(market.get("tags", [])).lower() if market.get("tags") else ""
            
            # Combine all searchable text
            search_text = f"{title} {description} {tags}"
            
            score = 0
            for keyword in keywords:
                keyword_lower = keyword.lower()
                if keyword_lower in title:
                    score += 5  # Title match is most important
                elif keyword_lower in description:
                    score += 2
                elif keyword_lower in tags:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_market = market
        
        # Log top match for debugging
        if best_market:
            logger.debug(f"Best match (score={best_score}): {best_market.get('question', '')[:60]}")
        
        # Only return if we have a reasonable match (at least 1 keyword in title)
        if best_score >= 5:
            return best_market
        
        return None
    
    async def get_popular_markets(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get popular/trending prediction markets
        """
        try:
            response = await self.client.get(
                f"{self.polymarket_api}/markets",
                params={
                    "closed": "false",
                    "limit": limit,
                    "order": "volume",
                    "ascending": "false"
                }
            )
            
            if response.status_code == 200:
                markets = response.json()
                return [{
                    "title": m.get("question", ""),
                    "probability": m.get("outcomePrices", [0.5])[0] * 100,
                    "volume": m.get("volume", 0),
                    "end_date": m.get("endDate", "")
                } for m in markets]
            
        except Exception as e:
            logger.error(f"Failed to get popular markets: {e}")
        
        return []
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
market_oracle = MarketOracle()
