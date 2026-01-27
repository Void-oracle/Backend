"""
Price Oracle Service
Fetches real-time crypto prices and calculates required growth metrics
"""
import aiohttp
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class PriceOracle:
    """
    Fetches real-time price data from CoinGecko API (free, no API key needed)
    """
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    # Ticker mapping to CoinGecko IDs
    TICKER_MAP = {
        "SOL": "solana",
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "DOGE": "dogecoin",
        "TESLA": "TSLA",  # Note: CoinGecko doesn't have stocks
        "AVAX": "avalanche-2",
        "MATIC": "matic-network",
        "ADA": "cardano",
        "DOT": "polkadot",
    }
    
    def __init__(self):
        self.cache = {}  # Simple cache to avoid rate limits
        self.cache_duration = 60  # Cache for 1 minute
        logger.info("Price Oracle initialized")
    
    async def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current USD price for a ticker
        
        Args:
            ticker: Crypto ticker (e.g., "SOL", "BTC")
        
        Returns:
            Current price in USD or None if not found
        """
        # Check cache first
        cache_key = f"{ticker}_price"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if (datetime.now() - cached_time).seconds < self.cache_duration:
                logger.debug(f"Using cached price for {ticker}: ${cached_data}")
                return cached_data
        
        # Get CoinGecko ID
        coingecko_id = self.TICKER_MAP.get(ticker.upper())
        if not coingecko_id:
            logger.warning(f"Unknown ticker: {ticker}")
            return None
        
        try:
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": coingecko_id,
                "vs_currencies": "usd",
                "include_24h_change": "true",
                "include_market_cap": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        price = data.get(coingecko_id, {}).get("usd")
                        
                        if price:
                            # Cache the result
                            self.cache[cache_key] = (price, datetime.now())
                            logger.info(f"Fetched {ticker} price: ${price}")
                            return float(price)
                    else:
                        logger.error(f"CoinGecko API error: {response.status}")
                        return None
        
        except asyncio.TimeoutError:
            logger.error(f"Timeout fetching price for {ticker}")
            return None
        except Exception as e:
            logger.error(f"Error fetching price for {ticker}: {e}")
            return None
    
    async def get_historical_data(self, ticker: str, days: int = 30) -> Optional[Dict[str, Any]]:
        """
        Get historical price data
        
        Args:
            ticker: Crypto ticker
            days: Number of days to look back (1, 7, 30, 90, 365)
        
        Returns:
            Historical data including price changes
        """
        coingecko_id = self.TICKER_MAP.get(ticker.upper())
        if not coingecko_id:
            return None
        
        try:
            url = f"{self.BASE_URL}/coins/{coingecko_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days,
                "interval": "daily"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=15) as response:
                    if response.status == 200:
                        data = await response.json()
                        prices = data.get("prices", [])
                        
                        if len(prices) >= 2:
                            first_price = prices[0][1]
                            last_price = prices[-1][1]
                            price_change = ((last_price - first_price) / first_price) * 100
                            
                            # Calculate volatility (standard deviation of daily returns)
                            daily_returns = []
                            for i in range(1, len(prices)):
                                daily_return = ((prices[i][1] - prices[i-1][1]) / prices[i-1][1]) * 100
                                daily_returns.append(daily_return)
                            
                            avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
                            
                            return {
                                "price_change": price_change,
                                "avg_daily_return": avg_daily_return,
                                "first_price": first_price,
                                "last_price": last_price,
                                "days": days
                            }
        
        except Exception as e:
            logger.error(f"Error fetching historical data for {ticker}: {e}")
            return None
    
    async def calculate_market_metrics(
        self,
        ticker: str,
        target_price: float,
        deadline: datetime
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate comprehensive market metrics for AI analysis
        
        Args:
            ticker: Crypto ticker
            target_price: Target price for the prediction
            deadline: Deadline for the prediction
        
        Returns:
            Dict with all necessary metrics for AI analysis
        """
        current_price = await self.get_current_price(ticker)
        if not current_price:
            logger.warning(f"Could not fetch price for {ticker}, using mock data")
            # Return mock data to avoid breaking the system
            return {
                "current_price": None,
                "target_price": target_price,
                "data_available": False
            }
        
        # Calculate time metrics
        now = datetime.now()
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=None)
        if now.tzinfo is None:
            now = now.replace(tzinfo=None)
        
        days_left = (deadline - now).days
        if days_left < 0:
            days_left = 0
        
        # Calculate required growth
        required_growth_pct = ((target_price - current_price) / current_price) * 100
        
        # Calculate daily growth needed (compounded)
        if days_left > 0:
            daily_growth_needed = (((target_price / current_price) ** (1 / days_left)) - 1) * 100
        else:
            daily_growth_needed = 0
        
        # Get historical data
        hist_30d = await self.get_historical_data(ticker, 30)
        hist_90d = await self.get_historical_data(ticker, 90)
        
        return {
            "data_available": True,
            "current_price": current_price,
            "target_price": target_price,
            "required_growth_pct": required_growth_pct,
            "days_left": days_left,
            "daily_growth_needed": daily_growth_needed,
            "historical_30d": hist_30d,
            "historical_90d": hist_90d,
            "feasibility_score": self._calculate_feasibility(
                required_growth_pct,
                days_left,
                hist_30d,
                hist_90d
            )
        }
    
    def _calculate_feasibility(
        self,
        required_growth: float,
        days_left: int,
        hist_30d: Optional[Dict],
        hist_90d: Optional[Dict]
    ) -> float:
        """
        Calculate a feasibility score (0-1) for the price target
        
        This helps AI understand how realistic the target is
        """
        if days_left <= 0:
            return 0.0
        
        # Start with neutral score
        score = 0.5
        
        # Factor 1: Required daily growth vs historical
        if hist_30d:
            daily_growth_needed = (((1 + required_growth / 100) ** (1 / days_left)) - 1) * 100
            historical_daily = hist_30d.get("avg_daily_return", 0)
            
            if abs(historical_daily) > 0.1:  # Avoid division by zero
                growth_ratio = daily_growth_needed / historical_daily
                
                # If required growth is similar to historical, high feasibility
                if 0.5 <= growth_ratio <= 2.0:
                    score += 0.3
                elif 2.0 < growth_ratio <= 5.0:
                    score += 0.1
                else:
                    score -= 0.2
        
        # Factor 2: Time available
        if days_left < 7:
            score -= 0.3  # Very short timeframe = low feasibility
        elif days_left < 30:
            score -= 0.1
        elif days_left > 180:
            score += 0.2  # Long timeframe = more feasible
        
        # Factor 3: Absolute growth required
        if abs(required_growth) < 10:
            score += 0.2  # Small move = feasible
        elif abs(required_growth) < 50:
            score += 0.1
        elif abs(required_growth) > 200:
            score -= 0.3  # Massive move = unlikely
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, score))
    
    def extract_target_price(self, query: str, ticker: str) -> Optional[float]:
        """
        Extract target price from market query
        
        Examples:
            "Will SOL reach $300 by Feb?" -> 300
            "BTC to $150k?" -> 150000
            "Will ETH hit $5000?" -> 5000
        """
        import re
        
        # Try to find price in various formats
        patterns = [
            r'\$(\d+(?:,\d+)*(?:\.\d+)?)[kK]?',  # $300, $150k, $1.5k
            r'(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:USD|dollars?)',  # 300 USD
            r'reach\s+(\d+(?:,\d+)*(?:\.\d+)?)',  # reach 300
        ]
        
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(',', '')
                price = float(price_str)
                
                # Handle 'k' suffix
                if 'k' in query.lower() or 'K' in query:
                    # Check if the k is right after the number
                    if re.search(r'\d[kK]', query):
                        price *= 1000
                
                logger.info(f"Extracted target price from '{query}': ${price}")
                return price
        
        logger.warning(f"Could not extract target price from: {query}")
        return None


# Global instance
price_oracle = PriceOracle()
