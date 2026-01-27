"""
Technical & Historical Data Collector
Gathers objective data for rational probability assessment
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import asyncio
import logging

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


class TechnicalDataCollector:
    """
    Collects technical and historical data for rational market analysis
    
    Data sources:
    - Price data (CoinGecko, Binance)
    - On-chain metrics (Solana, Ethereum)
    - Historical events database
    - Technical indicators
    - Market statistics
    """
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        self.coingecko_base = "https://api.coingecko.com/api/v3"
    
    async def collect_technical_data(
        self,
        ticker: str,
        market_type: str = "crypto"
    ) -> Dict[str, Any]:
        """
        Collect comprehensive technical data
        
        Args:
            ticker: Asset ticker (e.g., "SOL", "BTC")
            market_type: Type of market ("crypto", "sports", "politics", etc.)
        
        Returns:
            Dictionary with technical indicators and metrics
        """
        logger.info(f"Collecting technical data for {ticker}")
        
        if market_type == "crypto":
            return await self._collect_crypto_technical_data(ticker)
        else:
            # For non-crypto markets, return basic template
            return self._get_generic_technical_data()
    
    async def _collect_crypto_technical_data(self, ticker: str) -> Dict[str, Any]:
        """
        Collect crypto-specific technical data
        """
        tasks = [
            self._get_price_data(ticker),
            self._get_market_stats(ticker),
            self._get_volatility_data(ticker),
            self._get_on_chain_metrics(ticker)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        price_data = results[0] if not isinstance(results[0], Exception) else {}
        market_stats = results[1] if not isinstance(results[1], Exception) else {}
        volatility = results[2] if not isinstance(results[2], Exception) else {}
        on_chain = results[3] if not isinstance(results[3], Exception) else {}
        
        return {
            "ticker": ticker,
            "current_price": price_data.get('current_price', 0),
            "price_change_24h": price_data.get('price_change_24h', 0),
            "price_change_7d": price_data.get('price_change_7d', 0),
            "price_change_30d": price_data.get('price_change_30d', 0),
            "volume_24h": market_stats.get('volume_24h', 0),
            "market_cap": market_stats.get('market_cap', 0),
            "volatility_30d": volatility.get('volatility_30d', 0),
            "volatility_90d": volatility.get('volatility_90d', 0),
            "all_time_high": price_data.get('ath', 0),
            "ath_date": price_data.get('ath_date'),
            "distance_from_ath": price_data.get('ath_change_percentage', 0),
            "support_levels": self._calculate_support_levels(price_data),
            "resistance_levels": self._calculate_resistance_levels(price_data),
            "on_chain_metrics": on_chain,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _get_price_data(self, ticker: str) -> Dict[str, Any]:
        """Get price data from CoinGecko"""
        try:
            # Map common tickers to CoinGecko IDs
            coin_id_map = {
                'SOL': 'solana',
                'BTC': 'bitcoin',
                'ETH': 'ethereum',
                'USDC': 'usd-coin'
            }
            
            coin_id = coin_id_map.get(ticker.upper(), ticker.lower())
            
            url = f"{self.coingecko_base}/coins/{coin_id}"
            params = {
                'localization': 'false',
                'tickers': 'false',
                'community_data': 'false',
                'developer_data': 'false'
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                logger.warning(f"CoinGecko API error: {response.status_code}")
                return {}
            
            data = response.json()
            market_data = data.get('market_data', {})
            
            return {
                'current_price': market_data.get('current_price', {}).get('usd', 0),
                'price_change_24h': market_data.get('price_change_percentage_24h', 0),
                'price_change_7d': market_data.get('price_change_percentage_7d', 0),
                'price_change_30d': market_data.get('price_change_percentage_30d', 0),
                'ath': market_data.get('ath', {}).get('usd', 0),
                'ath_date': market_data.get('ath_date', {}).get('usd'),
                'ath_change_percentage': market_data.get('ath_change_percentage', {}).get('usd', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get price data: {e}")
            return {}
    
    async def _get_market_stats(self, ticker: str) -> Dict[str, Any]:
        """Get market statistics"""
        try:
            coin_id_map = {
                'SOL': 'solana',
                'BTC': 'bitcoin',
                'ETH': 'ethereum'
            }
            
            coin_id = coin_id_map.get(ticker.upper(), ticker.lower())
            
            url = f"{self.coingecko_base}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '30'
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            volumes = data.get('total_volumes', [])
            market_caps = data.get('market_caps', [])
            
            # Calculate average volume and market cap
            avg_volume = sum(v[1] for v in volumes) / len(volumes) if volumes else 0
            current_mcap = market_caps[-1][1] if market_caps else 0
            
            return {
                'volume_24h': volumes[-1][1] if volumes else 0,
                'avg_volume_30d': avg_volume,
                'market_cap': current_mcap
            }
            
        except Exception as e:
            logger.error(f"Failed to get market stats: {e}")
            return {}
    
    async def _get_volatility_data(self, ticker: str) -> Dict[str, Any]:
        """Calculate volatility from historical prices"""
        try:
            coin_id_map = {
                'SOL': 'solana',
                'BTC': 'bitcoin',
                'ETH': 'ethereum'
            }
            
            coin_id = coin_id_map.get(ticker.upper(), ticker.lower())
            
            # Get 90 days of data
            url = f"{self.coingecko_base}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': '90'
            }
            
            response = await self.http_client.get(url, params=params)
            
            if response.status_code != 200:
                return {}
            
            data = response.json()
            prices = [p[1] for p in data.get('prices', [])]
            
            if not prices:
                return {}
            
            # Calculate daily returns
            returns = []
            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)
            
            # Calculate volatility (standard deviation of returns)
            import statistics
            volatility_30d = statistics.stdev(returns[-30:]) * 100 if len(returns) >= 30 else 0
            volatility_90d = statistics.stdev(returns) * 100 if returns else 0
            
            return {
                'volatility_30d': round(volatility_30d, 2),
                'volatility_90d': round(volatility_90d, 2),
                'max_drawdown_30d': self._calculate_max_drawdown(prices[-30:]),
                'max_drawdown_90d': self._calculate_max_drawdown(prices)
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate volatility: {e}")
            return {}
    
    def _calculate_max_drawdown(self, prices: List[float]) -> float:
        """Calculate maximum drawdown from peak"""
        if not prices:
            return 0
        
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            dd = (peak - price) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        return round(max_dd, 2)
    
    def _calculate_support_levels(self, price_data: Dict[str, Any]) -> List[float]:
        """Calculate support levels (simplified)"""
        current = price_data.get('current_price', 0)
        if current == 0:
            return []
        
        # Simple support levels at -5%, -10%, -20% from current
        return [
            round(current * 0.95, 2),
            round(current * 0.90, 2),
            round(current * 0.80, 2)
        ]
    
    def _calculate_resistance_levels(self, price_data: Dict[str, Any]) -> List[float]:
        """Calculate resistance levels (simplified)"""
        current = price_data.get('current_price', 0)
        ath = price_data.get('ath', current * 2)
        
        if current == 0:
            return []
        
        # Resistance at +5%, +10%, +20%, and ATH
        levels = [
            round(current * 1.05, 2),
            round(current * 1.10, 2),
            round(current * 1.20, 2)
        ]
        
        if ath > current * 1.2:
            levels.append(round(ath, 2))
        
        return levels
    
    async def _get_on_chain_metrics(self, ticker: str) -> Dict[str, Any]:
        """Get on-chain metrics (placeholder - integrate with actual APIs)"""
        # TODO: Integrate with:
        # - Solana Beach API for Solana metrics
        # - Etherscan API for Ethereum metrics
        # - Glassnode API for advanced metrics
        
        return {
            'active_addresses_24h': 0,
            'transaction_count_24h': 0,
            'note': 'On-chain metrics integration pending'
        }
    
    def _get_generic_technical_data(self) -> Dict[str, Any]:
        """Generic technical data template for non-crypto markets"""
        return {
            'data_type': 'generic',
            'note': 'Limited technical data available for this market type',
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def get_historical_precedents(
        self,
        event_type: str,
        query: str
    ) -> Dict[str, Any]:
        """
        Find historical precedents for similar events
        
        This could integrate with:
        - Wikipedia API
        - News archives
        - Historical prediction market outcomes
        - Statistical databases
        """
        # Placeholder - implement actual historical search
        return {
            'event_type': event_type,
            'query': query,
            'similar_events': [],
            'historical_probability': None,
            'note': 'Historical precedent search not yet implemented'
        }
    
    async def close(self):
        """Close HTTP client"""
        await self.http_client.aclose()
