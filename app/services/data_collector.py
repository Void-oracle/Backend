"""
Data Collection Service
Collects data from Twitter API exclusively
"""
from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta

import httpx

from app.config import settings


logger = logging.getLogger(__name__)


class DataCollector:
    """Collects data from Twitter API for sentiment analysis"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30.0)
        
        # Twitter configuration
        self.twitter_bearer = settings.TWITTER_BEARER_TOKEN
        self.twitter_api_base = "https://api.twitter.com/2"
        
        if self.twitter_bearer:
            logger.info("Data collector initialized with Twitter API")
        else:
            logger.warning("Twitter API not configured - will use mock data")
    
    async def collect_social_data(
        self,
        ticker: str,
        query: str,
        time_range_hours: int = 24
    ) -> List[Dict[str, Any]]:
        """
        Collect social media data from all available sources
        
        Args:
            ticker: Token ticker or market identifier
            query: Search query or market question
            time_range_hours: How many hours back to search
            
        Returns:
            List of posts with metadata
        """
        logger.info(f"Collecting data for {ticker} over {time_range_hours} hours")
        
        # Construct search queries
        search_queries = self._build_search_queries(ticker, query)
        
        # Collect from Twitter only (Tavily removed - using Twitter API exclusively)
        tasks = []
        
        # Twitter search (if available)
        if self.twitter_bearer:
            for search_query in search_queries[:1]:
                tasks.append(
                    self._collect_from_twitter(search_query, time_range_hours)
                )
        else:
            # No Twitter API available, will use mock data as fallback
            logger.warning("Twitter API not available, will use mock data")
        
        # Gather all results
        results = await asyncio.gather(*tasks, return_exceptions=True) if tasks else []
        
        # Flatten and deduplicate
        all_posts = []
        seen_texts = set()
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Collection error: {result}")
                continue
            
            for post in result:
                text = post.get('text', '')
                # Simple deduplication
                text_hash = hash(text[:100])
                if text_hash not in seen_texts:
                    seen_texts.add(text_hash)
                    all_posts.append(post)
        
        logger.info(f"Collected {len(all_posts)} unique posts")
        
        # No fallback - only real data
        if len(all_posts) == 0:
            logger.error(f"Failed to collect any real data for {ticker}")
        
        return all_posts
    
    def _build_search_queries(self, ticker: str, query: str) -> List[str]:
        """Build search queries for different platforms"""
        queries = [
            f"{ticker} crypto prediction",
            f"{ticker} price analysis",
            f"{ticker} sentiment",
            query,
        ]
        
        # Add ticker variations
        if ticker:
            queries.append(f"${ticker}")
            queries.append(f"#{ticker}")
        
        return queries
    
    async def _collect_from_tavily(
        self,
        query: str,
        time_range_hours: int
    ) -> List[Dict[str, Any]]:
        """Collect data using Tavily search API"""
        try:
            # Tavily supports web + social search
            # Note: Tavily client is sync, but we run it in async context
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=20,
                include_domains=["twitter.com", "x.com", "reddit.com"],
                topic="news"
            )
            
            posts = []
            for result in response.get('results', []):
                # Convert Tavily result to post format
                post = {
                    'text': result.get('content', ''),
                    'url': result.get('url', ''),
                    'source': 'tavily',
                    'created_at': datetime.utcnow().isoformat(),
                    'engagement': result.get('score', 0.5) * 100,  # Convert score to engagement
                    'author': {
                        'username': self._extract_username_from_url(result.get('url', '')),
                        'followers': 0,
                        'verified': False
                    }
                }
                posts.append(post)
            
            logger.info(f"Tavily returned {len(posts)} results for '{query}'")
            return posts
            
        except Exception as e:
            logger.error(f"Tavily collection failed: {e}")
            return []
    
    async def _collect_from_twitter(
        self,
        query: str,
        time_range_hours: int
    ) -> List[Dict[str, Any]]:
        """Collect data from Twitter/X API"""
        if not self.twitter_bearer:
            return []
        
        try:
            # Calculate time range
            start_time = datetime.utcnow() - timedelta(hours=time_range_hours)
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            
            # Twitter API v2 search endpoint
            url = f"{self.twitter_api_base}/tweets/search/recent"
            
            headers = {
                "Authorization": f"Bearer {self.twitter_bearer}"
            }
            
            params = {
                "query": query,
                "max_results": 100,
                "start_time": start_time_str,
                "tweet.fields": "created_at,public_metrics,author_id",
                "user.fields": "username,verified,public_metrics",
                "expansions": "author_id"
            }
            
            response = await self.http_client.get(url, headers=headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Twitter API error: {response.status_code} - {response.text}")
                return []
            
            data = response.json()
            
            # Parse tweets
            posts = []
            tweets = data.get('data', [])
            users = {u['id']: u for u in data.get('includes', {}).get('users', [])}
            
            for tweet in tweets:
                author_id = tweet.get('author_id')
                author = users.get(author_id, {})
                metrics = tweet.get('public_metrics', {})
                
                engagement = (
                    metrics.get('like_count', 0) +
                    metrics.get('retweet_count', 0) * 2 +  # Weight retweets more
                    metrics.get('reply_count', 0)
                )
                
                post = {
                    'text': tweet.get('text', ''),
                    'url': f"https://twitter.com/user/status/{tweet.get('id')}",
                    'source': 'twitter',
                    'created_at': tweet.get('created_at'),
                    'engagement': engagement,
                    'author': {
                        'username': author.get('username', 'unknown'),
                        'followers': author.get('public_metrics', {}).get('followers_count', 0),
                        'following': author.get('public_metrics', {}).get('following_count', 0),
                        'verified': author.get('verified', False),
                        'created_at': None  # Not available in basic response
                    }
                }
                posts.append(post)
            
            logger.info(f"Twitter returned {len(posts)} tweets for '{query}'")
            return posts
            
        except Exception as e:
            logger.error(f"Twitter collection failed: {e}")
            return []
    
    def _extract_username_from_url(self, url: str) -> str:
        """Extract username from social media URL"""
        import re
        
        # Try to extract from twitter.com/username pattern
        match = re.search(r'(?:twitter\.com|x\.com)/([^/]+)', url)
        if match:
            return match.group(1)
        
        # Try to extract from reddit.com/u/username
        match = re.search(r'reddit\.com/u/([^/]+)', url)
        if match:
            return match.group(1)
        
        return 'unknown'
    
    async def close(self):
        """Close HTTP clients"""
        await self.http_client.aclose()
