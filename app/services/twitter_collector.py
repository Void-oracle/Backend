"""
Twitter Data Collector for VOID Oracle
Collects real-time tweets and historical data about crypto markets
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import re

try:
    import tweepy
    TWEEPY_AVAILABLE = True
except ImportError:
    TWEEPY_AVAILABLE = False

from app.config import settings
from app.models.schemas import Post

logger = logging.getLogger(__name__)


class TwitterCollector:
    """
    Collects Twitter data for sentiment analysis
    
    Features:
    - Initial bulk collection (10k+ tweets)
    - Real-time streaming for new tweets
    - Influencer tracking
    - Event detection
    """
    
    def __init__(self):
        self.enabled = False
        self.client = None
        
        # DISABLED: Twitter API requires payment. Using Gemini/Claude instead.
        logger.info("Twitter API DISABLED - using Gemini/Claude for analysis instead")
        return
        
        # Old code (kept for reference):
        # if not TWEEPY_AVAILABLE:
        #     logger.info("Twitter API disabled (tweepy not installed)")
        #     return
        #     
        # token = settings.TWITTER_BEARER_TOKEN
        # if not token or token.strip() == "" or len(token) < 20:
        #     logger.info("Twitter API disabled (no bearer token)")
        #     return
        # 
        # try:
        #     self.client = tweepy.Client(bearer_token=token, wait_on_rate_limit=True)
        #     self.enabled = True
        #     logger.info("Twitter API client initialized")
        # except Exception as e:
        #     logger.error(f"Failed to initialize Twitter client: {e}")
    
    async def initial_analysis(
        self,
        ticker: str,
        query: str,
        target_count: int = 10000
    ) -> List[Post]:
        """
        Initial deep analysis - collect large dataset
        
        Args:
            ticker: Crypto ticker (e.g., "SOL")
            query: Search query
            target_count: Target number of tweets to collect
        
        Returns:
            List of posts for analysis
        """
        if not self.client:
            logger.error("Twitter API not available - cannot collect data")
            return []
        
        logger.info(f"Starting initial analysis for {ticker} (target: {target_count} tweets)")
        
        all_tweets = []
        max_results = 100  # Max per request for Twitter API v2
        
        # Build search query
        search_query = self._build_search_query(ticker, query)
        
        try:
            # Collect tweets in batches
            for i in range(0, target_count, max_results):
                batch_size = min(max_results, target_count - len(all_tweets))
                
                logger.info(f"   Collecting batch {i//max_results + 1}, tweets: {len(all_tweets)}/{target_count}")
                
                tweets = self.client.search_recent_tweets(
                    query=search_query,
                    max_results=batch_size,
                    tweet_fields=['created_at', 'public_metrics', 'author_id'],
                    expansions=['author_id'],
                    user_fields=['username', 'verified', 'public_metrics']
                )
                
                if not tweets.data:
                    logger.info("   No more tweets available")
                    break
                
                # Process tweets
                for tweet in tweets.data:
                    user = self._get_user_from_includes(tweet.author_id, tweets.includes)
                    post = self._tweet_to_post(tweet, user)
                    all_tweets.append(post)
                
                # Rate limit protection
                await asyncio.sleep(1)
                
                if len(all_tweets) >= target_count:
                    break
            
            logger.info(f"Initial analysis complete: {len(all_tweets)} tweets collected")
            return all_tweets
            
        except Exception as e:
            logger.error(f"Error during initial analysis: {e}")
            return []
    
    async def collect_recent(
        self,
        ticker: str,
        query: str,
        since_minutes: int = 10
    ) -> List[Post]:
        """
        Collect recent tweets (for live monitoring)
        
        Args:
            ticker: Crypto ticker
            query: Search query
            since_minutes: Collect tweets from last N minutes
        
        Returns:
            List of recent posts
        """
        if not self.client:
            logger.error("Twitter API not available")
            return []
        
        search_query = self._build_search_query(ticker, query)
        
        try:
            tweets = self.client.search_recent_tweets(
                query=search_query,
                max_results=100,
                tweet_fields=['created_at', 'public_metrics', 'author_id'],
                expansions=['author_id'],
                user_fields=['username', 'verified', 'public_metrics']
            )
            
            if not tweets.data:
                return []
            
            posts = []
            cutoff_time = datetime.utcnow() - timedelta(minutes=since_minutes)
            
            for tweet in tweets.data:
                if tweet.created_at.replace(tzinfo=None) >= cutoff_time:
                    user = self._get_user_from_includes(tweet.author_id, tweets.includes)
                    post = self._tweet_to_post(tweet, user)
                    posts.append(post)
            
            logger.info(f"Collected {len(posts)} recent tweets for {ticker}")
            return posts
            
        except Exception as e:
            logger.error(f"Error collecting recent tweets: {e}")
            return []
    
    async def monitor_influencers(
        self,
        ticker: str,
        influencer_usernames: List[str]
    ) -> List[Post]:
        """
        Monitor specific influencers for tweets about ticker
        
        Args:
            ticker: Crypto ticker
            influencer_usernames: List of Twitter usernames to monitor
        
        Returns:
            List of influencer posts
        """
        if not self.client:
            return []
        
        posts = []
        
        for username in influencer_usernames:
            try:
                # Get user ID
                user = self.client.get_user(username=username)
                if not user.data:
                    continue
                
                # Get their recent tweets
                tweets = self.client.get_users_tweets(
                    id=user.data.id,
                    max_results=10,
                    tweet_fields=['created_at', 'public_metrics']
                )
                
                if not tweets.data:
                    continue
                
                # Filter tweets mentioning the ticker
                for tweet in tweets.data:
                    if ticker.upper() in tweet.text.upper():
                        post = self._tweet_to_post(tweet, user.data)
                        post.source = "influencer"
                        posts.append(post)
                
            except Exception as e:
                logger.warning(f"Error monitoring @{username}: {e}")
                continue
        
        if posts:
            logger.info(f"Found {len(posts)} influencer tweets about {ticker}")
        
        return posts
    
    def _build_search_query(self, ticker: str, query: str) -> str:
        """Build Twitter search query with available operators"""
        # Build query using only operators available in Pay-per-use tier:
        # ✅ OR - available
        # ✅ lang: - available  
        # ❌ -is: - NOT available (Pro/Enterprise only)
        
        base = f"({ticker} OR ${ticker})"
        lang = "lang:en"
        
        # Note: -is:retweet and -is:reply are not available in Pay-per-use tier
        # Results will include retweets and replies, but that's acceptable
        full_query = f"{base} {lang}"
        
        return full_query
    
    def _tweet_to_post(self, tweet: Any, user: Any) -> Post:
        """Convert Twitter tweet to Post model"""
        metrics = tweet.public_metrics
        
        # Calculate engagement score
        engagement = (
            metrics.get('like_count', 0) +
            metrics.get('retweet_count', 0) * 2 +  # Retweets worth more
            metrics.get('reply_count', 0) * 1.5
        )
        
        # Author credibility (followers, verified)
        author_followers = 0
        verified = False
        
        if user and hasattr(user, 'public_metrics'):
            author_followers = user.public_metrics.get('followers_count', 0)
            verified = getattr(user, 'verified', False)
        
        return Post(
            text=tweet.text,
            author=getattr(user, 'username', 'unknown') if user else 'unknown',
            timestamp=tweet.created_at.isoformat() if tweet.created_at else datetime.utcnow().isoformat(),
            source="twitter",
            engagement=engagement,
            metadata={
                "tweet_id": tweet.id,
                "likes": metrics.get('like_count', 0),
                "retweets": metrics.get('retweet_count', 0),
                "replies": metrics.get('reply_count', 0),
                "author_followers": author_followers,
                "verified": verified
            }
        )
    
    def _get_user_from_includes(self, author_id: str, includes: Any) -> Optional[Any]:
        """Extract user data from includes"""
        if not includes or not hasattr(includes, 'users'):
            return None
        
        for user in includes.users:
            if user.id == author_id:
                return user
        
        return None
    
