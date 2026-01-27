"""
Event Detection for VOID Oracle
Detects important events that should trigger prediction updates
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import Counter
from app.models.schemas import Post

logger = logging.getLogger(__name__)


class EventDetector:
    """
    Detects significant events that warrant prediction updates
    
    Events:
    - Breaking news (sudden spike in tweets)
    - Influencer activity (verified users with large following)
    - Sentiment shift (rapid change in sentiment)
    - Volume spike (unusual activity)
    """
    
    def __init__(self):
        self.baseline_activity = {}  # Track normal activity levels
    
    async def detect_events(
        self,
        ticker: str,
        recent_posts: List[Post],
        historical_baseline: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Detect if any significant events occurred
        
        Args:
            ticker: Crypto ticker
            recent_posts: Recent posts (last 10-30 min)
            historical_baseline: Normal activity level (tweets per hour)
        
        Returns:
            Event detection result
        """
        if not recent_posts:
            return {
                "has_event": False,
                "event_type": None,
                "severity": 0,
                "should_update": False
            }
        
        events = []
        max_severity = 0
        
        # 1. Volume spike detection
        volume_event = self._detect_volume_spike(ticker, recent_posts, historical_baseline)
        if volume_event["detected"]:
            events.append(volume_event)
            max_severity = max(max_severity, volume_event["severity"])
        
        # 2. Influencer activity
        influencer_event = self._detect_influencer_activity(recent_posts)
        if influencer_event["detected"]:
            events.append(influencer_event)
            max_severity = max(max_severity, influencer_event["severity"])
        
        # 3. Sentiment shift
        sentiment_event = self._detect_sentiment_shift(recent_posts)
        if sentiment_event["detected"]:
            events.append(sentiment_event)
            max_severity = max(max_severity, sentiment_event["severity"])
        
        # 4. Viral content
        viral_event = self._detect_viral_content(recent_posts)
        if viral_event["detected"]:
            events.append(viral_event)
            max_severity = max(max_severity, viral_event["severity"])
        
        # Determine if should trigger update
        should_update = max_severity >= 3  # Severity 3+ triggers update
        
        result = {
            "has_event": len(events) > 0,
            "event_type": events[0]["type"] if events else None,
            "events": events,
            "severity": max_severity,
            "should_update": should_update,
            "reason": self._generate_reason(events)
        }
        
        if should_update:
            logger.info(f"Event detected for {ticker}: {result['reason']} (severity: {max_severity})")
        
        return result
    
    def _detect_volume_spike(
        self,
        ticker: str,
        posts: List[Post],
        baseline: Optional[int]
    ) -> Dict[str, Any]:
        """Detect unusual volume of tweets"""
        current_volume = len(posts)
        
        # If no baseline, use current as baseline
        if baseline is None:
            baseline = self.baseline_activity.get(ticker, current_volume)
            self.baseline_activity[ticker] = current_volume
        
        # Calculate spike ratio
        spike_ratio = current_volume / max(baseline, 1)
        
        # Severity levels:
        # 2x baseline = severity 2
        # 3x baseline = severity 3 (triggers update)
        # 5x baseline = severity 4 (major event)
        # 10x baseline = severity 5 (breaking news)
        
        if spike_ratio >= 10:
            severity = 5
        elif spike_ratio >= 5:
            severity = 4
        elif spike_ratio >= 3:
            severity = 3
        elif spike_ratio >= 2:
            severity = 2
        else:
            severity = 0
        
        return {
            "detected": spike_ratio >= 2,
            "type": "volume_spike",
            "severity": severity,
            "details": {
                "current_volume": current_volume,
                "baseline": baseline,
                "spike_ratio": round(spike_ratio, 2)
            }
        }
    
    def _detect_influencer_activity(self, posts: List[Post]) -> Dict[str, Any]:
        """Detect activity from influential accounts"""
        influencer_posts = []
        
        for post in posts:
            metadata = post.metadata or {}
            followers = metadata.get("author_followers", 0)
            verified = metadata.get("verified", False)
            
            # Influencer criteria:
            # - Verified account, OR
            # - 10k+ followers, OR
            # - 50k+ followers (mega influencer)
            
            is_influencer = verified or followers >= 10000
            
            if is_influencer:
                influencer_posts.append({
                    "author": post.author,
                    "followers": followers,
                    "verified": verified,
                    "engagement": post.engagement
                })
        
        if not influencer_posts:
            return {"detected": False, "type": "influencer", "severity": 0}
        
        # Calculate severity based on influencer tier
        max_followers = max(p["followers"] for p in influencer_posts)
        
        if max_followers >= 1000000:  # 1M+ followers
            severity = 5
        elif max_followers >= 500000:  # 500k+ followers
            severity = 4
        elif max_followers >= 100000:  # 100k+ followers
            severity = 3
        else:  # 10k+ followers
            severity = 2
        
        return {
            "detected": True,
            "type": "influencer_activity",
            "severity": severity,
            "details": {
                "influencer_count": len(influencer_posts),
                "top_influencer": influencer_posts[0]["author"],
                "max_followers": max_followers
            }
        }
    
    def _detect_sentiment_shift(self, posts: List[Post]) -> Dict[str, Any]:
        """Detect rapid sentiment changes"""
        if len(posts) < 10:
            return {"detected": False, "type": "sentiment_shift", "severity": 0}
        
        # Split posts into time buckets
        sorted_posts = sorted(posts, key=lambda p: p.timestamp)
        mid_point = len(sorted_posts) // 2
        
        early_posts = sorted_posts[:mid_point]
        late_posts = sorted_posts[mid_point:]
        
        # Simple sentiment scoring based on keywords
        def score_sentiment(posts_subset):
            bullish_keywords = ['bullish', 'moon', 'pump', 'breakout', 'buy', 'long', 'up', 'rocket', 'up']
            bearish_keywords = ['bearish', 'dump', 'crash', 'sell', 'short', 'down', 'rip']
            
            bullish_count = sum(
                1 for post in posts_subset
                if any(kw in post.text.lower() for kw in bullish_keywords)
            )
            bearish_count = sum(
                1 for post in posts_subset
                if any(kw in post.text.lower() for kw in bearish_keywords)
            )
            
            total = len(posts_subset)
            return (bullish_count - bearish_count) / max(total, 1)
        
        early_sentiment = score_sentiment(early_posts)
        late_sentiment = score_sentiment(late_posts)
        
        sentiment_change = abs(late_sentiment - early_sentiment)
        
        # Severity based on magnitude of shift
        if sentiment_change >= 0.5:
            severity = 4
        elif sentiment_change >= 0.3:
            severity = 3
        elif sentiment_change >= 0.2:
            severity = 2
        else:
            severity = 0
        
        return {
            "detected": sentiment_change >= 0.2,
            "type": "sentiment_shift",
            "severity": severity,
            "details": {
                "early_sentiment": round(early_sentiment, 2),
                "late_sentiment": round(late_sentiment, 2),
                "shift_magnitude": round(sentiment_change, 2)
            }
        }
    
    def _detect_viral_content(self, posts: List[Post]) -> Dict[str, Any]:
        """Detect viral tweets (high engagement)"""
        if not posts:
            return {"detected": False, "type": "viral_content", "severity": 0}
        
        # Find highest engagement tweet
        max_engagement = max(post.engagement for post in posts)
        avg_engagement = sum(post.engagement for post in posts) / len(posts)
        
        # Viral if engagement is significantly above average
        viral_ratio = max_engagement / max(avg_engagement, 1)
        
        # Also check absolute engagement
        high_engagement = max_engagement >= 1000
        
        if high_engagement and viral_ratio >= 5:
            severity = 4
        elif viral_ratio >= 10:
            severity = 3
        elif viral_ratio >= 5:
            severity = 2
        else:
            severity = 0
        
        return {
            "detected": viral_ratio >= 5,
            "type": "viral_content",
            "severity": severity,
            "details": {
                "max_engagement": max_engagement,
                "avg_engagement": round(avg_engagement, 2),
                "viral_ratio": round(viral_ratio, 2)
            }
        }
    
    def _generate_reason(self, events: List[Dict[str, Any]]) -> str:
        """Generate human-readable reason for event"""
        if not events:
            return "No significant events detected"
        
        reasons = []
        for event in sorted(events, key=lambda e: e["severity"], reverse=True):
            event_type = event["type"]
            details = event.get("details", {})
            
            if event_type == "volume_spike":
                spike = details.get("spike_ratio", 0)
                reasons.append(f"{spike:.1f}x volume spike")
            
            elif event_type == "influencer_activity":
                count = details.get("influencer_count", 0)
                reasons.append(f"{count} influencer(s) posted")
            
            elif event_type == "sentiment_shift":
                shift = details.get("shift_magnitude", 0)
                reasons.append(f"{shift:.0%} sentiment shift")
            
            elif event_type == "viral_content":
                engagement = details.get("max_engagement", 0)
                reasons.append(f"viral tweet ({engagement} engagement)")
        
        return ", ".join(reasons)
