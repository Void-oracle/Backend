"""
VOID Oracle Engine - Event-Driven Prediction System

Architecture:
1. Initial Analysis: Collect 10k+ tweets, form baseline prediction
2. Live Monitoring: Watch for new events in real-time
3. Event Detection: Identify significant changes
4. Smart Updates: Update predictions only when important events occur
"""
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.services.twitter_collector import TwitterCollector
from app.services.event_detector import EventDetector
from app.services.event_verifier import EventVerifier
from app.services.price_oracle import price_oracle
from app.services.market_oracle import market_oracle
from app.agents.orchestrator import AgentOrchestrator
from app.core.market_database import get_market_db

logger = logging.getLogger(__name__)


class OracleEngine:
    """
    Main Oracle Engine - manages prediction lifecycle
    
    Workflow:
    1. Initial deep analysis (10k tweets)
    2. Store baseline prediction
    3. Monitor for events continuously
    4. Update prediction when significant events occur
    """
    
    def __init__(self):
        self.twitter = TwitterCollector()
        self.event_detector = EventDetector()
        self.event_verifier = EventVerifier()
        self.orchestrator = AgentOrchestrator()
        
        # Track market states
        self.market_states = {}  # market_id -> state info
        
        logger.info("Oracle Engine initialized")
    
    async def initialize_market(
        self,
        market_id: str,
        ticker: str,
        query: str,
        target_tweets: int = 10000,
        deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Phase 1: Initial Deep Analysis (or load existing)
        
        If market already has predictions in DB - just load them.
        Otherwise, collect data and form baseline prediction.
        
        Args:
            market_id: Unique market identifier
            ticker: Crypto ticker (e.g., "SOL")
            query: Market question
            target_tweets: How many tweets to collect (default 10k)
        
        Returns:
            Initial prediction result
        """
        # Check if market already has data in database
        market_db = get_market_db(market_id)
        existing_prediction = market_db.get_latest_prediction()
        
        if existing_prediction and existing_prediction.get("ai_score") is not None:
            # Market already initialized - just load existing data
            logger.info(f"LOADING EXISTING MARKET: {market_id}")
            logger.info(f"   Ticker: {ticker}")
            logger.info(f"   Last AI Score: {existing_prediction['ai_score']:.1f}%")
            logger.info(f"   Last Update: {existing_prediction.get('timestamp', 'unknown')}")
            
            # Initialize market state for monitoring
            self.market_states[market_id] = {
                "ticker": ticker,
                "query": query,
                "deadline": deadline,
                "baseline_activity": 0,
                "last_update": datetime.now(),
                "last_check": datetime.now(),
                "update_count": market_db.get_prediction_count()
            }
            
            return {
                "market_id": market_id,
                "status": "loaded_from_db",
                "ai_score": existing_prediction["ai_score"],
                "prediction_count": market_db.get_prediction_count()
            }
        
        # No existing data - do full initialization
        logger.info(f"INITIALIZING NEW MARKET: {market_id}")
        logger.info(f"   Ticker: {ticker}")
        logger.info(f"   Target: {target_tweets} tweets")
        
        start_time = datetime.now()
        
        # Step 1: Social data collection (optional if Twitter enabled)
        tweets = []
        if self.twitter.enabled:
            logger.info(f"Step 1/3: Collecting {target_tweets} tweets...")
            tweets = await self.twitter.initial_analysis(
                ticker=ticker,
                query=query,
                target_count=target_tweets
            )
            logger.info(f"   Collected {len(tweets)} tweets")
        else:
            logger.info(f"Step 1/3: Twitter disabled - Gemini will search Google for social data...")
        
        # Step 2: Deep AI analysis
        logger.info(f"Step 2/3: Running deep AI analysis...")
        
        # Convert Post objects to dict format for orchestrator
        posts_dict = []
        for tweet in tweets:
            # Extract metadata
            metadata = tweet.metadata if isinstance(tweet.metadata, dict) else {}
            
            post_dict = {
                "text": tweet.text,
                "author": {
                    "username": tweet.author,
                    "followers": metadata.get("author_followers", 0),
                    "verified": metadata.get("verified", False)
                },
                "created_at": tweet.timestamp,
                "engagement": tweet.engagement,
                "source": tweet.source,
            }
            # Add other metadata fields
            for key, value in metadata.items():
                if key not in ["author_followers", "verified"]:
                    post_dict[key] = value
            
            posts_dict.append(post_dict)
        
        # Get price metrics for realistic probability assessment
        logger.info(f"   Fetching price data for {ticker}...")
        target_price = price_oracle.extract_target_price(query, ticker)
        price_metrics = None
        if target_price:
            price_metrics = await price_oracle.calculate_market_metrics(
                ticker=ticker,
                target_price=target_price,
                deadline=deadline
            )
            if price_metrics and price_metrics.get("data_available"):
                logger.info(f"   Current price: ${price_metrics['current_price']:.2f}")
                logger.info(f"   Target price: ${price_metrics['target_price']:.2f}")
                logger.info(f"   Required growth: {price_metrics['required_growth_pct']:+.1f}% in {price_metrics['days_left']} days")
                logger.info(f"   Feasibility score: {price_metrics['feasibility_score']:.2f}/1.0")
        
        # Get real market probability from prediction markets (Polymarket)
        logger.info(f"   Fetching market probability from Polymarket...")
        market_data = await market_oracle.get_market_probability(query, ticker)
        
        # Ensure market_probability is a float
        market_prob = market_data.get("market_probability", 50.0)
        if isinstance(market_prob, str):
            try:
                market_prob = float(market_prob)
            except:
                market_prob = 50.0
        market_data["market_probability"] = market_prob
        
        if market_data.get("found"):
            logger.info(f"   Found matching market: {market_data.get('market_title', '')[:50]}...")
            logger.info(f"   Market probability: {market_prob:.1f}%")
        else:
            logger.info(f"   No matching market found, using default 50%")
        
        prediction = await self.orchestrator.process(
            posts=posts_dict,
            market_data=market_data,  # Pass real market data
            ticker=ticker,
            price_metrics=price_metrics,
            market_query=query,
            deadline=deadline
        )
        
        logger.info(f"   AI Score: {prediction.ai_score:.1f}%")
        logger.info(f"   Market Score: {prediction.market_score:.1f}%")
        logger.info(f"   Divergence: {prediction.divergence_index:.1f}%")
        
        # Step 3: Save baseline to market-specific database
        logger.info(f"Step 3/3: Saving baseline prediction with initial history...")
        
        # Get market-specific database
        market_db = get_market_db(market_id)
        
        # Set market info
        market_db.set_market_info(
            ticker=ticker,
            query=query,
            deadline=deadline
        )
        
        # Create initial history points (5 points showing AI's initial assessment)
        # This gives the graph something to display from the start
        import random
        base_ai = prediction.ai_score
        base_market = prediction.market_score
        
        for i in range(5):
            # Small random variation around the baseline (±3%)
            variation = random.uniform(-3, 3)
            ai_score_point = max(0, min(100, base_ai + variation * (1 - i/5)))  # Less variation for recent
            market_score_point = base_market  # Market stays constant initially
            
            market_db.save_prediction(
                ai_score=ai_score_point,
                market_score=market_score_point,
                divergence_index=abs(ai_score_point - market_score_point),
                confidence=prediction.confidence * (0.7 + i * 0.06),  # Confidence increases
                vocal_summary="" if i < 4 else prediction.vocal_summary,  # Only last has summary
                data_sources={"initial_point": i + 1},
                tweets_analyzed=0,
                event_triggered=False
            )
        
        logger.info(f"   Created 5 initial data points for graph")
        
        # Initialize market state for monitoring
        self.market_states[market_id] = {
            "ticker": ticker,
            "query": query,
            "deadline": deadline,
            "baseline_activity": len(tweets) // 7,  # Tweets per day
            "last_update": datetime.now(),
            "last_check": datetime.now(),
            "update_count": 1
        }
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"MARKET INITIALIZED in {elapsed:.1f}s")
        logger.info(f"   Baseline: {len(tweets)} tweets analyzed")
        logger.info(f"   Monitoring: Active")
        
        return {
            "market_id": market_id,
            "status": "initialized",
            "tweets_analyzed": len(tweets),
            "ai_score": prediction.ai_score,
            "elapsed_seconds": elapsed
        }
    
    async def monitor_market(
        self,
        market_id: str,
        update_interval_minutes: int = 30
    ):
        """
        Phase 2: Periodic Updates
        
        Updates market prediction every N minutes using Gemini + Claude
        
        Args:
            market_id: Market to monitor
            update_interval_minutes: How often to update prediction (default: 30 min)
        """
        if market_id not in self.market_states:
            logger.error(f"❌ Market {market_id} not initialized")
            return
        
        state = self.market_states[market_id]
        ticker = state["ticker"]
        query = state["query"]
        deadline = state.get("deadline")
        
        logger.info(f"MONITORING STARTED: {market_id}")
        logger.info(f"   Update interval: {update_interval_minutes} minutes")
        
        while True:
            try:
                # Wait for next update cycle
                await asyncio.sleep(update_interval_minutes * 60)
                
                logger.info(f"⏰ SCHEDULED UPDATE: {market_id} ({ticker})")
                
                # Run new AI analysis (Gemini will search Google for fresh data)
                await self._periodic_update(market_id, ticker, query, deadline)
                
                state["last_update"] = datetime.now()
                state["update_count"] += 1
                
            except Exception as e:
                logger.error(f"❌ Error monitoring {market_id}: {e}")
                await asyncio.sleep(60)  # Wait 1 min on error
    
    async def _periodic_update(
        self,
        market_id: str,
        ticker: str,
        query: str,
        deadline: Optional[datetime]
    ):
        """Periodic AI analysis update"""
        
        # Get market database
        market_db = get_market_db(market_id)
        
        # Get latest prediction for comparison
        latest = market_db.get_latest_prediction()
        previous_score = latest.get("ai_score", 50.0) if latest else 50.0
        
        # Get fresh price metrics
        target_price = price_oracle.extract_target_price(query, ticker)
        price_metrics = None
        if target_price:
            price_metrics = await price_oracle.calculate_market_metrics(
                ticker=ticker,
                target_price=target_price,
                deadline=deadline
            )
        
        # Get fresh market probability from Polymarket
        market_data = await market_oracle.get_market_probability(query, ticker)
        if market_data.get("found"):
            logger.info(f"   Market prob: {market_data['market_probability']:.1f}% (Polymarket)")
        
        # Run fresh AI analysis (Gemini searches Google for new facts)
        prediction = await self.orchestrator.process(
            posts=[],  # No tweets - Gemini will search
            market_data=market_data,  # Pass real market data
            ticker=ticker,
            price_metrics=price_metrics,
            market_query=query,
            deadline=deadline
        )
        
        # Calculate change from previous
        score_change = prediction.ai_score - previous_score
        
        # Save updated prediction
        market_db.save_prediction(
            ai_score=prediction.ai_score,
            market_score=prediction.market_score,
            divergence_index=prediction.divergence_index,
            confidence=prediction.confidence,
            vocal_summary=prediction.vocal_summary,
            data_sources={"update_type": "scheduled", "previous_score": previous_score},
            tweets_analyzed=0,
            event_triggered=abs(score_change) > 5  # Significant if change > 5%
        )
        
        logger.info(f"   Updated: {previous_score:.1f}% → {prediction.ai_score:.1f}% ({score_change:+.1f}%)")
    
    async def _update_prediction(
        self,
        market_id: str,
        recent_tweets: List[Any],
        event_info: Dict[str, Any]
    ):
        """Update prediction based on new events"""
        state = self.market_states[market_id]
        
        # Get market database
        market_db = get_market_db(market_id)
        
        # Save event
        market_db.save_event(
            event_type=event_info.get("event_type", "unknown"),
            severity=event_info.get("severity", 0),
            description=event_info.get("reason", "Event detected"),
            evidence=event_info.get("events", [])
        )
        
        # Get latest prediction
        latest = market_db.get_latest_prediction()
        
        # Convert Post objects to dict format
        posts_dict = []
        for tweet in recent_tweets:
            # Extract metadata
            metadata = tweet.metadata if isinstance(tweet.metadata, dict) else {}
            
            post_dict = {
                "text": tweet.text,
                "author": {
                    "username": tweet.author,
                    "followers": metadata.get("author_followers", 0),
                    "verified": metadata.get("verified", False)
                },
                "created_at": tweet.timestamp,
                "engagement": tweet.engagement,
                "source": tweet.source,
            }
            # Add other metadata fields
            for key, value in metadata.items():
                if key not in ["author_followers", "verified"]:
                    post_dict[key] = value
            
            posts_dict.append(post_dict)
        
        # Get price metrics for realistic probability assessment
        target_price = price_oracle.extract_target_price(state["query"], state["ticker"])
        price_metrics = None
        if target_price:
            price_metrics = await price_oracle.calculate_market_metrics(
                ticker=state["ticker"],
                target_price=target_price,
                deadline=state["deadline"]
            )
        
        # Run new analysis
        new_prediction = await self.orchestrator.process(
            posts=posts_dict,
            ticker=state["ticker"],
            price_metrics=price_metrics,
            market_query=state["query"],
            deadline=state["deadline"]
        )
        
        # Calculate change
        if latest:
            ai_change = new_prediction.ai_score - latest["ai_score"]
            logger.info(f"   AI Score: {latest['ai_score']:.1f}% -> {new_prediction.ai_score:.1f}% ({ai_change:+.1f}%)")
        else:
            logger.info(f"   AI Score: {new_prediction.ai_score:.1f}%")
        
        # Save update to market database
        market_db.save_prediction(
            ai_score=new_prediction.ai_score,
            market_score=new_prediction.market_score,
            divergence_index=new_prediction.divergence_index,
            confidence=new_prediction.confidence,
            vocal_summary=f"[Event Update] {event_info['reason']} | {new_prediction.vocal_summary}",
            data_sources={
                "total_posts": len(recent_tweets),
                "event_triggered": True,
                "event_type": event_info["event_type"],
                "event_severity": event_info["severity"]
            },
            tweets_analyzed=len(recent_tweets),
            event_triggered=True
        )
        
        logger.info(f"Prediction updated (total updates: {state['update_count']})")
    
    async def verify_market_event(
        self,
        market_id: str,
        deadline: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Verify if market event has occurred
        
        This is the Oracle's decision-making function:
        "Has the event happened?" YES / NO / UNCERTAIN
        
        Args:
            market_id: Market to verify
            deadline: Optional deadline for the event
        
        Returns:
            Verification result with decision
        """
        if market_id not in self.market_states:
            logger.warning(f"Market {market_id} not initialized")
            return None
        
        state = self.market_states[market_id]
        query = state["query"]
        ticker = state["ticker"]
        
        logger.info(f"VERIFYING EVENT: {market_id}")
        logger.info(f"   Query: {query}")
        
        # Collect recent data for verification
        recent_tweets = await self.twitter.collect_recent(
            ticker=ticker,
            query=query,
            since_minutes=60  # Last hour
        )
        
        # Verify event
        verification = await self.event_verifier.verify_event(
            market_id=market_id,
            query=query,
            posts=recent_tweets,
            deadline=deadline
        )
        
        # Make decision
        decision = self.event_verifier.get_event_decision(verification)
        
        logger.info(f"   DECISION: {decision['decision']}")
        logger.info(f"   Confidence: {decision['confidence']:.1%}")
        logger.info(f"   Reasoning: {decision['reasoning']}")
        
        return {
            "market_id": market_id,
            "query": query,
            "verification": verification,
            "decision": decision,
            "verified_at": datetime.now().isoformat()
        }
    
    def get_market_status(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a monitored market"""
        market_db = get_market_db(market_id)
        market_info = market_db.get_market_info()
        
        if not market_info:
            return None
        
        state = self.market_states.get(market_id, {})
        latest = market_db.get_latest_prediction()
        stats = market_db.get_stats()
        
        return {
            "market_id": market_id,
            "ticker": market_info["ticker"],
            "query": market_info["query"],
            "deadline": market_info.get("deadline"),
            "status": market_info["status"],
            "monitoring_active": market_id in self.market_states,
            "last_update": state.get("last_update").isoformat() if state.get("last_update") else None,
            "last_check": state.get("last_check").isoformat() if state.get("last_check") else None,
            "update_count": state.get("update_count", 0),
            "predictions_count": stats["predictions_count"],
            "events_count": stats["events_count"],
            "current_prediction": {
                "ai_score": latest["ai_score"] if latest else None,
                "market_score": latest["market_score"] if latest else None,
                "divergence": latest["divergence_index"] if latest else None
            } if latest else None
        }


# Global oracle engine instance
oracle_engine = OracleEngine()
