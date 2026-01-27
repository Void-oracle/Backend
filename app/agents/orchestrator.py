"""
Agent Orchestrator
Coordinates multiple AI agents to produce final oracle predictions
"""
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import time

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.agents.base_agent import BaseAgent
from app.agents.sentiment_agent import SentimentAgent
from app.agents.bot_detector import BotDetectorAgent
from app.agents.claude_oracle_agent import ClaudeOracleAgent
from app.agents.gemini_news_agent import gemini_agent
from app.models.schemas import (
    OraclePredictionResponse,
    SentimentAnalysis,
    BotDetectionResult,
    DataSourceStats
)
from app.config import settings


class AgentOrchestrator(BaseAgent):
    """Orchestrates multiple agents to produce final predictions"""
    
    def __init__(self):
        super().__init__("AgentOrchestrator")
        self.sentiment_agent = SentimentAgent()
        self.bot_detector = BotDetectorAgent()
        self.claude_oracle = ClaudeOracleAgent() # Claude for math + rational analysis
        self.gemini_agent = gemini_agent # Gemini for news + fact checking
        
        # Initialize AI client based on provider
        if settings.AI_PROVIDER == "claude" and settings.ANTHROPIC_API_KEY:
            self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model = settings.ANTHROPIC_MODEL
            self.provider = "claude"
        else:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
            self.provider = "openai"
    
    async def process(
        self,
        posts: List[Dict[str, Any]],
        market_data: Optional[Dict[str, Any]] = None,
        ticker: str = "UNKNOWN",
        price_metrics: Optional[Dict[str, Any]] = None,
        market_query: Optional[str] = None,
        deadline: Optional[Any] = None
    ) -> OraclePredictionResponse:
        """
        Orchestrate all agents to produce final oracle prediction
        
        Args:
            posts: Social media posts
            market_data: Market probability data
            ticker: Token ticker or market identifier
            
        Returns:
            Complete oracle prediction response
        """
        start_time = time.time()
        
        self.logger.info(f"Orchestrating prediction for {ticker} with {len(posts)} posts")
        
        # Run agents in parallel
        sentiment_task = asyncio.create_task(
            self.sentiment_agent.process(posts)
        )
        bot_detection_task = asyncio.create_task(
            self.bot_detector.process(posts)
        )
        
        # Wait for both agents
        sentiment_analysis, bot_detection = await asyncio.gather(
            sentiment_task,
            bot_detection_task
        )
        
        # Calculate data source statistics
        data_sources = self._calculate_data_sources(posts, bot_detection)
        
        # Get market score
        market_score = market_data.get('market_probability', 50.0) if market_data else 50.0
        
        # Determine event type for appropriate agent selection
        event_type = "generic"
        if market_query and self.gemini_agent.enabled:
            event_type = self.gemini_agent.classify_event_type(market_query)
            self.logger.info(f"Event type detected: {event_type}")
        
        # Use Claude for ALL events (Gemini disabled due to quota limits)
        # Claude can understand context and do deep research on any topic
        if self.provider == "claude" and market_query:
            self.logger.info(f"Using Gemini for {event_type} event with Google Search")
            try:
                gemini_analysis = await self.gemini_agent.analyze_event(
                    query=market_query or ticker,
                    event_type=event_type,
                    deadline=deadline,
                    current_market_prob=market_score,
                    price_metrics=price_metrics
                )
                
                # Use Gemini's fact-based probability
                ai_score = gemini_analysis.get("probability", 50.0)
                confidence = gemini_analysis.get("confidence", 0.75)
                
                # Generate summary from Gemini's findings
                vocal_summary = f"**🔍 Gemini News Analysis ({ai_score:.1f}% probability)**\n\n"
                vocal_summary += f"**Status:** {gemini_analysis.get('current_status', 'Unknown')}\n\n"
                vocal_summary += f"**Reasoning:**\n{gemini_analysis.get('reasoning', '')}\n\n"
                
                if gemini_analysis.get("key_facts"):
                    vocal_summary += "**Key Facts:**\n"
                    for fact in gemini_analysis["key_facts"][:5]:
                        vocal_summary += f"- {fact}\n"
                    vocal_summary += "\n"
                
                if gemini_analysis.get("sources"):
                    vocal_summary += "**Sources:**\n"
                    for source in gemini_analysis["sources"][:3]:
                        vocal_summary += f"- {source}\n"
                
                self.logger.info(f"Gemini: {ai_score:.1f}% (market: {market_score:.1f}%, verdict: {gemini_analysis.get('verdict')})")
            
            except Exception as e:
                self.logger.error(f"Gemini analysis failed, falling back to Claude: {e}")
                # Always fallback to Claude for any event type
                if self.provider == "claude":
                    ai_score, confidence, vocal_summary = await self._use_claude_analysis(
                        ticker, market_query, market_score, posts, price_metrics, 
                        sentiment_analysis, bot_detection, data_sources
                    )
                else:
                    ai_score, confidence, vocal_summary = await self._use_sentiment_analysis(
                        ticker, market_score, sentiment_analysis, bot_detection, data_sources
                    )
        
        # Use ClaudeOracleAgent for price targets (needs mathematical analysis)
        elif price_metrics and price_metrics.get("data_available") and self.provider == "claude":
            self.logger.info(f"Using ClaudeOracleAgent for deep analysis with price context")
            try:
                claude_analysis = await self.claude_oracle.analyze_market(
                    market_question=ticker,  # TODO: Pass actual market question
                    market_probability=market_score,
                    social_data=posts,
                    technical_data=None,
                    historical_data=None,
                    price_metrics=price_metrics
                )
                
                # Use Claude's rational probability
                ai_score = claude_analysis.get("rational_probability", 50.0)
                confidence = claude_analysis.get("confidence", 0.75)
                
                # Generate summary from Claude's reasoning
                vocal_summary = f"**AI Analysis ({ai_score:.1f}% probability)**\n\n"
                vocal_summary += claude_analysis.get("reasoning", "")
                vocal_summary += f"\n\n**Key Factors:**\n"
                for factor in claude_analysis.get("key_factors", []):
                    vocal_summary += f"- {factor}\n"
                
                self.logger.info(f"Claude Oracle: {ai_score:.1f}% (market: {market_score:.1f}%)")
            except Exception as e:
                self.logger.error(f"Claude Oracle failed, falling back to sentiment-based score: {e}")
                # Fallback to simple sentiment calculation
                ai_score = self._calculate_ai_score(sentiment_analysis, bot_detection)
                confidence = self._calculate_confidence(sentiment_analysis, bot_detection, len(posts))
                
                # Generate standard vocal summary
                vocal_summary = await self._generate_vocal_summary(
                    ticker=ticker,
                    ai_score=ai_score,
                    market_score=market_score,
                    divergence_index=abs(ai_score - market_score),
                    sentiment_analysis=sentiment_analysis,
                    bot_detection=bot_detection,
                    data_sources=data_sources
                )
        else:
            # Standard sentiment-based calculation
            ai_score = self._calculate_ai_score(sentiment_analysis, bot_detection)
            confidence = self._calculate_confidence(sentiment_analysis, bot_detection, len(posts))
            
            # Generate standard vocal summary
            vocal_summary = await self._generate_vocal_summary(
                ticker=ticker,
                ai_score=ai_score,
                market_score=market_score,
                divergence_index=abs(ai_score - market_score),
                sentiment_analysis=sentiment_analysis,
                bot_detection=bot_detection,
                data_sources=data_sources
            )
        
        # Calculate divergence
        divergence_index = abs(ai_score - market_score)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return OraclePredictionResponse(
            ticker=ticker,
            ai_score=ai_score,
            market_score=market_score,
            divergence_index=divergence_index,
            vocal_summary=vocal_summary,
            confidence=confidence,
            data_sources=data_sources,
            sentiment_analysis=sentiment_analysis,
            bot_detection=bot_detection,
            processing_time_ms=processing_time
        )
    
    def _calculate_ai_score(
        self,
        sentiment: SentimentAnalysis,
        bot_detection: BotDetectionResult
    ) -> float:
        """
        Calculate AI probability score (0-100) from sentiment and bot detection
        
        Formula:
        - Base score from sentiment (0-100)
        - Adjusted by bot detection (reduce if high bot activity)
        - Weighted by authentic influencer engagement
        """
        # Convert sentiment from -1,1 to 0,100
        sentiment_score = (sentiment.sentiment_score.overall + 1.0) * 50.0
        
        # Adjust for bot activity
        bot_penalty = bot_detection.bot_probability * 20  # Up to -20 points
        adjusted_score = sentiment_score - bot_penalty
        
        # Boost from authentic influencers
        influencer_boost = 0.0
        if bot_detection.top_influencers:
            # More credible influencers = more boost
            avg_credibility = sum(
                inf.credibility_score for inf in bot_detection.top_influencers[:5]
            ) / min(5, len(bot_detection.top_influencers))
            influencer_boost = avg_credibility * 10  # Up to +10 points
        
        final_score = adjusted_score + influencer_boost
        
        # Apply momentum
        momentum_adjustment = sentiment.sentiment_momentum * 5  # Up to ±5 points
        final_score += momentum_adjustment
        
        # Clamp to 0-100
        return max(0.0, min(100.0, final_score))
    
    def _calculate_data_sources(
        self,
        posts: List[Dict[str, Any]],
        bot_detection: BotDetectionResult
    ) -> DataSourceStats:
        """Calculate statistics about data sources"""
        total_posts = len(posts)
        
        # Count influencer posts
        influencer_usernames = {inf.username for inf in bot_detection.top_influencers}
        influencer_posts = sum(
            1 for post in posts
            if post.get('author', {}).get('username') in influencer_usernames
        )
        
        # Calculate total engagement
        total_engagement = sum(post.get('engagement', 0) for post in posts)
        
        # Count unique accounts
        unique_accounts = len(set(
            post.get('author', {}).get('username', '')
            for post in posts
        ))
        
        # Calculate time span
        timestamps = [post.get('created_at') for post in posts if post.get('created_at')]
        time_span_hours = 0.0
        if timestamps:
            try:
                from datetime import datetime
                dates = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                time_span = max(dates) - min(dates)
                time_span_hours = time_span.total_seconds() / 3600
            except:
                pass
        
        return DataSourceStats(
            twitter_posts=total_posts,
            influencer_posts=influencer_posts,
            bot_ratio=bot_detection.bot_probability,
            total_engagement=int(total_engagement),  # Convert to int
            unique_accounts=unique_accounts,
            time_span_hours=time_span_hours
        )
    
    async def _generate_vocal_summary(
        self,
        ticker: str,
        ai_score: float,
        market_score: float,
        divergence_index: float,
        sentiment_analysis: SentimentAnalysis,
        bot_detection: BotDetectionResult,
        data_sources: DataSourceStats
    ) -> str:
        """Generate human-readable executive summary using AI"""
        
        # Prepare context for AI
        sentiment = sentiment_analysis.sentiment_score
        sentiment_desc = (
            "strongly bullish" if sentiment.overall > 0.6 else
            "bullish" if sentiment.overall > 0.2 else
            "neutral" if sentiment.overall > -0.2 else
            "bearish" if sentiment.overall > -0.6 else
            "strongly bearish"
        )
        
        bot_level = (
            "minimal" if bot_detection.bot_probability < 0.2 else
            "moderate" if bot_detection.bot_probability < 0.5 else
            "high"
        )
        
        divergence_level = (
            "significant" if divergence_index > settings.DIVERGENCE_HIGH_THRESHOLD else
            "moderate" if divergence_index > 10 else
            "low"
        )
        
        influencer_count = len(bot_detection.top_influencers)
        
        prompt = f"""Generate a concise executive summary (2-3 sentences) for an oracle prediction analysis.

Context:
- Ticker/Market: {ticker}
- AI Probability Score: {ai_score:.1f}%
- Market Probability: {market_score:.1f}%
- Divergence: {divergence_index:.1f}% ({divergence_level})
- Sentiment: {sentiment_desc} (overall: {sentiment.overall:.2f})
- Bot Activity: {bot_level} ({bot_detection.bot_probability:.1%})
- Data Sources: {data_sources.twitter_posts} posts, {influencer_count} top influencers
- Key Topics: {', '.join(sentiment_analysis.trending_topics[:3])}
- Sentiment Momentum: {"increasing" if sentiment_analysis.sentiment_momentum > 0.2 else "decreasing" if sentiment_analysis.sentiment_momentum < -0.2 else "stable"}

Write a professional summary explaining:
1. The main sentiment finding
2. Key factors (bot activity, influencer engagement, unusual patterns)
3. Whether market is underpricing or overpricing based on AI analysis

Keep it factual, concise, and actionable. Use specific numbers when relevant.
"""
        
        try:
            if self.provider == "claude":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0.4,
                    system="You are VOID, an AI oracle that analyzes market sentiment. Provide clear, data-driven executive summaries.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                summary = response.content[0].text.strip()
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are VOID, an AI oracle that analyzes market sentiment. Provide clear, data-driven executive summaries."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    max_tokens=200
                )
                summary = response.choices[0].message.content.strip()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate vocal summary: {e}")
            # Fallback to template
            return self._fallback_summary(
                ticker, ai_score, market_score, divergence_index,
                sentiment_desc, bot_level, divergence_level
            )
    
    def _fallback_summary(
        self,
        ticker: str,
        ai_score: float,
        market_score: float,
        divergence_index: float,
        sentiment_desc: str,
        bot_level: str,
        divergence_level: str
    ) -> str:
        """Fallback summary template if AI generation fails"""
        direction = "overpricing" if ai_score < market_score else "underpricing"
        
        return (
            f"Analysis of {ticker} shows {sentiment_desc} sentiment with AI probability "
            f"at {ai_score:.1f}% vs market {market_score:.1f}% ({divergence_level} divergence of {divergence_index:.1f}%). "
            f"Bot activity is {bot_level}. Market appears to be {direction} the probability."
        )
    
    def _calculate_confidence(
        self,
        sentiment: SentimentAnalysis,
        bot_detection: BotDetectionResult,
        num_posts: int
    ) -> float:
        """Calculate overall confidence in the prediction"""
        # Start with sentiment confidence
        confidence = sentiment.sentiment_score.confidence
        
        # Reduce confidence if high bot activity
        confidence *= (1.0 - bot_detection.bot_probability * 0.5)
        
        # Increase confidence with more data
        if num_posts > 100:
            confidence = min(1.0, confidence * 1.2)
        elif num_posts < 20:
            confidence *= 0.7
        
        # Increase confidence with more influencers
        if len(bot_detection.top_influencers) > 10:
            confidence = min(1.0, confidence * 1.1)
        
        return max(0.0, min(1.0, confidence))
    
    async def _use_claude_analysis(
        self,
        ticker: str,
        market_query: str,
        market_score: float,
        posts: List[Dict[str, Any]],
        price_metrics: Optional[Dict[str, Any]],
        sentiment_analysis: SentimentAnalysis,
        bot_detection: BotDetectionResult,
        data_sources: DataSourceStats
    ) -> Tuple[float, float, str]:
        """Use Claude for mathematical + rational analysis"""
        try:
            # Use market_query if available, otherwise use ticker
            question = market_query if market_query else ticker
            
            claude_analysis = await self.claude_oracle.analyze_market(
                market_question=question,
                market_probability=market_score,
                social_data=posts,
                technical_data=None,
                historical_data=None,
                price_metrics=price_metrics
            )
            
            ai_score = claude_analysis.get("rational_probability", 50.0)
            confidence = claude_analysis.get("confidence", 0.75)
            
            vocal_summary = f"**AI Analysis ({ai_score:.1f}% probability)**\n\n"
            vocal_summary += claude_analysis.get("reasoning", "")
            vocal_summary += f"\n\n**Key Factors:**\n"
            for factor in claude_analysis.get("key_factors", []):
                vocal_summary += f"- {factor}\n"
            
            self.logger.info(f"Claude Oracle: {ai_score:.1f}% (market: {market_score:.1f}%)")
            return ai_score, confidence, vocal_summary
        
        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")
            raise
    
    async def _use_sentiment_analysis(
        self,
        ticker: str,
        market_score: float,
        sentiment_analysis: SentimentAnalysis,
        bot_detection: BotDetectionResult,
        data_sources: DataSourceStats
    ) -> Tuple[float, float, str]:
        """Fallback to sentiment-based analysis"""
        ai_score = self._calculate_ai_score(sentiment_analysis, bot_detection)
        confidence = self._calculate_confidence(sentiment_analysis, bot_detection, 
                                               data_sources.twitter_posts)
        
        vocal_summary = await self._generate_vocal_summary(
            ticker=ticker,
            ai_score=ai_score,
            market_score=market_score,
            divergence_index=abs(ai_score - market_score),
            sentiment_analysis=sentiment_analysis,
            bot_detection=bot_detection,
            data_sources=data_sources
        )
        
        return ai_score, confidence, vocal_summary
    
    async def health_check(self) -> bool:
        """Check health of all agents"""
        try:
            sentiment_health = await self.sentiment_agent.health_check()
            bot_health = await self.bot_detector.health_check()
            return sentiment_health and bot_health
        except:
            return False
