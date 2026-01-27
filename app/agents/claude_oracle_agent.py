"""
Claude-based Oracle Agent for Prediction Markets
Specialized for finding overvalued/undervalued markets using technical & historical data
"""
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime
import json

from anthropic import AsyncAnthropic

from app.agents.base_agent import BaseAgent
from app.models.schemas import SentimentAnalysis, SentimentScore
from app.config import settings


class ClaudeOracleAgent(BaseAgent):
    """
    Advanced oracle agent using Claude 3.5 Sonnet
    
    Specialized for:
    - Technical data analysis
    - Historical pattern recognition
    - Rational probability assessment
    - Detecting crowd overvaluation
    
    Perfect for finding markets where people emotionally overestimate probabilities
    """
    
    def __init__(self):
        super().__init__("ClaudeOracleAgent")
        self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        self.model = settings.ANTHROPIC_MODEL
    
    async def process(self, data: Any) -> Any:
        """
        Required by BaseAgent abstract class.
        Use analyze_market() instead for full market analysis.
        """
        return {"message": "Use analyze_market() for market analysis"}
    
    async def analyze_market(
        self,
        market_question: str,
        market_probability: float,
        social_data: List[Dict[str, Any]],
        technical_data: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
        price_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Deep market analysis using Claude
        
        Returns rational probability vs market probability to find mispricing
        """
        self.logger.info(f"Analyzing market with Claude: {market_question}")
        
        # Prepare comprehensive context
        context = self._prepare_analysis_context(
            market_question,
            market_probability,
            social_data,
            technical_data,
            historical_data,
            price_metrics
        )
        
        # Get Claude's rational analysis
        analysis = await self._claude_rational_analysis(context, market_probability)
        
        return analysis
    
    def _prepare_analysis_context(
        self,
        question: str,
        market_prob: float,
        social_data: List[Dict[str, Any]],
        technical_data: Optional[Dict[str, Any]],
        historical_data: Optional[Dict[str, Any]],
        price_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Prepare comprehensive context for Claude analysis
        """
        # PRICE ANALYSIS (MOST IMPORTANT!)
        price_summary = ""
        if price_metrics and price_metrics.get("data_available"):
            current_price = price_metrics.get("current_price")
            target_price = price_metrics.get("target_price")
            required_growth = price_metrics.get("required_growth_pct", 0)
            days_left = price_metrics.get("days_left", 0)
            daily_growth_needed = price_metrics.get("daily_growth_needed", 0)
            feasibility_score = price_metrics.get("feasibility_score", 0.5)
            
            hist_30d = price_metrics.get("historical_30d", {})
            hist_90d = price_metrics.get("historical_90d", {})
            
            price_summary = f"""
═══════════════════════════════════════════════════════════
🎯 PRICE TARGET ANALYSIS (CRITICAL FOR PROBABILITY!)
═══════════════════════════════════════════════════════════

Current Price: ${current_price:.2f}
Target Price:  ${target_price:.2f}
Required Growth: {required_growth:+.1f}%
Time Remaining: {days_left} days

📊 MATHEMATICAL REQUIREMENTS:
- Daily Growth Needed (compounded): {daily_growth_needed:.2f}% per day
- If growth stops for even a few days, target becomes impossible

📈 HISTORICAL PERFORMANCE (last 30 days):
- Actual 30d Growth: {hist_30d.get('price_change', 0):+.1f}%
- Average Daily Return: {hist_30d.get('avg_daily_return', 0):+.2f}%

📈 HISTORICAL PERFORMANCE (last 90 days):
- Actual 90d Growth: {hist_90d.get('price_change', 0):+.1f}%
- Average Daily Return: {hist_90d.get('avg_daily_return', 0):+.2f}%

⚠️ FEASIBILITY ASSESSMENT:
- Algorithmic Feasibility Score: {feasibility_score:.2f}/1.0
  (based on required growth vs historical volatility)

🧮 REALITY CHECK:
- If historical average daily return = {hist_30d.get('avg_daily_return', 0):+.2f}%
- But need {daily_growth_needed:.2f}% per day
- That's a {abs(daily_growth_needed / max(abs(hist_30d.get('avg_daily_return', 0.01)), 0.01)):.1f}x increase in growth rate!

⚡ KEY INSIGHT:
Social sentiment does NOT change mathematical requirements!
Even 100% bullish sentiment cannot override physics of price movement.

CRITICAL INSTRUCTION: Your probability MUST be primarily based on this
mathematical analysis, NOT on Twitter sentiment. Sentiment is a minor factor.
"""
        else:
            price_summary = f"""
⚠️ PRICE DATA UNAVAILABLE
Market Question: {question}
Current Market Probability: {market_prob}%

Note: Without price data, base probability on historical precedents and logic.
"""
        
        # Social sentiment summary
        total_posts = len(social_data)
        bullish_count = sum(1 for p in social_data if 'bull' in p.get('text', '').lower())
        bearish_count = sum(1 for p in social_data if 'bear' in p.get('text', '').lower())
        
        social_summary = f"""
SOCIAL SENTIMENT DATA (SECONDARY FACTOR):
- Total posts analyzed: {total_posts}
- Bullish mentions: {bullish_count} ({bullish_count/max(total_posts,1)*100:.1f}%)
- Bearish mentions: {bearish_count} ({bearish_count/max(total_posts,1)*100:.1f}%)
- Average engagement: {sum(p.get('engagement', 0) for p in social_data) / max(total_posts,1):.0f}

Top posts sentiment:
"""
        # Add top 3 posts (reduced from 5 to save tokens)
        top_posts = sorted(social_data, key=lambda x: x.get('engagement', 0), reverse=True)[:3]
        for i, post in enumerate(top_posts, 1):
            social_summary += f"{i}. {post.get('text', '')[:150]}...\n"
        
        # Technical data summary
        technical_summary = ""
        if technical_data:
            technical_summary = f"""
TECHNICAL DATA:
{json.dumps(technical_data, indent=2)}
"""
        
        # Historical data summary
        historical_summary = ""
        if historical_data:
            historical_summary = f"""
HISTORICAL CONTEXT:
{json.dumps(historical_data, indent=2)}
"""
        
        context = f"""
PREDICTION MARKET ANALYSIS REQUEST

{price_summary}

{social_summary}

{technical_summary}

{historical_summary}

Your task: Provide a RATIONAL, MATHEMATICALLY-GROUNDED probability assessment.

PRIORITY OF FACTORS:
1. 🎯 Mathematical feasibility (price requirements vs historical data) - 70% weight
2. 📊 Technical indicators and trends - 20% weight  
3. 💬 Social sentiment - 10% weight (sentiment often wrong!)

BE CONSERVATIVE. If math says it's unlikely, probability should be LOW regardless of hype.
"""
        return context
    
    async def _claude_rational_analysis(self, context: str, market_probability: float = 50.0) -> Dict[str, Any]:
        """
        Get Claude's rational probability analysis
        """
        system_prompt = """You are a HYPER-RATIONAL prediction market oracle that analyzes events based on MATHEMATICS FIRST.

🎯 ANALYSIS HIERARCHY (in order of importance):

1. **MATHEMATICAL FEASIBILITY** (70% weight)
   - Required price movement vs time available
   - Historical volatility vs required volatility
   - Compound growth mathematics
   - If math says <10% chance, your probability should be <15% MAX
   - If math says >70% chance, consider 60-80%
   
2. **TECHNICAL INDICATORS** (20% weight)
   - Price trends, momentum, support/resistance
   - Volume analysis
   - Historical patterns
   
3. **SOCIAL SENTIMENT** (10% weight)
   - Twitter sentiment is often WRONG and emotionally driven
   - Use only as minor confirmation, not primary signal
   - Crowds are notoriously bad at probability estimation

🧮 CRITICAL RULES:

- If target requires +100% growth in <60 days → probability <20% unless extraordinary evidence
- If target requires +200% growth in <90 days → probability <10%
- If target requires +50% growth in <30 days → probability <30%
- If historical 30d average daily return is X%, but need 5X% → very unlikely

Example of CORRECT analysis:
- Market says: 50% chance SOL reaches $300 (current $130)
- Required: +130% growth in 37 days = +2.3% daily compounded
- Historical 30d avg: +0.1% daily
- Math: Need 23x higher daily return than recent average!
- Technical: No parabolic setup visible
- Sentiment: Slightly positive (irrelevant given math)
- → **Your rational assessment: 5-10%** (market MASSIVELY OVERVALUED!)

BE BRUTALLY HONEST. Disappoint the bulls if math doesn't support their dreams.

Your reputation depends on ACCURACY, not making traders feel good."""
        
        prompt = f"""{context}

Provide your analysis in JSON format:

{{
  "rational_probability": 0-100,
  "market_probability": {market_probability},
  "divergence": "overvalued" | "undervalued" | "fair",
  "divergence_magnitude": 0-100,
  "confidence": 0.0-1.0,
  "key_factors": [
    "Factor 1 that influenced your assessment",
    "Factor 2...",
    ...
  ],
  "reasoning": "Detailed explanation of your rational probability",
  "historical_precedents": "Similar past events and their outcomes",
  "technical_analysis": "Technical indicators analysis",
  "crowd_psychology": "Why the crowd might be wrong",
  "risk_factors": ["Risk 1", "Risk 2", ...],
  "trading_signal": "STRONG_BUY" | "BUY" | "HOLD" | "SELL" | "STRONG_SELL"
}}

Be analytical, data-driven, and contrarian when needed."""
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                temperature=0.3,  # Low temperature for rational analysis
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract JSON from response
            content = response.content[0].text
            
            # Try to parse JSON
            # Claude sometimes wraps JSON in markdown, so clean it
            if '```json' in content:
                content = content.split('```json')[1].split('```')[0]
            elif '```' in content:
                content = content.split('```')[1].split('```')[0]
            
            analysis = json.loads(content.strip())
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Claude analysis failed: {e}")
            # Return conservative estimate
            return {
                "rational_probability": 50.0,
                "market_probability": 50.0,
                "divergence": "unknown",
                "divergence_magnitude": 0,
                "confidence": 0.0,
                "key_factors": ["Analysis failed"],
                "reasoning": f"Error: {str(e)}",
                "trading_signal": "HOLD"
            }
    
    async def batch_analyze_markets(
        self,
        markets: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyze multiple markets to find best opportunities
        
        Returns markets sorted by divergence magnitude (biggest mispricing first)
        """
        tasks = []
        for market in markets:
            task = self.analyze_market(
                market_question=market.get('question', ''),
                market_probability=market.get('market_probability', 50),
                social_data=market.get('social_data', []),
                technical_data=market.get('technical_data'),
                historical_data=market.get('historical_data')
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results and sort by divergence
        valid_results = []
        for i, result in enumerate(results):
            if not isinstance(result, Exception):
                result['market_id'] = markets[i].get('id')
                result['market_question'] = markets[i].get('question')
                valid_results.append(result)
        
        # Sort by divergence magnitude (biggest opportunities first)
        valid_results.sort(
            key=lambda x: x.get('divergence_magnitude', 0),
            reverse=True
        )
        
        return valid_results
    
    async def get_vocal_summary(
        self,
        analysis: Dict[str, Any],
        market_question: str
    ) -> str:
        """
        Generate human-readable summary of the analysis
        """
        rational_prob = analysis.get('rational_probability', 50)
        market_prob = analysis.get('market_probability', 50)
        divergence = analysis.get('divergence', 'fair')
        magnitude = analysis.get('divergence_magnitude', 0)
        
        prompt = f"""Based on this analysis, write a clear 2-3 sentence executive summary:

Market: {market_question}
Rational Probability: {rational_prob}%
Market Price: {market_prob}%
Assessment: Market is {divergence} by {magnitude} points

Key Factors: {', '.join(analysis.get('key_factors', [])[:3])}

Write a professional summary explaining the mispricing opportunity."""
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.4,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return f"Market appears {divergence} by {magnitude} points. Rational assessment: {rational_prob}% vs market {market_prob}%."


# Example usage functions

async def find_overvalued_markets(
    markets: List[Dict[str, Any]],
    min_divergence: float = 15.0
) -> List[Dict[str, Any]]:
    """
    Find markets where crowd is overvaluing (good SHORT opportunities)
    
    Args:
        markets: List of market data
        min_divergence: Minimum divergence to consider (default 15%)
    
    Returns:
        List of overvalued markets sorted by opportunity size
    """
    agent = ClaudeOracleAgent()
    
    results = await agent.batch_analyze_markets(markets)
    
    # Filter overvalued markets
    overvalued = [
        r for r in results
        if r.get('divergence') == 'overvalued'
        and r.get('divergence_magnitude', 0) >= min_divergence
    ]
    
    return overvalued


async def find_undervalued_markets(
    markets: List[Dict[str, Any]],
    min_divergence: float = 15.0
) -> List[Dict[str, Any]]:
    """
    Find markets where crowd is undervaluing (good LONG opportunities)
    
    Args:
        markets: List of market data
        min_divergence: Minimum divergence to consider
    
    Returns:
        List of undervalued markets sorted by opportunity size
    """
    agent = ClaudeOracleAgent()
    
    results = await agent.batch_analyze_markets(markets)
    
    # Filter undervalued markets
    undervalued = [
        r for r in results
        if r.get('divergence') == 'undervalued'
        and r.get('divergence_magnitude', 0) >= min_divergence
    ]
    
    return undervalued
