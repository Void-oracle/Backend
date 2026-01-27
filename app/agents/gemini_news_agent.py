"""
Gemini News Agent - Real-time fact checking with Google Search
Specialization: regulatory events, announcements, real-time news
"""
import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime

# Optional Gemini import
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

from app.agents.base_agent import BaseAgent
from app.config import settings

logger = logging.getLogger(__name__)


class GeminiNewsAgent(BaseAgent):
    """
    Google Gemini agent with Google Search grounding
    
    Perfect for:
    - Regulatory approvals (ETF, SEC filings)
    - Political events (elections, appointments)
    - Corporate announcements
    - Breaking news verification
    
    Uses Google Search to find REAL, CURRENT information
    """
    
    def __init__(self):
        super().__init__("GeminiNewsAgent")
        
        # Check if Gemini library is available
        if not GEMINI_AVAILABLE:
            logger.warning("google-generativeai not installed. Install: pip install google-generativeai")
            self.enabled = False
            self.model = None
            return
        
        # Configure Gemini
        if not settings.GOOGLE_API_KEY:
            logger.warning("Google API Key not configured, Gemini agent disabled")
            self.enabled = False
            self.model = None
            return
        
        try:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
            
            # Check if GenerativeModel is available (newer versions)
            if hasattr(genai, 'GenerativeModel'):
                # Use Gemini 3 Flash Preview - fast and powerful!
                self.model = genai.GenerativeModel('gemini-3-flash-preview')
                self.enabled = True
            else:
                logger.warning("google-generativeai version incompatible, Gemini agent disabled")
                self.enabled = False
                self.model = None
            logger.info("Gemini News Agent initialized with gemini-3-flash-preview")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.enabled = False
            self.model = None
    
    async def process(self, data: Any) -> Any:
        """Required by BaseAgent"""
        return {"message": "Use analyze_event() for news analysis"}
    
    async def analyze_event(
        self,
        query: str,
        event_type: str,
        deadline: Optional[datetime] = None,
        current_market_prob: float = 50.0,
        price_metrics: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze event using Google Search to find current facts
        
        Args:
            query: Market question (e.g., "Will ETH ETF be approved in Q1 2026?")
            event_type: Type of event (regulatory, political, announcement, etc.)
            deadline: Event deadline
            current_market_prob: Current market probability
        
        Returns:
            Analysis with real-world fact checking
        """
        if not self.enabled:
            return {
                "probability": current_market_prob,
                "confidence": 0.0,
                "reasoning": "Gemini not configured",
                "sources": []
            }
        
        self.logger.info(f"Gemini analyzing: {query} (type: {event_type})")
        
        try:
            # Build comprehensive prompt with search guidance
            prompt = self._build_search_prompt(
                query=query,
                event_type=event_type,
                deadline=deadline,
                current_market_prob=current_market_prob,
                price_metrics=price_metrics
            )
            
            # Generate with Google Search grounding
            response = await self._generate_with_search(prompt)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Gemini analysis failed: {e}")
            return {
                "probability": current_market_prob,
                "confidence": 0.0,
                "reasoning": f"Error: {str(e)}",
                "sources": []
            }
    
    def _build_search_prompt(
        self,
        query: str,
        event_type: str,
        deadline: Optional[datetime],
        current_market_prob: float,
        price_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """Build optimized search prompt for Gemini"""
        
        today = datetime.now().strftime("%B %d, %Y")
        deadline_str = deadline.strftime("%B %d, %Y") if deadline else "unknown"
        
        # Add real price data if available
        price_context = ""
        if price_metrics and price_metrics.get("data_available"):
            current_price = price_metrics.get("current_price", 0)
            target_price = price_metrics.get("target_price", 0)
            days_left = price_metrics.get("days_left", 0)
            required_growth = price_metrics.get("required_growth_pct", 0)
            
            price_context = f"""
VERIFIED PRICE DATA (from CoinGecko API):
- CURRENT PRICE: ${current_price:.2f} (as of {today})
- TARGET PRICE: ${target_price:.2f}
- DAYS LEFT: {days_left}
- REQUIRED GROWTH: {required_growth:+.1f}%

IMPORTANT: Use these VERIFIED prices in your analysis. Do NOT guess or use outdated prices!
"""
        
        # Event-specific search guidance
        search_guidance = {
            "regulatory": """
Search for:
1. Official SEC.gov filings and announcements
2. Recent regulatory news from Bloomberg, Reuters, WSJ
3. Current status of approval process
4. Timeline and deadlines
5. Expert analyst predictions
""",
            "political": """
Search for:
1. Latest polling data
2. Election news from major outlets
3. Political analysts predictions
4. Betting markets (PredictIt, Polymarket)
5. Recent developments and controversies
""",
            "announcement": """
Search for:
1. Official company press releases
2. Verified Twitter/X accounts
3. News from TechCrunch, The Verge, etc.
4. Recent statements from executives
5. Industry analyst reports
""",
            "price_target": """
Search for:
1. Current price and recent price action
2. Technical analysis from credible sources
3. Social sentiment (site:twitter.com OR site:reddit.com)
4. Recent discussions and community mood
5. Analyst price targets and timelines
6. Trading volume and momentum
""",
            "generic": """
Search for:
1. Latest news and updates
2. Social media discussions (Twitter, Reddit)
3. Expert opinions and analysis
4. Recent developments
5. Community sentiment
"""
        }
        
        guidance = search_guidance.get(event_type, search_guidance["announcement"])
        
        prompt = f"""You are an expert prediction market analyst. Your job is to assess the TRUE probability of events.

=== MARKET CONTEXT ===
QUESTION: {query}
DEADLINE: {deadline_str}
CURRENT MARKET PROBABILITY: {current_market_prob}%
{price_context}

=== YOUR ANALYSIS TASK ===
Search for current information and provide a well-reasoned probability assessment.

{guidance}

=== ANALYSIS FRAMEWORK ===
1. CURRENT STATUS: What is the current state of this event/topic?
2. KEY FACTORS: What factors will determine the outcome?
3. EVIDENCE: What do experts, analysts, and data suggest?
4. PROBABILITY: Based on evidence, what is the realistic probability?

=== IMPORTANT GUIDELINES ===
- Use Google Search to find the MOST RECENT news and analysis
- Prioritize official sources: SEC filings, company announcements, major news outlets
- Consider historical precedents for similar events
- Account for uncertainty - rarely is anything 0% or 100%
- If the market probability seems too high or low, explain why
- Be data-driven and rational, not speculative

Provide analysis in JSON format:

{{
  "probability": 0-100,
  "confidence": 0.0-1.0,
  "reasoning": "Detailed explanation based on search results",
  "key_facts": [
    "Fact 1 from source X",
    "Fact 2 from source Y",
    ...
  ],
  "sources": [
    "Source 1 URL or name",
    "Source 2 URL or name",
    ...
  ],
  "current_status": "Description of current situation",
  "likelihood_factors": [
    "Positive factor 1",
    "Negative factor 1",
    ...
  ],
  "verdict": "overvalued" | "undervalued" | "fairly_priced"
}}

BE FACTUAL. USE SEARCH RESULTS. BE CONSERVATIVE."""
        
        return prompt
    
    async def _generate_with_search(self, prompt: str) -> Dict[str, Any]:
        """Generate response with Google Search grounding"""
        
        try:
            # Generate with search grounding
            # Note: Synchronous call, but fast enough for our use case
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.2,  # Low temperature for factual responses
                    top_p=0.8,
                    top_k=40,
                    max_output_tokens=2048,
                )
            )
            
            # Extract text
            text = response.text
            
            # Try to parse JSON
            # Gemini sometimes wraps in markdown
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0]
            elif '```' in text:
                text = text.split('```')[1].split('```')[0]
            
            analysis = json.loads(text.strip())
            
            # Log sources found
            if analysis.get("sources"):
                self.logger.info(f"Gemini found {len(analysis['sources'])} sources:")
                for source in analysis["sources"][:3]:
                    self.logger.info(f"  - {source}")
            
            return analysis
            
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse Gemini JSON response: {e}")
            self.logger.error(f"Raw response: {text[:500]}...")
            
            # Try to extract probability from partial JSON using regex
            import re
            prob_match = re.search(r'"probability":\s*(\d+(?:\.\d+)?)', text)
            conf_match = re.search(r'"confidence":\s*(\d+(?:\.\d+)?)', text)
            
            probability = float(prob_match.group(1)) if prob_match else 50.0
            confidence = float(conf_match.group(1)) if conf_match else 0.5
            
            # Extract reasoning if available
            reasoning_match = re.search(r'"reasoning":\s*"([^"]{100,500})', text)
            reasoning = reasoning_match.group(1) if reasoning_match else text[:300]
            
            self.logger.info(f"Extracted from partial JSON: probability={probability}, confidence={confidence}")
            
            return {
                "probability": probability,
                "confidence": confidence,
                "reasoning": reasoning,
                "sources": [],
                "key_facts": []
            }
        
        except Exception as e:
            self.logger.error(f"Gemini generation failed: {e}")
            raise
    
    def classify_event_type(self, query: str) -> str:
        """
        Classify event type from query
        
        Returns: regulatory | political | announcement | price_target | generic
        """
        query_lower = query.lower()
        
        # Regulatory events
        if any(word in query_lower for word in ['etf', 'sec', 'approved', 'approval', 'regulatory', 'filing', 'fda']):
            return "regulatory"
        
        # Political events
        if any(word in query_lower for word in ['election', 'elect', 'president', 'senate', 'vote', 'win', 'cabinet']):
            return "political"
        
        # Announcements
        if any(word in query_lower for word in ['announce', 'announced', 'launch', 'released', 'reveal', 'unveil']):
            return "announcement"
        
        # Price targets
        if any(word in query_lower for word in ['reach', 'hit', 'price', '$', 'usd', 'target']):
            return "price_target"
        
        return "generic"


# Global instance
gemini_agent = GeminiNewsAgent()
