"""
Local Sentiment Analysis Agent using Ollama/Llama
No API costs, runs locally on your machine
"""
from typing import List, Dict, Any
import httpx
import json

from app.agents.base_agent import BaseAgent
from app.models.schemas import SentimentAnalysis, SentimentScore
from app.config import settings


class LocalSentimentAgent(BaseAgent):
    """
    Sentiment agent using local LLM (Ollama/Llama)
    
    Benefits:
    - Free (no API costs)
    - Fast (if you have GPU)
    - Private (data stays local)
    - No rate limits
    
    Setup:
    1. Install Ollama: https://ollama.ai
    2. Download model: ollama pull llama3.1:8b
    3. Run: ollama serve
    """
    
    def __init__(self, model: str = "llama3.1:8b"):
        super().__init__("LocalSentimentAgent")
        self.model = model
        self.ollama_url = "http://localhost:11434"
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Sentiment keywords (same as OpenAI version)
        self.bullish_keywords = {
            'bullish', 'moon', 'pump', 'buy', 'long', 'hodl', 'up', 'gain',
            'profit', 'surge', 'rally', 'breakout', 'accumulate', 'soaring',
            'bullrun', 'parabolic', 'undervalued', 'gem', 'alpha'
        }
        
        self.bearish_keywords = {
            'bearish', 'dump', 'sell', 'short', 'down', 'loss', 'crash',
            'drop', 'fall', 'decline', 'plunge', 'overvalued', 'bubble',
            'scam', 'rug', 'dead', 'rekt', 'capitulation'
        }
    
    async def process(self, posts: List[Dict[str, Any]]) -> SentimentAnalysis:
        """
        Process posts using local LLM
        """
        if not posts:
            return self._create_empty_analysis()
        
        self.logger.info(f"Analyzing sentiment for {len(posts)} posts using {self.model}")
        
        # Fast keyword analysis (backup)
        keyword_sentiment = self._keyword_based_sentiment(posts)
        
        # Try local LLM analysis
        try:
            llm_sentiment = await self._local_llm_analysis(posts)
        except Exception as e:
            self.logger.warning(f"Local LLM failed, using keyword-only: {e}")
            llm_sentiment = keyword_sentiment
        
        # Combine scores
        combined_score = self._combine_sentiment_scores(
            keyword_sentiment,
            llm_sentiment
        )
        
        # Extract additional features
        key_phrases = self._extract_key_phrases(posts)
        trending = self._extract_trending_topics(posts)
        momentum = self._calculate_momentum(posts)
        
        return SentimentAnalysis(
            sentiment_score=combined_score,
            key_phrases=key_phrases,
            trending_topics=trending,
            sentiment_momentum=momentum
        )
    
    async def _local_llm_analysis(self, posts: List[Dict[str, Any]]) -> SentimentScore:
        """
        Analyze sentiment using local Ollama/Llama model
        """
        # Take top posts by engagement
        top_posts = sorted(
            posts,
            key=lambda x: x.get('engagement', 0),
            reverse=True
        )[:30]  # Analyze top 30
        
        posts_text = "\n\n".join([
            f"Post {i+1}: {post.get('text', '')}"
            for i, post in enumerate(top_posts)
        ])
        
        prompt = f"""Analyze the sentiment of these cryptocurrency social media posts.

Posts:
{posts_text}

Provide sentiment analysis with scores from 0.0 to 1.0:
- bullish: How bullish (positive, optimistic)
- bearish: How bearish (negative, pessimistic)
- neutral: How neutral
- overall: Overall sentiment from -1.0 (very bearish) to 1.0 (very bullish)
- confidence: Your confidence in this analysis (0.0-1.0)

Consider:
- Explicit bullish/bearish language
- Implicit optimism/pessimism
- Sarcasm and irony
- Context and reasoning

Respond ONLY with valid JSON in this exact format:
{{"bullish": 0.0, "bearish": 0.0, "neutral": 0.0, "overall": 0.0, "confidence": 0.0}}"""
        
        try:
            response = await self.client.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama API error: {response.status_code}")
            
            result = response.json()
            response_text = result.get('response', '{}')
            
            # Parse JSON response
            sentiment_data = json.loads(response_text)
            
            return SentimentScore(
                bullish=sentiment_data.get('bullish', 0.5),
                bearish=sentiment_data.get('bearish', 0.5),
                neutral=sentiment_data.get('neutral', 0.0),
                overall=sentiment_data.get('overall', 0.0),
                confidence=sentiment_data.get('confidence', 0.5)
            )
            
        except Exception as e:
            self.logger.error(f"Local LLM analysis failed: {e}")
            raise
    
    def _keyword_based_sentiment(self, posts: List[Dict[str, Any]]) -> SentimentScore:
        """Fast keyword-based sentiment analysis (same as OpenAI version)"""
        bullish_count = 0
        bearish_count = 0
        total_weight = 0
        
        for post in posts:
            text = post.get('text', '').lower()
            engagement = post.get('engagement', 1)
            weight = max(1, engagement / 10)
            
            bullish_matches = sum(1 for word in self.bullish_keywords if word in text)
            bearish_matches = sum(1 for word in self.bearish_keywords if word in text)
            
            bullish_count += bullish_matches * weight
            bearish_count += bearish_matches * weight
            total_weight += weight
        
        if total_weight == 0:
            return SentimentScore(neutral=1.0, confidence=0.0)
        
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return SentimentScore(neutral=1.0, confidence=0.3)
        
        bullish_ratio = bullish_count / (bullish_count + bearish_count)
        bearish_ratio = bearish_count / (bullish_count + bearish_count)
        neutral_ratio = 1.0 - (bullish_ratio + bearish_ratio)
        
        overall = bullish_ratio - bearish_ratio
        confidence = min(0.8, total_signals / len(posts))
        
        return SentimentScore(
            bullish=bullish_ratio,
            bearish=bearish_ratio,
            neutral=max(0.0, neutral_ratio),
            overall=overall,
            confidence=confidence
        )
    
    def _combine_sentiment_scores(
        self,
        keyword_score: SentimentScore,
        llm_score: SentimentScore
    ) -> SentimentScore:
        """Combine keyword and LLM sentiment scores"""
        llm_weight = llm_score.confidence
        keyword_weight = 1.0 - llm_weight
        
        return SentimentScore(
            bullish=(keyword_score.bullish * keyword_weight + llm_score.bullish * llm_weight),
            bearish=(keyword_score.bearish * keyword_weight + llm_score.bearish * llm_weight),
            neutral=(keyword_score.neutral * keyword_weight + llm_score.neutral * llm_weight),
            overall=(keyword_score.overall * keyword_weight + llm_score.overall * llm_weight),
            confidence=(keyword_score.confidence + llm_score.confidence) / 2
        )
    
    def _extract_key_phrases(self, posts: List[Dict[str, Any]]) -> List[str]:
        """Extract key phrases (hashtags, cashtags)"""
        import re
        from collections import Counter
        
        all_text = " ".join([post.get('text', '') for post in posts])
        
        hashtags = re.findall(r'#\w+', all_text)
        cashtags = re.findall(r'\$\w+', all_text)
        
        counter = Counter(hashtags + cashtags)
        return [phrase for phrase, _ in counter.most_common(10)]
    
    def _extract_trending_topics(self, posts: List[Dict[str, Any]]) -> List[str]:
        """Extract trending topics"""
        import re
        from collections import Counter
        
        all_text = " ".join([post.get('text', '').lower() for post in posts])
        
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\b\w{4,}\b', all_text)
        words = [w for w in words if w not in stop_words]
        
        counter = Counter(words)
        return [word for word, _ in counter.most_common(5)]
    
    def _calculate_momentum(self, posts: List[Dict[str, Any]]) -> float:
        """Calculate sentiment momentum"""
        if len(posts) < 10:
            return 0.0
        
        sorted_posts = sorted(posts, key=lambda x: x.get('created_at', 0))
        
        mid = len(sorted_posts) // 2
        first_half = sorted_posts[:mid]
        second_half = sorted_posts[mid:]
        
        first_sentiment = self._keyword_based_sentiment(first_half).overall
        second_sentiment = self._keyword_based_sentiment(second_half).overall
        
        momentum = second_sentiment - first_sentiment
        return max(-1.0, min(1.0, momentum))
    
    def _create_empty_analysis(self) -> SentimentAnalysis:
        """Create empty analysis when no data"""
        return SentimentAnalysis(
            sentiment_score=SentimentScore(neutral=1.0, confidence=0.0),
            key_phrases=[],
            trending_topics=[],
            sentiment_momentum=0.0
        )
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Convenience function to check if Ollama is available
async def check_ollama_available() -> bool:
    """Check if Ollama server is running"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:11434/api/tags")
            return response.status_code == 200
    except:
        return False
