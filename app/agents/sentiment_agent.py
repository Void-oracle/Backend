"""
Sentiment Analysis Agent
Analyzes social media sentiment using NLP and AI models
"""
from typing import List, Dict, Any, Optional
import asyncio
import re
from collections import Counter

from openai import AsyncOpenAI
from anthropic import AsyncAnthropic

from app.agents.base_agent import BaseAgent
from app.models.schemas import SentimentAnalysis, SentimentScore
from app.config import settings


class SentimentAgent(BaseAgent):
    """Agent for analyzing sentiment from social media posts"""
    
    def __init__(self):
        super().__init__("SentimentAgent")
        
        # Initialize AI client based on provider
        if settings.AI_PROVIDER == "claude" and settings.ANTHROPIC_API_KEY:
            self.client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.model = settings.ANTHROPIC_MODEL
            self.provider = "claude"
        else:
            self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            self.model = settings.OPENAI_MODEL
            self.provider = "openai"
        
        # Sentiment keywords
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
        Process social media posts and return sentiment analysis
        
        Args:
            posts: List of social media posts with text content
            
        Returns:
            SentimentAnalysis with detailed sentiment scores
        """
        if not posts:
            return self._create_empty_analysis()
        
        self.logger.info(f"Analyzing sentiment for {len(posts)} posts")
        
        # Run analyses in parallel
        keyword_sentiment = self._keyword_based_sentiment(posts)
        ai_sentiment = await self._ai_sentiment_analysis(posts)
        key_phrases = self._extract_key_phrases(posts)
        trending = self._extract_trending_topics(posts)
        momentum = self._calculate_momentum(posts)
        
        # Combine scores (weighted average)
        combined_score = self._combine_sentiment_scores(
            keyword_sentiment, 
            ai_sentiment
        )
        
        return SentimentAnalysis(
            sentiment_score=combined_score,
            key_phrases=key_phrases,
            trending_topics=trending,
            sentiment_momentum=momentum
        )
    
    def _keyword_based_sentiment(self, posts: List[Dict[str, Any]]) -> SentimentScore:
        """Fast keyword-based sentiment analysis"""
        bullish_count = 0
        bearish_count = 0
        total_weight = 0
        
        for post in posts:
            text = post.get('text', '').lower()
            engagement = post.get('engagement', 1)  # likes + retweets + replies
            weight = max(1, engagement / 10)  # Weight by engagement
            
            # Count keyword occurrences
            bullish_matches = sum(1 for word in self.bullish_keywords if word in text)
            bearish_matches = sum(1 for word in self.bearish_keywords if word in text)
            
            bullish_count += bullish_matches * weight
            bearish_count += bearish_matches * weight
            total_weight += weight
        
        if total_weight == 0:
            return SentimentScore(neutral=1.0, confidence=0.0)
        
        # Normalize scores
        total_signals = bullish_count + bearish_count
        if total_signals == 0:
            return SentimentScore(neutral=1.0, confidence=0.3)
        
        bullish_ratio = bullish_count / (bullish_count + bearish_count)
        bearish_ratio = bearish_count / (bullish_count + bearish_count)
        neutral_ratio = 1.0 - (bullish_ratio + bearish_ratio)
        
        # Overall sentiment: -1 (bearish) to +1 (bullish)
        overall = (bullish_ratio - bearish_ratio)
        confidence = min(0.8, total_signals / len(posts))
        
        return SentimentScore(
            bullish=bullish_ratio,
            bearish=bearish_ratio,
            neutral=max(0.0, neutral_ratio),
            overall=overall,
            confidence=confidence
        )
    
    async def _ai_sentiment_analysis(self, posts: List[Dict[str, Any]]) -> SentimentScore:
        """Deep AI-powered sentiment analysis using GPT-4"""
        # Take top posts by engagement for detailed analysis
        top_posts = sorted(
            posts, 
            key=lambda x: x.get('engagement', 0), 
            reverse=True
        )[:50]
        
        posts_text = "\n\n".join([
            f"Post {i+1}: {post.get('text', '')}"
            for i, post in enumerate(top_posts)
        ])
        
        prompt = f"""Analyze the sentiment of these social media posts about a cryptocurrency or market event.

Posts:
{posts_text}

Provide:
1. Bullish sentiment score (0.0 to 1.0)
2. Bearish sentiment score (0.0 to 1.0)
3. Neutral sentiment score (0.0 to 1.0)
4. Overall sentiment (-1.0 to 1.0, where -1 is very bearish, 0 is neutral, 1 is very bullish)
5. Confidence in your analysis (0.0 to 1.0)

Consider:
- Explicit sentiment (bullish/bearish language)
- Implicit sentiment (optimism, fear, uncertainty)
- Sarcasm and irony
- Context and reasoning provided

Respond in JSON format:
{{"bullish": 0.0, "bearish": 0.0, "neutral": 0.0, "overall": 0.0, "confidence": 0.0}}
"""
        
        try:
            import json
            
            if self.provider == "claude":
                response = await self.client.messages.create(
                    model=self.model,
                    max_tokens=200,
                    temperature=0.3,
                    system="You are an expert sentiment analyst for cryptocurrency markets. Always respond with valid JSON.",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                content = response.content[0].text
                # Claude sometimes wraps JSON in markdown
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0]
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0]
                result = json.loads(content.strip())
            else:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert sentiment analyst for cryptocurrency markets."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=200,
                    response_format={"type": "json_object"}
                )
                result = json.loads(response.choices[0].message.content)
            
            return SentimentScore(
                bullish=result.get('bullish', 0.5),
                bearish=result.get('bearish', 0.5),
                neutral=result.get('neutral', 0.0),
                overall=result.get('overall', 0.0),
                confidence=result.get('confidence', 0.5)
            )
            
        except Exception as e:
            self.logger.error(f"AI sentiment analysis failed: {e}")
            # Fallback to neutral
            return SentimentScore(neutral=1.0, confidence=0.0)
    
    def _combine_sentiment_scores(
        self, 
        keyword_score: SentimentScore, 
        ai_score: SentimentScore
    ) -> SentimentScore:
        """Combine keyword and AI sentiment scores with weighted average"""
        # Weight AI analysis more heavily if it's confident
        ai_weight = ai_score.confidence
        keyword_weight = 1.0 - ai_weight
        
        return SentimentScore(
            bullish=(keyword_score.bullish * keyword_weight + ai_score.bullish * ai_weight),
            bearish=(keyword_score.bearish * keyword_weight + ai_score.bearish * ai_weight),
            neutral=(keyword_score.neutral * keyword_weight + ai_score.neutral * ai_weight),
            overall=(keyword_score.overall * keyword_weight + ai_score.overall * ai_weight),
            confidence=(keyword_score.confidence + ai_score.confidence) / 2
        )
    
    def _extract_key_phrases(self, posts: List[Dict[str, Any]]) -> List[str]:
        """Extract key phrases and important mentions"""
        all_text = " ".join([post.get('text', '') for post in posts])
        
        # Extract hashtags
        hashtags = re.findall(r'#\w+', all_text)
        # Extract cashtags
        cashtags = re.findall(r'\$\w+', all_text)
        # Extract mentions
        mentions = re.findall(r'@\w+', all_text)
        
        # Count frequencies
        counter = Counter(hashtags + cashtags)
        
        # Return top 10 most common
        top_phrases = [phrase for phrase, _ in counter.most_common(10)]
        return top_phrases
    
    def _extract_trending_topics(self, posts: List[Dict[str, Any]]) -> List[str]:
        """Extract trending topics from posts"""
        # Simple implementation - extract common words
        all_text = " ".join([post.get('text', '').lower() for post in posts])
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        words = re.findall(r'\b\w{4,}\b', all_text)
        words = [w for w in words if w not in stop_words]
        
        counter = Counter(words)
        trending = [word for word, _ in counter.most_common(5)]
        
        return trending
    
    def _calculate_momentum(self, posts: List[Dict[str, Any]]) -> float:
        """Calculate sentiment momentum (rate of change)"""
        if len(posts) < 10:
            return 0.0
        
        # Sort by timestamp
        sorted_posts = sorted(posts, key=lambda x: x.get('created_at', 0))
        
        # Split into two halves
        mid = len(sorted_posts) // 2
        first_half = sorted_posts[:mid]
        second_half = sorted_posts[mid:]
        
        # Calculate sentiment for each half
        first_sentiment = self._keyword_based_sentiment(first_half).overall
        second_sentiment = self._keyword_based_sentiment(second_half).overall
        
        # Momentum is the change
        momentum = second_sentiment - first_sentiment
        
        return max(-1.0, min(1.0, momentum))
    
    def _create_empty_analysis(self) -> SentimentAnalysis:
        """Create empty analysis when no data available"""
        return SentimentAnalysis(
            sentiment_score=SentimentScore(neutral=1.0, confidence=0.0),
            key_phrases=[],
            trending_topics=[],
            sentiment_momentum=0.0
        )
