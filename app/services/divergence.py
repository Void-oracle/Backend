"""
Divergence Service
Calculates and analyzes divergence between AI predictions and market probabilities
"""
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta

from app.models.schemas import (
    OraclePredictionResponse,
    MarketData
)


logger = logging.getLogger(__name__)


class DivergenceService:
    """Service for calculating and analyzing divergence indices"""
    
    def __init__(self):
        self.divergence_history: List[Dict[str, Any]] = []
        self.high_divergence_threshold = 20.0
    
    def calculate_divergence(
        self,
        ai_score: float,
        market_score: float
    ) -> float:
        """
        Calculate divergence index using the formula: D = |P_AI - P_Market|
        
        Args:
            ai_score: AI probability (0-100)
            market_score: Market probability (0-100)
            
        Returns:
            Divergence index (0-100)
        """
        divergence = abs(ai_score - market_score)
        return divergence
    
    def analyze_divergence(
        self,
        divergence_index: float,
        ai_score: float,
        market_score: float
    ) -> Dict[str, Any]:
        """
        Analyze divergence and provide insights
        
        Args:
            divergence_index: Calculated divergence
            ai_score: AI probability
            market_score: Market probability
            
        Returns:
            Analysis dict with insights
        """
        # Determine divergence level
        if divergence_index > 30:
            level = "extreme"
            opportunity = "very_high"
        elif divergence_index > 20:
            level = "high"
            opportunity = "high"
        elif divergence_index > 10:
            level = "moderate"
            opportunity = "moderate"
        else:
            level = "low"
            opportunity = "low"
        
        # Determine market mispricing direction
        if ai_score > market_score:
            direction = "underpriced"
            bias = "bearish"
            recommendation = "Market appears bearish compared to AI sentiment"
        elif ai_score < market_score:
            direction = "overpriced"
            bias = "bullish"
            recommendation = "Market appears bullish compared to AI sentiment"
        else:
            direction = "aligned"
            bias = "neutral"
            recommendation = "Market and AI are aligned"
        
        # Calculate potential alpha (trading opportunity)
        potential_alpha = divergence_index / 100.0  # Normalize to 0-1
        
        return {
            "divergence_level": level,
            "opportunity_rating": opportunity,
            "market_direction": direction,
            "market_bias": bias,
            "recommendation": recommendation,
            "potential_alpha": potential_alpha,
            "confidence_adjustment": self._calculate_confidence_adjustment(divergence_index)
        }
    
    def _calculate_confidence_adjustment(self, divergence_index: float) -> float:
        """
        Adjust confidence based on divergence
        
        Higher divergence might indicate:
        - Strong signal (new information)
        - Or data quality issues
        """
        # Moderate divergence increases confidence (signal)
        # Extreme divergence slightly decreases confidence (might be data issue)
        if 15 <= divergence_index <= 35:
            return 1.1  # Boost confidence
        elif divergence_index > 50:
            return 0.9  # Slight reduction
        else:
            return 1.0  # No adjustment
    
    def track_divergence(
        self,
        market_id: str,
        prediction: OraclePredictionResponse
    ) -> None:
        """
        Track divergence over time for historical analysis
        
        Args:
            market_id: Market identifier
            prediction: Oracle prediction response
        """
        record = {
            "market_id": market_id,
            "timestamp": datetime.utcnow(),
            "ai_score": prediction.ai_score,
            "market_score": prediction.market_score,
            "divergence_index": prediction.divergence_index,
            "confidence": prediction.confidence
        }
        
        self.divergence_history.append(record)
        
        # Keep only last 30 days
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        self.divergence_history = [
            r for r in self.divergence_history
            if r["timestamp"] > cutoff_date
        ]
    
    def get_divergence_trends(
        self,
        market_id: Optional[str] = None,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get divergence trends over time
        
        Args:
            market_id: Optional market filter
            days: Number of days to analyze
            
        Returns:
            Trend analysis
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        # Filter records
        records = [
            r for r in self.divergence_history
            if r["timestamp"] > cutoff_date
        ]
        
        if market_id:
            records = [r for r in records if r["market_id"] == market_id]
        
        if not records:
            return {
                "trend": "no_data",
                "avg_divergence": 0.0,
                "max_divergence": 0.0,
                "divergence_increasing": False
            }
        
        # Calculate statistics
        divergences = [r["divergence_index"] for r in records]
        avg_divergence = sum(divergences) / len(divergences)
        max_divergence = max(divergences)
        min_divergence = min(divergences)
        
        # Calculate trend (simple linear)
        if len(records) >= 2:
            recent_avg = sum(divergences[-len(divergences)//2:]) / max(1, len(divergences)//2)
            older_avg = sum(divergences[:len(divergences)//2]) / max(1, len(divergences)//2)
            divergence_increasing = recent_avg > older_avg
        else:
            divergence_increasing = False
        
        return {
            "trend": "increasing" if divergence_increasing else "decreasing",
            "avg_divergence": avg_divergence,
            "max_divergence": max_divergence,
            "min_divergence": min_divergence,
            "divergence_increasing": divergence_increasing,
            "sample_size": len(records)
        }
    
    def identify_anomalies(
        self,
        prediction: OraclePredictionResponse
    ) -> List[str]:
        """
        Identify potential anomalies in the prediction
        
        Args:
            prediction: Oracle prediction to analyze
            
        Returns:
            List of anomaly descriptions
        """
        anomalies = []
        
        # Check for extreme divergence
        if prediction.divergence_index > 50:
            anomalies.append(
                f"Extreme divergence detected ({prediction.divergence_index:.1f}%)"
            )
        
        # Check for low confidence with high divergence
        if prediction.divergence_index > 30 and prediction.confidence < 0.5:
            anomalies.append(
                "High divergence but low confidence - data quality may be poor"
            )
        
        # Check for high bot activity
        if prediction.bot_detection and prediction.bot_detection.bot_probability > 0.6:
            anomalies.append(
                f"High bot activity detected ({prediction.bot_detection.bot_probability:.1%})"
            )
        
        # Check for insufficient data
        if prediction.data_sources.twitter_posts < 20:
            anomalies.append(
                f"Low data volume ({prediction.data_sources.twitter_posts} posts)"
            )
        
        # Check for extreme sentiment
        if prediction.sentiment_analysis:
            sentiment = prediction.sentiment_analysis.sentiment_score.overall
            if abs(sentiment) > 0.9:
                anomalies.append(
                    f"Extreme sentiment detected ({sentiment:.2f})"
                )
        
        return anomalies
    
    def generate_trading_signal(
        self,
        prediction: OraclePredictionResponse
    ) -> Dict[str, Any]:
        """
        Generate a trading signal based on divergence
        
        Args:
            prediction: Oracle prediction
            
        Returns:
            Trading signal with action and strength
        """
        divergence = prediction.divergence_index
        confidence = prediction.confidence
        
        # Determine action
        if prediction.ai_score > prediction.market_score:
            action = "BUY"  # Market underpricing
        elif prediction.ai_score < prediction.market_score:
            action = "SELL"  # Market overpricing
        else:
            action = "HOLD"
        
        # Calculate signal strength (0-100)
        strength = min(100, divergence * confidence * 2)
        
        # Determine signal quality
        if strength > 70 and confidence > 0.7:
            quality = "strong"
        elif strength > 40 and confidence > 0.5:
            quality = "moderate"
        else:
            quality = "weak"
        
        return {
            "action": action,
            "strength": strength,
            "quality": quality,
            "confidence": confidence,
            "expected_edge": divergence * confidence / 100.0,
            "risk_level": "high" if confidence < 0.5 else "moderate" if confidence < 0.7 else "low"
        }
