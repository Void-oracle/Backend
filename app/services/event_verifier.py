"""
Event Verification System for VOID Oracle

This module determines if prediction market conditions have been met.
For example:
- "Bitcoin reached $150,000" - check if BTC price >= $150,000
- "Trump tweeted the n-word" - search for specific content
- "ETF approved" - search for official announcements
"""
import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum

from app.models.schemas import Post

logger = logging.getLogger(__name__)


class EventStatus(Enum):
    """Status of event verification"""
    NOT_YET = "not_yet"  # Event hasn't occurred yet
    OCCURRED = "occurred"  # Event definitely happened
    FAILED = "failed"  # Event deadline passed, didn't occur
    UNCERTAIN = "uncertain"  # Unclear if event occurred


class EventVerifier:
    """
    Verifies if prediction market conditions have been met
    
    The Oracle's job is to:
    1. Monitor relevant data sources
    2. Detect when events occur
    3. Provide confidence score on event occurrence
    4. Make final YES/NO decision at deadline
    """
    
    def __init__(self):
        self.verified_events = {}  # Cache verified events
        logger.info("Event Verifier initialized")
    
    async def verify_event(
        self,
        market_id: str,
        query: str,
        posts: List[Post],
        deadline: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Verify if market condition has been met
        
        Args:
            market_id: Unique market identifier
            query: Market question (e.g., "Will BTC reach $150k by June 2026?")
            posts: Recent social media posts
            deadline: Market deadline (if any)
        
        Returns:
            Verification result with confidence score
        """
        # Parse query to understand what we're looking for
        event_type = self._classify_event_type(query)
        
        logger.info(f"Verifying event for {market_id}")
        logger.info(f"   Event type: {event_type}")
        logger.info(f"   Query: {query}")
        
        # Route to appropriate verification method
        if event_type == "content_search":
            result = await self._verify_content(query, posts)
        elif event_type == "price_target":
            result = await self._verify_price_target(query, posts)
        elif event_type == "announcement":
            result = await self._verify_announcement(query, posts)
        elif event_type == "political_outcome":
            result = await self._verify_political_outcome(query, posts)
        elif event_type == "sports_outcome":
            result = await self._verify_sports_outcome(query, posts)
        elif event_type == "milestone":
            result = await self._verify_milestone(query, posts)
        elif event_type == "time_based":
            result = await self._verify_time_based(query, posts, deadline)
        else:
            result = await self._verify_generic(query, posts)
        
        # Check if deadline passed
        if deadline and datetime.utcnow() > deadline:
            if result["status"] != EventStatus.OCCURRED:
                result["status"] = EventStatus.FAILED
                result["reason"] = "Deadline passed, event did not occur"
        
        # Cache result if event occurred
        if result["status"] == EventStatus.OCCURRED:
            self.verified_events[market_id] = {
                "timestamp": datetime.utcnow(),
                "result": result
            }
        
        logger.info(f"   Status: {result['status'].value}")
        logger.info(f"   Confidence: {result['confidence']:.1%}")
        logger.info(f"   Reason: {result['reason']}")
        
        return result
    
    def _classify_event_type(self, query: str) -> str:
        """Classify the type of event to verify"""
        query_lower = query.lower()
        
        # Content search (specific tweet/statement) - HIGHEST PRIORITY
        # Examples: "Trump says X", "Elon tweets Y", "Someone posts Z"
        if any(word in query_lower for word in ['say', 'said', 'tweet', 'tweeted', 'post', 'posted', 'publish', 'write', 'claim', 'stated']):
            return "content_search"
        
        # Price targets (crypto/stocks)
        if any(word in query_lower for word in ['reach', 'hit', 'price', '$', 'usd', 'above', 'below', 'exceed']):
            return "price_target"
        
        # Official announcements/approvals (regulatory, company news)
        if any(word in query_lower for word in ['approved', 'approve', 'announced', 'announce', 'launched', 'launch', 'released', 'release', 'confirmed', 'confirm']):
            return "announcement"
        
        # Elections/Political outcomes
        if any(word in query_lower for word in ['win', 'election', 'elect', 'vote', 'president', 'senate', 'congress']):
            return "political_outcome"
        
        # Sports outcomes
        if any(word in query_lower for word in ['championship', 'win', 'lose', 'final', 'playoff', 'tournament', 'match', 'game']):
            return "sports_outcome"
        
        # Technical milestones (space, tech, science)
        if any(word in query_lower for word in ['land', 'launch', 'achieve', 'complete', 'breakthrough', 'discover']):
            return "milestone"
        
        # Time-based events
        if any(word in query_lower for word in ['by', 'before', 'within', 'during', 'end of']):
            return "time_based"
        
        return "generic"
    
    async def _verify_price_target(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """Verify if price target was reached"""
        # Extract price target from query
        price_match = re.search(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', query)
        if not price_match:
            return {
                "status": EventStatus.UNCERTAIN,
                "confidence": 0.0,
                "reason": "Could not parse price target from query"
            }
        
        target_price = float(price_match.group(1).replace(',', ''))
        
        # Search posts for price mentions
        price_confirmations = []
        for post in posts:
            # Look for price mentions in format "$XXX" or "XXX USD"
            prices = re.findall(r'\$(\d+(?:,\d+)*(?:\.\d+)?)', post.text)
            for price_str in prices:
                price = float(price_str.replace(',', ''))
                if price >= target_price * 0.95:  # Within 5% of target
                    price_confirmations.append({
                        "post": post.text,
                        "price": price,
                        "author": post.author,
                        "engagement": post.engagement
                    })
        
        if not price_confirmations:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.1,
                "reason": f"No confirmation of ${target_price:,.0f} price target",
                "target_price": target_price
            }
        
        # Calculate confidence based on evidence
        high_engagement_count = sum(1 for c in price_confirmations if c["engagement"] > 100)
        confidence = min(0.95, 0.5 + (high_engagement_count * 0.1))
        
        return {
            "status": EventStatus.OCCURRED,
            "confidence": confidence,
            "reason": f"Found {len(price_confirmations)} posts confirming ${target_price:,.0f} reached",
            "evidence": price_confirmations[:5],  # Top 5 evidence
            "target_price": target_price
        }
    
    async def _verify_announcement(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """Verify if announcement was made"""
        # Extract key terms from query
        keywords = self._extract_keywords(query)
        
        # Look for authoritative sources
        official_posts = []
        for post in posts:
            metadata = post.metadata or {}
            verified = metadata.get("verified", False)
            followers = metadata.get("author_followers", 0)
            
            # Check if post matches keywords and is from credible source
            text_lower = post.text.lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            
            if keyword_matches >= 2 and (verified or followers > 50000):
                official_posts.append({
                    "post": post.text,
                    "author": post.author,
                    "verified": verified,
                    "followers": followers,
                    "keyword_matches": keyword_matches
                })
        
        if not official_posts:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.2,
                "reason": "No official announcements found"
            }
        
        # High confidence if multiple verified sources
        verified_count = sum(1 for p in official_posts if p["verified"])
        confidence = min(0.95, 0.6 + (verified_count * 0.15))
        
        return {
            "status": EventStatus.OCCURRED,
            "confidence": confidence,
            "reason": f"Found {len(official_posts)} official posts announcing event",
            "evidence": official_posts[:3]
        }
    
    async def _verify_content(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """
        Verify if specific content was posted (tweets, statements)
        
        CRITICAL FOR SOCIAL EVENTS:
        Example: "Trump tweets the word 'crypto'"
        This needs REAL-TIME verification from Twitter
        """
        # Extract what we're looking for
        keywords = self._extract_keywords(query)
        
        # Extract who we're looking for (if mentioned)
        target_authors = self._extract_target_authors(query)
        
        # Search for exact or near-exact matches
        matches = []
        for post in posts:
            text_lower = post.text.lower()
            author_lower = post.author.lower()
            
            # Check if this is from the target author (if specified)
            author_match = True
            if target_authors:
                author_match = any(target in author_lower for target in target_authors)
            
            if not author_match:
                continue
            
            # Check keyword matches
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            
            if keyword_matches >= len(keywords) * 0.6:  # 60% keyword match (more lenient)
                metadata = post.metadata or {}
                matches.append({
                    "post": post.text,
                    "author": post.author,
                    "verified": metadata.get("verified", False),
                    "followers": metadata.get("author_followers", 0),
                    "match_score": keyword_matches / len(keywords),
                    "timestamp": post.timestamp
                })
        
        if not matches:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.1,
                "reason": "No matching content found",
                "searched_posts": len(posts),
                "target_authors": target_authors
            }
        
        # Sort by best match (verified + high followers + match score)
        matches.sort(key=lambda m: (m["verified"], m["followers"], m["match_score"]), reverse=True)
        best_match = matches[0]
        
        # High confidence if:
        # - From verified account, OR
        # - Multiple matches found, OR
        # - High follower count
        confidence = 0.3
        if best_match["verified"]:
            confidence = 0.9
        elif best_match["followers"] > 100000:
            confidence = 0.7
        elif len(matches) >= 3:
            confidence = 0.6
        
        return {
            "status": EventStatus.OCCURRED,
            "confidence": confidence,
            "reason": f"Found matching content from @{best_match['author']} (verified: {best_match['verified']})",
            "evidence": matches[:5],
            "match_count": len(matches)
        }
    
    async def _verify_time_based(
        self,
        query: str,
        posts: List[Post],
        deadline: Optional[datetime]
    ) -> Dict[str, Any]:
        """Verify time-based events"""
        # For time-based events, we can't verify until deadline
        if not deadline:
            return {
                "status": EventStatus.UNCERTAIN,
                "confidence": 0.5,
                "reason": "No deadline specified for time-based event"
            }
        
        now = datetime.utcnow()
        
        if now < deadline:
            # Event is still ongoing, provide probability estimate
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.5,
                "reason": f"Event deadline not reached yet (deadline: {deadline.isoformat()})"
            }
        else:
            # Deadline passed, verify if condition was met
            # Delegate to appropriate verification method
            event_type = self._classify_event_type(query)
            if event_type == "price_target":
                return await self._verify_price_target(query, posts)
            else:
                return await self._verify_generic(query, posts)
    
    async def _verify_generic(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """Generic verification for unclear event types"""
        keywords = self._extract_keywords(query)
        
        # Count posts mentioning keywords
        relevant_posts = []
        for post in posts:
            text_lower = post.text.lower()
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                relevant_posts.append({
                    "post": post.text,
                    "matches": matches
                })
        
        if len(relevant_posts) == 0:
            confidence = 0.1
        elif len(relevant_posts) < 10:
            confidence = 0.3
        elif len(relevant_posts) < 50:
            confidence = 0.5
        else:
            confidence = 0.7
        
        return {
            "status": EventStatus.UNCERTAIN,
            "confidence": confidence,
            "reason": f"Found {len(relevant_posts)} posts discussing the topic",
            "relevant_posts_count": len(relevant_posts)
        }
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract meaningful keywords from query"""
        # Remove common words
        stop_words = {'will', 'the', 'be', 'by', 'in', 'to', 'a', 'an', 'or', 'and', 'of', 'on', 'at', 'for', 'does', 'do', 'is', 'was'}
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def _extract_target_authors(self, query: str) -> List[str]:
        """Extract target author names from query"""
        query_lower = query.lower()
        
        # Known public figures (extend this list based on common markets)
        known_figures = [
            'trump', 'donald trump', 'donald',
            'elon', 'musk', 'elon musk',
            'biden', 'joe biden', 'joe',
            'vitalik', 'buterin', 'vitalik buterin',
            'cz', 'changpeng', 'changpeng zhao',
            'sbf', 'sam bankman', 'bankman-fried',
            'satoshi', 'nakamoto', 'satoshi nakamoto',
            'jack', 'dorsey', 'jack dorsey',
            'cathie', 'wood', 'cathie wood',
            'michael', 'saylor', 'michael saylor',
            'mark', 'zuckerberg', 'zuck',
            'bezos', 'jeff bezos',
            'gates', 'bill gates',
            'lebron', 'james', 'lebron james',
            'curry', 'stephen curry', 'steph curry'
        ]
        
        found = []
        for name in known_figures:
            if name in query_lower:
                # Add only the primary identifier (avoid duplicates)
                if ' ' not in name:  # Single word name
                    found.append(name)
                else:  # Full name
                    # Only add if we haven't already added a part of it
                    parts = name.split()
                    if not any(part in found for part in parts):
                        found.append(name)
        
        # Remove duplicates and sort by length (longer = more specific)
        found = list(set(found))
        found.sort(key=len, reverse=True)
        
        return found
    
    async def _verify_political_outcome(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """
        Verify political outcomes (elections, votes, policies)
        
        Examples:
        - "Trump wins 2024 election"
        - "Senate approves bill"
        - "Governor resigns"
        """
        keywords = self._extract_keywords(query)
        
        # Look for official/credible sources
        credible_posts = []
        for post in posts:
            metadata = post.metadata or {}
            verified = metadata.get("verified", False)
            followers = metadata.get("author_followers", 0)
            
            # Political news requires high credibility
            if not verified and followers < 100000:
                continue
            
            text_lower = post.text.lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            
            if keyword_matches >= len(keywords) * 0.5:
                credible_posts.append({
                    "post": post.text,
                    "author": post.author,
                    "verified": verified,
                    "followers": followers,
                    "match_score": keyword_matches / len(keywords)
                })
        
        if not credible_posts:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.2,
                "reason": "No credible reports of political outcome found"
            }
        
        # High confidence only with multiple verified sources
        verified_count = sum(1 for p in credible_posts if p["verified"])
        confidence = min(0.95, 0.5 + (verified_count * 0.15))
        
        # Require at least 3 verified sources for political claims
        if verified_count < 3:
            confidence = min(confidence, 0.6)
        
        return {
            "status": EventStatus.OCCURRED if verified_count >= 3 else EventStatus.UNCERTAIN,
            "confidence": confidence,
            "reason": f"Found {verified_count} verified sources confirming outcome",
            "evidence": credible_posts[:5]
        }
    
    async def _verify_sports_outcome(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """
        Verify sports outcomes (championships, games, tournaments)
        
        Examples:
        - "Lakers win NBA Championship"
        - "Messi scores hat trick"
        - "Warriors beat Celtics"
        """
        keywords = self._extract_keywords(query)
        
        # Look for sports news accounts and official team accounts
        sports_posts = []
        for post in posts:
            metadata = post.metadata or {}
            verified = metadata.get("verified", False)
            
            text_lower = post.text.lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            
            # Sports results spread quickly, so we're more lenient
            if keyword_matches >= len(keywords) * 0.4:
                sports_posts.append({
                    "post": post.text,
                    "author": post.author,
                    "verified": verified,
                    "match_score": keyword_matches / len(keywords),
                    "engagement": post.engagement
                })
        
        if not sports_posts:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.15,
                "reason": "No posts about sports outcome found"
            }
        
        # Sports news is reliable with high engagement even if not verified
        high_engagement = sum(1 for p in sports_posts if p["engagement"] > 50)
        verified_count = sum(1 for p in sports_posts if p["verified"])
        
        confidence = 0.4
        if verified_count >= 2:
            confidence = 0.85
        elif high_engagement >= 5:
            confidence = 0.7
        elif len(sports_posts) >= 10:
            confidence = 0.6
        
        return {
            "status": EventStatus.OCCURRED if confidence >= 0.7 else EventStatus.UNCERTAIN,
            "confidence": confidence,
            "reason": f"Found {len(sports_posts)} posts about outcome ({verified_count} verified)",
            "evidence": sports_posts[:5]
        }
    
    async def _verify_milestone(
        self,
        query: str,
        posts: List[Post]
    ) -> Dict[str, Any]:
        """
        Verify technical/scientific milestones
        
        Examples:
        - "SpaceX lands on Mars"
        - "Quantum computer breakthrough"
        - "Fusion reactor achieves net positive"
        """
        keywords = self._extract_keywords(query)
        
        # Technical milestones require authoritative sources
        milestone_posts = []
        for post in posts:
            metadata = post.metadata or {}
            verified = metadata.get("verified", False)
            followers = metadata.get("author_followers", 0)
            
            # Require credible sources for scientific claims
            if not verified and followers < 50000:
                continue
            
            text_lower = post.text.lower()
            keyword_matches = sum(1 for kw in keywords if kw in text_lower)
            
            if keyword_matches >= len(keywords) * 0.5:
                milestone_posts.append({
                    "post": post.text,
                    "author": post.author,
                    "verified": verified,
                    "followers": followers,
                    "match_score": keyword_matches / len(keywords)
                })
        
        if not milestone_posts:
            return {
                "status": EventStatus.NOT_YET,
                "confidence": 0.1,
                "reason": "No credible reports of milestone achievement"
            }
        
        # Scientific milestones need strong verification
        verified_count = sum(1 for p in milestone_posts if p["verified"])
        high_follower_count = sum(1 for p in milestone_posts if p["followers"] > 500000)
        
        confidence = 0.3
        if verified_count >= 5 and high_follower_count >= 2:
            confidence = 0.9
        elif verified_count >= 3:
            confidence = 0.7
        elif verified_count >= 1:
            confidence = 0.5
        
        return {
            "status": EventStatus.OCCURRED if confidence >= 0.7 else EventStatus.UNCERTAIN,
            "confidence": confidence,
            "reason": f"Found {verified_count} verified sources reporting milestone",
            "evidence": milestone_posts[:5]
        }
    
    def get_event_decision(
        self,
        verification_result: Dict[str, Any],
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Make final binary YES/NO decision on event occurrence
        
        Args:
            verification_result: Result from verify_event()
            threshold: Confidence threshold for YES decision
        
        Returns:
            Decision with reasoning
        """
        status = verification_result["status"]
        confidence = verification_result["confidence"]
        
        if status == EventStatus.OCCURRED and confidence >= threshold:
            decision = "YES"
            reasoning = f"Event occurred (confidence: {confidence:.1%}). {verification_result['reason']}"
        elif status == EventStatus.FAILED:
            decision = "NO"
            reasoning = "Event deadline passed without occurrence"
        else:
            decision = "UNCERTAIN"
            reasoning = f"Insufficient evidence (confidence: {confidence:.1%})"
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": reasoning,
            "status": status.value
        }
