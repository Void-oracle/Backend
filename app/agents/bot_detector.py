"""
Bot Detection Agent
Distinguishes real influencers from bots and paid shilling
"""
from typing import List, Dict, Any
from datetime import datetime, timedelta

from openai import AsyncOpenAI

from app.agents.base_agent import BaseAgent
from app.models.schemas import BotDetectionResult, InfluencerProfile
from app.config import settings


class BotDetectorAgent(BaseAgent):
    """Agent for detecting bots and identifying authentic influencers"""
    
    def __init__(self):
        super().__init__("BotDetectorAgent")
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        
        # Bot detection thresholds
        self.min_account_age_days = 30
        self.max_daily_posts = 100
        self.min_followers_ratio = 0.1  # followers / following
        self.suspicious_username_patterns = [
            r'\d{4,}$',  # Ends with 4+ digits
            r'^[a-z]{8}\d{8}$',  # Random chars + numbers
        ]
    
    async def process(self, posts: List[Dict[str, Any]]) -> BotDetectionResult:
        """
        Analyze accounts and detect bots vs authentic influencers
        
        Args:
            posts: List of posts with author metadata
            
        Returns:
            BotDetectionResult with bot probability and influencer analysis
        """
        if not posts:
            return self._create_empty_result()
        
        self.logger.info(f"Analyzing {len(posts)} posts for bot activity")
        
        # Group posts by author
        accounts = self._group_by_author(posts)
        
        # Analyze each account
        account_analyses = []
        for username, user_posts in accounts.items():
            analysis = self._analyze_account(username, user_posts)
            account_analyses.append(analysis)
        
        # Calculate overall statistics
        total_accounts = len(account_analyses)
        bot_accounts = sum(1 for a in account_analyses if a['is_bot'])
        authentic_ratio = 1.0 - (bot_accounts / total_accounts if total_accounts > 0 else 0)
        
        # Identify top influencers
        influencers = self._identify_influencers(account_analyses)
        
        # Collect suspicious patterns
        suspicious_patterns = self._collect_patterns(account_analyses)
        
        # AI-powered bot detection for suspicious accounts
        if settings.OPENAI_API_KEY:
            await self._ai_bot_verification(account_analyses[:20])  # Top 20 accounts
        
        return BotDetectionResult(
            bot_probability=1.0 - authentic_ratio,
            suspicious_patterns=list(set(suspicious_patterns)),
            authentic_ratio=authentic_ratio,
            top_influencers=influencers[:10]  # Top 10 influencers
        )
    
    def _group_by_author(self, posts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group posts by author username"""
        accounts = {}
        for post in posts:
            author = post.get('author', {})
            username = author.get('username', 'unknown')
            
            if username not in accounts:
                accounts[username] = []
            accounts[username].append(post)
        
        return accounts
    
    def _analyze_account(self, username: str, posts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a single account for bot indicators"""
        if not posts:
            return {'username': username, 'is_bot': True, 'bot_score': 1.0}
        
        # Get account metadata from first post
        author = posts[0].get('author', {})
        followers = author.get('followers', 0)
        following = author.get('following', 0)
        verified = author.get('verified', False)
        account_created = author.get('created_at')
        
        bot_score = 0.0
        bot_indicators = []
        
        # 1. Check account age
        if account_created:
            try:
                created_date = datetime.fromisoformat(account_created.replace('Z', '+00:00'))
                account_age_days = (datetime.utcnow() - created_date.replace(tzinfo=None)).days
                
                if account_age_days < self.min_account_age_days:
                    bot_score += 0.3
                    bot_indicators.append(f"New account ({account_age_days} days)")
            except:
                pass
        
        # 2. Follower/following ratio
        if following > 0:
            follower_ratio = followers / following
            if follower_ratio < self.min_followers_ratio:
                bot_score += 0.2
                bot_indicators.append(f"Low follower ratio ({follower_ratio:.2f})")
        
        # 3. Posting frequency
        post_count = len(posts)
        if post_count > 50:  # If many posts in short time
            bot_score += 0.15
            bot_indicators.append(f"High posting frequency ({post_count} posts)")
        
        # 4. Username patterns
        import re
        for pattern in self.suspicious_username_patterns:
            if re.search(pattern, username):
                bot_score += 0.15
                bot_indicators.append("Suspicious username pattern")
                break
        
        # 5. Content repetition
        unique_texts = len(set(post.get('text', '') for post in posts))
        if unique_texts < post_count * 0.5:
            bot_score += 0.2
            bot_indicators.append("Repetitive content")
        
        # Verified accounts get a pass
        if verified:
            bot_score *= 0.3
        
        # High follower count reduces bot score
        if followers > settings.MIN_INFLUENCER_FOLLOWERS:
            bot_score *= 0.5
        
        # Calculate engagement rate
        total_engagement = sum(post.get('engagement', 0) for post in posts)
        avg_engagement = total_engagement / post_count if post_count > 0 else 0
        engagement_rate = avg_engagement / max(followers, 1)
        
        return {
            'username': username,
            'followers': followers,
            'following': following,
            'verified': verified,
            'account_age_days': account_age_days if account_created else None,
            'post_count': post_count,
            'bot_score': min(1.0, bot_score),
            'is_bot': bot_score >= settings.BOT_DETECTION_THRESHOLD,
            'bot_indicators': bot_indicators,
            'engagement_rate': engagement_rate,
            'total_engagement': total_engagement
        }
    
    def _identify_influencers(self, account_analyses: List[Dict[str, Any]]) -> List[InfluencerProfile]:
        """Identify top authentic influencers"""
        # Filter out bots
        authentic = [a for a in account_analyses if not a['is_bot']]
        
        # Filter for minimum followers
        influencers = [
            a for a in authentic 
            if a['followers'] >= settings.MIN_INFLUENCER_FOLLOWERS
        ]
        
        # Sort by engagement and followers
        def influence_score(acc: Dict[str, Any]) -> float:
            followers = acc['followers']
            engagement = acc['total_engagement']
            verified_bonus = 1.5 if acc['verified'] else 1.0
            return (followers * 0.3 + engagement * 0.7) * verified_bonus
        
        influencers.sort(key=influence_score, reverse=True)
        
        # Convert to InfluencerProfile
        profiles = []
        for inf in influencers:
            # Calculate credibility score
            credibility = 1.0 - inf['bot_score']
            if inf['verified']:
                credibility = min(1.0, credibility * 1.2)
            
            profiles.append(InfluencerProfile(
                username=inf['username'],
                followers=inf['followers'],
                verified=inf['verified'],
                engagement_rate=inf['engagement_rate'],
                credibility_score=credibility
            ))
        
        return profiles
    
    def _collect_patterns(self, account_analyses: List[Dict[str, Any]]) -> List[str]:
        """Collect all suspicious patterns detected"""
        patterns = []
        for analysis in account_analyses:
            if analysis.get('is_bot'):
                patterns.extend(analysis.get('bot_indicators', []))
        return patterns
    
    async def _ai_bot_verification(self, account_analyses: List[Dict[str, Any]]) -> None:
        """Use AI to verify bot detection for edge cases"""
        # This is a more sophisticated check for borderline cases
        suspicious = [
            a for a in account_analyses 
            if 0.4 < a['bot_score'] < 0.7
        ]
        
        if not suspicious:
            return
        
        # Prepare account summaries
        summaries = []
        for acc in suspicious[:10]:  # Limit to 10 for API costs
            summary = f"""
Username: {acc['username']}
Followers: {acc['followers']}
Following: {acc['following']}
Verified: {acc['verified']}
Post Count: {acc['post_count']}
Engagement Rate: {acc['engagement_rate']:.4f}
Bot Indicators: {', '.join(acc['bot_indicators'])}
"""
            summaries.append(summary)
        
        prompt = f"""Analyze these social media accounts and determine if they are likely bots or authentic users.

Accounts:
{chr(10).join(summaries)}

For each account, provide:
1. Is it likely a bot? (true/false)
2. Confidence level (0.0 to 1.0)
3. Brief reasoning

Respond in JSON format as an array:
[{{"username": "...", "is_bot": true, "confidence": 0.8, "reason": "..."}}]
"""
        
        try:
            response = await self.client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting bot accounts and fake influencers on social media."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            import json
            results = json.loads(response.choices[0].message.content)
            
            # Update bot scores based on AI analysis
            if isinstance(results, list):
                for result in results:
                    username = result.get('username')
                    for acc in account_analyses:
                        if acc['username'] == username:
                            if result.get('is_bot'):
                                acc['bot_score'] = max(acc['bot_score'], result.get('confidence', 0.7))
                                acc['is_bot'] = acc['bot_score'] >= settings.BOT_DETECTION_THRESHOLD
            
        except Exception as e:
            self.logger.error(f"AI bot verification failed: {e}")
    
    def _create_empty_result(self) -> BotDetectionResult:
        """Create empty result when no data available"""
        return BotDetectionResult(
            bot_probability=0.0,
            suspicious_patterns=[],
            authentic_ratio=1.0,
            top_influencers=[]
        )
