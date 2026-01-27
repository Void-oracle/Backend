"""
Redis cache utilities
Caching layer for API responses and data
"""
import json
import logging
from typing import Optional, Any
import redis.asyncio as redis

from app.config import settings


logger = logging.getLogger(__name__)


class Cache:
    """Redis cache manager"""
    
    redis_client: Optional[redis.Redis] = None
    
    @classmethod
    async def connect(cls) -> None:
        """Connect to Redis"""
        try:
            logger.info(f"Connecting to Redis: {settings.REDIS_HOST}:{settings.REDIS_PORT}")
            
            cls.redis_client = await redis.from_url(
                f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DB}",
                password=settings.REDIS_PASSWORD if settings.REDIS_PASSWORD else None,
                encoding="utf-8",
                decode_responses=True
            )
            
            # Test connection
            await cls.redis_client.ping()
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Don't raise - cache is optional
            cls.redis_client = None
    
    @classmethod
    async def disconnect(cls) -> None:
        """Disconnect from Redis"""
        if cls.redis_client:
            await cls.redis_client.close()
            logger.info("Redis connection closed")
    
    @classmethod
    async def get(cls, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not cls.redis_client:
            return None
        
        try:
            value = await cls.redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    @classmethod
    async def set(
        cls,
        key: str,
        value: Any,
        ttl: int = None
    ) -> bool:
        """Set value in cache"""
        if not cls.redis_client:
            return False
        
        try:
            ttl = ttl or settings.CACHE_TTL
            await cls.redis_client.setex(
                key,
                ttl,
                json.dumps(value)
            )
            return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    @classmethod
    async def delete(cls, key: str) -> bool:
        """Delete key from cache"""
        if not cls.redis_client:
            return False
        
        try:
            await cls.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False


# Global cache instance
cache = Cache()


async def get_cache() -> Cache:
    """Dependency for getting cache in routes"""
    return cache
