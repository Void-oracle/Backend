"""
Base agent class for VOID Oracle
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import logging

from app.config import settings


logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"agent.{name}")
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize agent resources"""
        self.logger.info(f"Initializing agent: {self.name}")
    
    @abstractmethod
    async def process(self, data: Any) -> Any:
        """Process data and return results"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "name": self.name,
            "status": "active",
            "config": self.config
        }
    
    async def health_check(self) -> bool:
        """Check if agent is healthy"""
        return True
