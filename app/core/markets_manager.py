"""
Dynamic Markets Manager
Manages market creation, storage, and lifecycle without server restart
"""
import sqlite3
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Markets registry database
MARKETS_DB = Path(__file__).parent.parent.parent / "data" / "markets_registry.db"
MARKETS_DB.parent.mkdir(parents=True, exist_ok=True)


class MarketsManager:
    """
    Central manager for all prediction markets
    
    Features:
    - Dynamic market creation (no restart needed)
    - Persistent storage
    - Market status tracking
    - Automatic cleanup
    """
    
    def __init__(self):
        self._init_database()
        logger.info("Markets Manager initialized")
    
    def _init_database(self):
        """Initialize markets registry database with auto-increment ID"""
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        # Main markets table with auto-increment numeric ID
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT UNIQUE NOT NULL,
                ticker TEXT NOT NULL,
                query TEXT NOT NULL,
                description TEXT,
                category TEXT DEFAULT 'markets',
                deadline DATETIME,
                target_tweets INTEGER DEFAULT 500,
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                created_by TEXT DEFAULT 'system',
                monitoring_active BOOLEAN DEFAULT 1,
                check_interval_minutes INTEGER DEFAULT 30,
                external_market_url TEXT,
                resolution TEXT
            )
        """)
        
        # Add columns if they don't exist (for existing databases)
        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN external_market_url TEXT")
        except:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE markets ADD COLUMN resolution TEXT")
        except:
            pass  # Column already exists
        
        # Counter table to track next ID (persists across restarts)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_counter (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_id INTEGER DEFAULT 1
            )
        """)
        
        # Initialize counter if not exists
        cursor.execute("INSERT OR IGNORE INTO market_counter (id, next_id) VALUES (1, 1)")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markets_status ON markets(status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_markets_category ON markets(category)")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Markets registry database initialized: {MARKETS_DB}")
    
    def _get_next_market_id(self) -> str:
        """Get next market ID and increment counter"""
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        # Get current counter
        cursor.execute("SELECT next_id FROM market_counter WHERE id = 1")
        row = cursor.fetchone()
        next_num = row[0] if row else 1
        
        # Increment counter
        cursor.execute("UPDATE market_counter SET next_id = ? WHERE id = 1", (next_num + 1,))
        conn.commit()
        conn.close()
        
        return f"market_{next_num}"
    
    def create_market(
        self,
        ticker: str,
        query: str,
        description: Optional[str] = None,
        category: str = "markets",
        deadline: Optional[datetime] = None,
        target_tweets: int = 500,
        check_interval_minutes: int = 30,
        created_by: str = "system",
        market_id: Optional[str] = None,  # Optional - auto-generated if not provided
        external_market_url: Optional[str] = None  # Polymarket/PredictFun URL
    ) -> Dict[str, Any]:
        """
        Create a new market with auto-generated ID
        
        Args:
            external_market_url: URL to external market (Polymarket, PredictFun, etc.)
                               If provided, will fetch real market probability
        
        Returns market data
        """
        # Auto-generate market_id if not provided
        if not market_id:
            market_id = self._get_next_market_id()
        
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO markets (
                    market_id, ticker, query, description, category,
                    deadline, target_tweets, created_by, check_interval_minutes,
                    external_market_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                market_id,
                ticker,
                query,
                description,
                category,
                deadline.isoformat() if deadline else None,
                target_tweets,
                created_by,
                check_interval_minutes,
                external_market_url
            ))
            
            conn.commit()
            logger.info(f"Market created: {market_id} ({ticker})")
            
            return self.get_market(market_id)
            
        except sqlite3.IntegrityError:
            raise ValueError(f"Market {market_id} already exists")
        finally:
            conn.close()
    
    def get_market(self, market_id: str) -> Optional[Dict[str, Any]]:
        """Get market by ID"""
        conn = sqlite3.connect(str(MARKETS_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM markets WHERE market_id = ?", (market_id,))
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def list_markets(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        monitoring_active: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """List all markets with optional filters"""
        conn = sqlite3.connect(str(MARKETS_DB))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = "SELECT * FROM markets WHERE 1=1"
        params = []
        
        if status:
            query += " AND status = ?"
            params.append(status)
        
        if category:
            query += " AND category = ?"
            params.append(category)
        
        if monitoring_active is not None:
            query += " AND monitoring_active = ?"
            params.append(1 if monitoring_active else 0)
        
        query += " ORDER BY created_at DESC"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def update_market_status(
        self,
        market_id: str,
        status: str,
        monitoring_active: Optional[bool] = None
    ):
        """Update market status"""
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        if monitoring_active is not None:
            cursor.execute("""
                UPDATE markets
                SET status = ?, monitoring_active = ?
                WHERE market_id = ?
            """, (status, 1 if monitoring_active else 0, market_id))
        else:
            cursor.execute("""
                UPDATE markets
                SET status = ?
                WHERE market_id = ?
            """, (status, market_id))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Market {market_id} status updated to: {status}")
    
    def complete_market(self, market_id: str, outcome: Optional[str] = None):
        """
        Mark market as completed (keeps history, stops monitoring)
        
        Args:
            market_id: Market to complete
            outcome: Optional outcome description ("yes", "no", etc.)
        """
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE markets
            SET status = 'completed', 
                monitoring_active = 0,
                completed_at = CURRENT_TIMESTAMP
            WHERE market_id = ?
        """, (market_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Market {market_id} completed (outcome: {outcome})")
    
    def delete_market(self, market_id: str):
        """Delete a market from registry (use complete_market to preserve history)"""
        conn = sqlite3.connect(str(MARKETS_DB))
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM markets WHERE market_id = ?", (market_id,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Market {market_id} deleted from registry")
    
    def get_active_markets(self) -> List[Dict[str, Any]]:
        """Get all active markets that should be monitored"""
        return self.list_markets(status="active", monitoring_active=True)
    
    def initialize_default_markets(self):
        """Initialize default markets if none exist"""
        existing = self.list_markets()
        
        if len(existing) > 0:
            logger.info(f"Found {len(existing)} existing markets")
            return
        
        logger.info("No markets found, creating default markets...")
        
        # 5 FUTURE markets (2026-2027) - market_id auto-generated
        default_markets = [
            {
                "ticker": "BTC-200K",
                "query": "Will Bitcoin reach $200,000 by end of 2026?",
                "description": "Bitcoin price target for 2026 bull cycle",
                "category": "markets",
                "deadline": datetime(2026, 12, 31, 23, 59, 59)
            },
            {
                "ticker": "ETH-10K",
                "query": "Will Ethereum reach $10,000 by Q4 2026?",
                "description": "Ethereum price target prediction",
                "category": "markets",
                "deadline": datetime(2026, 12, 31, 23, 59, 59)
            },
            {
                "ticker": "SOL-ETF",
                "query": "Will a Solana spot ETF be approved in 2026?",
                "description": "SEC approval for Solana ETF",
                "category": "markets",
                "deadline": datetime(2026, 12, 31, 23, 59, 59)
            },
            {
                "ticker": "GPT5",
                "query": "Will OpenAI release GPT-5 before July 2026?",
                "description": "Next generation AI model release",
                "category": "tech",
                "deadline": datetime(2026, 7, 1, 23, 59, 59)
            },
            {
                "ticker": "FED-2026",
                "query": "Will the Fed cut rates at least 2 times in 2026?",
                "description": "Federal Reserve monetary policy 2026",
                "category": "politics",
                "deadline": datetime(2026, 12, 31, 23, 59, 59)
            }
        ]
        
        for market_data in default_markets:
            try:
                self.create_market(**market_data)
            except Exception as e:
                logger.error(f"Failed to create default market {market_data['market_id']}: {e}")
        
        logger.info(f"Created {len(default_markets)} default markets")


# Global instance
markets_manager = MarketsManager()
