"""
Market-specific database management
Each market gets its own SQLite database file
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import logging

logger = logging.getLogger(__name__)

# Base directory for market databases
DB_DIR = Path(__file__).parent.parent.parent / "data" / "markets"
DB_DIR.mkdir(parents=True, exist_ok=True)


class MarketDatabase:
    """
    Manages database for a specific market
    
    Each market has:
    - predictions: AI predictions over time
    - events: Detected events
    - status: Market status and metadata
    """
    
    def __init__(self, market_id: str):
        self.market_id = market_id
        self.db_path = DB_DIR / f"{market_id}.db"
        self._init_database()
        logger.info(f"MarketDatabase initialized for {market_id}: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Predictions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                ai_score REAL NOT NULL,
                market_score REAL NOT NULL,
                divergence_index REAL NOT NULL,
                confidence REAL NOT NULL,
                vocal_summary TEXT,
                data_sources TEXT,
                sentiment_analysis TEXT,
                bot_detection TEXT,
                tweets_analyzed INTEGER,
                event_triggered BOOLEAN DEFAULT 0
            )
        """)
        
        # Events table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                severity INTEGER,
                description TEXT,
                evidence TEXT
            )
        """)
        
        # Market status table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_status (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                query TEXT NOT NULL,
                deadline DATETIME,
                status TEXT DEFAULT 'active',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME,
                verification_result TEXT
            )
        """)
        
        # Indexes for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_predictions_timestamp 
            ON predictions(timestamp DESC)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_events_timestamp 
            ON events(timestamp DESC)
        """)
        
        conn.commit()
        conn.close()
    
    def save_prediction(
        self,
        ai_score: float,
        market_score: float,
        divergence_index: float,
        confidence: float,
        vocal_summary: str,
        data_sources: Dict[str, Any],
        sentiment_analysis: Optional[Dict[str, Any]] = None,
        bot_detection: Optional[Dict[str, Any]] = None,
        tweets_analyzed: int = 0,
        event_triggered: bool = False
    ) -> int:
        """Save prediction to market database"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (
                ai_score, market_score, divergence_index, confidence,
                vocal_summary, data_sources, sentiment_analysis, bot_detection,
                tweets_analyzed, event_triggered
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            ai_score,
            market_score,
            divergence_index,
            confidence,
            vocal_summary,
            json.dumps(data_sources),
            json.dumps(sentiment_analysis) if sentiment_analysis else None,
            json.dumps(bot_detection) if bot_detection else None,
            tweets_analyzed,
            event_triggered
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"[{self.market_id}] Saved prediction #{prediction_id}")
        return prediction_id
    
    def save_event(
        self,
        event_type: str,
        severity: int,
        description: str,
        evidence: Optional[Dict[str, Any]] = None
    ) -> int:
        """Save detected event"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO events (event_type, severity, description, evidence)
            VALUES (?, ?, ?, ?)
        """, (
            event_type,
            severity,
            description,
            json.dumps(evidence) if evidence else None
        ))
        
        event_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"[{self.market_id}] Saved event #{event_id}: {event_type}")
        return event_id
    
    def get_predictions(
        self,
        limit: int = 100,
        since_hours: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get prediction history"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if since_hours:
            cursor.execute("""
                SELECT * FROM predictions
                WHERE timestamp >= datetime('now', '-' || ? || ' hours')
                ORDER BY timestamp DESC
                LIMIT ?
            """, (since_hours, limit))
        else:
            cursor.execute("""
                SELECT * FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            # Parse JSON fields
            if result.get("data_sources"):
                result["data_sources"] = json.loads(result["data_sources"])
            if result.get("sentiment_analysis"):
                result["sentiment_analysis"] = json.loads(result["sentiment_analysis"])
            if result.get("bot_detection"):
                result["bot_detection"] = json.loads(result["bot_detection"])
            results.append(result)
        
        return results
    
    def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction"""
        predictions = self.get_predictions(limit=1)
        return predictions[0] if predictions else None
    
    def get_prediction_count(self) -> int:
        """Get total number of predictions for this market"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM predictions")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get event history"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM events
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        results = []
        for row in rows:
            result = dict(row)
            if result.get("evidence"):
                result["evidence"] = json.loads(result["evidence"])
            results.append(result)
        
        return results
    
    def set_market_info(
        self,
        ticker: str,
        query: str,
        deadline: Optional[datetime] = None
    ):
        """Set or update market information"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Check if info exists
        cursor.execute("SELECT id FROM market_status LIMIT 1")
        exists = cursor.fetchone()
        
        if exists:
            cursor.execute("""
                UPDATE market_status
                SET ticker = ?, query = ?, deadline = ?
                WHERE id = 1
            """, (ticker, query, deadline.isoformat() if deadline else None))
        else:
            cursor.execute("""
                INSERT INTO market_status (id, ticker, query, deadline)
                VALUES (1, ?, ?, ?)
            """, (ticker, query, deadline.isoformat() if deadline else None))
        
        conn.commit()
        conn.close()
    
    def get_market_info(self) -> Optional[Dict[str, Any]]:
        """Get market information"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM market_status WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        
        return dict(row) if row else None
    
    def mark_completed(self, verification_result: Optional[Dict[str, Any]] = None):
        """Mark market as completed"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE market_status
            SET status = 'completed',
                completed_at = datetime('now'),
                verification_result = ?
            WHERE id = 1
        """, (json.dumps(verification_result) if verification_result else None,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"[{self.market_id}] Marked as completed")
    
    def get_status(self) -> str:
        """Get current market status"""
        info = self.get_market_info()
        return info["status"] if info else "unknown"
    
    def archive(self):
        """Archive the database (rename with timestamp)"""
        if not self.db_path.exists():
            return
        
        archive_name = f"{self.market_id}_archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        archive_path = DB_DIR / "archive" / archive_name
        archive_path.parent.mkdir(exist_ok=True)
        
        self.db_path.rename(archive_path)
        logger.info(f"[{self.market_id}] Archived to {archive_path}")
    
    def delete(self):
        """Delete the database file"""
        if self.db_path.exists():
            self.db_path.unlink()
            logger.info(f"[{self.market_id}] Database deleted")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Count predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        prediction_count = cursor.fetchone()[0]
        
        # Count events
        cursor.execute("SELECT COUNT(*) FROM events")
        event_count = cursor.fetchone()[0]
        
        # Get first and last prediction timestamps
        cursor.execute("""
            SELECT MIN(timestamp), MAX(timestamp) FROM predictions
        """)
        first_ts, last_ts = cursor.fetchone()
        
        # Get database file size
        file_size = self.db_path.stat().st_size if self.db_path.exists() else 0
        
        conn.close()
        
        return {
            "market_id": self.market_id,
            "predictions_count": prediction_count,
            "events_count": event_count,
            "first_prediction": first_ts,
            "last_prediction": last_ts,
            "file_size_kb": file_size / 1024,
            "status": self.get_status()
        }


def get_market_db(market_id: str) -> MarketDatabase:
    """Factory function to get or create market database"""
    return MarketDatabase(market_id)


def cleanup_completed_markets():
    """Clean up databases for completed markets"""
    if not DB_DIR.exists():
        return
    
    for db_file in DB_DIR.glob("market_*.db"):
        market_id = db_file.stem
        db = MarketDatabase(market_id)
        
        info = db.get_market_info()
        if not info:
            continue
        
        # Check if market is completed
        if info["status"] == "completed":
            # Archive instead of delete (safer)
            db.archive()
            logger.info(f"Cleaned up completed market: {market_id}")
