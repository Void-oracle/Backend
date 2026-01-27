"""
SQLite Database for storing prediction history
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
from app.config import settings

# Database file path
DB_PATH = Path(__file__).parent.parent.parent / "data" / "predictions.db"
DB_PATH.parent.mkdir(exist_ok=True)

def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

def init_database():
    """Initialize database schema"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create predictions history table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            query TEXT NOT NULL,
            time_range_hours INTEGER NOT NULL,
            
            ai_score REAL NOT NULL,
            market_score REAL NOT NULL,
            divergence_index REAL NOT NULL,
            confidence REAL NOT NULL,
            vocal_summary TEXT,
            
            data_sources TEXT,
            sentiment_analysis TEXT,
            bot_detection TEXT,
            
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(market_id, timestamp)
        )
    """)
    
    # Create index for faster queries
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_market_timestamp 
        ON prediction_history(market_id, timestamp DESC)
    """)
    
    conn.commit()
    conn.close()

def save_prediction(
    market_id: str,
    ticker: str,
    query: str,
    time_range_hours: int,
    prediction_result: Dict[str, Any]
) -> int:
    """Save prediction to database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO prediction_history (
                market_id, ticker, query, time_range_hours,
                ai_score, market_score, divergence_index, confidence,
                vocal_summary, data_sources, sentiment_analysis, bot_detection,
                timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            market_id,
            ticker,
            query,
            time_range_hours,
            prediction_result["ai_score"],
            prediction_result["market_score"],
            prediction_result["divergence_index"],
            prediction_result["confidence"],
            prediction_result["vocal_summary"],
            json.dumps(prediction_result.get("data_sources", {})),
            json.dumps(prediction_result.get("sentiment_analysis", {})),
            json.dumps(prediction_result.get("bot_detection", {})),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        return cursor.lastrowid
    except sqlite3.IntegrityError:
        # Duplicate entry (same market_id and timestamp)
        return None
    finally:
        conn.close()

def get_prediction_history(
    market_id: str,
    limit: int = 100,
    time_range_hours: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Get prediction history for a market"""
    conn = get_connection()
    cursor = conn.cursor()
    
    if time_range_hours:
        cursor.execute("""
            SELECT * FROM prediction_history
            WHERE market_id = ? AND time_range_hours = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (market_id, time_range_hours, limit))
    else:
        cursor.execute("""
            SELECT * FROM prediction_history
            WHERE market_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (market_id, limit))
    
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

def get_latest_prediction(market_id: str, time_range_hours: int = 24) -> Optional[Dict[str, Any]]:
    """Get latest prediction for a market"""
    history = get_prediction_history(market_id, limit=1, time_range_hours=time_range_hours)
    return history[0] if history else None

def get_all_markets() -> List[str]:
    """Get list of all market IDs in database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT DISTINCT market_id FROM prediction_history")
    rows = cursor.fetchall()
    conn.close()
    
    return [row["market_id"] for row in rows]

# Initialize database on import
init_database()
