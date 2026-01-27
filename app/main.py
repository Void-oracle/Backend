"""
VOID Backend - Main FastAPI Application
Vocal Oracle of Intelligent Data
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.api.v1.router import api_router
from app.services.oracle_engine import OracleEngine
from app.core.market_database import cleanup_completed_markets
from app.core.markets_manager import markets_manager


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global Oracle Engine and monitoring tasks
oracle_engine = None
monitoring_tasks = {}  # market_id -> task


async def start_monitoring_market(market: dict):
    """
    Start monitoring a specific market
    Can be called dynamically without restart!
    """
    global oracle_engine, monitoring_tasks
    
    if not oracle_engine:
        logger.error("Oracle engine not initialized")
        return
    
    market_id = market["market_id"]
    
    # Stop existing monitoring if any
    if market_id in monitoring_tasks:
        monitoring_tasks[market_id].cancel()
        try:
            await monitoring_tasks[market_id]
        except asyncio.CancelledError:
            pass
    
    logger.info(f"Initializing market: {market_id}")
    
    # Parse deadline
    deadline = None
    if market.get("deadline"):
        if isinstance(market["deadline"], str):
            deadline = datetime.fromisoformat(market["deadline"].replace('Z', '+00:00'))
        else:
            deadline = market["deadline"]
    
    # Initialize market
    try:
        await oracle_engine.initialize_market(
            market_id=market_id,
            ticker=market["ticker"],
            query=market["query"],
            target_tweets=market.get("target_tweets", 500),
            deadline=deadline
        )
    except Exception as e:
        logger.error(f"Failed to initialize {market_id}: {e}")
        return
    
    # Start monitoring (updates every 30 minutes)
    update_interval = market.get("update_interval_minutes", 30)
    task = asyncio.create_task(
        oracle_engine.monitor_market(
            market_id=market_id,
            update_interval_minutes=update_interval
        )
    )
    monitoring_tasks[market_id] = task
    
    logger.info(f"Started monitoring: {market_id} (updates every {update_interval}min)")


async def initialize_oracle_system():
    """Initialize oracle system with all markets from database"""
    global oracle_engine, monitoring_tasks
    
    oracle_engine = OracleEngine()
    
    logger.info("=" * 60)
    logger.info("VOID ORACLE - INITIALIZATION PHASE")
    logger.info("=" * 60)
    
    # Initialize default markets if none exist
    markets_manager.initialize_default_markets()
    
    # Load all active markets from database
    active_markets = markets_manager.get_active_markets()
    
    logger.info(f"Markets to initialize: {len(active_markets)}")
    logger.info("")
    
    # Initialize all markets
    for i, market in enumerate(active_markets, 1):
        logger.info(f"[{i}/{len(active_markets)}] Initializing {market['market_id']}...")
        try:
            await start_monitoring_market(market)
            await asyncio.sleep(2)  # Small delay between markets
        except Exception as e:
            logger.error(f"Failed to initialize {market['market_id']}: {e}")
        logger.info("")
    
    logger.info("=" * 60)
    logger.info("INITIALIZATION COMPLETE")
    logger.info("=" * 60)
    logger.info("")
    logger.info(f"Monitoring {len(monitoring_tasks)} active markets")
    logger.info("Oracle system fully operational!")
    
    # Schedule periodic cleanup of completed markets
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(3600)  # Every hour
            try:
                cleanup_completed_markets()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
    
    asyncio.create_task(periodic_cleanup())


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    global monitoring_tasks
    
    # Startup
    logger.info("Starting VOID Oracle Backend...")
    logger.info(f"Environment: {settings.APP_ENV}")
    logger.info(f"API Prefix: {settings.API_PREFIX}")
    logger.info("")
    
    # Initialize oracle system in background
    asyncio.create_task(initialize_oracle_system())
    
    yield
    
    # Shutdown
    logger.info("Shutting down VOID Oracle Backend...")
    for task in monitoring_tasks.values():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("All monitoring tasks stopped")


# Create FastAPI application
app = FastAPI(
    title="VOID Oracle API",
    description="""
    **VOID - Vocal Oracle of Intelligent Data**
    
    AI-powered sentiment oracle for Solana prediction markets.
    
    Features:
    - Multi-agent sentiment analysis
    - Bot detection and influencer identification
    - Real-time divergence calculation
    - Vocal AI summaries
    
    ## Main Endpoint
    
    **POST /api/v1/oracle/predict** - Get oracle prediction with divergence index
    
    ## About
    
    VOID analyzes social media sentiment using advanced AI agents and compares 
    it with prediction market probabilities to identify divergence opportunities.
    """,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)


# CORS middleware
cors_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
]

# Add production domain from environment if set
if hasattr(settings, "FRONTEND_URL") and settings.FRONTEND_URL:
    domain = settings.FRONTEND_URL
    cors_origins.extend([
        domain,
        f"https://{domain}",
        f"http://{domain}",
        f"https://www.{domain}",
        f"http://www.{domain}",
    ])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins if cors_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include API router
app.include_router(api_router, prefix=settings.API_PREFIX)


# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "VOID Oracle API",
        "version": "0.1.0",
        "description": "Vocal Oracle of Intelligent Data - AI Sentiment Oracle for Solana",
        "docs": "/docs",
        "health": f"{settings.API_PREFIX}/oracle/health",
        "main_endpoint": f"{settings.API_PREFIX}/oracle/predict"
    }


# Health check endpoint
@app.get("/health", tags=["health"])
async def health() -> Dict[str, str]:
    """Simple health check"""
    return {"status": "healthy", "service": "void-oracle"}


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "body": exc.body
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc) if settings.DEBUG else "An error occurred"
        }
    )


# Middleware for logging requests
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests"""
    logger.info(f"{request.method} {request.url.path}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
