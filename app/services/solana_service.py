"""
Solana Service
Integrates with Solana blockchain for market data from prediction markets
"""
from typing import Optional, Dict, Any
import logging
from datetime import datetime
import json

from solana.rpc.async_api import AsyncClient
from solders.keypair import Keypair
from solders.pubkey import Pubkey

from app.config import settings
from app.models.schemas import MarketData


logger = logging.getLogger(__name__)


class SolanaService:
    """Service for interacting with Solana blockchain"""
    
    def __init__(self):
        self.rpc_url = settings.SOLANA_RPC_URL
        self.network = settings.SOLANA_NETWORK
        self.client = AsyncClient(self.rpc_url)
        
        # Oracle contract address
        self.oracle_address = settings.ORACLE_CONTRACT_ADDRESS
        
        logger.info(f"Initialized Solana service for {self.network}")
    
    async def get_market_data(self, market_id: str) -> Optional[MarketData]:
        """
        Fetch market data from Solana prediction market
        
        Args:
            market_id: Unique market identifier
            
        Returns:
            MarketData or None if not found
        """
        try:
            logger.info(f"Fetching market data for {market_id}")
            
            # In production, this would query the actual smart contract
            # For now, return mock data structure
            
            if not self.oracle_address:
                logger.warning("Oracle contract address not configured, using mock data")
                return self._get_mock_market_data(market_id)
            
            # Query smart contract
            market_data = await self._query_contract(market_id)
            
            return market_data
            
        except Exception as e:
            logger.error(f"Failed to fetch market data: {e}")
            return self._get_mock_market_data(market_id)
    
    async def _query_contract(self, market_id: str) -> Optional[MarketData]:
        """Query the oracle smart contract"""
        try:
            # Convert market_id to program account address
            # This is a simplified example - actual implementation depends on your contract
            
            if not self.oracle_address:
                return None
            
            program_id = Pubkey.from_string(self.oracle_address)
            
            # Get account info
            response = await self.client.get_account_info(program_id)
            
            if response.value is None:
                logger.error(f"No account found for {self.oracle_address}")
                return None
            
            # Parse account data (structure depends on your contract)
            account_data = response.value.data
            
            # This is where you'd decode the contract data
            # For now, return mock data
            return self._get_mock_market_data(market_id)
            
        except Exception as e:
            logger.error(f"Contract query failed: {e}")
            return None
    
    def _get_mock_market_data(self, market_id: str) -> MarketData:
        """
        Generate mock market data for testing
        
        In production, this would come from the smart contract
        """
        # Mock data based on market_id
        mock_markets = {
            "solana-300-feb-2026": {
                "market_probability": 42.0,
                "volume": "$1.2M",
                "volume_usd": 1200000.0,
                "participants": 1248,
                "liquidity": 850000.0,
                "end_date": "Feb 28, 2026"
            },
            "eth-etf-q1-2026": {
                "market_probability": 65.0,
                "volume": "$890K",
                "volume_usd": 890000.0,
                "participants": 892,
                "liquidity": 650000.0,
                "end_date": "Mar 31, 2026"
            },
            "btc-150k-summer-2026": {
                "market_probability": 28.0,
                "volume": "$2.1M",
                "volume_usd": 2100000.0,
                "participants": 2341,
                "liquidity": 1500000.0,
                "end_date": "Jun 21, 2026"
            }
        }
        
        # Get specific market or use default
        market_info = mock_markets.get(
            market_id,
            {
                "market_probability": 50.0,
                "volume": "$500K",
                "volume_usd": 500000.0,
                "participants": 500,
                "liquidity": 300000.0,
                "end_date": "Dec 31, 2026"
            }
        )
        
        return MarketData(
            market_id=market_id,
            market_probability=market_info["market_probability"],
            volume=market_info["volume"],
            volume_usd=market_info["volume_usd"],
            participants=market_info["participants"],
            liquidity=market_info["liquidity"],
            end_date=market_info["end_date"],
            created_at=datetime.utcnow()
        )
    
    async def submit_prediction(
        self,
        market_id: str,
        ai_score: float,
        divergence_index: float
    ) -> Optional[str]:
        """
        Submit oracle prediction to Solana smart contract
        
        Args:
            market_id: Market identifier
            ai_score: AI probability score
            divergence_index: Calculated divergence
            
        Returns:
            Transaction signature or None if failed
        """
        try:
            logger.info(f"Submitting prediction for {market_id}: AI={ai_score}, Div={divergence_index}")
            
            if not self.oracle_address or not settings.SOLANA_PRIVATE_KEY:
                logger.warning("Solana credentials not configured, skipping submission")
                return None
            
            # In production, this would:
            # 1. Create transaction with prediction data
            # 2. Sign with oracle keypair
            # 3. Submit to network
            # 4. Return transaction signature
            
            # Mock transaction signature
            mock_signature = f"mock_tx_{market_id}_{int(datetime.utcnow().timestamp())}"
            
            logger.info(f"Prediction submitted: {mock_signature}")
            return mock_signature
            
        except Exception as e:
            logger.error(f"Failed to submit prediction: {e}")
            return None
    
    async def get_transaction_status(self, signature: str) -> Dict[str, Any]:
        """Get status of a transaction"""
        try:
            # In production, query the transaction
            # For now, return mock status
            return {
                "signature": signature,
                "status": "confirmed",
                "slot": 12345678,
                "block_time": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get transaction status: {e}")
            return {"signature": signature, "status": "unknown"}
    
    async def close(self):
        """Close connection"""
        await self.client.close()
