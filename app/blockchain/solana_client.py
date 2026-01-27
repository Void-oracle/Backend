"""
Solana RPC Client for VOID Oracle

This module handles all Solana blockchain interactions:
- Reading on-chain market data
- Submitting oracle results
- Managing oracle keypair
"""
import logging
from typing import Optional, Dict, Any
from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import Transaction
from solana.rpc.types import TxOpts

from app.config import settings

logger = logging.getLogger(__name__)


class SolanaClient:
    """
    Solana blockchain client for oracle operations
    """
    
    def __init__(self):
        self.rpc_url = settings.SOLANA_RPC_URL
        self.client = AsyncClient(self.rpc_url, commitment=Confirmed)
        self.oracle_keypair = self._load_oracle_keypair()
        logger.info(f"Solana client initialized: {self.rpc_url}")
        if self.oracle_keypair:
            logger.info(f"Oracle public key: {self.oracle_keypair.pubkey()}")
    
    def _load_oracle_keypair(self) -> Optional[Keypair]:
        """Load oracle's keypair from environment"""
        try:
            if settings.SOLANA_PRIVATE_KEY:
                # Expects base58 string or JSON array
                if settings.SOLANA_PRIVATE_KEY.startswith('['):
                    # JSON array format: [1,2,3,...]
                    import json
                    key_bytes = bytes(json.loads(settings.SOLANA_PRIVATE_KEY))
                    return Keypair.from_bytes(key_bytes)
                else:
                    # Base58 string format
                    return Keypair.from_base58_string(settings.SOLANA_PRIVATE_KEY)
            
            logger.warning("SOLANA_PRIVATE_KEY not configured, oracle functions disabled")
            return None
            
        except Exception as e:
            logger.error(f"Failed to load oracle keypair: {e}")
            return None
    
    async def get_balance(self, pubkey: str) -> Optional[float]:
        """
        Get SOL balance for a public key
        
        Returns balance in SOL (not lamports)
        """
        try:
            pubkey_obj = Pubkey.from_string(pubkey)
            response = await self.client.get_balance(pubkey_obj)
            
            if response.value is not None:
                # Convert lamports to SOL
                sol_balance = response.value / 1_000_000_000
                return sol_balance
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting balance for {pubkey}: {e}")
            return None
    
    async def get_oracle_balance(self) -> Optional[float]:
        """Get oracle's own SOL balance"""
        if not self.oracle_keypair:
            return None
        
        return await self.get_balance(str(self.oracle_keypair.pubkey()))
    
    async def get_recent_blockhash(self) -> str:
        """Get recent blockhash for transactions"""
        response = await self.client.get_latest_blockhash()
        return str(response.value.blockhash)
    
    async def get_signature_statuses(self, signatures: list) -> Dict[str, Any]:
        """Get status of transaction signatures"""
        try:
            response = await self.client.get_signature_statuses(signatures)
            return response.value
        except Exception as e:
            logger.error(f"Error getting signature statuses: {e}")
            return {}
    
    async def confirm_transaction(self, signature: str, max_retries: int = 30) -> bool:
        """
        Wait for transaction confirmation
        
        Returns True if confirmed, False if failed/timeout
        """
        import asyncio
        
        for i in range(max_retries):
            try:
                statuses = await self.get_signature_statuses([signature])
                if statuses and len(statuses) > 0:
                    status = statuses[0]
                    if status:
                        if status.confirmation_status in ["confirmed", "finalized"]:
                            logger.info(f"Transaction {signature} confirmed")
                            return True
                        elif status.err:
                            logger.error(f"Transaction {signature} failed: {status.err}")
                            return False
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error checking transaction status: {e}")
                await asyncio.sleep(1)
        
        logger.warning(f"Transaction {signature} confirmation timeout")
        return False
    
    async def airdrop(self, pubkey: str, amount_sol: float = 1.0) -> Optional[str]:
        """
        Request airdrop (devnet/testnet only)
        
        Args:
            pubkey: Target public key
            amount_sol: Amount in SOL (not lamports)
        
        Returns:
            Transaction signature if successful
        """
        try:
            pubkey_obj = Pubkey.from_string(pubkey)
            lamports = int(amount_sol * 1_000_000_000)
            
            response = await self.client.request_airdrop(pubkey_obj, lamports)
            signature = str(response.value)
            
            logger.info(f"Airdrop requested: {amount_sol} SOL to {pubkey}")
            logger.info(f"Signature: {signature}")
            
            # Wait for confirmation
            confirmed = await self.confirm_transaction(signature)
            
            if confirmed:
                return signature
            else:
                logger.error("Airdrop transaction failed")
                return None
            
        except Exception as e:
            logger.error(f"Airdrop error: {e}")
            return None
    
    async def send_sol(
        self,
        to_pubkey: str,
        amount_sol: float,
        from_keypair: Optional[Keypair] = None
    ) -> Optional[str]:
        """
        Send SOL from oracle wallet to another address
        
        Args:
            to_pubkey: Destination public key
            amount_sol: Amount in SOL
            from_keypair: Source keypair (defaults to oracle keypair)
        
        Returns:
            Transaction signature if successful
        """
        try:
            keypair = from_keypair or self.oracle_keypair
            if not keypair:
                logger.error("No keypair available for sending SOL")
                return None
            
            to_pubkey_obj = Pubkey.from_string(to_pubkey)
            lamports = int(amount_sol * 1_000_000_000)
            
            # Create transfer instruction
            transfer_ix = transfer(
                TransferParams(
                    from_pubkey=keypair.pubkey(),
                    to_pubkey=to_pubkey_obj,
                    lamports=lamports
                )
            )
            
            # Get recent blockhash
            blockhash = await self.get_recent_blockhash()
            
            # Create and sign transaction
            tx = Transaction.new_with_payer(
                [transfer_ix],
                keypair.pubkey()
            )
            tx.recent_blockhash = blockhash
            tx.sign([keypair])
            
            # Send transaction
            response = await self.client.send_transaction(
                tx,
                opts=TxOpts(skip_preflight=False)
            )
            signature = str(response.value)
            
            logger.info(f"Sent {amount_sol} SOL to {to_pubkey}")
            logger.info(f"Signature: {signature}")
            
            # Wait for confirmation
            confirmed = await self.confirm_transaction(signature)
            
            if confirmed:
                return signature
            else:
                logger.error("Transfer transaction failed")
                return None
            
        except Exception as e:
            logger.error(f"Error sending SOL: {e}")
            return None
    
    async def get_account_info(self, pubkey: str) -> Optional[Dict[str, Any]]:
        """Get account info for a public key"""
        try:
            pubkey_obj = Pubkey.from_string(pubkey)
            response = await self.client.get_account_info(pubkey_obj)
            
            if response.value:
                return {
                    "lamports": response.value.lamports,
                    "owner": str(response.value.owner),
                    "executable": response.value.executable,
                    "rent_epoch": response.value.rent_epoch,
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return None
    
    async def close(self):
        """Close the RPC client connection"""
        await self.client.close()


# Global instance
solana_client = SolanaClient()
