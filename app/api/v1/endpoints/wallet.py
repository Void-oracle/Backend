"""
Wallet & Blockchain API Endpoints

Provides information about Solana blockchain state and wallet operations
"""
from fastapi import APIRouter, HTTPException
from typing import Optional
from pydantic import BaseModel
import logging

from app.blockchain.solana_client import solana_client

logger = logging.getLogger(__name__)
router = APIRouter()


class WalletBalanceResponse(BaseModel):
    """Wallet balance information"""
    pubkey: str
    balance_sol: float
    balance_lamports: int
    network: str


class AirdropRequest(BaseModel):
    """Request airdrop (devnet only)"""
    pubkey: str
    amount_sol: float = 1.0


@router.get("/balance/{pubkey}")
async def get_wallet_balance(pubkey: str):
    """
    Get SOL balance for a wallet address
    
    - **pubkey**: Solana public key (base58 string)
    
    Returns balance in both SOL and lamports
    """
    try:
        balance = await solana_client.get_balance(pubkey)
        
        if balance is None:
            raise HTTPException(status_code=404, detail="Account not found or invalid pubkey")
        
        return {
            "pubkey": pubkey,
            "balance_sol": balance,
            "balance_lamports": int(balance * 1_000_000_000),
            "network": "devnet"  # TODO: Get from config
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid public key: {str(e)}")
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get balance")


@router.get("/oracle/balance")
async def get_oracle_balance():
    """
    Get Oracle's own wallet balance
    
    This is the backend oracle's wallet that submits results to the blockchain
    """
    try:
        if not solana_client.oracle_keypair:
            raise HTTPException(
                status_code=503,
                detail="Oracle keypair not configured. Set SOLANA_PRIVATE_KEY in .env"
            )
        
        balance = await solana_client.get_oracle_balance()
        pubkey = str(solana_client.oracle_keypair.pubkey())
        
        if balance is None:
            raise HTTPException(status_code=404, detail="Failed to get oracle balance")
        
        return {
            "pubkey": pubkey,
            "balance_sol": balance,
            "balance_lamports": int(balance * 1_000_000_000),
            "network": "devnet",
            "role": "oracle_authority"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting oracle balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get oracle balance")


@router.post("/airdrop")
async def request_airdrop(request: AirdropRequest):
    """
    Request SOL airdrop (Devnet/Testnet only)
    
    - **pubkey**: Destination wallet address
    - **amount_sol**: Amount of SOL to airdrop (max 5 SOL per request)
    
    ⚠️ Only works on devnet/testnet networks
    """
    try:
        # Validate amount
        if request.amount_sol <= 0 or request.amount_sol > 5:
            raise HTTPException(
                status_code=400,
                detail="Amount must be between 0 and 5 SOL"
            )
        
        # Request airdrop
        signature = await solana_client.airdrop(request.pubkey, request.amount_sol)
        
        if not signature:
            raise HTTPException(
                status_code=500,
                detail="Airdrop request failed. May be rate limited or network issue."
            )
        
        return {
            "success": True,
            "signature": signature,
            "pubkey": request.pubkey,
            "amount_sol": request.amount_sol,
            "message": f"Airdrop of {request.amount_sol} SOL confirmed",
            "explorer_url": f"https://explorer.solana.com/tx/{signature}?cluster=devnet"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Airdrop error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/network-info")
async def get_network_info():
    """
    Get Solana network information
    
    Returns current RPC endpoint, network name, and oracle status
    """
    try:
        oracle_balance = None
        oracle_pubkey = None
        
        if solana_client.oracle_keypair:
            oracle_balance = await solana_client.get_oracle_balance()
            oracle_pubkey = str(solana_client.oracle_keypair.pubkey())
        
        return {
            "rpc_url": solana_client.rpc_url,
            "network": "devnet",
            "oracle": {
                "configured": solana_client.oracle_keypair is not None,
                "pubkey": oracle_pubkey,
                "balance_sol": oracle_balance
            },
            "endpoints": {
                "devnet": "https://api.devnet.solana.com",
                "testnet": "https://api.testnet.solana.com",
                "mainnet": "https://api.mainnet-beta.solana.com"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting network info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get network info")


@router.get("/account/{pubkey}")
async def get_account_info(pubkey: str):
    """
    Get detailed account information
    
    - **pubkey**: Solana account address
    
    Returns lamports, owner, executable status, etc.
    """
    try:
        account_info = await solana_client.get_account_info(pubkey)
        
        if not account_info:
            raise HTTPException(status_code=404, detail="Account not found")
        
        return {
            "pubkey": pubkey,
            **account_info,
            "balance_sol": account_info["lamports"] / 1_000_000_000
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
