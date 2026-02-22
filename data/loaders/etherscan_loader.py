"""
Etherscan API Loader — fetches real Ethereum transaction data.

Supports:
  - Wallet transaction history (normal + internal)
  - ERC-20 token transfers
  - Live transaction lookup by hash
  - Simple caching via a local parquet store
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

from config.config import settings


class EtherscanLoader:
    """Fetches Ethereum data from the Etherscan public API."""

    BASE_URL = settings.blockchain.etherscan_base_url

    def __init__(self, api_key: Optional[str] = None, cache_dir: Optional[Path] = None):
        self.api_key = api_key or settings.blockchain.etherscan_api_key
        self.cache_dir = cache_dir or Path("./data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._rate_limit_delay = 0.2  # 5 calls/sec on free tier

    # ─────────────────────────────────────────────────────────── Internal ────

    def _call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single API call with basic retry logic."""
        params["apikey"] = self.api_key
        for attempt in range(3):
            try:
                resp = requests.get(
                    self.BASE_URL,
                    params=params,
                    timeout=settings.blockchain.request_timeout,
                )
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "1":
                    return data
                if data.get("message") == "No transactions found":
                    return {"result": []}
                logger.warning(f"Etherscan API warning: {data.get('message')}")
                return {"result": []}
            except requests.RequestException as exc:
                logger.warning(f"Attempt {attempt + 1}/3 failed: {exc}")
                time.sleep(2 ** attempt)
        return {"result": []}

    def _to_dataframe(self, records: List[Dict]) -> pd.DataFrame:
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records)
        # Numeric conversions
        for col in ["value", "gas", "gasPrice", "gasUsed", "blockNumber", "timeStamp"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "timeStamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timeStamp"], unit="s", utc=True)
        return df

    # ─────────────────────────────────────────────────────────── Public ─────

    def get_wallet_transactions(
        self,
        address: str,
        start_block: int = 0,
        end_block: int = 99_999_999,
        limit: int = 10_000,
    ) -> pd.DataFrame:
        """Fetch normal ETH transactions for a wallet address."""
        logger.info(f"Fetching transactions for {address[:10]}...")
        cache_file = self.cache_dir / f"txs_{address[:10]}_{limit}.parquet"
        if cache_file.exists():
            logger.info("Returning cached results")
            return pd.read_parquet(cache_file)

        all_records: List[Dict] = []
        page = 1
        while len(all_records) < limit:
            time.sleep(self._rate_limit_delay)
            batch_size = min(10_000, limit - len(all_records))
            data = self._call({
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "endblock": end_block,
                "page": page,
                "offset": batch_size,
                "sort": "desc",
            })
            records = data.get("result", [])
            if not records:
                break
            all_records.extend(records)
            if len(records) < batch_size:
                break
            page += 1

        df = self._to_dataframe(all_records)
        if not df.empty:
            df.to_parquet(cache_file, index=False)
        logger.info(f"Loaded {len(df)} transactions for {address[:10]}")
        return df

    def get_transaction_by_hash(self, tx_hash: str) -> Optional[Dict[str, Any]]:
        """Fetch a single transaction by its hash."""
        data = self._call({
            "module": "proxy",
            "action": "eth_getTransactionByHash",
            "txhash": tx_hash,
        })
        return data.get("result")

    def get_token_transfers(
        self,
        address: str,
        contract_address: Optional[str] = None,
        limit: int = 5_000,
    ) -> pd.DataFrame:
        """Fetch ERC-20 token transfers for a wallet."""
        params: Dict[str, Any] = {
            "module": "account",
            "action": "tokentx",
            "address": address,
            "sort": "desc",
            "offset": limit,
            "page": 1,
        }
        if contract_address:
            params["contractaddress"] = contract_address

        time.sleep(self._rate_limit_delay)
        data = self._call(params)
        return self._to_dataframe(data.get("result", []))

    def get_internal_transactions(self, address: str, limit: int = 5_000) -> pd.DataFrame:
        """Fetch internal (contract-to-contract) transactions."""
        time.sleep(self._rate_limit_delay)
        data = self._call({
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "sort": "desc",
            "offset": limit,
            "page": 1,
        })
        return self._to_dataframe(data.get("result", []))

    def get_multiple_wallets(
        self, addresses: List[str], limit_per_wallet: int = 1_000
    ) -> pd.DataFrame:
        """Batch-fetch transactions for multiple wallets and concatenate."""
        frames = []
        for addr in addresses:
            df = self.get_wallet_transactions(addr, limit=limit_per_wallet)
            if not df.empty:
                df["queried_address"] = addr
                frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).drop_duplicates(
            subset=["hash"] if "hash" in frames[0].columns else None
        )

    def enrich_with_eth_price(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Approximate USD value using Etherscan's ETH price endpoint.
        Only fetches current price (historical requires paid tier).
        """
        time.sleep(self._rate_limit_delay)
        data = self._call({"module": "stats", "action": "ethprice"})
        result = data.get("result", {})
        eth_usd = float(result.get("ethusd", 0))
        if "value" in df.columns:
            df["value_eth"] = df["value"] / 1e18
            df["value_usd"] = df["value_eth"] * eth_usd
        return df
