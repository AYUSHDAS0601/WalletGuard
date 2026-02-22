"""
Flash Loan Exploit Detector — identifies multi-protocol single-block attacks and MEV patterns.

Signals:
  - Large borrowed amount repaid within same block
  - Transaction touches ≥2 DeFi protocols in one TX
  - MEV sandwich patterns (buy before and sell after a victim TX)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from loguru import logger

from config.config import settings

# Known DeFi protocol addresses (add more as needed)
KNOWN_PROTOCOLS: Dict[str, str] = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d": "Uniswap v2",
    "0xe592427a0aece92de3edee1f18e0157c05861564": "Uniswap v3",
    "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2": "Aave v3",
    "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9": "Aave v2",
    "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b": "Compound",
    "0xc0a47dfe034b400b47bdad5fecda2621de6c4d95": "Uniswap v1",
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff": "0x Protocol",
    "0x1111111254fb6c44bac0bed2854e76f90643097d": "1inch",
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f": "SushiSwap",
}


@dataclass
class FlashLoanPattern:
    tx_hash: str
    from_address: str
    block_number: int
    protocols_touched: List[str]
    estimated_profit_eth: float
    pattern_type: str               # "flash_loan" | "mev_sandwich" | "arbitrage"
    confidence: float
    details: dict = field(default_factory=dict)

    @property
    def risk_level(self) -> str:
        if self.confidence >= 0.85:
            return "CRITICAL"
        if self.confidence >= 0.70:
            return "HIGH"
        return "MEDIUM"

    def to_dict(self) -> dict:
        return {
            "pattern": f"flash_loan_{self.pattern_type}",
            "tx_hash": self.tx_hash,
            "from_address": self.from_address,
            "block_number": self.block_number,
            "protocols_touched": self.protocols_touched,
            "estimated_profit_eth": round(self.estimated_profit_eth, 6),
            "pattern_type": self.pattern_type,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "details": self.details,
        }


class FlashLoanDetector:
    """
    Detects flash loan exploits and MEV sandwich attacks.

    Works on both individual transactions and block-level DataFrames.
    Uses heuristics + protocol address matching.
    """

    def __init__(
        self,
        min_protocols: int = None,
        min_value_eth: float = 0.5,
    ):
        self.min_protocols = min_protocols or settings.detection.flash_loan_min_protocols
        self.min_value_eth = min_value_eth
        self._protocol_addrs: Set[str] = {
            addr.lower() for addr in KNOWN_PROTOCOLS
        }

    # ─────────────────────────────────────────────────── Public API ───────────

    def detect(self, df: pd.DataFrame) -> List[FlashLoanPattern]:
        """Analyze a DataFrame of transactions for flash loan patterns."""
        if df.empty:
            return []

        patterns = []

        # Group by block for MEV analysis
        block_col = "blockNumber" if "blockNumber" in df.columns else None

        if block_col:
            for block_num, block_df in df.groupby(block_col):
                patterns.extend(self._analyze_block(block_df, int(block_num)))
        else:
            patterns.extend(self._analyze_block(df, -1))

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info(f"Flash loan detector found {len(patterns)} patterns")
        return patterns

    def detect_single_tx(self, tx: Dict) -> Optional[FlashLoanPattern]:
        """Lightweight detection for a single transaction dict."""
        df = pd.DataFrame([tx])
        results = self.detect(df)
        return results[0] if results else None

    # ─────────────────────────────────────────────────── Internals ────────────

    def _analyze_block(
        self, block_df: pd.DataFrame, block_number: int
    ) -> List[FlashLoanPattern]:
        """Detect flash loans and MEV within a single block."""
        patterns = []

        # ── Multi-protocol flash loans ────────────────────────────────────────
        for addr, addr_df in block_df.groupby(
            "from" if "from" in block_df.columns else block_df.columns[0]
        ):
            p = self._check_flash_loan(addr_df, str(addr), block_number)
            if p:
                patterns.append(p)

        # ── MEV sandwich detection ─────────────────────────────────────────────
        mev = self._detect_mev_sandwich(block_df, block_number)
        patterns.extend(mev)

        return patterns

    def _check_flash_loan(
        self, addr_df: pd.DataFrame, address: str, block_number: int
    ) -> Optional[FlashLoanPattern]:
        """Check if an address engaged in a flash loan within a block."""
        if "to" not in addr_df.columns:
            return None

        to_addrs = addr_df["to"].fillna("").str.lower()
        touched = [
            KNOWN_PROTOCOLS[addr]
            for addr in to_addrs
            if addr in KNOWN_PROTOCOLS
        ]

        if len(set(touched)) < self.min_protocols:
            return None

        value_col = "value_eth" if "value_eth" in addr_df.columns else "value"
        values = addr_df[value_col].fillna(0)
        if value_col == "value":
            values = values / 1e18

        total_value = float(values.sum())
        if total_value < self.min_value_eth:
            return None

        # Estimate profit as: received value - sent value (net flow)
        received = addr_df.get("value_received", values)
        net_flow = float(received.sum() - values.sum())

        confidence = self._flash_loan_confidence(
            n_protocols=len(set(touched)),
            total_value=total_value,
            net_profit=net_flow,
        )

        if confidence < 0.5:
            return None

        tx_hash = str(addr_df.get("hash", pd.Series(["unknown"])).iloc[0])

        return FlashLoanPattern(
            tx_hash=tx_hash,
            from_address=address,
            block_number=block_number,
            protocols_touched=list(set(touched)),
            estimated_profit_eth=net_flow,
            pattern_type="flash_loan",
            confidence=confidence,
            details={
                "total_value_eth": round(total_value, 4),
                "tx_count_in_block": len(addr_df),
            },
        )

    def _detect_mev_sandwich(
        self, block_df: pd.DataFrame, block_number: int
    ) -> List[FlashLoanPattern]:
        """
        Detect MEV sandwich attacks:
        Pattern: [bot buy] → [victim swap] → [bot sell]
        within the same block, ordered by tx_position.
        """
        patterns = []

        if "transactionIndex" not in block_df.columns:
            return patterns

        block_df = block_df.copy()
        block_df["tx_idx"] = pd.to_numeric(block_df["transactionIndex"], errors="coerce")
        block_df = block_df.sort_values("tx_idx")

        to_col = "to" if "to" in block_df.columns else None
        if not to_col:
            return patterns

        # Find addresses touching DEX routers
        dex_routers = {
            "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap v2
            "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap v3
            "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap
        }

        dex_txs = block_df[
            block_df["to"].fillna("").str.lower().isin(dex_routers)
        ].copy()

        if len(dex_txs) < 3:
            return patterns

        # Simple heuristic: same sender appears at tx_idx i and i+2,
        # different sender at i+1 (the victim) all touching same DEX
        senders = dex_txs["from"].values if "from" in dex_txs.columns else []
        for i in range(len(dex_txs) - 2):
            if senders[i] == senders[i + 2] and senders[i] != senders[i + 1]:
                attacker = str(senders[i])
                confidence = 0.75  # heuristic confidence
                tx_hash = str(dex_txs.iloc[i].get("hash", "unknown"))
                patterns.append(
                    FlashLoanPattern(
                        tx_hash=tx_hash,
                        from_address=attacker,
                        block_number=block_number,
                        protocols_touched=["DEX Router"],
                        estimated_profit_eth=0.0,  # would need simulation
                        pattern_type="mev_sandwich",
                        confidence=confidence,
                        details={
                            "victim_tx_idx": int(dex_txs.iloc[i + 1]["tx_idx"]),
                        },
                    )
                )

        return patterns

    @staticmethod
    def _flash_loan_confidence(
        n_protocols: int, total_value: float, net_profit: float
    ) -> float:
        """Heuristic confidence score for flash loan detection."""
        proto_score = min(n_protocols / 4.0, 1.0)  # 4+ protocols = max score
        value_score = min(total_value / 100.0, 1.0)  # 100 ETH+ = max score
        profit_score = 0.5 + 0.5 * min(max(net_profit / 10.0, 0), 1.0)
        return round(0.4 * proto_score + 0.3 * value_score + 0.3 * profit_score, 4)
