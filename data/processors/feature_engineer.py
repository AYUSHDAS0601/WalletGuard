"""
Feature Engineering Pipeline for Blockchain Transaction Data.

Produces:
  - Transaction-level features (per TX row)
  - Wallet-level aggregate features (per address)
  - Combined feature matrix ready for ML models
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler


# ── Column name constants ────────────────────────────────────────────────────
TX_VALUE_COL = "value"          # raw Wei or smallest unit
GAS_COL = "gas"
GAS_PRICE_COL = "gasPrice"
GAS_USED_COL = "gasUsed"
TIMESTAMP_COL = "timestamp"     # must be a datetime column
FROM_COL = "from"
TO_COL = "to"
HASH_COL = "hash"
BLOCK_COL = "blockNumber"


class FeatureEngineer:
    """
    Transforms raw transaction DataFrames into ML-ready feature matrices.

    Usage:
        fe = FeatureEngineer()
        tx_features   = fe.build_transaction_features(df)
        wallet_features = fe.build_wallet_features(df)
        combined      = fe.build_combined(df)
    """

    KNOWN_MIXERS = frozenset({
        "0xba214c1c1928a32bffe790263e38b4af9bfcd659",  # Tornado Cash
        "0xd90e2f925da726b50c4ed8d0fb90ad053324f31b",
        "0x722122df12d4e14e13ac3b6895a86e84145b6967",
    })

    KNOWN_EXCHANGES = frozenset({
        "0xdac17f958d2ee523a2206206994597c13d831ec7",  # USDT
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap v2 router
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap v3 router
    })

    def __init__(self, scaler: Optional[StandardScaler] = None):
        self.scaler = scaler or StandardScaler()
        self._fitted = False
        self._last_feature_names: List[str] = []

    # ──────────────────────────────────────────────── Transaction-level ──────

    def build_transaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Derive per-transaction features from a raw transactions DataFrame.
        Expects columns: value, gas, gasPrice/gasUsed, timestamp, from, to.
        """
        df = df.copy()
        df = self._ensure_timestamp(df)

        # ── Value features ────────────────────────────────────────────────────
        df["value_eth"] = df.get(TX_VALUE_COL, 0) / 1e18
        df["tx_fee_eth"] = (
            df.get(GAS_USED_COL, df.get(GAS_COL, 0)) * df.get(GAS_PRICE_COL, 0) / 1e18
        )
        df["gas_price_gwei"] = df.get(GAS_PRICE_COL, 0) / 1e9

        # Relative percentiles within this batch
        df["value_percentile"] = df["value_eth"].rank(pct=True)
        df["gas_percentile"] = df["gas_price_gwei"].rank(pct=True)

        # ── Temporal features ─────────────────────────────────────────────────
        df["hour_of_day"] = df[TIMESTAMP_COL].dt.hour
        df["day_of_week"] = df[TIMESTAMP_COL].dt.dayofweek
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # ── Transaction type flags ────────────────────────────────────────────
        df["is_contract_creation"] = (
            df.get(TO_COL, pd.Series([""] * len(df))).isna()
            | (df.get(TO_COL, pd.Series([""] * len(df))) == "")
        ).astype(int)

        df["is_zero_value"] = (df["value_eth"] == 0).astype(int)
        df["is_self_transfer"] = (
            df.get(FROM_COL, pd.Series()) == df.get(TO_COL, pd.Series())
        ).astype(int)

        # Mixer / exchange interaction
        to_col = df.get(TO_COL, pd.Series([""] * len(df))).fillna("").str.lower()
        from_col = df.get(FROM_COL, pd.Series([""] * len(df))).fillna("").str.lower()
        df["to_mixer"] = to_col.isin(self.KNOWN_MIXERS).astype(int)
        df["from_mixer"] = from_col.isin(self.KNOWN_MIXERS).astype(int)
        df["to_exchange"] = to_col.isin(self.KNOWN_EXCHANGES).astype(int)

        # ── MEV indicator: tx position in block ──────────────────────────────
        if "transactionIndex" in df.columns:
            df["tx_position_in_block"] = pd.to_numeric(
                df["transactionIndex"], errors="coerce"
            ).fillna(0)
        else:
            df["tx_position_in_block"] = 0

        logger.debug(f"Transaction features built — shape {df.shape}")
        return df

    # ──────────────────────────────────────────────────── Wallet-level ───────

    def build_wallet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate per-wallet statistics from the transaction DataFrame.
        Returns a DataFrame indexed by wallet address.
        """
        df = df.copy()
        df = self._ensure_timestamp(df)

        if "value_eth" not in df.columns:
            df["value_eth"] = df.get(TX_VALUE_COL, 0) / 1e18

        records: List[Dict] = []

        for addr in pd.concat([df.get(FROM_COL, pd.Series()), df.get(TO_COL, pd.Series())]).dropna().unique():
            sent = df[df.get(FROM_COL, pd.Series([""] * len(df))) == addr]
            received = df[df.get(TO_COL, pd.Series([""] * len(df))) == addr]
            all_txs = pd.concat([sent, received]).drop_duplicates(
                subset=[HASH_COL] if HASH_COL in df.columns else None
            )

            if all_txs.empty:
                continue

            total_txs = len(all_txs)
            total_sent = sent["value_eth"].sum()
            total_received = received["value_eth"].sum()
            avg_tx_value = all_txs["value_eth"].mean()
            std_tx_value = all_txs["value_eth"].std(ddof=0)

            # Temporal
            min_ts = all_txs[TIMESTAMP_COL].min()
            max_ts = all_txs[TIMESTAMP_COL].max()
            span_days = max(1, (max_ts - min_ts).total_seconds() / 86400)
            tx_frequency = total_txs / span_days

            # Counterparties
            sent_to = set(sent.get(TO_COL, pd.Series()).dropna())
            recv_from = set(received.get(FROM_COL, pd.Series()).dropna())
            unique_counterparties = len(sent_to | recv_from)

            # Behavioural flags
            to_series = sent.get(TO_COL, pd.Series([])).fillna("").str.lower()
            mixer_interactions = to_series.isin(self.KNOWN_MIXERS).sum()
            self_transfers = (
                (sent.get(FROM_COL, pd.Series()) == addr)
                & (sent.get(TO_COL, pd.Series()) == addr)
            ).sum()

            # Active hours distribution (entropy-like spread)
            hour_counts = all_txs[TIMESTAMP_COL].dt.hour.value_counts()
            hour_probs = hour_counts / hour_counts.sum()
            hour_entropy = float(-(hour_probs * np.log2(hour_probs + 1e-9)).sum())

            records.append({
                "address": addr,
                "total_txs": total_txs,
                "total_value_sent": total_sent,
                "total_value_received": total_received,
                "avg_tx_value": avg_tx_value,
                "std_tx_value": float(std_tx_value) if not np.isnan(std_tx_value) else 0.0,
                "tx_frequency_per_day": tx_frequency,
                "unique_counterparties": unique_counterparties,
                "mixer_interaction_count": int(mixer_interactions),
                "self_transfer_count": int(self_transfers),
                "hour_entropy": hour_entropy,
                "span_days": span_days,
            })

        wallet_df = pd.DataFrame(records).set_index("address")
        logger.debug(f"Wallet features built for {len(wallet_df)} addresses")
        return wallet_df

    # ────────────────────────────────────────────── Combined feature matrix ──

    def build_combined(
        self,
        df: pd.DataFrame,
        tx_feature_cols: Optional[List[str]] = None,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Build a flat feature matrix (rows = transactions) for ML ingestion.
        Merges transaction-level and sender wallet-level features.
        """
        tx_df = self.build_transaction_features(df)
        wallet_df = self.build_wallet_features(df)

        # Merge sender wallet features into tx rows
        merged = tx_df.merge(
            wallet_df.add_prefix("sender_"),
            left_on=FROM_COL,
            right_index=True,
            how="left",
        )

        default_tx_cols = [
            "value_eth", "tx_fee_eth", "gas_price_gwei",
            "value_percentile", "gas_percentile",
            "hour_of_day", "day_of_week", "is_weekend",
            "is_contract_creation", "is_zero_value", "is_self_transfer",
            "to_mixer", "from_mixer", "to_exchange",
            "tx_position_in_block",
        ]
        default_wallet_cols = [
            f"sender_{c}" for c in [
                "total_txs", "total_value_sent", "total_value_received",
                "avg_tx_value", "std_tx_value", "tx_frequency_per_day",
                "unique_counterparties", "mixer_interaction_count",
                "self_transfer_count", "hour_entropy",
            ]
        ]

        cols = tx_feature_cols or (default_tx_cols + default_wallet_cols)
        available = [c for c in cols if c in merged.columns]
        self._last_feature_names = available
        X = merged[available].fillna(0).values.astype(np.float32)

        if normalize:
            if not self._fitted:
                X = self.scaler.fit_transform(X)
                self._fitted = True
            else:
                X = self.scaler.transform(X)

        logger.info(f"Combined feature matrix: {X.shape}")
        return X

    # ─────────────────────────────────────────────────────────── Helpers ─────

    @staticmethod
    def _ensure_timestamp(df: pd.DataFrame) -> pd.DataFrame:
        """Ensure TIMESTAMP_COL exists as a proper datetime column."""
        if TIMESTAMP_COL not in df.columns:
            if "timeStamp" in df.columns:
                df[TIMESTAMP_COL] = pd.to_datetime(
                    pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", utc=True
                )
            elif "block_timestamp" in df.columns:
                df[TIMESTAMP_COL] = pd.to_datetime(df["block_timestamp"], utc=True)
            else:
                df[TIMESTAMP_COL] = pd.Timestamp.now(tz="UTC")
        elif not pd.api.types.is_datetime64_any_dtype(df[TIMESTAMP_COL]):
            df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL], utc=True)
        return df
