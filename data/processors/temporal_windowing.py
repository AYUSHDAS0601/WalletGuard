"""
Temporal Windowing — sliding window aggregations over transaction time series.

Produces time-bucketed statistics used as features for the BiLSTM model
and for anomaly scoring.

Windows: 1h, 24h, 7d, 30d
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger


WINDOWS: Dict[str, str] = {
    "1h": "1H",
    "24h": "24H",
    "7d": "7D",
    "30d": "30D",
}


class TemporalWindowing:
    """
    Generates sliding window aggregations for both global network stats
    and per-wallet time series.

    Usage:
        tw = TemporalWindowing()
        network_df     = tw.aggregate_network(df)
        wallet_series  = tw.build_wallet_sequences(df, address)
        tensor, labels = tw.build_sequence_dataset(df, labels_df)
    """

    def __init__(
        self,
        windows: Optional[Dict[str, str]] = None,
        max_seq_length: int = 100,
        stride: int = 10,
    ):
        self.windows = windows or WINDOWS
        self.max_seq_length = max_seq_length
        self.stride = stride

    # ─────────────────────────────────────── Network-wide aggregations ────────

    def aggregate_network(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Compute time-windowed statistics across the entire transaction network.
        Returns a DataFrame indexed by (window_end, window_size).
        """
        df = self._ensure_timestamp(df, timestamp_col)
        df = df.set_index(timestamp_col).sort_index()

        if "value_eth" not in df.columns and "value" in df.columns:
            df["value_eth"] = df["value"] / 1e18
        if "gasPrice" in df.columns:
            df["gas_price_gwei"] = df["gasPrice"] / 1e9

        agg_cols = {
            "tx_count": ("hash", "count") if "hash" in df.columns else ("value_eth", "count"),
            "volume_eth": ("value_eth", "sum"),
            "avg_value_eth": ("value_eth", "mean"),
            "max_value_eth": ("value_eth", "max"),
        }
        if "gas_price_gwei" in df.columns:
            agg_cols["avg_gas_gwei"] = ("gas_price_gwei", "mean")
        if "from" in df.columns:
            agg_cols["unique_senders"] = ("from", "nunique")
        if "to" in df.columns:
            agg_cols["unique_receivers"] = ("to", "nunique")

        frames = []
        for window_name, freq in self.windows.items():
            resampled = df.resample(freq).agg(**agg_cols).reset_index()
            resampled["window"] = window_name
            resampled = resampled.rename(columns={timestamp_col: "window_end"})
            frames.append(resampled)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result = result.fillna(0)
        logger.debug(f"Network temporal aggregation — shape {result.shape}")
        return result

    # ────────────────────────────────────────── Per-wallet time series ────────

    def build_wallet_sequences(
        self,
        df: pd.DataFrame,
        address: str,
        timestamp_col: str = "timestamp",
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Build a time-indexed feature sequence for a specific wallet.
        Used as input to the BiLSTM temporal model.
        """
        df = self._ensure_timestamp(df, timestamp_col)

        # Filter rows involving this address
        from_mask = df.get("from", pd.Series([])) == address
        to_mask = df.get("to", pd.Series([])) == address
        wallet_df = df[from_mask | to_mask].copy().sort_values(timestamp_col)

        if wallet_df.empty:
            logger.warning(f"No transactions found for {address[:10]}")
            return pd.DataFrame()

        if "value_eth" not in wallet_df.columns:
            wallet_df["value_eth"] = wallet_df.get("value", 0) / 1e18

        wallet_df["is_sender"] = (
            wallet_df.get("from", pd.Series()) == address
        ).astype(float)

        if "gasPrice" in wallet_df.columns:
            wallet_df["gas_price_gwei"] = wallet_df["gasPrice"] / 1e9

        default_cols = [
            "value_eth", "is_sender",
            "gas_price_gwei" if "gas_price_gwei" in wallet_df.columns else "value_eth",
        ]
        cols = feature_cols or [c for c in default_cols if c in wallet_df.columns]
        return wallet_df[cols + [timestamp_col]].reset_index(drop=True)

    # ──────────────────────────────── Sequence dataset for BiLSTM training ───

    def build_sequence_dataset(
        self,
        df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
        timestamp_col: str = "timestamp",
        address_col: str = "from",
        feature_cols: Optional[List[str]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Construct (X, y) tensors for BiLSTM training.

        X shape: (n_wallets, max_seq_length, n_features)
        y shape: (n_wallets,)  [if labels_df provided]

        Uses zero-padding for sequences shorter than max_seq_length.
        """
        df = self._ensure_timestamp(df, timestamp_col)
        if "value_eth" not in df.columns:
            df["value_eth"] = df.get("value", 0) / 1e18

        addresses = df[address_col].dropna().unique() if address_col in df.columns else []

        sequences = []
        for addr in addresses:
            seq_df = self.build_wallet_sequences(
                df, addr, timestamp_col, feature_cols
            )
            if seq_df.empty:
                continue

            feat_cols = [c for c in seq_df.columns if c != timestamp_col]
            seq = seq_df[feat_cols].values.astype(np.float32)

            # Truncate to max length from the most recent end
            if len(seq) > self.max_seq_length:
                seq = seq[-self.max_seq_length:]

            # Zero-pad to the left if shorter
            if len(seq) < self.max_seq_length:
                pad = np.zeros(
                    (self.max_seq_length - len(seq), seq.shape[1]), dtype=np.float32
                )
                seq = np.vstack([pad, seq])

            sequences.append((addr, seq))

        if not sequences:
            return torch.zeros((0, self.max_seq_length, 1)), None

        addrs, seqs = zip(*sequences)
        X = torch.tensor(np.stack(seqs), dtype=torch.float)  # (N, T, F)

        y = None
        if labels_df is not None and "address" in labels_df.columns:
            label_map = labels_df.set_index("address")["label"].to_dict()
            y_vals = [label_map.get(a, -1) for a in addrs]
            y = torch.tensor(y_vals, dtype=torch.float)

        logger.info(f"Sequence dataset built — X: {X.shape}, y: {y.shape if y is not None else 'None'}")
        return X, y

    # ──────────────────────────────────────── Rolling anomaly velocity ────────

    def compute_velocity_features(
        self, df: pd.DataFrame, timestamp_col: str = "timestamp"
    ) -> pd.DataFrame:
        """
        Compute transaction velocity features useful for burst/flash-loan detection.
        Returns per-row features about the recent N-minute activity level.
        """
        df = self._ensure_timestamp(df, timestamp_col).sort_values(timestamp_col)
        if "value_eth" not in df.columns:
            df["value_eth"] = df.get("value", 0) / 1e18

        df = df.set_index(timestamp_col)

        for window_label, freq in [("1min", "1min"), ("5min", "5min"), ("1h", "1H")]:
            df[f"tx_count_{window_label}"] = (
                df["value_eth"].rolling(window=freq, min_periods=1).count()
            )
            df[f"volume_{window_label}"] = (
                df["value_eth"].rolling(window=freq, min_periods=1).sum()
            )

        return df.reset_index()

    # ─────────────────────────────────────────────────────── Helpers ─────────

    @staticmethod
    def _ensure_timestamp(df: pd.DataFrame, col: str) -> pd.DataFrame:
        if col not in df.columns:
            if "timeStamp" in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(
                    pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", utc=True
                )
            elif "block_timestamp" in df.columns:
                df = df.copy()
                df[col] = pd.to_datetime(df["block_timestamp"], utc=True)
            else:
                df = df.copy()
                df[col] = pd.Timestamp.now(tz="UTC")
        elif not pd.api.types.is_datetime64_any_dtype(df[col]):
            df = df.copy()
            df[col] = pd.to_datetime(df[col], utc=True)
        return df
