"""
Market Manipulation Detector — pump-and-dump, coordinated buying/selling.

Uses volume + wallet activity correlation to flag:
  - Sudden volume spikes ≥3σ above rolling baseline
  - Coordinated buy/sell waves across multiple wallets
  - Pump-and-dump timing patterns
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class ManipulationPattern:
    pattern_type: str       # "pump_dump" | "coordinated_buy" | "spoofing"
    affected_wallets: List[str]
    start_time: Optional[str]
    end_time: Optional[str]
    volume_spike_factor: float   # How many σ above baseline
    coordinated_wallet_count: int
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
            "pattern": f"market_manipulation_{self.pattern_type}",
            "pattern_type": self.pattern_type,
            "affected_wallets": self.affected_wallets[:20],
            "start_time": self.start_time,
            "end_time": self.end_time,
            "volume_spike_factor": round(self.volume_spike_factor, 2),
            "coordinated_wallet_count": self.coordinated_wallet_count,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "details": self.details,
        }


class MarketManipulationDetector:
    """
    Detects market manipulation patterns using statistical volume analysis
    and wallet activity correlation.

    Usage:
        detector = MarketManipulationDetector()
        patterns = detector.detect(df)
    """

    def __init__(
        self,
        volume_spike_threshold_sigma: float = 3.0,
        min_coordinated_wallets: int = 3,
        time_window: str = "1H",
        pump_window: str = "4H",
    ):
        self.sigma_threshold = volume_spike_threshold_sigma
        self.min_coord_wallets = min_coordinated_wallets
        self.time_window = time_window
        self.pump_window = pump_window

    # ─────────────────────────────────────────────────── Public API ───────────

    def detect(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        value_col: str = "value_eth",
        from_col: str = "from",
        to_col: str = "to",
    ) -> List[ManipulationPattern]:
        """Run all manipulation detectors on the DataFrame."""
        df = self._prep(df, timestamp_col, value_col)
        if df.empty:
            return []

        patterns: List[ManipulationPattern] = []

        # ── Volume spike detection ────────────────────────────────────────────
        spike_windows = self._find_volume_spikes(df, timestamp_col, value_col)
        for window_start, window_end, spike_factor in spike_windows:
            window_df = df[
                (df[timestamp_col] >= window_start) &
                (df[timestamp_col] <= window_end)
            ]
            unique_senders = (
                window_df[from_col].unique().tolist()
                if from_col in window_df.columns else []
            )

            # ── Pump-and-dump check ───────────────────────────────────────────
            p = self._check_pump_dump(
                df, window_start, window_end,
                timestamp_col, value_col, from_col, to_col,
                spike_factor, unique_senders
            )
            if p:
                patterns.append(p)

        # ── Coordinated activity ──────────────────────────────────────────────
        coord = self._detect_coordinated_activity(df, timestamp_col, from_col, value_col)
        patterns.extend(coord)

        patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info(f"Market manipulation detector found {len(patterns)} patterns")
        return patterns

    # ─────────────────────────────────────────────────── Internals ────────────

    def _prep(self, df: pd.DataFrame, ts_col: str, val_col: str) -> pd.DataFrame:
        df = df.copy()
        if ts_col not in df.columns:
            if "timeStamp" in df.columns:
                df[ts_col] = pd.to_datetime(
                    pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", utc=True
                )
            else:
                return pd.DataFrame()

        if not pd.api.types.is_datetime64_any_dtype(df[ts_col]):
            df[ts_col] = pd.to_datetime(df[ts_col], utc=True)

        if val_col not in df.columns and "value" in df.columns:
            df[val_col] = df["value"] / 1e18

        return df.sort_values(ts_col)

    def _find_volume_spikes(
        self, df: pd.DataFrame, timestamp_col: str, value_col: str
    ) -> List[Tuple]:
        """Return (start, end, spike_factor) tuples for anomalous volume windows."""
        if value_col not in df.columns:
            return []

        df_ts = df.set_index(timestamp_col)
        resampled = df_ts[value_col].resample(self.time_window).sum()

        if len(resampled) < 4:
            return []

        rolling_mean = resampled.rolling(window=6, min_periods=1).mean()
        rolling_std = resampled.rolling(window=6, min_periods=1).std().fillna(0)

        spikes = []
        for ts, vol in resampled.items():
            mean = rolling_mean.get(ts, vol)
            std = rolling_std.get(ts, 0) + 1e-6
            z_score = (vol - mean) / std
            if z_score >= self.sigma_threshold:
                end_ts = ts
                start_ts = ts - pd.Timedelta(self.time_window)
                spikes.append((start_ts, end_ts, float(z_score)))

        return spikes

    def _check_pump_dump(
        self,
        df: pd.DataFrame,
        spike_start,
        spike_end,
        timestamp_col: str,
        value_col: str,
        from_col: str,
        to_col: str,
        spike_factor: float,
        spike_wallets: List[str],
    ) -> Optional[ManipulationPattern]:
        """
        After a volume spike (pump), check if the same wallets appear in
        a subsequent sell wave (dump) within pump_window.
        """
        if not spike_wallets:
            return None

        dump_end = spike_end + pd.Timedelta(self.pump_window)
        post_window = df[
            (df[timestamp_col] > spike_end) &
            (df[timestamp_col] <= dump_end)
        ]

        if post_window.empty or from_col not in post_window.columns:
            return None

        # Wallets that bought during spike and sold during post-window
        buy_wallets = set(spike_wallets)
        sell_wallets = set(post_window[from_col].dropna().tolist())

        overlap = buy_wallets & sell_wallets
        if len(overlap) < self.min_coord_wallets:
            return None

        # Confidence increases with overlap size and spike magnitude
        confidence = min(
            0.5 + 0.3 * (len(overlap) / max(len(buy_wallets), 1)) + 0.2 * (spike_factor / 10),
            0.99,
        )

        return ManipulationPattern(
            pattern_type="pump_dump",
            affected_wallets=list(overlap),
            start_time=str(spike_start),
            end_time=str(dump_end),
            volume_spike_factor=spike_factor,
            coordinated_wallet_count=len(overlap),
            confidence=confidence,
            details={
                "pump_volume_eth": round(
                    df[
                        (df[timestamp_col] >= spike_start) &
                        (df[timestamp_col] <= spike_end)
                    ][value_col].sum(), 4
                ) if value_col in df.columns else 0,
                "sigma_above_baseline": round(spike_factor, 2),
            },
        )

    def _detect_coordinated_activity(
        self, df: pd.DataFrame, timestamp_col: str, from_col: str, value_col: str
    ) -> List[ManipulationPattern]:
        """
        Detect groups of wallets that transact in near-simultaneous bursts
        within a short time window.
        """
        if from_col not in df.columns or timestamp_col not in df.columns:
            return []

        patterns = []
        df_ts = df.set_index(timestamp_col)

        # 30-minute windows
        for window_start in pd.date_range(
            df[timestamp_col].min(),
            df[timestamp_col].max(),
            freq="30min",
        ):
            window_end = window_start + pd.Timedelta("30min")
            window_df = df[
                (df[timestamp_col] >= window_start) &
                (df[timestamp_col] < window_end)
            ]

            unique_senders = window_df[from_col].dropna().unique()
            if len(unique_senders) < self.min_coord_wallets:
                continue

            # High density of senders in a short window = coordinated
            volume = float(window_df[value_col].sum()) if value_col in window_df.columns else 0
            tx_per_wallet = len(window_df) / len(unique_senders)

            if tx_per_wallet < 1.5:  # Each wallet sent ~1 tx — very coordinated
                confidence = min(0.5 + 0.4 * (len(unique_senders) / 20), 0.9)
                patterns.append(
                    ManipulationPattern(
                        pattern_type="coordinated_buy",
                        affected_wallets=unique_senders.tolist()[:50],
                        start_time=str(window_start),
                        end_time=str(window_end),
                        volume_spike_factor=0.0,
                        coordinated_wallet_count=len(unique_senders),
                        confidence=confidence,
                        details={"volume_eth": round(volume, 4), "txs_in_window": len(window_df)},
                    )
                )

        return patterns
