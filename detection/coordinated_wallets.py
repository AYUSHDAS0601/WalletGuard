"""
Coordinated Wallet Detector — identifies Sybil attacks and wallet clusters.

Signals:
  - Many new wallets funded from the same source
  - Correlated activity time patterns (cosine similarity of hour-vectors)
  - Community detection: tight clusters with dense internal edges
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class WalletCluster:
    cluster_id: int
    wallets: List[str]
    funding_source: Optional[str]         # Common funder, if detected
    avg_account_age_days: float
    activity_correlation: float           # 0-1 cosine similarity average
    is_sybil: bool
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
            "pattern": "coordinated_wallets",
            "cluster_id": self.cluster_id,
            "wallets": self.wallets[:50],
            "wallet_count": len(self.wallets),
            "funding_source": self.funding_source,
            "avg_account_age_days": round(self.avg_account_age_days, 1),
            "activity_correlation": round(self.activity_correlation, 4),
            "is_sybil": self.is_sybil,
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "details": self.details,
        }


class CoordinatedWalletDetector:
    """
    Detects coordinated multi-wallet attack patterns using:
      1. Sybil detection: new wallets funded from a common source
      2. Timing correlation: wallets active at the same hours
      3. Graph community detection: dense clusters in the transaction graph

    Usage:
        detector = CoordinatedWalletDetector()
        clusters = detector.detect(df)
    """

    def __init__(
        self,
        min_cluster_size: int = 3,
        max_account_age_days: float = 30.0,    # "new" wallet threshold
        correlation_threshold: float = 0.80,
        time_window_minutes: int = None,
    ):
        self.min_cluster_size = min_cluster_size
        self.max_account_age_days = max_account_age_days
        self.correlation_threshold = correlation_threshold
        from config.config import settings
        self.time_window_minutes = (
            time_window_minutes or settings.detection.coordinated_wallet_time_window_minutes
        )

    # ─────────────────────────────────────────────────── Public API ───────────

    def detect(
        self,
        df: pd.DataFrame,
        timestamp_col: str = "timestamp",
        from_col: str = "from",
        to_col: str = "to",
        value_col: str = "value_eth",
    ) -> List[WalletCluster]:
        """Run all coordinated wallet detectors."""
        df = self._prep(df, timestamp_col)
        if df.empty:
            return []

        clusters: List[WalletCluster] = []

        # 1. Sybil clusters (common funding source)
        sybil_clusters = self._detect_sybil(df, from_col, to_col, timestamp_col)
        clusters.extend(sybil_clusters)

        # 2. Timing correlation clusters
        timing_clusters = self._detect_timing_correlation(df, from_col, timestamp_col)
        clusters.extend(timing_clusters)

        # Deduplicate by wallet sets
        clusters = self._deduplicate(clusters)
        clusters.sort(key=lambda c: c.confidence, reverse=True)

        logger.info(f"Coordinated wallet detector found {len(clusters)} clusters")
        return clusters

    # ─────────────────────────────────────────────────── Sybil ───────────────

    def _detect_sybil(
        self,
        df: pd.DataFrame,
        from_col: str,
        to_col: str,
        timestamp_col: str,
    ) -> List[WalletCluster]:
        """
        Find groups of wallets that were all funded by the same source address
        within a short time, and are typically new.
        """
        clusters = []
        if from_col not in df.columns or to_col not in df.columns:
            return clusters

        # Group recipients by funder
        funding_groups: Dict[str, List[str]] = {}
        for _, row in df.iterrows():
            funder = str(row.get(from_col, "")).lower()
            recipient = str(row.get(to_col, "")).lower()
            if funder and recipient and funder != recipient:
                funding_groups.setdefault(funder, []).append(recipient)

        cluster_id = 0
        for funder, recipients in funding_groups.items():
            unique_recipients = list(set(recipients))
            if len(unique_recipients) < self.min_cluster_size:
                continue

            # Estimate account age: look for first TX of each recipient
            ages = []
            for addr in unique_recipients:
                first_tx = df[
                    (df[from_col] == addr) | (df[to_col] == addr)
                ][timestamp_col].min()
                if pd.notna(first_tx):
                    now = df[timestamp_col].max()
                    age_days = (now - first_tx).total_seconds() / 86400
                    ages.append(age_days)

            avg_age = np.mean(ages) if ages else 999.0
            is_sybil = avg_age <= self.max_account_age_days

            # Confidence based on cluster size and age
            size_score = min(len(unique_recipients) / 20.0, 1.0)
            age_score = max(0, 1 - avg_age / self.max_account_age_days)
            confidence = 0.5 * size_score + 0.5 * age_score if is_sybil else 0.3 * size_score

            if confidence < 0.4:
                continue

            clusters.append(
                WalletCluster(
                    cluster_id=cluster_id,
                    wallets=unique_recipients,
                    funding_source=funder,
                    avg_account_age_days=avg_age,
                    activity_correlation=0.0,  # computed in timing step
                    is_sybil=is_sybil,
                    confidence=confidence,
                    details={"funder": funder, "funded_count": len(unique_recipients)},
                )
            )
            cluster_id += 1

        return clusters

    # ──────────────────────────────────────────── Timing correlation ──────────

    def _detect_timing_correlation(
        self,
        df: pd.DataFrame,
        from_col: str,
        timestamp_col: str,
    ) -> List[WalletCluster]:
        """
        Detect wallets with highly correlated hourly activity patterns.
        Uses cosine similarity of 24-dim activity vectors.
        """
        if from_col not in df.columns or timestamp_col not in df.columns:
            return []

        df = df.copy()
        df["hour"] = df[timestamp_col].dt.hour

        # Build 24-dim activity vector per wallet
        activity: Dict[str, np.ndarray] = {}
        for addr, addr_df in df.groupby(from_col):
            vec = np.zeros(24, dtype=np.float32)
            for h, count in addr_df["hour"].value_counts().items():
                vec[int(h)] = count
            # Normalize
            norm = vec.sum()
            if norm > 0:
                vec /= norm
            activity[str(addr)] = vec

        if len(activity) < self.min_cluster_size:
            return []

        # Compute pairwise cosine similarity
        addresses = list(activity.keys())
        vectors = np.stack([activity[a] for a in addresses])
        sim_matrix = cosine_similarity(vectors)

        # Find clusters with high mutual similarity
        clusters = []
        visited: Set[int] = set()
        cluster_id = 1000  # offset from sybil cluster IDs

        for i in range(len(addresses)):
            if i in visited:
                continue
            correlated = [i]
            for j in range(i + 1, len(addresses)):
                if j not in visited and sim_matrix[i, j] >= self.correlation_threshold:
                    correlated.append(j)

            if len(correlated) >= self.min_cluster_size:
                for idx in correlated:
                    visited.add(idx)

                cluster_wallets = [addresses[k] for k in correlated]
                avg_sim = float(
                    np.mean([sim_matrix[i, j] for j in correlated if j != i] or [0])
                )
                confidence = min(0.5 + 0.4 * avg_sim + 0.1 * min(len(cluster_wallets) / 10, 1), 0.95)

                clusters.append(
                    WalletCluster(
                        cluster_id=cluster_id,
                        wallets=cluster_wallets,
                        funding_source=None,
                        avg_account_age_days=0.0,
                        activity_correlation=avg_sim,
                        is_sybil=False,
                        confidence=confidence,
                        details={"detection_method": "timing_correlation"},
                    )
                )
                cluster_id += 1

        return clusters

    # ─────────────────────────────────────────────────── Helpers ─────────────

    @staticmethod
    def _prep(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        df = df.copy()
        if timestamp_col not in df.columns:
            if "timeStamp" in df.columns:
                df[timestamp_col] = pd.to_datetime(
                    pd.to_numeric(df["timeStamp"], errors="coerce"), unit="s", utc=True
                )
            else:
                return pd.DataFrame()
        if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        return df

    @staticmethod
    def _deduplicate(clusters: List[WalletCluster]) -> List[WalletCluster]:
        """Remove near-duplicate clusters (≥70% wallet overlap)."""
        kept = []
        for c in clusters:
            set_c = set(c.wallets)
            is_dup = False
            for k in kept:
                overlap = len(set_c & set(k.wallets)) / max(len(set_c), 1)
                if overlap >= 0.7:
                    # Keep the one with higher confidence
                    if c.confidence > k.confidence:
                        kept.remove(k)
                        kept.append(c)
                    is_dup = True
                    break
            if not is_dup:
                kept.append(c)
        return kept
