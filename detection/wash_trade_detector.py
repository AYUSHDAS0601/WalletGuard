"""
Wash Trade Detector — identifies circular transaction patterns between wallets.

Pattern: A → B → C → A (with minimal price variance + short time window)
Signals artificially inflated trading volume.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd
from loguru import logger

from config.config import settings


@dataclass
class WashTradePattern:
    cycle: List[str]                    # Wallet addresses forming the ring
    total_volume_eth: float
    tx_count: int
    time_span_hours: float
    confidence: float                   # 0-1
    tx_hashes: List[str] = field(default_factory=list)

    @property
    def risk_level(self) -> str:
        if self.confidence >= 0.85:
            return "CRITICAL"
        if self.confidence >= 0.70:
            return "HIGH"
        if self.confidence >= 0.50:
            return "MEDIUM"
        return "LOW"

    def to_dict(self) -> dict:
        return {
            "pattern": "wash_trading",
            "cycle": self.cycle,
            "cycle_length": len(self.cycle),
            "total_volume_eth": round(self.total_volume_eth, 6),
            "tx_count": self.tx_count,
            "time_span_hours": round(self.time_span_hours, 2),
            "confidence": round(self.confidence, 4),
            "risk_level": self.risk_level,
            "tx_hashes": self.tx_hashes,
        }


class WashTradeDetector:
    """
    Detects wash trading by finding directed cycles in the transaction graph
    and scoring them based on:
      - Cycle length (shorter = more suspicious)
      - Value symmetry (near-equal values in/out = more suspicious)
      - Time compactness (tighter window = more suspicious)
      - Volume significance

    Usage:
        detector = WashTradeDetector()
        patterns = detector.detect(df)           # from DataFrame
        patterns = detector.detect_from_graph(G) # from NetworkX graph
    """

    def __init__(
        self,
        max_cycle_depth: int = None,
        time_window_hours: float = 24.0,
        min_volume_eth: float = 0.01,
    ):
        self.max_cycle_depth = max_cycle_depth or settings.detection.wash_trade_cycle_depth
        self.time_window_hours = time_window_hours
        self.min_volume_eth = min_volume_eth

    # ─────────────────────────────────────────────────── Public API ───────────

    def detect(
        self,
        df: pd.DataFrame,
        from_col: str = "from",
        to_col: str = "to",
        value_col: str = "value_eth",
        timestamp_col: str = "timestamp",
        hash_col: str = "hash",
    ) -> List[WashTradePattern]:
        """Detect wash trading from a transaction DataFrame."""
        from data.processors.graph_builder import GraphBuilder

        if df.empty:
            return []

        builder = GraphBuilder()
        G = builder.build_networkx(df, from_col, to_col, value_col, timestamp_col, hash_col)

        # Attach timestamps to nodes for time-window filtering
        node_first_seen = {}
        node_last_seen = {}
        for _, row in df.iterrows():
            for addr in [row.get(from_col), row.get(to_col)]:
                if addr is None:
                    continue
                ts = row.get(timestamp_col)
                if ts is not None:
                    if addr not in node_first_seen or ts < node_first_seen[addr]:
                        node_first_seen[addr] = ts
                    if addr not in node_last_seen or ts > node_last_seen[addr]:
                        node_last_seen[addr] = ts

        return self.detect_from_graph(G, node_first_seen, node_last_seen)

    def detect_from_graph(
        self,
        G: nx.DiGraph,
        node_first_seen: Optional[Dict] = None,
        node_last_seen: Optional[Dict] = None,
    ) -> List[WashTradePattern]:
        """Detect cycles in a pre-built NetworkX DiGraph."""
        if G.number_of_nodes() == 0:
            return []

        cycles = self._find_cycles(G)
        logger.info(f"Found {len(cycles)} candidate cycles")

        patterns = []
        for cycle in cycles:
            pattern = self._score_cycle(G, cycle, node_first_seen, node_last_seen)
            if pattern is not None:
                patterns.append(pattern)

        # Sort by confidence descending
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        logger.info(f"Detected {len(patterns)} wash trade patterns")
        return patterns

    # ─────────────────────────────────────────────────── Internals ────────────

    def _find_cycles(self, G: nx.DiGraph) -> List[List[str]]:
        """Find all simple directed cycles between 3 and max_depth nodes."""
        result = []
        try:
            for cycle in nx.simple_cycles(G):
                if 3 <= len(cycle) <= self.max_cycle_depth:
                    result.append(cycle)
                    if len(result) > 10_000:   # Safety cap
                        logger.warning("Too many cycles — truncating at 10,000")
                        break
        except Exception as e:
            logger.warning(f"Cycle detection error: {e}")
        return result

    def _score_cycle(
        self,
        G: nx.DiGraph,
        cycle: List[str],
        node_first_seen: Optional[Dict],
        node_last_seen: Optional[Dict],
    ) -> Optional[WashTradePattern]:
        """
        Score a cycle and return a WashTradePattern if suspicious enough.
        Returns None if the cycle does not meet minimum criteria.
        """
        # Collect edge attributes for the cycle edges
        edges_in_cycle = []
        tx_hashes = []
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            if not G.has_edge(u, v):
                return None
            edge_data = G[u][v]
            edges_in_cycle.append(edge_data)
            tx_hashes.extend(edge_data.get("tx_hashes", []))

        # ── Feature 1: Value symmetry ─────────────────────────────────────────
        volumes = [e.get("weight", 0) for e in edges_in_cycle]
        total_volume = sum(volumes)
        if total_volume < self.min_volume_eth:
            return None

        if max(volumes) > 0:
            symmetry_score = min(volumes) / max(volumes)  # 1.0 = perfectly symmetric
        else:
            return None

        # ── Feature 2: Time compactness ───────────────────────────────────────
        time_score = 0.5  # default if no timestamps
        time_span_hours = 0.0
        if node_first_seen and node_last_seen:
            first_times = [node_first_seen.get(n) for n in cycle if node_first_seen.get(n)]
            last_times = [node_last_seen.get(n) for n in cycle if node_last_seen.get(n)]
            if first_times and last_times:
                span = max(last_times) - min(first_times)
                if hasattr(span, "total_seconds"):
                    time_span_hours = span.total_seconds() / 3600
                else:
                    time_span_hours = float(span) / 3600

                if time_span_hours > self.time_window_hours:
                    time_score = 0.3   # less suspicious if spread over a long period
                elif time_span_hours < 1:
                    time_score = 1.0   # very tight window = very suspicious
                else:
                    time_score = 1 - (time_span_hours / self.time_window_hours)

        # ── Feature 3: Cycle length (shorter = more suspicious) ───────────────
        length_score = 1 - (len(cycle) - 3) / max(self.max_cycle_depth - 3, 1)
        length_score = max(0, min(1, length_score))

        # ── Composite confidence ──────────────────────────────────────────────
        confidence = 0.5 * symmetry_score + 0.3 * time_score + 0.2 * length_score

        if confidence < 0.4:  # Prune low-confidence detections
            return None

        return WashTradePattern(
            cycle=cycle,
            total_volume_eth=total_volume,
            tx_count=sum(e.get("tx_count", 1) for e in edges_in_cycle),
            time_span_hours=time_span_hours,
            confidence=float(confidence),
            tx_hashes=tx_hashes[:20],  # limit list size
        )
