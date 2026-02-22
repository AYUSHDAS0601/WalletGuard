"""
Detector integration tests — use synthetic data to validate detection logic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def make_wash_trade_df() -> pd.DataFrame:
    """Create a synthetic A→B→C→A wash trade dataset."""
    wallets = ["0xaaaa", "0xbbbb", "0xcccc"]
    base_time = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    records = [
        {"from": "0xaaaa", "to": "0xbbbb", "value_eth": 1.0, "hash": "0xhash1",
         "blockNumber": 1001, "timestamp": base_time},
        {"from": "0xbbbb", "to": "0xcccc", "value_eth": 1.0, "hash": "0xhash2",
         "blockNumber": 1002, "timestamp": base_time + pd.Timedelta(minutes=5)},
        {"from": "0xcccc", "to": "0xaaaa", "value_eth": 1.0, "hash": "0xhash3",
         "blockNumber": 1003, "timestamp": base_time + pd.Timedelta(minutes=10)},
    ]
    return pd.DataFrame(records)


def make_flash_loan_df() -> pd.DataFrame:
    """Create a synthetic multi-protocol transaction block."""
    base_time = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    records = [
        {
            "from": "0xattacker", "to": "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",
            "value_eth": 100.0, "hash": "0xfl1", "blockNumber": 1001,
            "transactionIndex": 1, "timestamp": base_time,
        },
        {
            "from": "0xattacker", "to": "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",
            "value_eth": 100.0, "hash": "0xfl2", "blockNumber": 1001,
            "transactionIndex": 2, "timestamp": base_time,
        },
        {
            "from": "0xattacker", "to": "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
            "value_eth": 100.0, "hash": "0xfl3", "blockNumber": 1001,
            "transactionIndex": 3, "timestamp": base_time,
        },
    ]
    return pd.DataFrame(records)


def make_coordinated_df() -> pd.DataFrame:
    """Create a synthetic common-funder Sybil wallet dataset."""
    base_time = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
    records = []
    for i in range(1, 6):
        records.append({
            "from": "0xfunder",
            "to": f"0xsybil{i:04d}",
            "value_eth": 0.1,
            "hash": f"0xseed{i}",
            "blockNumber": 1000 + i,
            "timestamp": base_time + pd.Timedelta(minutes=i),
        })
        # Each sybil also transacts
        records.append({
            "from": f"0xsybil{i:04d}",
            "to": "0xtarget",
            "value_eth": 0.05,
            "hash": f"0xact{i}",
            "blockNumber": 1010 + i,
            "timestamp": base_time + pd.Timedelta(hours=1, minutes=i),
        })
    return pd.DataFrame(records)


class TestWashTradeDetector:
    def test_detects_circular_pattern(self):
        from detection.wash_trade_detector import WashTradeDetector
        detector = WashTradeDetector()
        df = make_wash_trade_df()
        patterns = detector.detect(df)
        assert len(patterns) > 0, "Expected at least one wash trade pattern"
        cycle_lens = [len(p.cycle) for p in patterns]
        assert 3 in cycle_lens, f"Expected 3-node cycle, got {cycle_lens}"

    def test_no_false_positive_linear(self):
        """A simple linear A→B→C flow should NOT trigger wash trade."""
        from detection.wash_trade_detector import WashTradeDetector
        base_time = pd.Timestamp("2024-01-15", tz="UTC")
        df = pd.DataFrame([
            {"from": "0xaa", "to": "0xbb", "value_eth": 1.0, "hash": "0x1",
             "blockNumber": 1, "timestamp": base_time},
            {"from": "0xbb", "to": "0xcc", "value_eth": 1.0, "hash": "0x2",
             "blockNumber": 2, "timestamp": base_time + pd.Timedelta(minutes=5)},
        ])
        detector = WashTradeDetector()
        patterns = detector.detect(df)
        assert len(patterns) == 0, f"False positive: {patterns}"


class TestFlashLoanDetector:
    def test_detects_multi_protocol(self):
        from detection.flash_loan_detector import FlashLoanDetector
        detector = FlashLoanDetector(min_protocols=2, min_value_eth=0.5)
        df = make_flash_loan_df()
        patterns = detector.detect(df)
        types = [p.pattern_type for p in patterns]
        assert "flash_loan" in types, f"Expected flash_loan, got {types}"

    def test_empty_df_returns_empty(self):
        from detection.flash_loan_detector import FlashLoanDetector
        patterns = FlashLoanDetector().detect(pd.DataFrame())
        assert patterns == []


class TestCoordinatedWalletDetector:
    def test_detects_sybil_cluster(self):
        from detection.coordinated_wallets import CoordinatedWalletDetector
        detector = CoordinatedWalletDetector(min_cluster_size=3)
        df = make_coordinated_df()
        clusters = detector.detect(df)
        assert len(clusters) > 0, "Expected at least one Sybil cluster"
        cluster_wallets = [w for c in clusters for w in c.wallets]
        assert any("sybil" in w for w in cluster_wallets), "Sybil wallets not in cluster"

    def test_empty_df_returns_empty(self):
        from detection.coordinated_wallets import CoordinatedWalletDetector
        clusters = CoordinatedWalletDetector().detect(pd.DataFrame())
        assert clusters == []


class TestDetectionPipeline:
    def test_analyze_mock_returns_result(self):
        """Mock analysis should return a valid DetectionResult."""
        from detection.pipeline import DetectionPipeline
        pipeline = DetectionPipeline(load_models=False)
        result = pipeline.analyze_mock()

        assert result.anomaly_score >= 0.0
        assert result.anomaly_score <= 1.0
        assert result.risk_level in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        assert isinstance(result.detected_patterns, list)
        assert isinstance(result.explanations, dict)
        assert "summary" in result.explanations

    def test_analyze_dataframe_with_wash_trade(self):
        """DataFrame with circular trades should be flagged."""
        from detection.pipeline import DetectionPipeline
        pipeline = DetectionPipeline(load_models=False)
        df = make_wash_trade_df()
        result = pipeline.analyze_dataframe(df, address="0xaaaa")
        # Should detect wash trading
        pattern_types = " ".join(result.pattern_types)
        assert "wash_trading" in pattern_types or result.anomaly_score > 0.1
