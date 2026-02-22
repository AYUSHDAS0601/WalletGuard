"""
API endpoint integration tests using FastAPI TestClient.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="module")
def client():
    """Create a TestClient with a mocked DetectionPipeline."""
    from api.main import app
    from detection.pipeline import DetectionResult

    # Mock the pipeline so tests don't need real data/models
    mock_pipeline = MagicMock()
    mock_result = DetectionResult(
        address="0xaaaa000000000000000000000000000000000001",
        tx_hash="0x" + "a" * 64,
        anomaly_score=0.75,
        risk_level="HIGH",
        detected_patterns=[{
            "pattern": "wash_trading",
            "cycle": ["0xaaaa", "0xbbbb", "0xcccc"],
            "confidence": 0.82,
        }],
        pattern_types=["wash_trading"],
        gnn_score=0.7,
        temporal_score=0.8,
        ensemble_score=0.75,
        explanations={"summary": "Circular wash-trading detected.", "top_risk_factors": []},
        graph_summary={"num_wallets": 3, "num_transactions": 3},
    )
    mock_pipeline.analyze_transaction.return_value = mock_result
    mock_pipeline.analyze_wallet.return_value = mock_result
    mock_pipeline.analyze_mock.return_value = mock_result
    mock_pipeline.gnn_model = None
    mock_pipeline.bilstm_model = None
    mock_pipeline.scorer = MagicMock(_xgb_fitted=False)

    # Mock graph query
    from data.processors.graph_builder import GraphBuilder
    mock_pipeline.graph_builder = GraphBuilder()

    # Mock etherscan
    mock_pipeline.etherscan = MagicMock()
    mock_pipeline.etherscan.get_wallet_transactions.return_value = __import__("pandas").DataFrame()

    app.state.pipeline = mock_pipeline

    with TestClient(app) as client:
        yield client


class TestHealthEndpoints:
    def test_root(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "name" in r.json()

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "gnn_loaded" in body
        assert "bilstm_loaded" in body


class TestAnalysisRoutes:
    VALID_TX_HASH = "0x" + "a" * 64
    VALID_WALLET = "0x" + "a" * 40

    def test_analyze_transaction_success(self, client):
        r = client.post(
            "/api/v1/analyze/transaction",
            json={"tx_hash": self.VALID_TX_HASH, "blockchain": "ethereum"},
        )
        assert r.status_code == 200
        body = r.json()
        assert "anomaly_score" in body
        assert "risk_level" in body
        assert "detected_patterns" in body
        assert 0 <= body["anomaly_score"] <= 1

    def test_analyze_transaction_invalid_hash(self, client):
        r = client.post(
            "/api/v1/analyze/transaction",
            json={"tx_hash": "invalid_hash", "blockchain": "ethereum"},
        )
        assert r.status_code == 422   # Validation error

    def test_analyze_wallet_success(self, client):
        r = client.get(f"/api/v1/analyze/wallet/{self.VALID_WALLET}")
        assert r.status_code == 200
        body = r.json()
        assert "address" in body
        assert "risk_level" in body
        assert "anomaly_score" in body

    def test_analyze_wallet_invalid_address(self, client):
        r = client.get("/api/v1/analyze/wallet/not_an_address")
        assert r.status_code == 400


class TestSearchRoutes:
    def test_search_patterns_valid(self, client):
        r = client.post(
            "/api/v1/search/patterns",
            json={"pattern_type": "wash_trading", "min_confidence": 0.5},
        )
        assert r.status_code == 200
        body = r.json()
        assert "total_matches" in body
        assert "results" in body

    def test_search_invalid_pattern_type(self, client):
        r = client.post(
            "/api/v1/search/patterns",
            json={"pattern_type": "invalid_type"},
        )
        assert r.status_code == 400


class TestGraphRoute:
    VALID_WALLET = "0x" + "a" * 40

    def test_graph_query_returns_structure(self, client):
        r = client.post(
            "/api/v1/graph/query",
            json={"center_address": self.VALID_WALLET, "depth": 2},
        )
        assert r.status_code == 200
        body = r.json()
        assert "nodes" in body
        assert "edges" in body
        assert "node_count" in body
