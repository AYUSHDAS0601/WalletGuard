"""
Unit tests for GNN, BiLSTM, and Ensemble models.
Tests run without real data — uses synthetic tensors.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── GNN tests ────────────────────────────────────────────────────────────────

class TestBlockchainGNN:
    def setup_method(self):
        from models.gnn.blockchain_gnn import BlockchainGNN
        self.model = BlockchainGNN(in_channels=64, hidden_channels=128, out_channels=32)
        self.model.eval()

    def test_forward_shape(self):
        """Output should be (N, 32) embeddings and (N, 1) probs."""
        N = 100
        x = torch.randn(N, 64)
        edge_index = torch.randint(0, N, (2, 200))
        emb, prob = self.model(x, edge_index)
        assert emb.shape == (N, 32), f"Expected (100, 32), got {emb.shape}"
        assert prob.shape == (N, 1), f"Expected (100, 1), got {prob.shape}"

    def test_output_range(self):
        """Probabilities must be in [0, 1]."""
        x = torch.randn(50, 64)
        edge_index = torch.randint(0, 50, (2, 100))
        _, prob = self.model(x, edge_index)
        assert (prob >= 0).all(), "Probabilities must be >= 0"
        assert (prob <= 1).all(), "Probabilities must be <= 1"

    def test_no_nan(self):
        """No NaN values in output."""
        x = torch.randn(30, 64)
        edge_index = torch.randint(0, 30, (2, 60))
        emb, prob = self.model(x, edge_index)
        assert not torch.isnan(emb).any(), "NaN in embeddings"
        assert not torch.isnan(prob).any(), "NaN in probabilities"

    def test_predict_method(self):
        """predict() returns 1D tensor of size N."""
        x = torch.randn(20, 64)
        edge_index = torch.randint(0, 20, (2, 40))
        scores = self.model.predict(x, edge_index)
        assert scores.shape == (20,)

    def test_save_load(self, tmp_path):
        """Model can be saved and reloaded with identical output."""
        from models.gnn.blockchain_gnn import BlockchainGNN
        path = str(tmp_path / "test_gnn.pth")
        self.model.save(path)
        loaded = BlockchainGNN.load(path)
        loaded.eval()

        x = torch.randn(10, 64)
        ei = torch.randint(0, 10, (2, 20))
        with torch.no_grad():
            _, p1 = self.model(x, ei)
            _, p2 = loaded(x, ei)

        assert torch.allclose(p1, p2, atol=1e-5), "Loaded model output mismatch"

    def test_empty_edges(self):
        """Model should handle graphs with no edges."""
        x = torch.randn(10, 64)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        emb, prob = self.model(x, edge_index)
        assert emb.shape == (10, 32)


# ── BiLSTM tests ─────────────────────────────────────────────────────────────

class TestTemporalAnomalyDetector:
    def setup_method(self):
        from models.temporal.bilstm_detector import TemporalAnomalyDetector
        self.model = TemporalAnomalyDetector(input_size=50, hidden_size=64, num_layers=2)
        self.model.eval()

    def test_forward_shape(self):
        """Probability output should be (B,)."""
        B, T, F = 16, 100, 50
        x = torch.randn(B, T, F)
        prob, attn = self.model(x)
        assert prob.shape == (B,), f"Expected ({B},), got {prob.shape}"

    def test_output_range(self):
        """Probabilities must be in [0, 1]."""
        x = torch.randn(8, 50, 50)
        prob, _ = self.model(x)
        assert (prob >= 0).all()
        assert (prob <= 1).all()

    def test_no_nan(self):
        """No NaN in output."""
        x = torch.randn(4, 100, 50)
        prob, attn = self.model(x)
        assert not torch.isnan(prob).any()
        assert not torch.isnan(attn).any()

    def test_padding_mask(self):
        """Padding-masked sequences should not crash."""
        from models.temporal.bilstm_detector import make_padding_mask
        x = torch.randn(8, 100, 50)
        x[:, :20, :] = 0.0   # Simulate left-padded zeros
        mask = make_padding_mask(x)
        prob, _ = self.model(x, src_key_padding_mask=mask)
        assert prob.shape == (8,)
        assert not torch.isnan(prob).any()

    def test_save_load(self, tmp_path):
        """Save and reload returns identical output."""
        from models.temporal.bilstm_detector import TemporalAnomalyDetector
        path = str(tmp_path / "test_bilstm.pth")
        self.model.save(path)
        loaded = TemporalAnomalyDetector.load(path)
        loaded.eval()

        x = torch.randn(4, 50, 50)
        with torch.no_grad():
            p1, _ = self.model(x)
            p2, _ = loaded(x)

        assert torch.allclose(p1, p2, atol=1e-5)


# ── Ensemble Scorer tests ─────────────────────────────────────────────────────

class TestEnsembleAnomalyScorer:
    def setup_method(self):
        from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer
        self.scorer = EnsembleAnomalyScorer()

    def test_predict_without_training(self):
        """predict() should return zeros gracefully when models not fitted."""
        X = np.random.randn(10, 25).astype(np.float32)
        scores = self.scorer.predict(X)
        assert len(scores) == 10
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_fit_and_predict(self):
        """fit then predict should return valid scores."""
        np.random.seed(42)
        X_train = np.random.randn(200, 25).astype(np.float32)
        y_train = (X_train[:, 0] > 0).astype(int)   # simple rule

        self.scorer.fit_traditional(X_train, y_train)

        X_test = np.random.randn(50, 25).astype(np.float32)
        scores = self.scorer.predict(X_test)
        assert scores.shape == (50,)
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_with_neural_scores(self):
        """predict() should combine neural scores correctly."""
        from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer
        scorer = EnsembleAnomalyScorer()
        X = np.random.randn(10, 25).astype(np.float32)
        gnn = np.full(10, 0.9, dtype=np.float32)
        temporal = np.full(10, 0.8, dtype=np.float32)
        scores = scorer.predict(X, gnn_scores=gnn, temporal_scores=temporal)
        assert (scores > 0.5).all(), "High neural scores should raise ensemble score"

    def test_risk_level(self):
        scorer = self.scorer
        assert scorer.risk_level(0.9) == "CRITICAL"
        assert scorer.risk_level(0.75) == "HIGH"
        assert scorer.risk_level(0.55) == "MEDIUM"
        assert scorer.risk_level(0.2) == "LOW"

    def test_save_load(self, tmp_path):
        """Fitted scorer can be saved and loaded."""
        from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer
        X = np.random.randn(100, 25).astype(np.float32)
        y = (np.random.rand(100) > 0.5).astype(int)
        self.scorer.fit_traditional(X, y)
        self.scorer.save(str(tmp_path))

        loaded = EnsembleAnomalyScorer.load(str(tmp_path))
        scores = loaded.predict(X)
        assert scores.shape == (100,)
