"""
Temporal Anomaly Detector — Bidirectional LSTM with Multi-Head Attention.

Models transaction sequence behavior per wallet to detect temporal patterns
characteristic of wash trading, pump-and-dump, and coordinated attacks.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TemporalAnomalyDetector(nn.Module):
    """
    BiLSTM + Multi-head Attention for per-wallet transaction sequence anomaly detection.

    Input:  (batch, seq_len, input_size)  — time-ordered transaction feature sequences
    Output: anomaly probability (batch,)

    Architecture:
      BiLSTM (2 layers)  →  MultiheadAttention  →  mean pool  →  FC head  →  Sigmoid

    Args:
        input_size:   Number of per-transaction features.
        hidden_size:  LSTM hidden state size (each direction).
        num_layers:   Number of stacked LSTM layers.
        dropout:      Dropout applied between LSTM layers and in FC head.
        num_heads:    Attention heads (must divide hidden_size * 2).
        max_seq_len:  Expected maximum sequence length (for positional info).
    """

    def __init__(
        self,
        input_size: int = 50,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_heads: int = 4,
        max_seq_len: int = 100,
    ) -> None:
        super().__init__()

        self.hidden_size = hidden_size
        self.bidirectional_dim = hidden_size * 2  # because bidirectional

        # ── Input projection (handles variable input_size) ────────────────────
        self.input_proj = nn.Linear(input_size, hidden_size)

        # ── BiLSTM ────────────────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        # ── Multi-head Self-Attention ─────────────────────────────────────────
        self.attention = nn.MultiheadAttention(
            embed_dim=self.bidirectional_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(self.bidirectional_dim)

        # ── Classification head ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(self.bidirectional_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    # ─────────────────────────────────────────────────────── Forward ──────────

    def forward(
        self,
        x: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:                    Input sequence  [B, T, input_size]
            src_key_padding_mask: Boolean mask    [B, T]  — True means padded (ignore)

        Returns:
            prob:         Anomaly probability    [B]
            attn_weights: Attention weight map   [B, T, T]
        """
        # Project inputs to LSTM dimension
        x = self.input_proj(x)                        # [B, T, H]

        # BiLSTM
        lstm_out, _ = self.lstm(x)                    # [B, T, H*2]

        # Self-attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out,
            key_padding_mask=src_key_padding_mask,
        )                                              # [B, T, H*2]

        # Residual + LayerNorm
        attn_out = self.layer_norm(lstm_out + attn_out)

        # Mean pooling over time (ignoring padded positions if mask provided)
        if src_key_padding_mask is not None:
            # Zero out padded tokens before mean
            pad_mask = (~src_key_padding_mask).float().unsqueeze(-1)  # [B, T, 1]
            pooled = (attn_out * pad_mask).sum(dim=1) / pad_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = attn_out.mean(dim=1)              # [B, H*2]

        prob = self.classifier(pooled).squeeze(-1)     # [B]
        return prob, attn_weights

    def predict(self, x: Tensor) -> Tensor:
        """Convenience method — returns anomaly probability only."""
        prob, _ = self.forward(x)
        return prob

    # ─────────────────────────────────────────────────────── Serialization ────

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "input_size": self.input_proj.in_features,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.lstm.num_layers,
                    "dropout": self.lstm.dropout,
                    "num_heads": self.attention.num_heads,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "TemporalAnomalyDetector":
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


# ── Padding mask builder helper ────────────────────────────────────────────

def make_padding_mask(sequences: Tensor, pad_value: float = 0.0) -> Tensor:
    """
    Create a boolean padding mask for sequences zero-padded from the left.

    Args:
        sequences: [B, T, F]
        pad_value: Value used for padding (default 0.0)

    Returns:
        mask: [B, T]  — True where positions are padding
    """
    # A position is padding if ALL features equal pad_value
    return (sequences == pad_value).all(dim=-1)
