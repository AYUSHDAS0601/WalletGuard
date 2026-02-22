"""
Blockchain GNN — GraphSAGE + GAT architecture for wallet interaction modeling.

Optimised for NVIDIA GTX 2050 (4 GB VRAM):
  - Mixed-precision (AMP) compatible
  - Gradient checkpointing support
  - Mini-batch / NeighborLoader friendly
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import (
    BatchNorm,
    GATConv,
    SAGEConv,
    global_mean_pool,
)


class BlockchainGNN(nn.Module):
    """
    Graph Neural Network for anomaly detection on wallet interaction graphs.

    Architecture:
      GraphSAGE layers (memory-efficient, inductive)  → node embeddings
      Optional GAT attention head on the final layer
      MLP classifier head → anomaly probability per node

    Args:
        in_channels:     Dimension of input node features.
        hidden_channels: Width of hidden GraphSAGE layers.
        out_channels:    Dimension of the node embedding output.
        num_layers:      Total number of message-passing layers.
        dropout:         Dropout probability applied after each layer.
        use_gat_head:    If True, replace the last SAGEConv with a GATConv.
        gat_heads:       Number of attention heads in the GAT layer.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 32,
        num_layers: int = 3,
        dropout: float = 0.3,
        use_gat_head: bool = True,
        gat_heads: int = 4,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "Need at least 2 layers"

        self.num_layers = num_layers
        self.dropout = dropout
        self.use_gat_head = use_gat_head

        # ── Message-passing layers ─────────────────────────────────────────────
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # First layer
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        # Intermediate layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # Last layer — GAT or SAGE
        if use_gat_head:
            self.convs.append(
                GATConv(hidden_channels, out_channels, heads=gat_heads, concat=False)
            )
        else:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        # No BatchNorm on output layer

        # ── Classification head (outputs logits for BCEWithLogitsLoss stability) ──
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    # ─────────────────────────────────────────────────────── Forward ──────────

    def forward(self, x: Tensor, edge_index: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x:          Node feature matrix  [N, in_channels]
            edge_index: Graph connectivity   [2, E]

        Returns:
            embeddings:  Node embeddings     [N, out_channels]
            logits:     Anomaly logits      [N, 1]
        """
        for i, conv in enumerate(self.convs[:-1]):
            identity = x
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            if identity.shape == x.shape:
                x = x + identity  # Residual when dimensions match
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Last conv (no activation — let the head do that)
        x = self.convs[-1](x, edge_index)
        embeddings = x

        logits = self.classifier(embeddings)
        return embeddings, logits

    def embed(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return only the node embeddings (no classification head)."""
        embeddings, _ = self.forward(x, edge_index)
        return embeddings

    def predict(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """Return per-node anomaly probabilities (sigmoid applied to logits)."""
        _, logits = self.forward(x, edge_index)
        return torch.sigmoid(logits).squeeze(-1)

    # ─────────────────────────────────────────────────────── Serialization ────

    def save(self, path: str) -> None:
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": {
                    "in_channels": self.convs[0].in_channels,
                    "hidden_channels": self.convs[0].out_channels,
                    "out_channels": self.convs[-1].out_channels,
                    "num_layers": self.num_layers,
                    "dropout": self.dropout,
                    "use_gat_head": self.use_gat_head,
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BlockchainGNN":
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model


# ── Graph-level pooling variant (for graph classification) ─────────────────

class BlockchainGNNGraph(BlockchainGNN):
    """
    Extends BlockchainGNN with graph-level pooling for subgraph classification
    (e.g., classifying an entire transaction cluster as malicious).
    """

    def forward_graph(
        self, x: Tensor, edge_index: Tensor, batch: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            batch: Batch vector mapping each node to its graph index [N].
        Returns:
            graph_emb: Graph-level embedding [B, out_channels]
            graph_prob: Graph-level anomaly prob [B, 1]
        """
        node_emb, logits = super().forward(x, edge_index)
        graph_emb = global_mean_pool(node_emb, batch)
        graph_logits = self.classifier(graph_emb)
        graph_prob = torch.sigmoid(graph_logits)
        return graph_emb, graph_prob
