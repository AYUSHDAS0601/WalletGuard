"""
Graph Builder — converts transaction DataFrames to NetworkX + PyTorch Geometric graphs.

Key outputs:
  - NetworkX DiGraph for algorithmic analysis (cycles, centrality, community)
  - PyTorch Geometric Data object for GNN training
  - Pre-computed graph metrics per node
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


class GraphBuilder:
    """
    Converts a DataFrame of transactions into graphs suitable for
    NetworkX analysis and PyTorch Geometric (GNN) training.

    Usage:
        builder = GraphBuilder()
        G          = builder.build_networkx(df)
        pyg_data   = builder.build_pyg(df, node_features)
        metrics_df = builder.compute_node_metrics(G)
    """

    def __init__(self, min_edge_weight: float = 0.0):
        """
        Args:
            min_edge_weight: Minimum ETH value to include an edge (filters dust TXs).
        """
        self.min_edge_weight = min_edge_weight

    # ───────────────────────────────────────────────────── NetworkX graph ────

    def build_networkx(
        self,
        df: pd.DataFrame,
        from_col: str = "from",
        to_col: str = "to",
        value_col: str = "value_eth",
        timestamp_col: str = "timestamp",
        hash_col: str = "hash",
    ) -> nx.DiGraph:
        """
        Build a directed transaction graph.

        Nodes  = wallet addresses
        Edges  = transactions (with weight, timestamp, tx_hash attributes)
        Multi-edges between the same pair are collapsed: weight = sum of all ETH,
        tx_count = number of transactions.
        """
        G = nx.DiGraph()

        if df.empty:
            return G

        # Ensure value column exists
        if value_col not in df.columns and "value" in df.columns:
            df = df.copy()
            df[value_col] = df["value"] / 1e18

        for _, row in df.iterrows():
            src = str(row.get(from_col, "")).lower().strip()
            dst = str(row.get(to_col, "")).lower().strip()
            val = float(row.get(value_col, 0) or 0)
            ts = row.get(timestamp_col)
            tx_hash = str(row.get(hash_col, ""))

            if not src or not dst or val < self.min_edge_weight:
                continue

            if G.has_edge(src, dst):
                G[src][dst]["weight"] += val
                G[src][dst]["tx_count"] += 1
                G[src][dst]["tx_hashes"].append(tx_hash)
            else:
                G.add_edge(
                    src,
                    dst,
                    weight=val,
                    tx_count=1,
                    tx_hashes=[tx_hash],
                    first_seen=ts,
                    last_seen=ts,
                )

        logger.info(
            f"Graph built — nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}"
        )
        return G

    # ───────────────────────────────────────── PyTorch Geometric conversion ──

    def build_pyg(
        self,
        df: pd.DataFrame,
        node_features: Optional[Dict[str, np.ndarray]] = None,
        feature_dim: int = 64,
        **nx_kwargs,
    ) -> Data:
        """
        Build a PyTorch Geometric Data object from transaction data.

        Args:
            df:            Transaction DataFrame.
            node_features: Optional dict {address: feature_vector (np.array)}.
                           If None, random features are used as placeholder.
            feature_dim:   Dimensionality of node feature vectors.

        Returns:
            PyG Data object with .x, .edge_index, .edge_attr, .node_ids
        """
        G = self.build_networkx(df, **nx_kwargs)

        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, returning placeholder Data object")
            return Data(
                x=torch.zeros((1, feature_dim)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
            )

        nodes = list(G.nodes())
        node_to_idx = {n: i for i, n in enumerate(nodes)}
        n = len(nodes)

        # ── Node features ─────────────────────────────────────────────────────
        if node_features is not None:
            x_list = [
                node_features.get(node, np.zeros(feature_dim)) for node in nodes
            ]
            x = torch.tensor(np.array(x_list, dtype=np.float32), dtype=torch.float)
        else:
            # Placeholder: will be replaced by proper feature engineering
            x = torch.zeros((n, feature_dim), dtype=torch.float)

        # ── Edge index + attributes ───────────────────────────────────────────
        src_list, dst_list, weights, tx_counts = [], [], [], []
        for u, v, data in G.edges(data=True):
            src_list.append(node_to_idx[u])
            dst_list.append(node_to_idx[v])
            weights.append(data.get("weight", 0.0))
            tx_counts.append(data.get("tx_count", 1))

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr = torch.tensor(
            np.column_stack([weights, tx_counts]).astype(np.float32),
            dtype=torch.float,
        )

        pyg_data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_ids=nodes,
        )
        logger.info(f"PyG Data — nodes: {pyg_data.num_nodes}, edges: {pyg_data.num_edges}")
        return pyg_data

    # ──────────────────────────────────────────────── Node metrics ────────────

    def compute_node_metrics(
        self, G: nx.DiGraph, top_k: int = 1000
    ) -> pd.DataFrame:
        """
        Compute per-node graph metrics used as features.

        For large graphs (> top_k nodes), computes metrics on a sampled subgraph
        of the highest-degree nodes to stay within memory budget.
        """
        if G.number_of_nodes() == 0:
            return pd.DataFrame()

        # Sample large graphs
        if G.number_of_nodes() > top_k:
            top_nodes = sorted(
                G.degree(), key=lambda x: x[1], reverse=True
            )[:top_k]
            G = G.subgraph([n for n, _ in top_nodes]).copy()

        nodes = list(G.nodes())
        logger.info(f"Computing metrics for {len(nodes)} nodes...")

        # Degree
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())

        # PageRank
        try:
            pagerank = nx.pagerank(G, max_iter=100)
        except Exception:
            pagerank = {n: 0.0 for n in nodes}

        # Clustering (on undirected view)
        G_undirected = G.to_undirected()
        clustering = nx.clustering(G_undirected)

        # Betweenness (expensive — skip for very large graphs)
        if len(nodes) <= 500:
            betweenness = nx.betweenness_centrality(G, normalized=True)
        else:
            betweenness = {n: 0.0 for n in nodes}

        # Community detection (Louvain via python-louvain)
        community_map = self._detect_communities(G_undirected)

        records = []
        for node in nodes:
            records.append({
                "address": node,
                "in_degree": in_degree.get(node, 0),
                "out_degree": out_degree.get(node, 0),
                "total_degree": in_degree.get(node, 0) + out_degree.get(node, 0),
                "pagerank_score": pagerank.get(node, 0.0),
                "clustering_coefficient": clustering.get(node, 0.0),
                "betweenness_centrality": betweenness.get(node, 0.0),
                "community_id": community_map.get(node, -1),
            })

        metrics_df = pd.DataFrame(records).set_index("address")
        logger.info(f"Graph metrics computed — shape {metrics_df.shape}")
        return metrics_df

    def detect_cycles(
        self, G: nx.DiGraph, max_length: int = 5
    ) -> List[List[str]]:
        """
        Find all simple directed cycles up to max_length.
        Used for wash-trade detection.
        """
        cycles = []
        try:
            for cycle in nx.simple_cycles(G):
                if 3 <= len(cycle) <= max_length:
                    cycles.append(cycle)
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")
        return cycles

    def get_subgraph(
        self,
        G: nx.DiGraph,
        center: str,
        depth: int = 2,
        min_weight: float = 0.0,
    ) -> nx.DiGraph:
        """Extract a depth-limited ego-network around a center address."""
        center = center.lower().strip()
        if center not in G:
            return nx.DiGraph()

        reachable: set = {center}
        frontier = {center}
        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(G.successors(node))
                next_frontier.update(G.predecessors(node))
            frontier = next_frontier - reachable
            reachable.update(frontier)

        sub = G.subgraph(reachable).copy()

        # Filter by edge weight
        if min_weight > 0:
            edges_to_remove = [
                (u, v) for u, v, d in sub.edges(data=True)
                if d.get("weight", 0) < min_weight
            ]
            sub.remove_edges_from(edges_to_remove)

        return sub

    # ─────────────────────────────────────────────────────── Helpers ─────────

    @staticmethod
    def _detect_communities(G_undirected: nx.Graph) -> Dict[str, int]:
        """Apply Louvain community detection. Falls back to stub if not available."""
        try:
            import community as community_louvain
            return community_louvain.best_partition(G_undirected)
        except ImportError:
            logger.warning(
                "python-louvain not installed. Community detection disabled. "
                "Install with: pip install python-louvain"
            )
            return {}
        except Exception as e:
            logger.warning(f"Community detection failed: {e}")
            return {}

    @staticmethod
    def to_serializable(G: nx.DiGraph) -> Dict:
        """Convert graph to JSON-serializable dict for API responses."""
        nodes = [
            {
                "id": n,
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
            }
            for n in G.nodes()
        ]
        edges = [
            {
                "source": u,
                "target": v,
                "weight": round(d.get("weight", 0), 6),
                "tx_count": d.get("tx_count", 1),
            }
            for u, v, d in G.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}
