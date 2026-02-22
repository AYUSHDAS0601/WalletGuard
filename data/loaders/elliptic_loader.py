"""
Elliptic Dataset Loader — Kaggle's labeled Bitcoin transaction graph.

Dataset structure:
  elliptic_txs_features.csv  → 166 node features (94 local + 72 aggregated)
  elliptic_txs_classes.csv   → label: 1=illicit, 2=licit, unknown
  elliptic_txs_edgelist.csv  → directed edges (txid1, txid2)

We load these into a PyTorch Geometric Data object suitable for GNN training.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

LABEL_MAP = {"1": 1, "2": 0, "unknown": -1}  # illicit=1 licit=0 unknown=-1


class EllipticLoader:
    """
    Loads the Elliptic dataset and produces a PyTorch Geometric Data object.

    Usage:
        loader = EllipticLoader(data_dir="./data/raw/elliptic")
        data, label_mask = loader.load()
    """

    FEATURE_FILE = "elliptic_txs_features.csv"
    CLASS_FILE = "elliptic_txs_classes.csv"
    EDGE_FILE = "elliptic_txs_edgelist.csv"

    def __init__(self, data_dir: str | Path = "./data/raw/elliptic"):
        self.data_dir = Path(data_dir)

    def _check_files(self) -> bool:
        """Return True if all required files are present."""
        required = [self.FEATURE_FILE, self.CLASS_FILE, self.EDGE_FILE]
        missing = [f for f in required if not (self.data_dir / f).exists()]
        if missing:
            logger.warning(f"Missing Elliptic files: {missing}")
            logger.info(
                "Download from: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set"
                f"\nExtract to: {self.data_dir.resolve()}"
            )
            return False
        return True

    def load(
        self, time_step: Optional[int] = None, normalize: bool = True
    ) -> Tuple[Data, torch.Tensor]:
        """
        Load and return (PyG Data object, labelled_mask).

        Args:
            time_step: If set, filter to a single time step (1-49).
            normalize:  Standardize node features (recommended).

        Returns:
            data:         PyG Data with .x, .edge_index, .y, .time_step
            labelled_mask: Boolean tensor marking rows with known labels
        """
        if not self._check_files():
            logger.warning("Returning synthetic mock data (Elliptic files not found)")
            return self._mock_data()

        # ── Features ──────────────────────────────────────────────────────────
        # First column = txId, second = time step, rest = features
        feat_df = pd.read_csv(self.data_dir / self.FEATURE_FILE, header=None)
        feat_df.columns = ["txId", "time_step"] + [
            f"feat_{i}" for i in range(feat_df.shape[1] - 2)
        ]
        if time_step is not None:
            feat_df = feat_df[feat_df["time_step"] == time_step].reset_index(drop=True)

        tx_ids = feat_df["txId"].values
        time_steps = feat_df["time_step"].values
        features = feat_df.drop(columns=["txId", "time_step"]).values.astype(np.float32)

        # Replace NaN/Inf
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize:
            scaler = StandardScaler()
            features = scaler.fit_transform(features).astype(np.float32)

        # ── Labels ────────────────────────────────────────────────────────────
        class_df = pd.read_csv(self.data_dir / self.CLASS_FILE)
        class_df.columns = ["txId", "class"]
        class_df["label"] = class_df["class"].astype(str).map(
            lambda c: LABEL_MAP.get(c, -1)
        )

        # Build txId → index map
        txid_to_idx = {txid: i for i, txid in enumerate(tx_ids)}

        labels = np.full(len(tx_ids), -1, dtype=np.long)
        for _, row in class_df.iterrows():
            idx = txid_to_idx.get(row["txId"])
            if idx is not None:
                labels[idx] = int(row["label"])

        # ── Edges ─────────────────────────────────────────────────────────────
        edge_df = pd.read_csv(self.data_dir / self.EDGE_FILE)
        edge_df.columns = ["txId1", "txId2"]

        # Keep only edges where both endpoints are in our node set
        mask = edge_df["txId1"].isin(txid_to_idx) & edge_df["txId2"].isin(txid_to_idx)
        edge_df = edge_df[mask]

        src = np.array([txid_to_idx[t] for t in edge_df["txId1"]])
        dst = np.array([txid_to_idx[t] for t in edge_df["txId2"]])
        edge_index = np.stack([src, dst], axis=0)

        # ── PyG Data object ───────────────────────────────────────────────────
        data = Data(
            x=torch.tensor(features, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            y=torch.tensor(labels, dtype=torch.long),
            time_step=torch.tensor(time_steps, dtype=torch.long),
        )

        labelled_mask = torch.tensor(labels != -1, dtype=torch.bool)

        logger.info(
            f"Elliptic dataset loaded — nodes: {data.num_nodes}, "
            f"edges: {data.num_edges}, labelled: {labelled_mask.sum().item()}"
        )
        return data, labelled_mask

    def load_time_splits(
        self, train_steps: int = 34, val_steps: int = 5, normalize: bool = True
    ) -> Tuple[Data, dict]:
        """
        Load all 49 time steps and return train/val/test masks.

        Default split (as used in Elliptic paper):
          train: steps 1-34, val: 35-39, test: 40-49
        """
        full_data, labelled_mask = self.load(time_step=None, normalize=normalize)
        ts = full_data.time_step

        train_mask = labelled_mask & (ts <= train_steps)
        val_mask = labelled_mask & (ts > train_steps) & (ts <= train_steps + val_steps)
        test_mask = labelled_mask & (ts > train_steps + val_steps)

        splits = {"train": train_mask, "val": val_mask, "test": test_mask}
        logger.info(
            f"Splits — train: {train_mask.sum()}, val: {val_mask.sum()}, test: {test_mask.sum()}"
        )
        return full_data, splits

    # ── Fallback mock data when dataset not downloaded ─────────────────────

    def _mock_data(self) -> Tuple[Data, torch.Tensor]:
        """Generate small synthetic graph for testing without real data."""
        n_nodes, n_edges = 500, 1_000
        features = torch.randn(n_nodes, 64).float()
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        edge_index = torch.stack([src, dst], dim=0)
        # 10% illicit
        labels = torch.zeros(n_nodes, dtype=torch.long)
        labels[: n_nodes // 10] = 1
        labelled_mask = torch.ones(n_nodes, dtype=torch.bool)
        # time_step 1-49 for load_time_splits compatibility
        time_steps = torch.randint(1, 50, (n_nodes,))
        data = Data(
            x=features,
            edge_index=edge_index,
            y=labels,
            time_step=time_steps,
        )
        return data, labelled_mask
