#!/usr/bin/env python3
"""
Extract Elliptic dataset features and labels for ensemble training.

Loads the Elliptic dataset via EllipticLoader, extracts node features and labels,
and saves them as .npy files for train_ensemble.py.

Usage:
    python scripts/extract_elliptic_features.py

    # Custom paths:
    python scripts/extract_elliptic_features.py \
        --data-dir ./data/raw/elliptic \
        --out-dir ./data/processed \
        --max-features 64

Outputs:
    data/processed/elliptic_features.npy   shape (N, F)
    data/processed/elliptic_labels.npy    shape (N,)  — 0=licit, 1=illicit, -1=unknown
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.loaders.elliptic_loader import EllipticLoader


def parse_args():
    p = argparse.ArgumentParser(description="Extract Elliptic features for ensemble")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "elliptic")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "processed",
        help="Directory for .npy outputs",
    )
    p.add_argument(
        "--max-features",
        type=int,
        default=64,
        help="Use first N features (Elliptic has 164; trim for compatibility)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    loader = EllipticLoader(data_dir=str(args.data_dir))
    data, labelled_mask = loader.load(time_step=None, normalize=True)

    X = data.x.numpy().astype(np.float32)
    y = data.y.numpy().astype(np.int32)

    if X.shape[1] > args.max_features:
        X = X[:, : args.max_features]
        logger.info(f"Trimming features to first {args.max_features} (was {data.x.shape[1]})")

    features_path = args.out_dir / "elliptic_features.npy"
    labels_path = args.out_dir / "elliptic_labels.npy"

    np.save(features_path, X)
    np.save(labels_path, y)

    n_illicit = (y == 1).sum()
    n_licit = (y == 0).sum()
    n_unknown = (y == -1).sum()

    logger.info(
        f"Saved:\n  {features_path}  shape={X.shape}\n  {labels_path}  shape={y.shape}"
    )
    logger.info(
        f"Label distribution: illicit={n_illicit}, licit={n_licit}, unknown={n_unknown}"
    )
    logger.info(
        f"Run ensemble training:\n  python training/train_ensemble.py "
        f"--features {features_path} --labels {labels_path} "
        f"--out-dir {ROOT / 'checkpoints'}"
    )


if __name__ == "__main__":
    main()
