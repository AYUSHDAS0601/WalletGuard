#!/usr/bin/env bash
# ============================================================
# scripts/download_elliptic.sh
# Download and pre-process the Elliptic Bitcoin dataset.
#
# Requirements:
#   pip install kaggle
#   Set KAGGLE_USERNAME and KAGGLE_KEY env vars, OR
#   place ~/.kaggle/kaggle.json (from Kaggle → Account → API)
#
# Usage:
#   chmod +x scripts/download_elliptic.sh
#   ./scripts/download_elliptic.sh
#
# Outputs:
#   data/raw/elliptic/                   (raw CSV files)
#   data/processed/elliptic_features.npy
#   data/processed/elliptic_labels.npy
#   data/processed/elliptic_edges.npy
# ============================================================

set -euo pipefail

DATASET="ellipticco/elliptic-bitcoin-dataset"
RAW_DIR="data/raw/elliptic"
PROCESSED_DIR="data/processed"

echo "=== Elliptic Dataset Download ==="

# ── Verify kaggle CLI is available ──────────────────────────────────────────
if ! command -v kaggle &> /dev/null; then
    echo "[ERROR] kaggle CLI not found. Install with: pip install kaggle"
    exit 1
fi

# ── Check credentials ────────────────────────────────────────────────────────
if [[ -z "${KAGGLE_USERNAME:-}" ]] && [[ ! -f "$HOME/.kaggle/kaggle.json" ]]; then
    echo "[ERROR] Kaggle credentials not found."
    echo "  Option 1: Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
    echo "  Option 2: Place ~/.kaggle/kaggle.json (from kaggle.com → Account → API)"
    exit 1
fi

# ── Download ─────────────────────────────────────────────────────────────────
mkdir -p "$RAW_DIR"
echo "[1/3] Downloading dataset from Kaggle..."
kaggle datasets download -d "$DATASET" -p "$RAW_DIR" --unzip

echo "✓  Raw files in $RAW_DIR:"
ls -lh "$RAW_DIR/"

# ── Pre-process ──────────────────────────────────────────────────────────────
mkdir -p "$PROCESSED_DIR"
echo "[2/3] Pre-processing CSVs → NumPy arrays..."

python - <<'PYEOF'
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path.cwd()
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RAW = ROOT / "data" / "raw" / "elliptic"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# ── Features ─────────────────────────────────────────────────────────────────
feat_path = RAW / "elliptic_txs_features.csv"
print(f"  Loading features from {feat_path} …")
feats = pd.read_csv(feat_path, header=None)
# Column 0 = txId, columns 1..N = features
tx_ids = feats.iloc[:, 0].values
X = feats.iloc[:, 1:].values.astype(np.float32)
print(f"  Features shape: {X.shape}")

# ── Labels ───────────────────────────────────────────────────────────────────
classes_path = RAW / "elliptic_txs_classes.csv"
print(f"  Loading classes from {classes_path} …")
classes = pd.read_csv(classes_path)
# Columns: txId, class  (values: '1'=illicit, '2'=licit, 'unknown')
class_map = {"1": 1, "2": 0, "unknown": -1}

# Align labels to feature order
id_to_label = dict(zip(classes["txId"].astype(str), classes["class"].astype(str)))
y = np.array([class_map.get(id_to_label.get(str(tid), "unknown"), -1)
              for tid in tx_ids], dtype=np.int8)
print(f"  Labels: illicit={( y==1).sum()}, licit={(y==0).sum()}, unknown={(y==-1).sum()}")

# ── Edges ────────────────────────────────────────────────────────────────────
edges_path = RAW / "elliptic_txs_edgelist.csv"
print(f"  Loading edges from {edges_path} …")
edges = pd.read_csv(edges_path)
edge_arr = edges.values.astype(np.int64)
print(f"  Edges shape: {edge_arr.shape}")

# ── Save ─────────────────────────────────────────────────────────────────────
np.save(str(OUT / "elliptic_features.npy"), X)
np.save(str(OUT / "elliptic_labels.npy"),   y)
np.save(str(OUT / "elliptic_edges.npy"),    edge_arr)
np.save(str(OUT / "elliptic_tx_ids.npy"),   tx_ids)
print(f"  Saved to {OUT}/")
PYEOF

echo "[3/3] Verifying outputs..."
for f in elliptic_features.npy elliptic_labels.npy elliptic_edges.npy; do
    fpath="$PROCESSED_DIR/$f"
    if [[ -f "$fpath" ]]; then
        echo "  ✓  $fpath  ($(du -sh "$fpath" | cut -f1))"
    else
        echo "  ✗  MISSING: $fpath"
        exit 1
    fi
done

echo ""
echo "=== Dataset ready! ==="
echo "  features : $PROCESSED_DIR/elliptic_features.npy"
echo "  labels   : $PROCESSED_DIR/elliptic_labels.npy"
echo "  edges    : $PROCESSED_DIR/elliptic_edges.npy"
echo ""
echo "Next steps:"
echo "  python training/train_gnn.py --dry-run"
echo "  python training/train_ensemble.py --features data/processed/elliptic_features.npy --labels data/processed/elliptic_labels.npy"
