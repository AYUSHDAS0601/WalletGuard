#!/bin/bash
# Install dependencies in correct order (torch before torch-scatter/torch-sparse).
# Run from project root: ./scripts/install_deps.sh
set -e
cd "$(dirname "$0")/.."

VENV="${1:-.venv}"
if [ -d "$VENV" ]; then
  source "$VENV/bin/activate"
fi

echo "=== Step 1: PyTorch ==="
pip install torch torchvision torchaudio

echo "=== Step 2: PyG extensions (pre-built wheels) ==="
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
echo "PyTorch version: $TORCH_VER"
pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cpu.html" || true
# If PyG wheels fail (e.g. no CPU build), try without CUDA suffix
pip install torch-scatter torch-sparse 2>/dev/null || true

echo "=== Step 3: PyTorch Geometric ==="
pip install torch-geometric

echo "=== Step 4: Rest of requirements ==="
pip install -r requirements-core.txt

echo "=== Done ==="
