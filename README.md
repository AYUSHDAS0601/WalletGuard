# Blockchain Transaction Anomaly Detection System

AI-powered Ethereum transaction monitoring using **Graph Neural Networks**, **BiLSTM temporal analysis**, and an **ensemble detection pipeline** to identify wash trading, flash loan exploits, market manipulation, and coordinated multi-wallet attacks.

---

## Quick Start (Local, No Docker)

### 1. Install dependencies

```bash
cd /home/pookie/Documents/Hacknovation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Option A: Use the install script (handles PyTorch + PyG order correctly)
./scripts/install_deps.sh

# Option B: Manual install
pip install torch torchvision torchaudio
# For CUDA 11.8 (GTX 2050): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter torch-sparse -f "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__.split(\"+\")[0])')+cpu.html"
pip install torch-geometric
pip install -r requirements-core.txt
```

### 2. Configure environment

```bash
cp config/.env.example .env
# Edit .env and set:
#   ETHERSCAN_API_KEY=your_key   ← get free at etherscan.io/apis
#   SECRET_KEY=some-random-string
```

### 3. Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

**API Docs:** http://localhost:8000/docs  
**Health check:** http://localhost:8000/health

---

## Dataset Setup (Elliptic — for GNN training)

1. Download from Kaggle: https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
2. Extract to `./data/raw/elliptic/` (3 CSV files)
3. Run training:

```bash
# Quick sanity check (2 batches, 2 epochs)
python training/train_gnn.py --dry-run

# Full training (optional: --focal-loss for class imbalance)
python training/train_gnn.py --epochs 100

# BiLSTM (uses neighbor time-ordering for sequences)
python training/train_temporal.py --epochs 50

# Ensemble: extract features then train
python scripts/extract_elliptic_features.py
python training/train_ensemble.py --features data/processed/elliptic_features.npy --labels data/processed/elliptic_labels.npy

# Evaluate all models (GNN, BiLSTM, Ensemble)
python training/evaluate.py

# Evaluation with accuracy + graphical output (ROC, PR, confusion matrix, metrics bar chart)
python scripts/run_evaluation_with_graphs.py
```

---

## Run Tests

```bash
pytest tests/ -v --tb=short

# Or specific suites
pytest tests/test_models.py -v      # GNN + BiLSTM + Ensemble
pytest tests/test_detectors.py -v   # Detection logic
pytest tests/test_api.py -v         # API endpoints
```

---

## Docker (Full Stack)

```bash
# Copy and edit env file
cp config/.env.example .env

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop
docker-compose down
```

Services started: **API** (8000) · **Celery** · **Redis** (6379) · **TimescaleDB** (5432) · **Neo4j** (7474 / 7687) · **Kafka** (9092)

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | System health + model status |
| `POST` | `/api/v1/analyze/transaction` | Analyze TX by hash |
| `GET`  | `/api/v1/analyze/wallet/{address}` | Wallet risk profile |
| `POST` | `/api/v1/analyze/transaction/async` | Queue TX analysis (returns task ID) |
| `GET`  | `/api/v1/analyze/task/{task_id}` | Poll async task result |
| `POST` | `/api/v1/graph/query` | Extract transaction subgraph |
| `POST` | `/api/v1/search/patterns` | Search historical anomaly patterns |
| `WS`   | `/api/v1/stream/alerts` | Real-time anomaly alert stream |

Full interactive docs at `/docs` (Swagger UI).

---

## Project Structure

```
Hacknovation/
├── config/             # Central config + .env.example
├── data/
│   ├── loaders/        # Etherscan API + Elliptic dataset loaders
│   ├── processors/     # Feature engineering + graph builder + temporal windowing
│   └── schemas/        # TimescaleDB init SQL
├── models/
│   ├── gnn/            # GraphSAGE + GAT (blockchain_gnn.py)
│   ├── temporal/       # BiLSTM + Attention (bilstm_detector.py)
│   └── ensemble/       # XGBoost + IsolationForest (anomaly_scorer.py)
├── training/           # train_gnn.py · train_temporal.py · evaluate.py
├── detection/
│   ├── wash_trade_detector.py
│   ├── flash_loan_detector.py
│   ├── market_manipulation.py
│   ├── coordinated_wallets.py
│   └── pipeline.py     # Unified detection orchestrator
├── explainability/     # SHAP explainer + report generator
├── api/
│   ├── main.py         # FastAPI app factory
│   ├── routes/         # analysis · graph · search · stream
│   ├── models/         # Pydantic schemas
│   └── tasks/          # Celery async tasks
├── tests/              # test_models · test_detectors · test_api
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Hardware Optimizations (GTX 2050 / 4GB VRAM)

| Technique | Effect |
|-----------|--------|
| Mixed Precision (AMP) | 2× speedup, −50% VRAM |
| Gradient Accumulation (4 steps) | Effective batch = 256 |
| NeighborLoader (10+5 sampling) | Fits large graphs in 4GB |
| GradScaler | Prevents FP16 underflow |
| Early Stopping | Saves time on overfit |
| Isolation Forest (CPU) | No GPU needed for ensemble |

---

## Detection Capabilities (Optimized for Two Main Targets)

**1. Coordinated Wallet Attacks** (most traceable) — many wallets move together; same timing, tokens, amounts; visible clusters. **2. Market Manipulation** — wash trading (circular trades, fake volume), pump & dump (volume spikes, buy-sell cycles).

| Pattern | Detector | Method |
|---------|----------|--------|
| **Wash Trading** | `WashTradeDetector` | Directed cycle A→B→C→A, value symmetry, time-window compactness |
| **Market Manipulation** | `MarketManipulationDetector` | Volume spike σ, pump-dump overlap, coordinated buy/sell waves |
| **Coordinated Multi-Wallet** | `CoordinatedWalletDetector` | Sybil (common funder), timing correlation (cosine similarity), cluster size |
| Flash Loan Exploits | `FlashLoanDetector` | Multi-protocol matching + MEV sandwich detection |
| General Anomaly | `EnsembleAnomalyScorer` | GNN 40% + BiLSTM 30% + XGBoost 20% + IsoForest 10% |

Detection thresholds are tuned via `config.config.DetectionConfig` (and env vars) for coordinated wallets and market manipulation.

---

## Evaluation with Graphical Output

After training, run:

```bash
python scripts/run_evaluation_with_graphs.py
```

This writes **accuracy** and full metrics to `checkpoints/evaluation_results.json`, and saves plots under `checkpoints/graphs/`:

| File | Description |
|------|--------------|
| `roc_curves.png` | ROC curves per model (AUC in legend) |
| `precision_recall_curves.png` | Precision-Recall curves (AP in legend) |
| `confusion_matrices.png` | Confusion matrix heatmap per model |
| `metrics_summary.png` | Bar chart: Accuracy, F1, Precision, Recall, AUC-ROC |

---

## Model Performance Targets

| Metric | Target | Model |
|--------|--------|-------|
| Accuracy | > 0.90 | Ensemble / GNN |
| Precision | > 0.85 | GNN on Elliptic |
| Recall | > 0.80 | GNN on Elliptic |
| F1-Score | > 0.82 | Ensemble |
| AUC-ROC | > 0.90 | Ensemble |
| Inference Latency | < 100ms | Full pipeline |
# WalletGuard
