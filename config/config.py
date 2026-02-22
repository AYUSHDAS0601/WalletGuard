"""
Central configuration for the Blockchain Anomaly Detection System.
All settings are loaded from environment variables with sensible defaults.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env from project root (one level up from config/)
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_PATH, override=False)


@dataclass
class AppConfig:
    env: str = os.getenv("APP_ENV", "development")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "8000"))
    secret_key: str = os.getenv("SECRET_KEY", "change-me-in-production")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("APP_ENV", "development") == "development"


@dataclass
class BlockchainConfig:
    etherscan_api_key: str = os.getenv("ETHERSCAN_API_KEY", "")
    alchemy_api_key: str = os.getenv("ALCHEMY_API_KEY", "")
    infura_project_id: str = os.getenv("INFURA_PROJECT_ID", "")
    rpc_url: str = os.getenv(
        "ETHEREUM_RPC_URL",
        "https://mainnet.infura.io/v3/demo",
    )
    etherscan_base_url: str = "https://api.etherscan.io/api"
    request_timeout: int = 30


@dataclass
class RedisConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    celery_broker: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
    celery_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
    cache_ttl_seconds: int = 3600


@dataclass
class DatabaseConfig:
    postgres_url: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/blockchain_anomaly",
    )
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "neo4j")
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    mongodb_db: str = os.getenv("MONGODB_DB", "blockchain_alerts")


@dataclass
class KafkaConfig:
    bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    transaction_topic: str = os.getenv("KAFKA_TRANSACTION_TOPIC", "ethereum-transactions")
    alert_topic: str = os.getenv("KAFKA_ALERT_TOPIC", "anomaly-alerts")
    consumer_group: str = "anomaly-detection-group"


@dataclass
class ModelConfig:
    checkpoint_dir: Path = Path(os.getenv("MODEL_CHECKPOINT_DIR", "./checkpoints"))
    gnn_model_path: Path = Path(os.getenv("GNN_MODEL_PATH", "./checkpoints/gnn_best.pth"))
    bilstm_model_path: Path = Path(
        os.getenv("BILSTM_MODEL_PATH", "./checkpoints/bilstm_best.pth")
    )
    xgboost_model_path: Path = Path(
        os.getenv("XGBOOST_MODEL_PATH", "./checkpoints/xgboost_model.json")
    )
    # GNN architecture
    gnn_in_channels: int = int(os.getenv("GNN_IN_CHANNELS", "64"))
    gnn_hidden_channels: int = int(os.getenv("GNN_HIDDEN_CHANNELS", "128"))
    gnn_out_channels: int = int(os.getenv("GNN_OUT_CHANNELS", "32"))
    gnn_num_layers: int = 3
    gnn_dropout: float = 0.3
    # BiLSTM architecture
    bilstm_input_size: int = int(os.getenv("BILSTM_INPUT_SIZE", "50"))
    bilstm_hidden_size: int = int(os.getenv("BILSTM_HIDDEN_SIZE", "128"))
    bilstm_num_layers: int = 2
    bilstm_dropout: float = 0.3
    # Training
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    accumulation_steps: int = 4       # effective batch = 256
    early_stopping_patience: int = 10
    num_workers: int = 4
    # Temporal
    max_sequence_length: int = 100
    sequence_stride: int = 10
    # Sampling (GraphSAGE)
    neighbor_sampling: List[int] = field(default_factory=lambda: [10, 5])

    def __post_init__(self):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class DetectionConfig:
    """
    Tuned for Coordinated Wallet Attacks + Market Manipulation (wash trading, pump & dump).
    - Coordinated: many wallets move together, same timing/tokens → graph ML + clustering.
    - Market manip: fake volume loops, circular trades, volume spikes, buy-sell cycles.
    """
    anomaly_threshold: float = float(os.getenv("ANOMALY_SCORE_THRESHOLD", "0.65"))
    high_risk_threshold: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.85"))
    # Wash trading: circular A→B→C→A, value symmetry, short time window
    wash_trade_cycle_depth: int = int(os.getenv("WASH_TRADE_CYCLE_DEPTH", "5"))
    wash_trade_time_window_hours: float = float(os.getenv("WASH_TRADE_TIME_WINDOW_HOURS", "24.0"))
    # Market manipulation: volume spike σ, pump-dump overlap
    volume_spike_sigma: float = float(os.getenv("VOLUME_SPIKE_SIGMA", "2.5"))
    pump_dump_window_hours: float = float(os.getenv("PUMP_DUMP_WINDOW_HOURS", "4.0"))
    min_coordinated_wallets: int = int(os.getenv("MIN_COORDINATED_WALLETS", "3"))
    # Coordinated wallets: Sybil + timing correlation (same hours, same tokens)
    flash_loan_min_protocols: int = int(os.getenv("FLASH_LOAN_MIN_PROTOCOLS", "2"))
    coordinated_wallet_time_window_minutes: int = int(
        os.getenv("COORDINATED_WALLET_TIME_WINDOW_MINUTES", "30")
    )
    correlation_threshold: float = float(os.getenv("COORDINATED_CORRELATION_THRESHOLD", "0.75"))
    max_account_age_days_sybil: float = float(os.getenv("MAX_ACCOUNT_AGE_DAYS_SYBIL", "45.0"))
    # Ensemble weights
    ensemble_weights: dict = field(
        default_factory=lambda: {
            "gnn": 0.4,
            "temporal": 0.3,
            "xgboost": 0.2,
            "isolation": 0.1,
        }
    )


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    blockchain: BlockchainConfig = field(default_factory=BlockchainConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    kafka: KafkaConfig = field(default_factory=KafkaConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)


# Singleton instance used throughout the project
settings = Config()
