-- TimescaleDB Schema for Blockchain Anomaly Detection System
-- Run this script to initialize the database

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ── Transactions table (hypertable for time-series queries) ──────────────────
CREATE TABLE IF NOT EXISTS transactions (
    id              BIGSERIAL,
    tx_hash         VARCHAR(66) NOT NULL,
    from_address    VARCHAR(42),
    to_address      VARCHAR(42),
    block_number    BIGINT,
    block_timestamp TIMESTAMPTZ NOT NULL,
    value_eth       DOUBLE PRECISION DEFAULT 0,
    gas_price_gwei  DOUBLE PRECISION DEFAULT 0,
    gas_used        BIGINT DEFAULT 0,
    tx_fee_eth      DOUBLE PRECISION DEFAULT 0,
    is_contract     BOOLEAN DEFAULT FALSE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

SELECT create_hypertable('transactions', 'block_timestamp', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions (from_address, block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tx_to   ON transactions (to_address,   block_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_tx_hash ON transactions (tx_hash);

-- ── Anomaly detections ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_detections (
    id              BIGSERIAL PRIMARY KEY,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    tx_hash         VARCHAR(66),
    wallet_address  VARCHAR(42),
    anomaly_score   DOUBLE PRECISION,
    risk_level      VARCHAR(10),
    pattern_types   TEXT[],
    gnn_score       DOUBLE PRECISION,
    temporal_score  DOUBLE PRECISION,
    ensemble_score  DOUBLE PRECISION,
    explanation     JSONB,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_anomaly_wallet ON anomaly_detections (wallet_address, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_score  ON anomaly_detections (anomaly_score DESC);
CREATE INDEX IF NOT EXISTS idx_anomaly_risk   ON anomaly_detections (risk_level);

-- ── Wallet feature cache ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS wallet_features (
    address             VARCHAR(42) PRIMARY KEY,
    total_txs           BIGINT DEFAULT 0,
    total_value_sent    DOUBLE PRECISION DEFAULT 0,
    total_value_recv    DOUBLE PRECISION DEFAULT 0,
    tx_frequency        DOUBLE PRECISION DEFAULT 0,
    unique_counterparties INT DEFAULT 0,
    mixer_interactions  INT DEFAULT 0,
    risk_score          DOUBLE PRECISION DEFAULT 0,
    last_updated        TIMESTAMPTZ DEFAULT NOW()
);

-- ── Pattern index ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pattern_index (
    id              BIGSERIAL PRIMARY KEY,
    detected_at     TIMESTAMPTZ DEFAULT NOW(),
    pattern_type    VARCHAR(50) NOT NULL,
    confidence      DOUBLE PRECISION,
    affected_wallets TEXT[],
    total_volume_eth DOUBLE PRECISION DEFAULT 0,
    tx_count        INT DEFAULT 0,
    details         JSONB
);

CREATE INDEX IF NOT EXISTS idx_pattern_type ON pattern_index (pattern_type, detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_pattern_conf ON pattern_index (confidence DESC);
