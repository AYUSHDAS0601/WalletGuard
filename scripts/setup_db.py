"""
scripts/setup_db.py
───────────────────
Initialize TimescaleDB (PostgreSQL) and Neo4j databases for the
Blockchain Anomaly Detection system.

Usage
-----
# From repo root (requires running services — see docker-compose.yml):
python scripts/setup_db.py

# Skip individual DBs:
python scripts/setup_db.py --skip-neo4j
python scripts/setup_db.py --skip-timescale

Environment variables (override docker-compose defaults):
  POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

Outputs
-------
  ✓  TimescaleDB schema applied (hypertables + indexes)
  ✓  Neo4j constraints + indexes created
  ✓  Redis ping confirmed
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
SQL_SCHEMA = ROOT / "data" / "schemas" / "db_init.sql"


# ─────────────────────────────────────────── Argument Parsing ─────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize databases for Hacknovation")
    p.add_argument("--skip-timescale", action="store_true", help="Skip PostgreSQL/TimescaleDB init")
    p.add_argument("--skip-neo4j",     action="store_true", help="Skip Neo4j init")
    p.add_argument("--skip-redis",     action="store_true", help="Skip Redis ping check")
    p.add_argument("--retries", type=int, default=5, help="Connection retries with backoff")
    return p.parse_args()


# ─────────────────────────────────────────── TimescaleDB ──────────────────────


def init_timescale(retries: int) -> bool:
    """Apply db_init.sql to the TimescaleDB instance."""
    try:
        import psycopg2
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return False

    pg_config = {
        "host":     os.getenv("POSTGRES_HOST",     "localhost"),
        "port":     int(os.getenv("POSTGRES_PORT", "5432")),
        "dbname":   os.getenv("POSTGRES_DB",       "blockchain_anomaly"),
        "user":     os.getenv("POSTGRES_USER",     "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", "password"),
    }

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[TimescaleDB] Connecting ({attempt}/{retries}) → "
                        f"{pg_config['host']}:{pg_config['port']}/{pg_config['dbname']}")
            conn = psycopg2.connect(**pg_config)
            conn.autocommit = True
            cur = conn.cursor()

            # ── Enable TimescaleDB extension ──────────────────────────────────
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            logger.info("[TimescaleDB] Extension enabled")

            # ── Load and execute schema SQL ───────────────────────────────────
            if SQL_SCHEMA.exists():
                sql = SQL_SCHEMA.read_text()
                cur.execute(sql)
                logger.info(f"[TimescaleDB] Schema applied from {SQL_SCHEMA}")
            else:
                logger.warning(f"[TimescaleDB] Schema file not found: {SQL_SCHEMA}. "
                               "Running minimal inline setup.")
                _apply_minimal_pg_schema(cur)

            # ── Verify hypertables ────────────────────────────────────────────
            cur.execute("SELECT hypertable_name FROM timescaledb_information.hypertables;")
            hypertables = [row[0] for row in cur.fetchall()]
            logger.info(f"[TimescaleDB] Hypertables: {hypertables}")

            cur.close()
            conn.close()
            logger.info("[TimescaleDB] ✓ Initialization complete")
            return True

        except psycopg2.OperationalError as e:
            logger.warning(f"[TimescaleDB] Connection failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                logger.info(f"[TimescaleDB] Retrying in {wait}s …")
                time.sleep(wait)

    logger.error("[TimescaleDB] ✗ Could not connect after all retries")
    return False


def _apply_minimal_pg_schema(cur) -> None:
    """Fallback minimal schema if db_init.sql is not found."""
    statements = [
        """CREATE TABLE IF NOT EXISTS transactions (
            tx_hash     TEXT        NOT NULL,
            timestamp   TIMESTAMPTZ NOT NULL,
            from_addr   TEXT        NOT NULL,
            to_addr     TEXT        NOT NULL,
            value_eth   DOUBLE PRECISION,
            gas_used    BIGINT,
            block_number BIGINT,
            anomaly_score FLOAT,
            risk_level  TEXT,
            patterns    JSONB,
            PRIMARY KEY (tx_hash, timestamp)
        );""",
        "SELECT create_hypertable('transactions', 'timestamp', if_not_exists => TRUE);",
        """CREATE TABLE IF NOT EXISTS wallet_profiles (
            address     TEXT PRIMARY KEY,
            risk_score  FLOAT,
            risk_level  TEXT,
            last_seen   TIMESTAMPTZ,
            tx_count    INTEGER,
            flagged_patterns JSONB,
            updated_at  TIMESTAMPTZ DEFAULT NOW()
        );""",
        """CREATE TABLE IF NOT EXISTS anomaly_alerts (
            id          SERIAL,
            timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            entity_type TEXT,    -- 'transaction' | 'wallet'
            entity_id   TEXT,
            alert_type  TEXT,
            severity    TEXT,
            details     JSONB,
            PRIMARY KEY (id, timestamp)
        );""",
        "SELECT create_hypertable('anomaly_alerts', 'timestamp', if_not_exists => TRUE);",
        "CREATE INDEX IF NOT EXISTS idx_tx_from ON transactions (from_addr);",
        "CREATE INDEX IF NOT EXISTS idx_tx_to   ON transactions (to_addr);",
        "CREATE INDEX IF NOT EXISTS idx_tx_risk ON transactions (anomaly_score DESC);",
    ]
    for stmt in statements:
        cur.execute(stmt)


# ─────────────────────────────────────────── Neo4j ────────────────────────────


def init_neo4j(retries: int) -> bool:
    """Create Neo4j constraints and indexes for the wallet graph."""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.error("neo4j driver not installed. Run: pip install neo4j")
        return False

    uri      = os.getenv("NEO4J_URI",      "bolt://localhost:7687")
    user     = os.getenv("NEO4J_USER",     "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")

    neo4j_statements = [
        # Node uniqueness constraints
        "CREATE CONSTRAINT wallet_address IF NOT EXISTS FOR (w:Wallet) REQUIRE w.address IS UNIQUE",
        "CREATE CONSTRAINT tx_hash IF NOT EXISTS FOR (t:Transaction) REQUIRE t.hash IS UNIQUE",
        # Indexes for performance
        "CREATE INDEX wallet_risk IF NOT EXISTS FOR (w:Wallet) ON (w.risk_score)",
        "CREATE INDEX tx_block IF NOT EXISTS FOR (t:Transaction) ON (t.block_number)",
        "CREATE INDEX tx_timestamp IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
        "CREATE INDEX tx_anomaly IF NOT EXISTS FOR (t:Transaction) ON (t.anomaly_score)",
    ]

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[Neo4j] Connecting ({attempt}/{retries}) → {uri}")
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()

            with driver.session() as session:
                for stmt in neo4j_statements:
                    session.run(stmt)
                    logger.debug(f"[Neo4j] Applied: {stmt[:60]}…")

            driver.close()
            logger.info("[Neo4j] ✓ Constraints and indexes created")
            return True

        except Exception as e:
            logger.warning(f"[Neo4j] Connection failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                logger.info(f"[Neo4j] Retrying in {wait}s …")
                time.sleep(wait)

    logger.error("[Neo4j] ✗ Could not connect after all retries")
    return False


# ─────────────────────────────────────────── Redis ────────────────────────────


def check_redis(retries: int) -> bool:
    """Verify Redis is reachable."""
    try:
        import redis
    except ImportError:
        logger.error("redis-py not installed. Run: pip install redis")
        return False

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    for attempt in range(1, retries + 1):
        try:
            logger.info(f"[Redis] Connecting ({attempt}/{retries}) → {redis_url}")
            r = redis.from_url(redis_url, socket_connect_timeout=5)
            r.ping()
            logger.info("[Redis] ✓ Reachable")
            return True
        except Exception as e:
            logger.warning(f"[Redis] Ping failed: {e}")
            if attempt < retries:
                wait = 2 ** attempt
                time.sleep(wait)

    logger.error("[Redis] ✗ Could not reach Redis after all retries")
    return False


# ─────────────────────────────────────────────────────────────── main ─────────


def main() -> None:
    args = parse_args()
    logger.info("=== Database Setup Started ===")

    results = {}

    if not args.skip_timescale:
        results["timescaledb"] = init_timescale(args.retries)
    else:
        logger.info("[TimescaleDB] Skipped")

    if not args.skip_neo4j:
        results["neo4j"] = init_neo4j(args.retries)
    else:
        logger.info("[Neo4j] Skipped")

    if not args.skip_redis:
        results["redis"] = check_redis(args.retries)
    else:
        logger.info("[Redis] Skipped")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=== Setup Summary ===")
    all_ok = True
    for service, ok in results.items():
        status = "✓" if ok else "✗"
        logger.info(f"  {status}  {service}")
        if not ok:
            all_ok = False

    if all_ok:
        logger.info("")
        logger.info("All databases initialized successfully!")
        logger.info("You can now start the API server:")
        logger.info("  uvicorn api.main:app --reload --port 8000")
    else:
        logger.error("")
        logger.error("Some initializations failed. Check service logs and retry.")
        sys.exit(1)


if __name__ == "__main__":
    main()
