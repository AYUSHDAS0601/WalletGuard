"""
Celery task definitions for async anomaly analysis.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from celery import Celery
from loguru import logger

from config.config import settings

# ── Celery app ────────────────────────────────────────────────────────────────
celery_app = Celery(
    "blockchain_anomaly",
    broker=settings.redis.celery_broker,
    backend=settings.redis.celery_backend,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    result_expires=3600,  # 1 hour
    worker_prefetch_multiplier=1,   # Fair task distribution
    task_acks_late=True,            # Retry on worker crash
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@celery_app.task(bind=True, name="tasks.analyze_transaction", max_retries=2)
def analyze_transaction_task(self, tx_hash: str, blockchain: str = "ethereum") -> dict:
    """
    Async transaction analysis task.

    Usage:
        result = analyze_transaction_task.delay("0x123...")
        result.get(timeout=30)
    """
    try:
        from detection.pipeline import DetectionPipeline
        pipeline = DetectionPipeline(load_models=True)
        result = pipeline.analyze_transaction(tx_hash)
        return result.to_dict()
    except Exception as exc:
        logger.exception(f"Task failed for tx {tx_hash}: {exc}")
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(bind=True, name="tasks.analyze_wallet", max_retries=2)
def analyze_wallet_task(self, address: str, tx_limit: int = 1000) -> dict:
    """Async wallet risk analysis task."""
    try:
        from detection.pipeline import DetectionPipeline
        pipeline = DetectionPipeline(load_models=True)
        result = pipeline.analyze_wallet(address, tx_limit=tx_limit)
        return result.to_dict()
    except Exception as exc:
        logger.exception(f"Task failed for wallet {address}: {exc}")
        raise self.retry(exc=exc, countdown=5)


@celery_app.task(name="tasks.batch_analyze", max_retries=1)
def batch_analyze_task(addresses: list, tx_limit: int = 500) -> list:
    """Batch analysis for multiple wallets."""
    from detection.pipeline import DetectionPipeline
    pipeline = DetectionPipeline(load_models=True)
    results = []
    for addr in addresses:
        try:
            r = pipeline.analyze_wallet(addr, tx_limit=tx_limit)
            results.append(r.to_dict())
        except Exception as e:
            results.append({"address": addr, "error": str(e)})
    return results
