"""
Analysis Routes — transaction and wallet risk analysis endpoints.
"""
from __future__ import annotations

import time
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from loguru import logger

from api.models.schemas import (
    ModelScores,
    TransactionAnalysisRequest,
    TransactionAnalysisResponse,
    WalletRiskResponse,
)

router = APIRouter(prefix="/api/v1/analyze", tags=["Analysis"])


def get_pipeline(request: Request):
    """Dependency to get the shared DetectionPipeline from app state."""
    return request.app.state.pipeline


@router.post(
    "/transaction",
    response_model=TransactionAnalysisResponse,
    summary="Analyze a transaction for anomalies",
)
async def analyze_transaction(
    body: TransactionAnalysisRequest,
    background_tasks: BackgroundTasks,
    pipeline=Depends(get_pipeline),
):
    """
    Analyze a single Ethereum transaction for malicious patterns.

    Returns an anomaly score (0-1), risk level, detected pattern types,
    and model explanation details.
    """
    t0 = time.perf_counter()

    try:
        result = pipeline.analyze_transaction(body.tx_hash)
    except Exception as e:
        logger.exception(f"Transaction analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result.error:
        raise HTTPException(status_code=404, detail=result.error)

    return TransactionAnalysisResponse(
        tx_hash=body.tx_hash,
        anomaly_score=result.anomaly_score,
        risk_level=result.risk_level,
        detected_patterns=result.detected_patterns,
        pattern_types=result.pattern_types,
        model_scores=ModelScores(
            gnn=result.gnn_score,
            temporal=result.temporal_score,
            ensemble=result.ensemble_score,
        ),
        explanations=result.explanations,
        graph_summary=result.graph_summary,
        processing_time_ms=round(elapsed_ms, 2),
    )


@router.get(
    "/wallet/{address}",
    response_model=WalletRiskResponse,
    summary="Get wallet risk profile",
)
async def analyze_wallet(
    address: str,
    tx_limit: int = 1000,
    pipeline=Depends(get_pipeline),
):
    """
    Analyze a wallet address and return a comprehensive risk profile.

    Fetches the wallet's recent transactions, computes behavioral features,
    and runs the full detection pipeline.
    """
    address = address.lower().strip()
    if not address.startswith("0x") or len(address) != 42:
        raise HTTPException(status_code=400, detail="Invalid Ethereum address format")

    try:
        result = pipeline.analyze_wallet(address, tx_limit=tx_limit)
    except Exception as e:
        logger.exception(f"Wallet analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Derive risk factors from detected patterns
    risk_factors = list({
        p.get("pattern", "").replace("_", " ")
        for p in result.detected_patterns
        if p.get("confidence", 0) >= 0.6
    })

    return WalletRiskResponse(
        address=address,
        anomaly_score=result.anomaly_score,
        risk_level=result.risk_level,
        risk_factors=risk_factors,
        wallet_stats=result.wallet_stats,
        network_metrics=result.graph_summary,
        model_scores=ModelScores(
            gnn=result.gnn_score,
            temporal=result.temporal_score,
            ensemble=result.ensemble_score,
        ),
        detected_patterns=result.detected_patterns,
        explanations=result.explanations,
        error=result.error,
    )


@router.post(
    "/transaction/async",
    summary="Queue transaction analysis (returns task ID)",
)
async def analyze_transaction_async(
    body: TransactionAnalysisRequest,
    background_tasks: BackgroundTasks,
):
    """Queue a transaction for async Celery analysis. Returns a task ID."""
    try:
        from api.tasks.celery_tasks import analyze_transaction_task
        task = analyze_transaction_task.delay(body.tx_hash, body.blockchain)
        return {"task_id": task.id, "status": "queued"}
    except Exception as e:
        logger.warning(f"Celery not available — running sync: {e}")
        raise HTTPException(status_code=503, detail="Async processing unavailable — use /transaction")


@router.get(
    "/task/{task_id}",
    summary="Get async task result",
)
async def get_task_result(task_id: str):
    """Poll the result of a previously queued async task."""
    try:
        from api.tasks.celery_tasks import celery_app
        result = celery_app.AsyncResult(task_id)
        if result.ready():
            return {"task_id": task_id, "status": "done", "result": result.get()}
        return {"task_id": task_id, "status": result.status}
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Task not found: {e}")
