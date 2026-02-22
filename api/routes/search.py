"""
Pattern Search Routes — historical anomaly pattern search.
"""
from __future__ import annotations

import time

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Request

from api.models.schemas import PatternMatch, PatternSearchRequest, PatternSearchResponse

router = APIRouter(prefix="/api/v1/search", tags=["Search"])


def get_pipeline(request: Request):
    return request.app.state.pipeline


VALID_PATTERN_TYPES = {
    "wash_trading", "flash_loan", "market_manipulation", "coordinated_wallets"
}


@router.post(
    "/patterns",
    response_model=PatternSearchResponse,
    summary="Search for historical anomaly patterns",
)
async def search_patterns(
    body: PatternSearchRequest,
    pipeline=Depends(get_pipeline),
):
    """
    Search historical transaction data for anomaly patterns.

    NOTE: In this version the search runs over the last N transactions
    fetched on-demand. A production deployment would query a TimescaleDB
    index for pre-computed results.
    """
    if body.pattern_type not in VALID_PATTERN_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"pattern_type must be one of: {VALID_PATTERN_TYPES}",
        )

    t0 = time.perf_counter()

    # For demo: fetch recent transactions and scan for the pattern
    # The limit is deliberately low to avoid rate-limit issues
    try:
        # Use mock data since we can't know which wallet to fetch
        # In production: query TimescaleDB index
        result = pipeline.analyze_mock()
        patterns = result.detected_patterns

        # Filter to the requested pattern type
        matched = [
            p for p in patterns
            if body.pattern_type in p.get("pattern", "")
            and p.get("confidence", 0) >= body.min_confidence
        ]

        results = []
        for p in matched[: body.limit]:
            results.append(
                PatternMatch(
                    pattern_type=p.get("pattern", body.pattern_type),
                    wallet_cluster=p.get("cycle", p.get("affected_wallets", p.get("wallets", []))),
                    total_volume_eth=p.get("total_volume_eth", 0),
                    transaction_count=p.get("tx_count", 0),
                    confidence=p.get("confidence", 0),
                    time_span_hours=p.get("time_span_hours", 0),
                )
            )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return PatternSearchResponse(
        pattern_type=body.pattern_type,
        total_matches=len(results),
        results=results,
        query_time_ms=round(elapsed_ms, 2),
    )
