"""
WebSocket Stream Route — real-time anomaly alert streaming.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from api.models.schemas import AlertMessage

router = APIRouter(tags=["Streaming"])

# Active WebSocket connections
_active_connections: Set[WebSocket] = set()


@router.websocket("/api/v1/stream/alerts")
async def stream_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time anomaly alert streaming.

    Connect with: ws://host:8000/api/v1/stream/alerts

    Messages:
      - ANOMALY_DETECTED: New anomaly identified
      - PING:             Keepalive (every 30s)
    """
    await websocket.accept()
    _active_connections.add(websocket)
    logger.info(f"WebSocket connected — {len(_active_connections)} active connections")

    try:
        await websocket.send_json({
            "type": "CONNECTED",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "message": "Blockchain anomaly detection stream connected.",
        })

        while True:
            try:
                # Receive client messages (e.g. subscribe/unsubscribe) with timeout
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                msg = json.loads(data)
                if msg.get("type") == "PING":
                    await websocket.send_json({"type": "PONG"})
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({
                    "type": "PING",
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                })
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        _active_connections.discard(websocket)
        logger.info(f"Connection removed — {len(_active_connections)} active")


async def broadcast_alert(alert: AlertMessage) -> None:
    """
    Broadcast an anomaly alert to ALL connected WebSocket clients.

    Call this from anywhere in the backend when a new alert is detected:
        await broadcast_alert(AlertMessage(...))
    """
    if not _active_connections:
        return

    payload = alert.model_dump()
    disconnected = set()

    for ws in _active_connections:
        try:
            await ws.send_json(payload)
        except Exception:
            disconnected.add(ws)

    for ws in disconnected:
        _active_connections.discard(ws)

    logger.debug(
        f"Alert broadcast to {len(_active_connections)} clients — {alert.pattern}"
    )
