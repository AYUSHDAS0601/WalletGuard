"""
FastAPI Application Entry Point.

Startup: initialises the DetectionPipeline (loads models if available).
Shutdown: cleanly closes any open resources.
"""
from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.models.schemas import HealthResponse
from api.routes import analysis, graph, search, stream
from config.config import settings

_START_TIME = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle manager."""
    logger.info("Starting Blockchain Anomaly Detection API...")

    # Initialise the detection pipeline (model loading is lazy/safe)
    from detection.pipeline import DetectionPipeline
    app.state.pipeline = DetectionPipeline(load_models=True)
    logger.info("DetectionPipeline initialised")

    yield

    logger.info("Shutting down API...")


# ── App factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Blockchain Transaction Anomaly Detection API",
        description=(
            "AI-powered anomaly detection for Ethereum transactions. "
            "Detects wash trading, flash loan exploits, market manipulation, "
            "and coordinated multi-wallet attacks using GNN + BiLSTM + ensemble models."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Request logging ───────────────────────────────────────────────────────
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - t0) * 1000
        logger.debug(f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.1f}ms)")
        return response

    # ── Exception handler ─────────────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "Internal server error", "error": str(exc)},
        )

    # ── Routes ────────────────────────────────────────────────────────────────
    app.include_router(analysis.router)
    app.include_router(graph.router)
    app.include_router(search.router)
    app.add_api_websocket_route("/api/v1/stream/alerts", stream.stream_alerts)

    # ── Health / info endpoints ───────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    async def health(request: Request):
        pipeline = request.app.state.pipeline
        return HealthResponse(
            status="ok",
            version="1.0.0",
            gnn_loaded=pipeline.gnn_model is not None,
            bilstm_loaded=pipeline.bilstm_model is not None,
            ensemble_fitted=pipeline.scorer._xgb_fitted,
            uptime_seconds=round(time.time() - _START_TIME, 1),
        )

    @app.get("/", tags=["Health"])
    async def root():
        return {
            "name": "Blockchain Anomaly Detection API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health",
        }

    return app


app = create_app()

# ── Dev server entrypoint ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
        log_level=settings.app.log_level.lower(),
    )
