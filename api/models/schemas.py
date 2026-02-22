"""
Pydantic schemas for all API request/response models.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Request models ────────────────────────────────────────────────────────────

class TransactionAnalysisRequest(BaseModel):
    tx_hash: str = Field(..., description="Ethereum transaction hash (0x...)")
    blockchain: str = Field(default="ethereum", description="Target blockchain")
    priority: str = Field(default="normal", description="normal | high")

    @field_validator("tx_hash")
    @classmethod
    def validate_tx_hash(cls, v: str) -> str:
        v = v.strip()
        if not v.startswith("0x") or len(v) != 66:
            raise ValueError("tx_hash must be a valid 66-character hex string starting with 0x")
        return v.lower()


class WalletAnalysisRequest(BaseModel):
    address: str
    tx_limit: int = Field(default=1000, ge=1, le=10000)
    include_graph: bool = Field(default=True)


class GraphQueryRequest(BaseModel):
    center_address: str
    depth: int = Field(default=2, ge=1, le=4)
    min_edge_weight_eth: float = Field(default=0.0, ge=0)
    time_range: str = Field(default="7d", description="e.g. 1d, 7d, 30d")


class PatternSearchRequest(BaseModel):
    pattern_type: str = Field(
        ...,
        description="wash_trading | flash_loan | market_manipulation | coordinated_wallets",
    )
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    min_confidence: float = Field(default=0.7, ge=0, le=1)
    blockchain: str = "ethereum"
    limit: int = Field(default=50, ge=1, le=500)


# ── Response models ───────────────────────────────────────────────────────────

class FeatureExplanation(BaseModel):
    feature: str
    shap_value: float
    feature_value: float
    contribution: str
    importance: float


class ModelScores(BaseModel):
    gnn: float = 0.0
    temporal: float = 0.0
    ensemble: float = 0.0


class TransactionAnalysisResponse(BaseModel):
    tx_hash: str
    anomaly_score: float
    risk_level: str
    detected_patterns: List[Dict[str, Any]] = []
    pattern_types: List[str] = []
    model_scores: ModelScores = Field(default_factory=ModelScores)
    explanations: Dict[str, Any] = {}
    graph_summary: Dict[str, Any] = {}
    processing_time_ms: Optional[float] = None


class WalletRiskResponse(BaseModel):
    address: str
    anomaly_score: float
    risk_level: str
    risk_factors: List[str] = []
    wallet_stats: Dict[str, Any] = {}
    network_metrics: Dict[str, Any] = {}
    model_scores: ModelScores = Field(default_factory=ModelScores)
    detected_patterns: List[Dict[str, Any]] = []
    explanations: Dict[str, Any] = {}
    error: Optional[str] = None


class GraphNode(BaseModel):
    id: str
    risk_score: float = 0.0
    node_type: str = "wallet"
    in_degree: int = 0
    out_degree: int = 0


class GraphEdge(BaseModel):
    source: str
    target: str
    weight: float
    tx_count: int


class GraphQueryResponse(BaseModel):
    center_address: str
    depth: int
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []
    node_count: int = 0
    edge_count: int = 0


class PatternMatch(BaseModel):
    pattern_type: str
    wallet_cluster: List[str] = []
    total_volume_eth: float = 0.0
    transaction_count: int = 0
    confidence: float
    time_span_hours: float = 0.0
    detected_at: Optional[str] = None


class PatternSearchResponse(BaseModel):
    pattern_type: str
    total_matches: int
    results: List[PatternMatch]
    query_time_ms: Optional[float] = None


# ── Alert / Stream models ─────────────────────────────────────────────────────

class AlertMessage(BaseModel):
    type: str = "ANOMALY_DETECTED"
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    tx_hash: Optional[str] = None
    wallet_address: Optional[str] = None
    severity: str = "HIGH"
    pattern: str
    anomaly_score: float
    affected_protocols: List[str] = []
    details: Dict[str, Any] = {}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    gnn_loaded: bool = False
    bilstm_loaded: bool = False
    ensemble_fitted: bool = False
    uptime_seconds: float = 0.0
