"""
Unified Detection Pipeline — orchestrates all detectors and models.

Single entry point: DetectionPipeline.analyze(tx_hash | wallet_address | dataframe)
Returns a structured DetectionResult with scores, patterns, and explanations.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import settings
from data.loaders.etherscan_loader import EtherscanLoader
from data.processors.feature_engineer import FeatureEngineer
from data.processors.graph_builder import GraphBuilder
from data.processors.temporal_windowing import TemporalWindowing
from detection.coordinated_wallets import CoordinatedWalletDetector
from detection.flash_loan_detector import FlashLoanDetector
from detection.market_manipulation import MarketManipulationDetector
from detection.wash_trade_detector import WashTradeDetector
from explainability.shap_explainer import SHAPExplainer
from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer


@dataclass
class DetectionResult:
    """Structured anomaly detection output for a wallet or transaction."""
    address: Optional[str] = None
    tx_hash: Optional[str] = None
    anomaly_score: float = 0.0
    risk_level: str = "LOW"
    detected_patterns: List[Dict[str, Any]] = field(default_factory=list)
    pattern_types: List[str] = field(default_factory=list)
    gnn_score: float = 0.0
    temporal_score: float = 0.0
    ensemble_score: float = 0.0
    graph_summary: Dict[str, Any] = field(default_factory=dict)
    explanations: Dict[str, Any] = field(default_factory=dict)
    wallet_stats: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "tx_hash": self.tx_hash,
            "anomaly_score": round(self.anomaly_score, 4),
            "risk_level": self.risk_level,
            "detected_patterns": self.detected_patterns,
            "pattern_types": self.pattern_types,
            "model_scores": {
                "gnn": round(self.gnn_score, 4),
                "temporal": round(self.temporal_score, 4),
                "ensemble": round(self.ensemble_score, 4),
            },
            "graph_summary": self.graph_summary,
            "explanations": self.explanations,
            "wallet_stats": self.wallet_stats,
        }


class DetectionPipeline:
    """
    Orchestrates the full anomaly detection workflow:
      1. Load data (Etherscan API)
      2. Build transaction graph
      3. Engineer features
      4. Run specialized detectors
      5. Score with ensemble (GNN + BiLSTM + XGBoost + IsoForest)
      6. Produce explanations

    Usage:
        pipeline = DetectionPipeline()
        result = pipeline.analyze_wallet("0xabc...")
        result = pipeline.analyze_dataframe(df)
        result = pipeline.analyze_mock()  # for testing
    """

    def __init__(
        self,
        load_models: bool = True,
        etherscan_key: Optional[str] = None,
    ):
        # ── Component initialization ──────────────────────────────────────────
        self.etherscan = EtherscanLoader(api_key=etherscan_key)
        self.feature_engineer = FeatureEngineer()
        self.graph_builder = GraphBuilder()
        self.temporal = TemporalWindowing()

        # Detectors (tuned for coordinated wallets + market manipulation / wash trading)
        self.wash_trader = WashTradeDetector(
            max_cycle_depth=settings.detection.wash_trade_cycle_depth,
            time_window_hours=settings.detection.wash_trade_time_window_hours,
        )
        self.flash_loan = FlashLoanDetector()
        self.market_manip = MarketManipulationDetector(
            volume_spike_threshold_sigma=settings.detection.volume_spike_sigma,
            min_coordinated_wallets=settings.detection.min_coordinated_wallets,
            pump_window=f"{int(settings.detection.pump_dump_window_hours)}H",
        )
        self.coord_wallets = CoordinatedWalletDetector(
            min_cluster_size=settings.detection.min_coordinated_wallets,
            correlation_threshold=settings.detection.correlation_threshold,
            max_account_age_days=settings.detection.max_account_age_days_sybil,
        )

        # Ensemble scorer (traditional models)
        self.scorer = EnsembleAnomalyScorer()
        self._shap_explainer: Optional[Any] = None
        self._models_loaded = False

        # GNN + BiLSTM (optional — loaded if checkpoints exist)
        self.gnn_model = None
        self.bilstm_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_models:
            self._load_neural_models()

    # ────────────────────────────────────────────────── Public interface ──────

    def analyze_wallet(
        self, address: str, tx_limit: int = 1_000
    ) -> DetectionResult:
        """Fetch and analyze a wallet address."""
        logger.info(f"Analyzing wallet {address[:10]}...")

        df = self.etherscan.get_wallet_transactions(address, limit=tx_limit)
        if df.empty:
            logger.warning(f"No transactions found for {address[:10]}")
            return DetectionResult(
                address=address,
                error="No transactions found for this address",
            )

        result = self.analyze_dataframe(df, address=address)
        result.address = address
        return result

    def analyze_transaction(self, tx_hash: str) -> DetectionResult:
        """Analyze a single transaction by hash."""
        logger.info(f"Analyzing transaction {tx_hash[:12]}...")

        tx = self.etherscan.get_transaction_by_hash(tx_hash)
        if not tx:
            return DetectionResult(tx_hash=tx_hash, error="Transaction not found")

        df = pd.DataFrame([tx])
        result = self.analyze_dataframe(df, address=tx.get("from"))
        result.tx_hash = tx_hash
        return result

    def analyze_dataframe(
        self, df: pd.DataFrame, address: Optional[str] = None
    ) -> DetectionResult:
        """Core analysis on any transaction DataFrame."""
        result = DetectionResult(address=address)

        try:
            # ── Feature engineering ───────────────────────────────────────────
            tx_df = self.feature_engineer.build_transaction_features(df)
            wallet_df = self.feature_engineer.build_wallet_features(df)

            # ── Build graph ───────────────────────────────────────────────────
            G = self.graph_builder.build_networkx(df)
            metrics_df = self.graph_builder.compute_node_metrics(G)

            result.graph_summary = {
                "num_wallets": G.number_of_nodes(),
                "num_transactions": G.number_of_edges(),
            }

            # ── Wallet statistics ─────────────────────────────────────────────
            if address and address.lower() in (wallet_df.index.str.lower()):
                wstats = wallet_df.loc[
                    wallet_df.index.str.lower() == address.lower()
                ].iloc[0].to_dict()
                result.wallet_stats = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                       for k, v in wstats.items()}

            # ── Run specialized detectors ─────────────────────────────────────
            all_patterns = []

            wash_patterns = self.wash_trader.detect(df)
            for p in wash_patterns:
                all_patterns.append(p.to_dict())

            flash_patterns = self.flash_loan.detect(df)
            for p in flash_patterns:
                all_patterns.append(p.to_dict())

            manip_patterns = self.market_manip.detect(df)
            for p in manip_patterns:
                all_patterns.append(p.to_dict())

            coord_patterns = self.coord_wallets.detect(df)
            for p in coord_patterns:
                all_patterns.append(p.to_dict())

            result.detected_patterns = all_patterns
            result.pattern_types = list({p.get("pattern", "") for p in all_patterns})

            # ── Ensemble scoring ──────────────────────────────────────────────
            X = self.feature_engineer.build_combined(df)

            gnn_scores: Optional[np.ndarray] = None
            temporal_scores: Optional[np.ndarray] = None

            # Neural model inference (if loaded)
            if self.gnn_model is not None and not df.empty:
                gnn_scores = self._run_gnn(df, X.shape[0])
                result.gnn_score = float(gnn_scores.mean())

            if self.bilstm_model is not None and not df.empty:
                temporal_scores = self._run_bilstm(df, X.shape[0])
                result.temporal_score = float(temporal_scores.mean())

            if self.scorer._iso_fitted or self.scorer._xgb_fitted:
                scores = self.scorer.predict(X, gnn_scores, temporal_scores)
            else:
                # Heuristic fallback from detector patterns
                scores = self._heuristic_score(all_patterns, X.shape[0])

            result.ensemble_score = float(scores.mean())
            result.anomaly_score = float(scores.max())  # Worst-case score
            result.risk_level = self.scorer.risk_level(result.anomaly_score)

            # ── Explanations ──────────────────────────────────────────────────
            result.explanations = self._build_explanations(
                result,
                X_features=X,
                feature_names=self.feature_engineer._last_feature_names,
            )

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
            result.error = str(e)

        return result

    def analyze_mock(self) -> DetectionResult:
        """
        Generate a mock analysis result for testing without real data.
        Injects synthetic wash-trade and flash-loan patterns.
        """
        import random
        wallets = [f"0x{'a' * i + 'b' * (40 - i)}" for i in range(1, 6)]

        # Synthetic transaction DataFrame
        records = []
        base_time = pd.Timestamp("2024-01-15 10:00:00", tz="UTC")
        for i, (frm, to) in enumerate(
            zip(wallets, wallets[1:] + [wallets[0]])
        ):
            records.append({
                "from": frm, "to": to,
                "value": int(1e18), "gas": 21000,
                "gasPrice": int(50e9), "hash": f"0xhash{i}",
                "blockNumber": 1000 + i,
                "timestamp": base_time + pd.Timedelta(minutes=i * 5),
            })

        df = pd.DataFrame(records)
        df["value_eth"] = 1.0

        result = self.analyze_dataframe(df, address=wallets[0])
        result.address = wallets[0]
        return result

    # ─────────────────────────────────────────────────── Model I/O ───────────

    def _load_neural_models(self) -> None:
        """Load GNN and BiLSTM from checkpoints if they exist."""
        from models.gnn.blockchain_gnn import BlockchainGNN
        from models.temporal.bilstm_detector import TemporalAnomalyDetector

        gnn_path = settings.model.gnn_model_path
        if gnn_path.exists():
            try:
                self.gnn_model = BlockchainGNN.load(str(gnn_path), device=str(self._device))
                self.gnn_model.eval()
                logger.info(f"GNN loaded from {gnn_path}")
            except Exception as e:
                logger.warning(f"Could not load GNN: {e}")

        bilstm_path = settings.model.bilstm_model_path
        if bilstm_path.exists():
            try:
                self.bilstm_model = TemporalAnomalyDetector.load(
                    str(bilstm_path), device=str(self._device)
                )
                self.bilstm_model.eval()
                logger.info(f"BiLSTM loaded from {bilstm_path}")
            except Exception as e:
                logger.warning(f"Could not load BiLSTM: {e}")

        # Load ensemble scorer (check checkpoint_dir and legacy models/saved/ensemble)
        scorer_dir = settings.model.checkpoint_dir
        for dir_path in [
            scorer_dir,
            Path(__file__).parent.parent / "models" / "saved" / "ensemble",
        ]:
            if (dir_path / "xgboost_model.json").exists() or (dir_path / "xgb_model.pkl").exists():
                self.scorer = EnsembleAnomalyScorer.load(str(dir_path))
                logger.info(f"Ensemble scorer loaded from {dir_path}")
                if self.scorer._xgb_fitted:
                    self._shap_explainer = SHAPExplainer(
                        self.scorer.xgb_model,
                        feature_names=None,
                    )
                break

    def _run_gnn(self, df: pd.DataFrame, batch_size: int) -> np.ndarray:
        """Run GNN inference on the transaction graph."""
        try:
            pyg_data = self.graph_builder.build_pyg(
                df, feature_dim=settings.model.gnn_in_channels
            )
            pyg_data = pyg_data.to(self._device)
            with torch.no_grad():
                scores = self.gnn_model.predict(pyg_data.x, pyg_data.edge_index)
            return scores.cpu().numpy()[:batch_size]
        except Exception as e:
            logger.warning(f"GNN inference failed: {e}")
            return np.zeros(batch_size)

    def _run_bilstm(self, df: pd.DataFrame, batch_size: int) -> np.ndarray:
        """Run BiLSTM inference on wallet sequences."""
        try:
            from models.temporal.bilstm_detector import make_padding_mask
            sequences, _ = self.temporal.build_sequence_dataset(df)
            if sequences.shape[0] == 0:
                return np.zeros(batch_size)
            sequences = sequences.to(self._device)
            mask = make_padding_mask(sequences).to(self._device)
            with torch.no_grad():
                scores, _ = self.bilstm_model(sequences, mask)
            result = scores.cpu().numpy()
            # Pad/trim to batch_size
            if len(result) < batch_size:
                result = np.pad(result, (0, batch_size - len(result)))
            return result[:batch_size]
        except Exception as e:
            logger.warning(f"BiLSTM inference failed: {e}")
            return np.zeros(batch_size)

    @staticmethod
    def _heuristic_score(patterns: List[Dict], n: int) -> np.ndarray:
        """Compute a heuristic score when models are not yet trained."""
        base = 0.1
        for p in patterns:
            conf = p.get("confidence", 0.0)
            ptype = p.get("pattern", "")
            if "flash_loan" in ptype:
                base = max(base, conf * 0.9)
            elif "wash_trading" in ptype:
                base = max(base, conf * 0.85)
            elif "market_manipulation" in ptype:
                base = max(base, conf * 0.80)
            elif "coordinated" in ptype:
                base = max(base, conf * 0.75)
        return np.full(n, float(base))

    def _build_explanations(
        self,
        result: DetectionResult,
        X_features: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Build human-readable explanation from detected patterns and SHAP (if available)."""
        explanations: Dict[str, Any] = {"summary": "", "top_risk_factors": [], "shap_features": []}

        # SHAP-based feature importance (when XGBoost fitted and features provided)
        if (
            self._shap_explainer is not None
            and X_features is not None
            and len(X_features) > 0
        ):
            try:
                explainer = self._shap_explainer
                if feature_names:
                    explainer.feature_names = feature_names
                shap_explanations = explainer.explain(X_features[: min(50, len(X_features))], top_k=8)
                explanations["shap_features"] = [
                    {
                        "feature": e["feature"],
                        "importance": e["importance"],
                        "contribution": e["contribution"],
                    }
                    for e in shap_explanations
                ]
            except Exception as e:
                logger.debug(f"SHAP explain failed: {e}")

        if not result.detected_patterns and not explanations["shap_features"]:
            explanations["summary"] = (
                "No significant anomalous patterns detected in the analyzed transactions."
            )
            return explanations

        # Build top factors from confidence scores
        factors = sorted(
            result.detected_patterns,
            key=lambda p: p.get("confidence", 0),
            reverse=True,
        )[:5]

        descriptions = {
            "wash_trading": "Circular trading pattern detected — possible wash trading.",
            "flash_loan_flash_loan": "Flash loan exploit pattern detected.",
            "flash_loan_mev_sandwich": "MEV sandwich attack detected.",
            "market_manipulation_pump_dump": "Pump-and-dump scheme detected.",
            "market_manipulation_coordinated_buy": "Coordinated buying pattern detected.",
            "coordinated_wallets": "Suspicious wallet cluster detected.",
        }

        explanations["top_risk_factors"] = [
            {
                "factor": p.get("pattern", "unknown"),
                "confidence": p.get("confidence", 0),
                "description": descriptions.get(
                    p.get("pattern", ""), "Anomalous pattern detected."
                ),
            }
            for p in factors
        ]

        if factors:
            top = factors[0]
            explanations["summary"] = descriptions.get(
                top.get("pattern", ""),
                f"Anomaly detected with {top.get('confidence', 0):.0%} confidence.",
            )
        elif explanations["shap_features"]:
            top_feat = explanations["shap_features"][0]["feature"].replace("_", " ")
            explanations["summary"] = (
                f"Model flags anomaly driven primarily by {top_feat}."
            )
        else:
            explanations["summary"] = "No significant anomalous patterns detected."

        explanations["risk_level"] = result.risk_level
        explanations["anomaly_score"] = result.anomaly_score

        return explanations
