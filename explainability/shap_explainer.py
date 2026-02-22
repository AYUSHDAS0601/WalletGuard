"""
SHAP Explainability — generates feature-level explanations for anomaly predictions.
"""
from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np
from loguru import logger

warnings.filterwarnings("ignore", category=UserWarning)

# Feature names matching the combined feature matrix in FeatureEngineer
FEATURE_NAMES = [
    "value_eth", "tx_fee_eth", "gas_price_gwei",
    "value_percentile", "gas_percentile",
    "hour_of_day", "day_of_week", "is_weekend",
    "is_contract_creation", "is_zero_value", "is_self_transfer",
    "to_mixer", "from_mixer", "to_exchange", "tx_position_in_block",
    # Wallet features
    "sender_total_txs", "sender_total_value_sent", "sender_total_value_received",
    "sender_avg_tx_value", "sender_std_tx_value", "sender_tx_frequency_per_day",
    "sender_unique_counterparties", "sender_mixer_interaction_count",
    "sender_self_transfer_count", "sender_hour_entropy",
]


class SHAPExplainer:
    """
    Wraps XGBoost SHAP TreeExplainer for feature-importance explanations.

    Usage:
        explainer = SHAPExplainer(scorer.xgb_model)
        explanations = explainer.explain(X)
        report = explainer.generate_report(tx_hash, X)
    """

    def __init__(self, model=None, feature_names: Optional[List[str]] = None):
        self.model = model
        self.feature_names = feature_names or FEATURE_NAMES
        self._shap_explainer = None

        if model is not None:
            self._init_explainer()

    def _init_explainer(self) -> None:
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP TreeExplainer initialized")
        except Exception as e:
            logger.warning(f"SHAP init failed — explanations will be limited: {e}")

    # ─────────────────────────────────────────────────── Public API ───────────

    def explain(
        self, X: np.ndarray, top_k: int = 10
    ) -> List[Dict]:
        """
        Compute SHAP values for a feature matrix.

        Returns a list of top-k feature explanations per sample (averaged if N > 1).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self._shap_explainer is not None:
            return self._shap_explain(X, top_k)
        else:
            return self._fallback_explain(X, top_k)

    def _shap_explain(self, X: np.ndarray, top_k: int) -> List[Dict]:
        """Full SHAP-based explanation."""
        try:
            import shap
            shap_values = self._shap_explainer.shap_values(X)

            # Handle multi-output (binary XGBoost returns list)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class

            # Average absolute SHAP across samples
            mean_abs = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(mean_abs)[::-1][:top_k]

            explanations = []
            for idx in top_indices:
                name = (
                    self.feature_names[idx]
                    if idx < len(self.feature_names)
                    else f"feature_{idx}"
                )
                sv = float(shap_values[:, idx].mean())
                fv = float(X[:, idx].mean())
                explanations.append({
                    "feature": name,
                    "shap_value": round(sv, 6),
                    "feature_value": round(fv, 6),
                    "contribution": "increases_risk" if sv > 0 else "decreases_risk",
                    "importance": round(float(mean_abs[idx]), 6),
                })

            return explanations

        except Exception as e:
            logger.warning(f"SHAP explain failed: {e}")
            return self._fallback_explain(X, top_k)

    def _fallback_explain(self, X: np.ndarray, top_k: int) -> List[Dict]:
        """
        Fallback: rank features by absolute deviation from mean
        (useful when SHAP is unavailable or model not fitted).
        """
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-9
        z_scores = np.abs((X.mean(axis=0) - mean) / std)
        top_indices = np.argsort(z_scores)[::-1][:top_k]

        return [
            {
                "feature": self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}",
                "shap_value": float(z_scores[i]),
                "feature_value": float(X[:, i].mean()),
                "contribution": "increases_risk",
                "importance": float(z_scores[i]),
            }
            for i in top_indices
        ]

    def generate_report(
        self,
        identifier: str,           # tx_hash or wallet address
        X: np.ndarray,
        anomaly_score: float,
        risk_level: str,
        pattern_types: Optional[List[str]] = None,
    ) -> Dict:
        """
        Generate a structured anomaly report with explanations.
        """
        explanations = self.explain(X, top_k=10)

        report = {
            "identifier": identifier,
            "anomaly_score": round(anomaly_score, 4),
            "risk_level": risk_level,
            "detected_patterns": pattern_types or [],
            "summary": self._generate_summary(risk_level, pattern_types or [], explanations),
            "top_features": explanations[:5],
            "all_features": explanations,
            "recommendations": self._recommendations(risk_level, pattern_types or []),
        }

        return report

    @staticmethod
    def _generate_summary(
        risk_level: str, patterns: List[str], explanations: List[Dict]
    ) -> str:
        """Natural language summary of the analysis."""
        if not patterns and risk_level == "LOW":
            return "No significant anomalous behaviour detected in the analysed transactions."

        pattern_desc = {
            "wash_trading": "circular wash-trading",
            "flash_loan_flash_loan": "flash loan exploitation",
            "flash_loan_mev_sandwich": "MEV sandwich attack",
            "market_manipulation_pump_dump": "a pump-and-dump scheme",
            "market_manipulation_coordinated_buy": "coordinated buying",
            "coordinated_wallets": "coordinated multi-wallet activity",
        }

        pattern_strs = [pattern_desc.get(p, p.replace("_", " ")) for p in patterns[:3]]
        top_feature = explanations[0]["feature"].replace("_", " ") if explanations else "transaction pattern"

        return (
            f"This {risk_level.lower()}-risk entity shows signs of "
            f"{', '.join(pattern_strs) if pattern_strs else 'anomalous behaviour'}, "
            f"driven primarily by {top_feature}."
        )

    @staticmethod
    def _recommendations(risk_level: str, patterns: List[str]) -> List[str]:
        recs = []
        if risk_level in ("CRITICAL", "HIGH"):
            recs.append("Flag for immediate manual review.")
            recs.append("Cross-reference with sanctions screening databases.")
        if "wash_trading" in patterns:
            recs.append("Report suspected wash trading to relevant exchange compliance teams.")
        if "flash_loan" in " ".join(patterns):
            recs.append("Notify affected DeFi protocol security teams.")
        if "coordinated" in " ".join(patterns):
            recs.append("Trace common funding sources across the wallet cluster.")
        if risk_level == "LOW":
            recs.append("Continue standard monitoring — no immediate action required.")
        return recs
