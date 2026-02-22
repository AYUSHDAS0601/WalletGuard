"""
Ensemble Anomaly Scorer — combines GNN, BiLSTM, XGBoost, and Isolation Forest.

Weighted ensemble:
  GNN       40%
  BiLSTM    30%
  XGBoost   20%
  IsoForest 10%
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False
    logger.warning("XGBoost not installed — falling back to sklearn GradientBoosting")
    from sklearn.ensemble import GradientBoostingClassifier


_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EnsembleAnomalyScorer:
    """
    Combines neural (GNN + BiLSTM) and traditional (XGBoost + IsoForest) models
    into a weighted ensemble for robust anomaly scoring.

    The GNN and BiLSTM scores are expected to be precomputed tensors/arrays
    (e.g., from a previous forward pass). The XGBoost and IsoForest are trained
    on tabular feature matrices.

    Attributes:
        weights:   Dict controlling each model's contribution.
        xgb_model: Trained XGBoost (or GradientBoosting) classifier.
        iso_forest: Trained Isolation Forest.
        scaler:    Feature scaler for traditional models.
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        contamination: float = 0.05,
        xgb_n_estimators: int = 200,
        xgb_max_depth: int = 6,
        iso_n_estimators: int = 100,
        random_state: int = 42,
    ) -> None:
        self.weights = weights or {
            "gnn": 0.4,
            "temporal": 0.3,
            "xgboost": 0.2,
            "isolation": 0.1,
        }
        # Normalise weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # ── Traditional models ─────────────────────────────────────────────────
        if _XGB_AVAILABLE:
            device_arg = "cuda" if _DEVICE == "cuda" else "cpu"
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device=device_arg,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=random_state,
            )
        else:
            from sklearn.ensemble import GradientBoostingClassifier
            self.xgb_model = GradientBoostingClassifier(
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                random_state=random_state,
            )

        self.iso_forest = IsolationForest(
            n_estimators=iso_n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )

        self.scaler = MinMaxScaler()
        self._xgb_fitted = False
        self._iso_fitted = False
        self._scaler_fitted = False

    # ─────────────────────────────────────────────── Training ────────────────

    def fit_traditional(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EnsembleAnomalyScorer":
        """
        Train XGBoost and Isolation Forest on tabular features.

        Args:
            X_train: Labelled training features [N, F]
            y_train: Binary labels (0=normal, 1=anomalous) [N]
            X_val, y_val: Optional validation set for XGBoost early stopping.
        """
        logger.info(f"Fitting traditional models on {len(X_train)} samples")

        X_train_scaled = self.scaler.fit_transform(X_train)
        self._scaler_fitted = True

        # XGBoost
        fit_kwargs = {}
        if X_val is not None and y_val is not None and _XGB_AVAILABLE:
            X_val_scaled = self.scaler.transform(X_val)
            fit_kwargs["eval_set"] = [(X_val_scaled, y_val)]
            fit_kwargs["verbose"] = False

        # Filter only labelled samples
        labelled = y_train != -1
        self.xgb_model.fit(X_train_scaled[labelled], y_train[labelled], **fit_kwargs)
        self._xgb_fitted = True

        # Isolation Forest (unsupervised — uses all data)
        self.iso_forest.fit(X_train_scaled)
        self._iso_fitted = True

        logger.info("Traditional models fitted")
        return self

    # ─────────────────────────────────────────────── Prediction ──────────────

    def predict(
        self,
        X_features: np.ndarray,
        gnn_scores: Optional[np.ndarray] = None,
        temporal_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute final ensemble anomaly score in [0, 1].

        Args:
            X_features:     Tabular feature matrix   [N, F]
            gnn_scores:     GNN anomaly probs         [N]  or None
            temporal_scores: BiLSTM anomaly probs    [N]  or None

        Returns:
            scores: Ensemble anomaly scores [N]  — higher = more anomalous
        """
        N = len(X_features)
        scores = np.zeros(N, dtype=np.float32)
        weight_sum = 0.0

        # ── XGBoost ───────────────────────────────────────────────────────────
        if self._xgb_fitted:
            X_sc = self.scaler.transform(X_features) if self._scaler_fitted else X_features
            xgb_prob = self.xgb_model.predict_proba(X_sc)[:, 1].astype(np.float32)
            scores += self.weights["xgboost"] * xgb_prob
            weight_sum += self.weights["xgboost"]

        # ── Isolation Forest ──────────────────────────────────────────────────
        if self._iso_fitted:
            X_sc = self.scaler.transform(X_features) if self._scaler_fitted else X_features
            # score_samples returns negative values; more negative = more anomalous
            raw = self.iso_forest.score_samples(X_sc)
            iso_score = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
            iso_score = iso_score.astype(np.float32)
            scores += self.weights["isolation"] * iso_score
            weight_sum += self.weights["isolation"]

        # ── GNN ───────────────────────────────────────────────────────────────
        if gnn_scores is not None:
            gnn = np.clip(np.asarray(gnn_scores, dtype=np.float32), 0, 1)
            scores += self.weights["gnn"] * gnn
            weight_sum += self.weights["gnn"]

        # ── BiLSTM ────────────────────────────────────────────────────────────
        if temporal_scores is not None:
            temp = np.clip(np.asarray(temporal_scores, dtype=np.float32), 0, 1)
            scores += self.weights["temporal"] * temp
            weight_sum += self.weights["temporal"]

        # Re-normalise to [0, 1] based on actually used weights
        if weight_sum > 0:
            scores /= weight_sum

        return np.clip(scores, 0, 1)

    def predict_labels(
        self,
        X_features: np.ndarray,
        threshold: float = 0.7,
        gnn_scores: Optional[np.ndarray] = None,
        temporal_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return binary labels (1=anomaly) given a threshold."""
        scores = self.predict(X_features, gnn_scores, temporal_scores)
        return (scores >= threshold).astype(int)

    def risk_level(self, score: float) -> str:
        """Map a float score to a human-readable risk level."""
        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.70:
            return "HIGH"
        if score >= 0.50:
            return "MEDIUM"
        return "LOW"

    # ─────────────────────────────────────────────── Persistence ─────────────

    def save(self, dir_path: str) -> None:
        """Save both models and scaler to a directory."""
        path = Path(dir_path)
        path.mkdir(parents=True, exist_ok=True)

        if _XGB_AVAILABLE and self._xgb_fitted:
            self.xgb_model.save_model(str(path / "xgboost_model.json"))
        elif self._xgb_fitted:
            with open(path / "xgb_model.pkl", "wb") as f:
                pickle.dump(self.xgb_model, f)

        if self._iso_fitted:
            with open(path / "iso_forest.pkl", "wb") as f:
                pickle.dump(self.iso_forest, f)

        if self._scaler_fitted:
            with open(path / "scaler.pkl", "wb") as f:
                pickle.dump(self.scaler, f)

        logger.info(f"Ensemble models saved to {path}")

    @classmethod
    def load(cls, dir_path: str, **init_kwargs) -> "EnsembleAnomalyScorer":
        """Load from a previously saved directory."""
        path = Path(dir_path)
        scorer = cls(**init_kwargs)

        xgb_json = path / "xgboost_model.json"
        xgb_pkl = path / "xgb_model.pkl"
        if xgb_json.exists() and _XGB_AVAILABLE:
            scorer.xgb_model.load_model(str(xgb_json))
            scorer._xgb_fitted = True
        elif xgb_pkl.exists():
            with open(xgb_pkl, "rb") as f:
                scorer.xgb_model = pickle.load(f)
            scorer._xgb_fitted = True

        iso_pkl = path / "iso_forest.pkl"
        if iso_pkl.exists():
            with open(iso_pkl, "rb") as f:
                scorer.iso_forest = pickle.load(f)
            scorer._iso_fitted = True

        scaler_pkl = path / "scaler.pkl"
        if scaler_pkl.exists():
            with open(scaler_pkl, "rb") as f:
                scorer.scaler = pickle.load(f)
            scorer._scaler_fitted = True

        logger.info(f"Ensemble models loaded from {path}")
        return scorer
