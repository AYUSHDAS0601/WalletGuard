"""
training/train_ensemble.py
──────────────────────────
Train the XGBoost + Isolation Forest ensemble on tabular node features
extracted from the Elliptic dataset.

Usage
-----
# Dry-run (synthetic data, 5 folds):
python training/train_ensemble.py --dry-run

# Real Elliptic features:
python training/train_ensemble.py \
    --features  data/processed/elliptic_features.npy \
    --labels    data/processed/elliptic_labels.npy \
    --out-dir   models/saved/ensemble \
    --n-folds   5

Outputs
-------
  models/saved/ensemble/
      xgboost_model.json   (or xgb_model.pkl)
      iso_forest.pkl
      scaler.pkl
      metrics.json          evaluation metrics per fold + overall
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

# Project root on PYTHONPATH when run from repo root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer  # noqa: E402


# ─────────────────────────────────────────────────────────────── CLI ──────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Ensemble Anomaly Scorer")
    p.add_argument("--features", type=Path, default=None,
                   help="Path to .npy feature matrix (N, F)")
    p.add_argument("--labels", type=Path, default=None,
                   help="Path to .npy label vector (N,)  — 0=legit, 1=illicit, -1=unknown")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "checkpoints",
                   help="Directory to save trained models (default: checkpoints for pipeline)")
    p.add_argument("--n-folds", type=int, default=5,
                   help="Number of stratified CV folds (default: 5)")
    p.add_argument("--contamination", type=float, default=0.05,
                   help="IsolationForest contamination ratio (default: 0.05)")
    p.add_argument("--xgb-estimators", type=int, default=300,
                   help="XGBoost n_estimators (default: 300)")
    p.add_argument("--xgb-depth", type=int, default=6,
                   help="XGBoost max_depth (default: 6)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Decision threshold for binary labels (default: 0.5)")
    p.add_argument("--dry-run", action="store_true",
                   help="Generate synthetic data and run a 2-fold test")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ─────────────────────────────────────────── Data helpers ─────────────────────


def load_elliptic_features(
    features_path: Optional[Path],
    labels_path: Optional[Path],
    dry_run: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) arrays.  Unknown labels (-1) kept for IsoForest; filtered for XGB."""
    if dry_run:
        rng = np.random.default_rng(seed)
        N = 2000
        X = rng.standard_normal((N, 64)).astype(np.float32)
        y = (rng.random(N) > 0.92).astype(int)          # ~8 % illicit
        y[rng.choice(N, size=200, replace=False)] = -1   # unlabelled
        logger.info(f"[dry-run] Synthetic dataset: {N} nodes, 64 features, "
                    f"{(y==1).sum()} illicit, {(y==-1).sum()} unknown")
        return X, y

    if features_path is None or labels_path is None:
        raise ValueError(
            "Provide --features and --labels paths, or use --dry-run."
        )

    X = np.load(features_path).astype(np.float32)
    y = np.load(labels_path).astype(int)
    logger.info(f"Loaded features {X.shape}, labels {y.shape}  "
                f"(illicit={( y==1).sum()}, legit={(y==0).sum()}, unknown={(y==-1).sum()})")
    return X, y


# ──────────────────────────────────────────── Evaluation helper ───────────────


def evaluate_fold(
    scorer: EnsembleAnomalyScorer,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float,
    fold: int,
) -> dict:
    labelled = y_test != -1
    X_l, y_l = X_test[labelled], y_test[labelled]

    scores = scorer.predict(X_test)
    scores_l = scores[labelled]

    preds = (scores_l >= threshold).astype(int)

    metrics = {
        "fold": fold,
        "n_test": int(labelled.sum()),
        "illicit_pct": float((y_l == 1).mean() * 100),
        "roc_auc":    float(roc_auc_score(y_l, scores_l)),
        "avg_precision": float(average_precision_score(y_l, scores_l)),
        "f1":         float(f1_score(y_l, preds, zero_division=0)),
        "precision":  float(precision_score(y_l, preds, zero_division=0)),
        "recall":     float(recall_score(y_l, preds, zero_division=0)),
    }
    logger.info(
        f"Fold {fold:2d} | AUC={metrics['roc_auc']:.4f}  "
        f"AP={metrics['avg_precision']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"P={metrics['precision']:.4f}  "
        f"R={metrics['recall']:.4f}"
    )
    return metrics


# ──────────────────────────────────────────────────────────────── main ────────


def main() -> None:
    args = parse_args()
    logger.info("=== Ensemble Training Started ===")
    logger.info(f"Config: folds={args.n_folds}, contamination={args.contamination}, "
                f"xgb_estimators={args.xgb_estimators}, threshold={args.threshold}")

    # ── Load data ──────────────────────────────────────────────────────────────
    X, y = load_elliptic_features(args.features, args.labels, args.dry_run, args.seed)

    n_folds = 2 if args.dry_run else args.n_folds

    # Only labelled samples can be stratified
    labelled_mask = y != -1
    X_lab, y_lab = X[labelled_mask], y[labelled_mask]
    X_unl = X[~labelled_mask]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=args.seed)

    fold_metrics: list[dict] = []
    best_auc = -1.0
    best_fold = -1

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_lab, y_lab), start=1):
        logger.info(f"── Fold {fold_idx}/{n_folds} ──────────────────────────────")

        X_train_l, y_train_l = X_lab[train_idx], y_lab[train_idx]
        X_val_l, y_val_l     = X_lab[val_idx],   y_lab[val_idx]

        # Augment training with all unlabelled data (IsoForest uses them all)
        X_train_full = np.vstack([X_train_l, X_unl]) if len(X_unl) > 0 else X_train_l
        y_train_full = np.concatenate([y_train_l, np.full(len(X_unl), -1)])

        scorer = EnsembleAnomalyScorer(
            contamination=args.contamination,
            xgb_n_estimators=args.xgb_estimators,
            xgb_max_depth=args.xgb_depth,
            random_state=args.seed,
        )

        scorer.fit_traditional(
            X_train=X_train_full,
            y_train=y_train_full,
            X_val=X_val_l,
            y_val=y_val_l,
        )

        m = evaluate_fold(scorer, X_val_l, y_val_l, args.threshold, fold_idx)
        fold_metrics.append(m)

        if m["roc_auc"] > best_auc:
            best_auc = m["roc_auc"]
            best_fold = fold_idx
            best_scorer = scorer

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    agg: dict = {}
    for key in ["roc_auc", "avg_precision", "f1", "precision", "recall"]:
        vals = [m[key] for m in fold_metrics]
        agg[f"mean_{key}"] = float(np.mean(vals))
        agg[f"std_{key}"]  = float(np.std(vals))

    logger.info("=== Cross-Validation Summary ===")
    logger.info(f"  Mean AUC:          {agg['mean_roc_auc']:.4f} ± {agg['std_roc_auc']:.4f}")
    logger.info(f"  Mean F1:           {agg['mean_f1']:.4f} ± {agg['std_f1']:.4f}")
    logger.info(f"  Mean Avg Precision:{agg['mean_avg_precision']:.4f} ± {agg['std_avg_precision']:.4f}")
    logger.info(f"  Best fold: {best_fold} (AUC={best_auc:.4f})")

    # ── Save best model ───────────────────────────────────────────────────────
    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_scorer.save(str(args.out_dir))

    metrics_out = {
        "args": {
            "n_folds": n_folds,
            "contamination": args.contamination,
            "xgb_n_estimators": args.xgb_estimators,
            "xgb_max_depth": args.xgb_depth,
            "threshold": args.threshold,
        },
        "folds": fold_metrics,
        "aggregate": agg,
        "best_fold": best_fold,
        "best_roc_auc": best_auc,
    }
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_out, indent=2))

    logger.info(f"Models saved → {args.out_dir}")
    logger.info(f"Metrics saved → {metrics_path}")
    logger.info("=== Training Complete ===")


if __name__ == "__main__":
    main()
