#!/usr/bin/env python3
"""
Run full evaluation (GNN, BiLSTM, Ensemble) and generate graphical results.

Optimized for: Coordinated Wallet Attacks + Market Manipulation (wash trading, pump & dump).
Produces:
  - evaluation_results.json (with accuracy, precision, recall, F1, AUC-ROC, AUC-PR)
  - checkpoints/graphs/roc_curves.png
  - checkpoints/graphs/precision_recall_curves.png
  - checkpoints/graphs/confusion_matrices.png
  - checkpoints/graphs/metrics_summary.png (accuracy + F1 + precision + recall + AUC)

Usage:
    python scripts/run_evaluation_with_graphs.py
    python scripts/run_evaluation_with_graphs.py --data-dir ./data/raw/elliptic --checkpoint-dir ./checkpoints
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config.config import settings
from data.loaders.elliptic_loader import EllipticLoader
from models.gnn.blockchain_gnn import BlockchainGNN
from models.temporal.bilstm_detector import TemporalAnomalyDetector, make_padding_mask
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def load_test_data(data_dir: str):
    """Load Elliptic test split and return (X, y, edge_index, test_mask, data) for GNN/BiLSTM/Ensemble."""
    loader = EllipticLoader(data_dir=data_dir)
    data, splits = loader.load_time_splits()
    test_mask = splits["test"].numpy()

    y_full = data.y.numpy()
    labelled = y_full != -1
    test_labelled = test_mask & labelled

    X_full = data.x.numpy().astype(np.float32)
    if X_full.shape[1] > 64:
        X_64 = X_full[:, :64]
    else:
        X_64 = X_full

    y_test = y_full[test_labelled]
    X_test = X_64[test_labelled]

    return {
        "data": data,
        "splits": splits,
        "test_mask": test_mask,
        "test_labelled": test_labelled,
        "X_full": X_full,
        "X_test": X_test,
        "y_test": y_test,
        "n_test": len(y_test),
    }


def evaluate_gnn(checkpoint_path: Path, test_data: dict, device: torch.device):
    """Return (y_true, y_prob, y_pred, metrics_dict)."""
    model = BlockchainGNN.load(str(checkpoint_path), device=str(device))
    model.eval()

    data = test_data["data"].to(device)
    test_mask = test_data["test_mask"]
    test_labelled = test_data["test_labelled"]
    y_full = test_data["data"].y.cpu().numpy()

    with torch.no_grad():
        probs = model.predict(data.x, data.edge_index).cpu().numpy()

    y_true = y_full[test_labelled]
    y_prob = probs[test_labelled]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _metrics_dict(y_true, y_pred, y_prob, "GNN")
    return y_true, y_prob, y_pred, metrics


def evaluate_bilstm(checkpoint_path: Path, test_data: dict, data_dir: str, device: torch.device):
    """Return (y_true, y_prob, y_pred, metrics_dict)."""
    from torch.utils.data import DataLoader, TensorDataset
    from training.train_temporal import build_sequence_tensors_from_elliptic

    result = build_sequence_tensors_from_elliptic(
        data_dir, max_seq_len=settings.model.max_sequence_length
    )
    _, _, _, _, X_test, y_test = result

    model = TemporalAnomalyDetector.load(str(checkpoint_path), device=str(device))
    model.eval()

    all_probs = []
    loader = DataLoader(TensorDataset(X_test), batch_size=64)
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            mask = make_padding_mask(X_batch).to(device)
            prob, _ = model(X_batch, src_key_padding_mask=mask)
            all_probs.extend(prob.cpu().numpy())

    y_true = y_test.numpy().astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = _metrics_dict(y_true, y_pred, y_prob, "BiLSTM")
    return y_true, y_prob, y_pred, metrics


def evaluate_ensemble(checkpoint_dir: Path, test_data: dict):
    """Return (y_true, y_prob, y_pred, metrics_dict)."""
    from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer

    scorer = EnsembleAnomalyScorer.load(str(checkpoint_dir))
    X_test = test_data["X_test"]
    y_true = test_data["y_test"]

    if len(X_test) == 0:
        return y_true, np.zeros_like(y_true, dtype=np.float32), np.zeros_like(y_true), {}

    y_prob = scorer.predict(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = _metrics_dict(y_true, y_pred, y_prob, "Ensemble")
    return y_true, y_prob, y_pred, metrics


def _metrics_dict(y_true, y_pred, y_prob, model_name: str):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    except ValueError:
        auc_roc = auc_pr = 0.5

    return {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n_samples": len(y_true),
        "n_anomalies": int(y_true.sum()),
    }


def plot_roc_curves(results: list, out_path: Path):
    """Plot ROC curves for each model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_true, y_prob in results:
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Coordinated Wallets & Market Manipulation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_pr_curves(results: list, out_path: Path):
    """Plot Precision-Recall curves."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    for name, y_true, y_prob in results:
        if len(np.unique(y_true)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, lw=2, label=f"{name} (AP = {auc_pr:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_confusion_matrices(metrics_per_model: dict, out_path: Path):
    """Plot confusion matrix heatmap per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    models = [k for k in metrics_per_model if "confusion_matrix" in metrics_per_model[k]]
    if not models:
        return

    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, name in zip(axes, models):
        cm = np.array(metrics_per_model[name]["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Normal", "Anomaly"],
            yticklabels=["Normal", "Anomaly"],
        )
        ax.set_title(f"{name}\nAccuracy = {metrics_per_model[name]['accuracy']:.2%}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")

    fig.suptitle("Confusion Matrices (Test Set)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def plot_metrics_summary(metrics_per_model: dict, out_path: Path):
    """Bar chart: Accuracy, F1, Precision, Recall, AUC-ROC per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(metrics_per_model.keys())
    metrics_names = ["accuracy", "f1_score", "precision", "recall", "auc_roc"]
    labels = ["Accuracy", "F1-Score", "Precision", "Recall", "AUC-ROC"]

    x = np.arange(len(labels))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [metrics_per_model[model].get(m, 0) for m in metrics_names]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=model)

    ax.set_ylabel("Score")
    ax.set_title("Model Performance — Coordinated Wallets & Market Manipulation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser(description="Run evaluation and generate graphs")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "elliptic"))
    p.add_argument("--checkpoint-dir", default=str(settings.model.checkpoint_dir))
    p.add_argument("--out-dir", default=None, help="Defaults to checkpoint-dir/graphs")
    args = p.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (checkpoint_dir / "graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    test_data = load_test_data(args.data_dir)
    logger.info(f"Test samples (labelled): {test_data['n_test']}")

    all_metrics = {}
    roc_data = []  # (model_name, y_true, y_prob)

    # GNN
    gnn_path = checkpoint_dir / "gnn_best.pth"
    if gnn_path.exists():
        y_true, y_prob, y_pred, all_metrics["gnn"] = evaluate_gnn(gnn_path, test_data, device)
        roc_data.append(("GNN", y_true, y_prob))
        logger.info(f"GNN Accuracy: {all_metrics['gnn']['accuracy']:.4f}")
    else:
        logger.warning(f"GNN checkpoint not found: {gnn_path}")

    # BiLSTM
    bilstm_path = checkpoint_dir / "bilstm_best.pth"
    if bilstm_path.exists():
        y_true, y_prob, y_pred, all_metrics["bilstm"] = evaluate_bilstm(
            bilstm_path, test_data, args.data_dir, device
        )
        roc_data.append(("BiLSTM", y_true, y_prob))
        logger.info(f"BiLSTM Accuracy: {all_metrics['bilstm']['accuracy']:.4f}")
    else:
        logger.warning(f"BiLSTM checkpoint not found: {bilstm_path}")

    # Ensemble
    for ensemble_dir in [checkpoint_dir, ROOT / "models" / "saved" / "ensemble"]:
        if (ensemble_dir / "xgboost_model.json").exists() or (ensemble_dir / "xgb_model.pkl").exists():
            y_true, y_prob, y_pred, all_metrics["ensemble"] = evaluate_ensemble(
                ensemble_dir, test_data
            )
            roc_data.append(("Ensemble", y_true, y_prob))
            logger.info(f"Ensemble Accuracy: {all_metrics['ensemble']['accuracy']:.4f}")
            break

    if not all_metrics:
        logger.error("No models found. Train GNN/BiLSTM/Ensemble first.")
        return 1

    # Save JSON
    json_path = checkpoint_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved {json_path}")

    # Plots
    if roc_data:
        plot_roc_curves(roc_data, out_dir / "roc_curves.png")
        plot_pr_curves(roc_data, out_dir / "precision_recall_curves.png")
    plot_confusion_matrices(all_metrics, out_dir / "confusion_matrices.png")
    plot_metrics_summary(all_metrics, out_dir / "metrics_summary.png")

    # Print summary table
    logger.info("\n" + "=" * 60)
    logger.info("  ACCURACY & METRICS SUMMARY (Coordinated + Market Manipulation)")
    logger.info("=" * 60)
    for name, m in all_metrics.items():
        logger.info(
            f"  {name:10} | Acc: {m['accuracy']:.4f} | F1: {m['f1_score']:.4f} | "
            f"P: {m['precision']:.4f} | R: {m['recall']:.4f} | AUC: {m['auc_roc']:.4f}"
        )
    logger.info("=" * 60)
    logger.info(f"Graphs saved to: {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
