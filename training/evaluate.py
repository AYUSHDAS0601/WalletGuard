"""
Evaluation Script — computes comprehensive metrics for all trained models.

Outputs:
  - Precision, Recall, F1, AUC-ROC, AUC-PR
  - Confusion matrix
  - Per-class metrics
  - Saves metrics JSON to checkpoint dir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import settings
from data.loaders.elliptic_loader import EllipticLoader
from models.gnn.blockchain_gnn import BlockchainGNN
from models.temporal.bilstm_detector import TemporalAnomalyDetector


def evaluate_gnn(checkpoint_path: str, data_dir: str) -> dict:
    """Evaluate saved GNN model on Elliptic test split."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = EllipticLoader(data_dir=data_dir)
    data, splits = loader.load_time_splits()
    test_mask = splits["test"]

    model = BlockchainGNN.load(checkpoint_path, device=str(device))
    model.eval()

    with torch.no_grad():
        data = data.to(device)
        probs = model.predict(data.x, data.edge_index).cpu().numpy()

    y_true = data.y.cpu().numpy()[test_mask.numpy()]
    y_prob = probs[test_mask.numpy()]
    y_pred = (y_prob >= 0.5).astype(int)

    # Only evaluate on labelled nodes
    labelled = y_true != -1
    y_true = y_true[labelled]
    y_prob = y_prob[labelled]
    y_pred = y_pred[labelled]

    metrics = _compute_metrics(y_true, y_pred, y_prob, model_name="GNN")
    return metrics


def evaluate_bilstm(checkpoint_path: str, data_dir: str) -> dict:
    """Evaluate saved BiLSTM model."""
    from training.train_temporal import build_sequence_tensors_from_elliptic
    from torch.utils.data import DataLoader, TensorDataset
    from models.temporal.bilstm_detector import make_padding_mask

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = build_sequence_tensors_from_elliptic(
        data_dir, max_seq_len=settings.model.max_sequence_length
    )
    _, _, _, _, X_test, y_test = result

    model = TemporalAnomalyDetector.load(checkpoint_path, device=str(device))
    model.eval()

    all_probs = []
    loader = DataLoader(TensorDataset(X_test), batch_size=64)
    with torch.no_grad():
        for (X_batch,) in loader:
            X_batch = X_batch.to(device)
            mask = make_padding_mask(X_batch).to(device)
            prob, _ = model(X_batch, src_key_padding_mask=mask)
            all_probs.extend(prob.cpu().numpy())

    y_prob = np.array(all_probs)
    y_true = y_test.numpy().astype(int)
    y_pred = (y_prob >= 0.5).astype(int)

    return _compute_metrics(y_true, y_pred, y_prob, model_name="BiLSTM")


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
) -> dict:
    """Compute all evaluation metrics including accuracy."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    except ValueError:
        auc_roc = auc_pr = 0.5

    cm = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=["Normal", "Anomaly"], output_dict=True)

    metrics = {
        "model": model_name,
        "accuracy": round(float(acc), 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "auc_roc": round(auc_roc, 4),
        "auc_pr": round(auc_pr, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(y_true),
        "n_anomalies": int(y_true.sum()),
        "anomaly_rate": round(float(y_true.mean()), 4),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"  {model_name} Evaluation")
    logger.info(f"{'='*50}")
    logger.info(f"  Accuracy:    {acc:.4f}")
    logger.info(f"  Precision:   {prec:.4f}")
    logger.info(f"  Recall:      {rec:.4f}")
    logger.info(f"  F1-Score:    {f1:.4f}")
    logger.info(f"  AUC-ROC:     {auc_roc:.4f}")
    logger.info(f"  AUC-PR:      {auc_pr:.4f}")
    logger.info(f"  Confusion Matrix: {cm}")
    logger.info(f"{'='*50}\n")

    return metrics


def evaluate_ensemble(checkpoint_dir: str, data_dir: str) -> dict:
    """Evaluate ensemble (XGBoost + IsoForest) on Elliptic test split."""
    from models.ensemble.anomaly_scorer import EnsembleAnomalyScorer

    try:
        scorer = EnsembleAnomalyScorer.load(checkpoint_dir)
    except Exception as e:
        logger.warning(f"Could not load ensemble: {e}")
        return {}

    loader = EllipticLoader(data_dir=data_dir)
    data, splits = loader.load_time_splits()
    test_mask = splits["test"]

    X_full = data.x.numpy().astype(np.float32)
    if X_full.shape[1] > 64:
        X_full = X_full[:, :64]
    y_full = data.y.numpy()

    test_idx = test_mask.numpy()
    X_test_all = X_full[test_idx]
    y_test_all = y_full[test_idx]

    labelled = y_test_all != -1
    X_test = X_test_all[labelled]
    y_test = y_test_all[labelled]

    if len(X_test) == 0:
        logger.warning("No labelled test samples for ensemble")
        return {}

    scores = scorer.predict(X_test)
    y_pred = (scores >= 0.5).astype(int)

    return _compute_metrics(y_test, y_pred, scores, model_name="Ensemble")


def main():
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default=str(settings.model.checkpoint_dir))
    p.add_argument("--data-dir", default="./data/raw/elliptic")
    args = p.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    all_metrics = {}

    gnn_path = checkpoint_dir / "gnn_best.pth"
    if gnn_path.exists():
        logger.info("Evaluating GNN...")
        all_metrics["gnn"] = evaluate_gnn(str(gnn_path), args.data_dir)
    else:
        logger.warning(f"GNN checkpoint not found at {gnn_path}")

    bilstm_path = checkpoint_dir / "bilstm_best.pth"
    if bilstm_path.exists():
        logger.info("Evaluating BiLSTM...")
        all_metrics["bilstm"] = evaluate_bilstm(str(bilstm_path), args.data_dir)
    else:
        logger.warning(f"BiLSTM checkpoint not found at {bilstm_path}")

    # Ensemble evaluation (check checkpoint_dir and models/saved/ensemble)
    for ensemble_dir in [checkpoint_dir, Path(__file__).parent.parent / "models" / "saved" / "ensemble"]:
        if (ensemble_dir / "xgboost_model.json").exists() or (ensemble_dir / "xgb_model.pkl").exists():
            logger.info("Evaluating Ensemble...")
            all_metrics["ensemble"] = evaluate_ensemble(str(ensemble_dir), args.data_dir)
            break

    out_path = checkpoint_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    logger.info(f"\nEvaluation results saved to {out_path}")


if __name__ == "__main__":
    main()
