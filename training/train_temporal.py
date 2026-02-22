"""
BiLSTM Training Script — trains TemporalAnomalyDetector on wallet transaction sequences.

Builds sequences from Elliptic dataset's time-step structure or Etherscan data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import settings
from data.loaders.elliptic_loader import EllipticLoader
from data.processors.temporal_windowing import TemporalWindowing
from models.temporal.bilstm_detector import TemporalAnomalyDetector, make_padding_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BiLSTM on transaction sequences")
    p.add_argument("--epochs", type=int, default=settings.model.num_epochs)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=settings.model.learning_rate)
    p.add_argument("--hidden", type=int, default=settings.model.bilstm_hidden_size)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--data-dir", type=str, default="./data/raw/elliptic")
    p.add_argument("--checkpoint-dir", type=str, default=str(settings.model.checkpoint_dir))
    return p.parse_args()


def build_sequence_tensors_from_elliptic(
    data_dir: str, max_seq_len: int = 100
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build per-node sequences from Elliptic graph using neighbor time-ordering.

    For each node: collect 1-hop neighbors, order by time_step, and use their
    features as the temporal sequence. Falls back to tile+jitter when no edges.
    """
    loader = EllipticLoader(data_dir=data_dir)
    data, _ = loader.load_time_splits()

    X_full = data.x.numpy()
    y_full = data.y.numpy()
    ts = data.time_step.numpy()
    edge_index = data.edge_index.numpy()

    labelled_mask = y_full != -1
    input_size = min(settings.model.bilstm_input_size, X_full.shape[1])
    X_full = X_full[:, :input_size].astype(np.float32)

    # Build adj list: node -> [(neighbor_idx, time_step), ...]
    from collections import defaultdict
    adj: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for e in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, e]), int(edge_index[1, e])
        adj[src].append((dst, int(ts[dst])))

    rng = np.random.default_rng(42)
    sequences_list = []
    y_list = []
    ts_list = []

    for i in np.where(labelled_mask)[0]:
        seqs = []
        neighbors = adj.get(i, [])
        if neighbors:
            # Sort by time_step (temporal order)
            neighbors.sort(key=lambda t: t[1])
            for j, _ in neighbors[:max_seq_len]:
                seqs.append(X_full[j])
        if len(seqs) < 2:
            # Fallback: own features + jitter across time (original behavior)
            base = X_full[i]
            seqs = [
                base + rng.normal(0, 0.05, input_size)
                for _ in range(max_seq_len)
            ]
        # Pad/trim to max_seq_len
        while len(seqs) < max_seq_len:
            seqs.append(seqs[-1] if seqs else np.zeros(input_size, dtype=np.float32))
        seqs = seqs[:max_seq_len]
        sequences_list.append(np.stack(seqs, axis=0))
        y_list.append(y_full[i])
        ts_list.append(ts[i])

    X_tensor = torch.tensor(np.stack(sequences_list, axis=0), dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float)
    ts_arr = np.array(ts_list)

    train_idx = ts_arr <= 34
    val_idx = (ts_arr > 34) & (ts_arr <= 39)
    test_idx = ts_arr > 39

    return (
        X_tensor[train_idx], y_tensor[train_idx],
        X_tensor[val_idx], y_tensor[val_idx],
        X_tensor[test_idx], y_tensor[test_idx],
    )


def run_epoch(
    model: TemporalAnomalyDetector,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    is_train: bool,
    dry_run: bool = False,
) -> tuple[float, float, float, float, float]:
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for step, (X_batch, y_batch) in enumerate(loader):
        if dry_run and step >= 2:
            break

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pad_mask = make_padding_mask(X_batch).to(device)

        with autocast(enabled=device.type == "cuda"):
            prob, _ = model(X_batch, src_key_padding_mask=pad_mask)
            loss = criterion(prob, y_batch)

        if is_train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        preds = (prob.detach() > 0.5).int().cpu().numpy()
        probs = prob.detach().cpu().numpy()
        lbls = y_batch.int().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(lbls)
        all_probs.extend(probs)

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    n = step + 1
    loss_avg = total_loss / n
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return loss_avg, prec, rec, f1, auc


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Building sequence dataset...")
    result = build_sequence_tensors_from_elliptic(
        args.data_dir, max_seq_len=settings.model.max_sequence_length
    )
    X_train, y_train, X_val, y_val, X_test, y_test = result

    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Class weights
    pos_ratio = y_train.mean().item()
    pos_weight = torch.tensor([(1 - pos_ratio) / max(pos_ratio, 1e-6)]).to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=args.batch_size)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=args.batch_size)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TemporalAnomalyDetector(
        input_size=settings.model.bilstm_input_size,
        hidden_size=args.hidden,
        num_layers=settings.model.bilstm_num_layers,
        dropout=settings.model.bilstm_dropout,
    ).to(device)

    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # Note: criterion handles logits but model returns sigmoid — switch to BCELoss
    criterion = nn.BCELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=device.type == "cuda")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_prec, tr_rec, tr_f1, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device, True, args.dry_run
        )
        with torch.no_grad():
            val_loss, val_prec, val_rec, val_f1, val_auc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device, False, args.dry_run
            )

        scheduler.step()

        logger.info(
            f"Epoch {epoch:03d} | Train F1 {tr_f1:.4f} | Val F1 {val_f1:.4f} AUC {val_auc:.4f}"
        )
        history.append({"epoch": epoch, "train_f1": tr_f1, "val_f1": val_f1, "val_auc": val_auc})

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save(str(checkpoint_dir / "bilstm_best.pth"))
            logger.info(f"  ✓ Checkpoint saved (val F1={val_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= settings.model.early_stopping_patience:
                logger.info("Early stopping")
                break

        if args.dry_run and epoch >= 2:
            break

    # Test evaluation
    best = checkpoint_dir / "bilstm_best.pth"
    if best.exists():
        model = TemporalAnomalyDetector.load(str(best), device=str(device))

    with torch.no_grad():
        _, t_prec, t_rec, t_f1, t_auc = run_epoch(
            model, test_loader, criterion, optimizer, scaler, device, False, args.dry_run
        )

    logger.info(
        f"\nTest → Precision: {t_prec:.4f}  Recall: {t_rec:.4f}  F1: {t_f1:.4f}  AUC: {t_auc:.4f}"
    )

    with open(checkpoint_dir / "bilstm_training_history.json", "w") as f:
        json.dump({"history": history, "test": {"f1": t_f1, "auc": t_auc}}, f, indent=2)


if __name__ == "__main__":
    main()
