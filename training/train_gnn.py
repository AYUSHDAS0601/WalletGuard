"""
GNN Training Script — trains BlockchainGNN on the Elliptic dataset.

Hardware-optimised for NVIDIA GTX 2050 (4 GB VRAM):
  • Mixed-precision (AMP) with GradScaler
  • Gradient accumulation (effective batch = 256)
  • NeighborLoader mini-batch sampling (2-hop: 10+5 neighbours)
  • Early stopping with best-model checkpointing

Usage:
    python training/train_gnn.py [--epochs N] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.cuda.amp import GradScaler, autocast
from torch_geometric.loader import NeighborLoader

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import settings
from data.loaders.elliptic_loader import EllipticLoader
from models.gnn.blockchain_gnn import BlockchainGNN


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, pos_weight: float = 1.0) -> torch.Tensor:
    """Focal loss for extreme class imbalance (~2% illicit in Elliptic)."""
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=torch.tensor([pos_weight], device=logits.device)
    )
    probs = torch.sigmoid(logits)
    pt = torch.where(targets == 1, probs, 1 - probs)
    focal_weight = (1 - pt) ** gamma
    return (focal_weight * bce).mean()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GNN on Elliptic dataset")
    p.add_argument("--epochs", type=int, default=settings.model.num_epochs)
    p.add_argument("--batch-size", type=int, default=settings.model.batch_size)
    p.add_argument("--lr", type=float, default=settings.model.learning_rate)
    p.add_argument("--hidden", type=int, default=settings.model.gnn_hidden_channels)
    p.add_argument("--dry-run", action="store_true", help="Run 2 batches only (sanity check)")
    p.add_argument("--data-dir", type=str, default="./data/raw/elliptic")
    p.add_argument("--checkpoint-dir", type=str, default=str(settings.model.checkpoint_dir))
    p.add_argument("--focal-loss", action="store_true", help="Use focal loss for class imbalance")
    return p.parse_args()


def build_loaders(data, splits, args, device):
    """Build NeighborLoader train/val/test data loaders."""
    common = dict(
        data=data,
        num_neighbors=settings.model.neighbor_sampling,
        batch_size=args.batch_size,
        num_workers=settings.model.num_workers if not args.dry_run else 0,
        pin_memory=device.type == "cuda",
    )
    train_loader = NeighborLoader(
        input_nodes=splits["train"].nonzero(as_tuple=True)[0],
        shuffle=True,
        **common,
    )
    val_loader = NeighborLoader(
        input_nodes=splits["val"].nonzero(as_tuple=True)[0],
        shuffle=False,
        **common,
    )
    test_loader = NeighborLoader(
        input_nodes=splits["test"].nonzero(as_tuple=True)[0],
        shuffle=False,
        **common,
    )
    return train_loader, val_loader, test_loader


def run_epoch(
    model: BlockchainGNN,
    loader: NeighborLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    accumulation_steps: int,
    is_train: bool,
    dry_run: bool = False,
    use_focal: bool = False,
    pos_weight: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """Run one epoch. Returns (loss, precision, recall, f1, auc)."""
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        if dry_run and step >= 2:
            break

        batch = batch.to(device)

        # Only the first batch_size nodes are the "seed" nodes
        batch_size = batch.batch_size

        # Filter to labelled nodes among the seeds
        labels = batch.y[:batch_size]
        labelled_mask = labels != -1
        if labelled_mask.sum() == 0:
            continue

        labels_labelled = labels[labelled_mask].float()

        with autocast(enabled=device.type == "cuda"):
            _, logits = model(batch.x, batch.edge_index)
            logits = logits.squeeze(-1)[:batch_size][labelled_mask]
            if use_focal:
                loss = focal_loss(logits, labels_labelled, gamma=2.0, pos_weight=pos_weight)
            else:
                loss = criterion(logits, labels_labelled)
            loss = loss / accumulation_steps

        if is_train:
            scaler.scale(loss).backward()
            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        prob = torch.sigmoid(logits.detach())
        preds = (prob > 0.5).int().cpu().numpy()
        probs = prob.cpu().numpy()
        lbls = labels_labelled.int().cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(lbls)
        all_probs.extend(probs)

    if not all_labels:
        return 0.0, 0.0, 0.0, 0.0, 0.5

    n_steps = step + 1
    loss_avg = total_loss / max(n_steps, 1)
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
    logger.info(f"Training config: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # ── Data ──────────────────────────────────────────────────────────────────
    loader = EllipticLoader(data_dir=args.data_dir)
    data, splits = loader.load_time_splits()

    # Pad feature dimension if needed
    in_channels = data.x.shape[1]
    logger.info(f"Feature dim: {in_channels}, Nodes: {data.num_nodes}")

    train_loader, val_loader, test_loader = build_loaders(data, splits, args, device)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = BlockchainGNN(
        in_channels=in_channels,
        hidden_channels=args.hidden,
        out_channels=settings.model.gnn_out_channels,
        num_layers=settings.model.gnn_num_layers,
        dropout=settings.model.gnn_dropout,
    ).to(device)

    logger.info(
        f"Model params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Handle class imbalance (~2% illicit in Elliptic)
    train_labels = data.y[splits["train"]]
    n_positive = (train_labels == 1).sum().item()
    n_negative = (train_labels == 0).sum().item()
    pos_weight_val = n_negative / max(n_positive, 1)
    pos_weight = torch.tensor([pos_weight_val]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) if not args.focal_loss else None

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=settings.model.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5, verbose=True
    )
    scaler = GradScaler(enabled=device.type == "cuda")

    # ── Training loop ─────────────────────────────────────────────────────────
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_f1 = 0.0
    patience_counter = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, tr_prec, tr_rec, tr_f1, tr_auc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, device,
            settings.model.accumulation_steps, is_train=True, dry_run=args.dry_run,
            use_focal=args.focal_loss, pos_weight=pos_weight_val,
        )

        with torch.no_grad():
            val_loss, val_prec, val_rec, val_f1, val_auc = run_epoch(
                model, val_loader, criterion, optimizer, scaler, device,
                1, is_train=False, dry_run=args.dry_run,
                use_focal=args.focal_loss, pos_weight=pos_weight_val,
            )

        scheduler.step(val_f1)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss {train_loss:.4f} F1 {tr_f1:.4f} | "
            f"Val Loss {val_loss:.4f} F1 {val_f1:.4f} AUC {val_auc:.4f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_f1": tr_f1,
            "val_loss": val_loss, "val_f1": val_f1, "val_auc": val_auc,
        })

        # ── Checkpoint ────────────────────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            model.save(str(checkpoint_dir / "gnn_best.pth"))
            logger.info(f"  ✓ New best val F1={val_f1:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= settings.model.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        if args.dry_run and epoch >= 2:
            logger.info("Dry run complete — exiting early")
            break

    # ── Final evaluation on test set ──────────────────────────────────────────
    logger.info("Loading best model for test evaluation...")
    best_path = checkpoint_dir / "gnn_best.pth"
    if best_path.exists():
        model = BlockchainGNN.load(str(best_path), device=str(device))

    with torch.no_grad():
        _, test_prec, test_rec, test_f1, test_auc = run_epoch(
            model, test_loader, criterion, optimizer, scaler, device,
            1, is_train=False, dry_run=args.dry_run,
        )

    logger.info(
        f"\n{'='*50}\nTest Results\n{'='*50}\n"
        f"Precision: {test_prec:.4f}\n"
        f"Recall:    {test_rec:.4f}\n"
        f"F1-Score:  {test_f1:.4f}\n"
        f"AUC-ROC:   {test_auc:.4f}\n"
        f"{'='*50}"
    )

    # Save training history
    with open(checkpoint_dir / "gnn_training_history.json", "w") as f:
        json.dump({"history": history, "test": {
            "precision": test_prec, "recall": test_rec,
            "f1": test_f1, "auc": test_auc,
        }}, f, indent=2)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
