"""
Linear probe on a frozen contrastive backbone.

The contrastive loss never trains the backbone's classifier head, so after pretraining the head is
fit on labels. Two things matter for that to be a fair measurement of the frozen representation:

* The backbone must run in eval mode. Fitting the head with the backbone in train mode uses
  BatchNorm batch statistics (and keeps updating the running ones), but the checkpoint is scored with
  running statistics, so the head is trained on features it will never see at test time.
* Running statistics collected during contrastive training come from augmented views (masked,
  cropped, mutated). Recalibrate them on clean data before extracting features.

Embeddings of a frozen backbone are computed once and the head is trained on the cached tensors, so
an epoch costs seconds and many epochs are affordable.
"""

from __future__ import annotations

import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from metapathpredict.experiment_tracking import MetricsSink, NullSink

logger = logging.getLogger(__name__)


def recalibrate_batchnorm(backbone: nn.Module, loader, device, max_batches: int = 200) -> int:
    """Recompute BatchNorm running statistics of `backbone` over clean data (cumulative average).

    Returns the number of batches used. The module is left in eval mode.
    """
    norms = [m for m in backbone.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    if not norms:
        backbone.eval()
        return 0
    for m in norms:
        m.reset_running_stats()
        m.momentum = None  # cumulative moving average over the batches below
    backbone.train()
    used = 0
    with torch.no_grad():
        for batch in loader:
            backbone.get_embeddings(batch[0].to(device))
            used += 1
            if used >= max_batches:
                break
    for m in norms:
        m.momentum = 0.1
    backbone.eval()
    return used


def cache_embeddings(backbone: nn.Module, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Backbone embeddings and labels for a whole loader, on `device`. Backbone stays in eval mode."""
    backbone.eval()
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            feats.append(backbone.get_embeddings(batch[0].to(device)))
            labels.append(batch[1].to(device))
    return torch.cat(feats), torch.cat(labels)


def _reset_head(head: nn.Module) -> None:
    for m in head.modules():
        if hasattr(m, "reset_parameters") and m is not head:
            m.reset_parameters()


def fit_linear_probe(
    backbone: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    patience: int = 15,
    lr: float = 1e-3,
    batch_size: int = 256,
    recalibrate_batches: int = 200,
    reinit_head: bool = True,
    sink: MetricsSink | None = None,
) -> dict:
    """
    Fit `backbone.classifier` on the frozen backbone's clean, eval-mode embeddings.

    The head keeps the weights of the epoch with the best validation accuracy. Backbone parameters
    are never changed; only its BatchNorm running statistics are recalibrated (see module docstring).
    Returns {"best_val_acc", "best_epoch", "epochs_run", "history"}.
    """
    sink = sink or NullSink()
    head = backbone.classifier

    used = recalibrate_batchnorm(backbone, train_loader, device, recalibrate_batches)
    logger.info(f"  BatchNorm statistics recalibrated on {used} clean batches")
    x_train, y_train = cache_embeddings(backbone, train_loader, device)
    x_val, y_val = cache_embeddings(backbone, val_loader, device)
    logger.info(f"  Cached embeddings: train {tuple(x_train.shape)}, val {tuple(x_val.shape)}")

    if reinit_head:
        _reset_head(head)
    head.to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)

    best_acc, best_epoch, best_state, stale = -1.0, 0, None, 0
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(len(x_train), device=device)
        total_loss, correct = 0.0, 0
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            logits = head(x_train[idx])
            loss = F.cross_entropy(logits, y_train[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(dim=1) == y_train[idx]).sum().item()

        head.eval()
        with torch.no_grad():
            val_acc = (head(x_val).argmax(dim=1) == y_val).float().mean().item()
        row = {"epoch": epoch, "train_loss": total_loss / len(x_train),
               "train_acc": correct / len(x_train), "val_acc": val_acc}
        history.append(row)
        sink.log_metrics({"probe/train_loss": row["train_loss"], "probe/train_acc": row["train_acc"],
                          "probe/val_acc": val_acc}, step=epoch)
        logger.info(
            f"  Linear probe epoch {epoch}/{epochs}: train_loss={row['train_loss']:.4f}, "
            f"train_acc={row['train_acc']:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = copy.deepcopy(head.state_dict())
        else:
            stale += 1
            if stale >= patience:
                logger.info(f"  Probe early stopping: no val improvement for {patience} epochs")
                break

    head.load_state_dict(best_state)
    head.eval()
    backbone.eval()
    return {"best_val_acc": best_acc, "best_epoch": best_epoch, "epochs_run": len(history), "history": history}
