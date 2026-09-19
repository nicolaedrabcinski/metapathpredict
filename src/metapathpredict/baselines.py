"""
Reference points for the contrastive pipeline: k-mer composition and a plain cross-entropy CNN.

Without them a contrastive result has nothing to be compared to. On the species-disjoint taxa8 split a
gradient-boosted 4-mer model already reaches about the accuracy of the contrastive encoder, so the
deep model has to be judged against it.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metapathpredict.models.contrastive import ContrastiveAugmentation, reverse_complement

logger = logging.getLogger(__name__)


def kmer_frequencies(sequences: np.ndarray, k: int = 4, chunk: int = 20000) -> np.ndarray:
    """
    Normalised k-mer counts of one-hot sequences [n, 4, length] (channels A, C, G, T).

    Windows that contain an all-zero column (N) are skipped. Returns float32 [n, 4**k]; a row sums
    to 1 (or 0 if the sequence has no complete window).
    """
    sequences = np.asarray(sequences)
    n, _, length = sequences.shape
    windows = length - k + 1
    out = np.zeros((n, 4 ** k), dtype=np.float32)
    for start in range(0, n, chunk):
        block = sequences[start:start + chunk]
        m = block.shape[0]
        base = block.argmax(axis=1).astype(np.int64)
        base[block.sum(axis=1) == 0] = -1
        code = np.zeros((m, windows), dtype=np.int64)
        invalid = np.zeros((m, windows), dtype=bool)
        for j in range(k):
            column = base[:, j:j + windows]
            invalid |= column < 0
            code = code * 4 + np.where(column < 0, 0, column)
        flat = (code + np.arange(m)[:, None] * 4 ** k)[~invalid]
        counts = np.bincount(flat, minlength=m * 4 ** k).reshape(m, 4 ** k).astype(np.float32)
        out[start:start + m] = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    return out


def load_backbone_weights(model: nn.Module, checkpoint_path) -> int:
    """
    Load the backbone of a contrastive checkpoint into a ConfigurableCNN classifier, leaving its
    classifier head freshly initialised. Returns the number of tensors loaded; raises if the
    architectures do not match (a silent partial load would make the comparison meaningless).
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    prefix = "encoder."
    state = {k[len(prefix):]: v for k, v in checkpoint["encoder_state_dict"].items()
             if k.startswith(prefix) and not k.startswith(prefix + "classifier.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith("classifier.")]
    if missing or unexpected:
        raise ValueError(f"backbone mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")
    return len(state)


def _augment(x: torch.Tensor, mode: str, augmentation: ContrastiveAugmentation | None) -> torch.Tensor:
    if mode == "none":
        return x
    if mode == "rc":
        flip = torch.rand(x.size(0), device=x.device) < 0.5
        return torch.where(flip[:, None, None], reverse_complement(x), x)
    if mode == "full":
        return augmentation(x)[0]
    raise ValueError(f"augment must be none, rc or full, got {mode!r}")


def evaluate_accuracy(model: nn.Module, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    """Accuracy, targets and predictions of `model` (eval mode) over a loader."""
    model.eval()
    targets, preds = [], []
    with torch.no_grad():
        for batch in loader:
            preds.append(model(batch[0].to(device)).argmax(dim=1).cpu())
            targets.append(batch[1])
    targets, preds = torch.cat(targets).numpy(), torch.cat(preds).numpy()
    return float((targets == preds).mean()), targets, preds


def train_supervised(
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    epochs: int = 20,
    patience: int = 7,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    augment: str = "rc",
    augmentation: ContrastiveAugmentation | None = None,
    sink=None,
) -> dict:
    """
    Train `model` (a classifier: [batch, 4, length] -> logits) with cross-entropy and keep the weights
    of the epoch with the best validation accuracy. `augment`: none | rc (random reverse complement) |
    full (the contrastive augmentation, first view only). Returns {"best_val_acc", "best_epoch", "history"}.
    """
    if augment == "full" and augmentation is None:
        augmentation = ContrastiveAugmentation()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_epoch, best_state, stale, history = -1.0, 0, None, 0, []

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            logits = model(_augment(x, augment, augmentation))
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            seen += len(y)

        val_acc, _, _ = evaluate_accuracy(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": total / seen, "train_acc": correct / seen, "val_acc": val_acc}
        history.append(row)
        if sink is not None:
            sink.log_metrics({"supervised/train_loss": row["train_loss"], "supervised/train_acc": row["train_acc"],
                              "supervised/val_acc": val_acc}, step=epoch)
        logger.info(f"[Supervised] Epoch {epoch}/{epochs} | loss {row['train_loss']:.4f} | "
                    f"train acc {row['train_acc']:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc, best_epoch, stale = val_acc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if patience and stale >= patience:
                logger.info(f"Early stopping: val accuracy has not improved for {patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    return {"best_val_acc": best_acc, "best_epoch": best_epoch, "history": history}
